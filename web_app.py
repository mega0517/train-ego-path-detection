#!/usr/bin/env python3
"""
TEP-Net Train Ego-Path Detection — Web Application

A browser-based front end for model inference, mirroring the desktop
``gui_app.py`` (model/device/crop selection, side-by-side single-frame vs RNN
comparison) but served over HTTP so it can be used remotely without an X server.

Run::

    python web_app.py --host 0.0.0.0 --port 5000

then open http://<host>:5000 in a browser.
"""

import argparse
import base64
import hmac
import io
import os
import subprocess
import sys
import threading

import torch
import yaml
from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from PIL import Image

# Register HEIC/HEIF support (iPhone photos) with PIL when available.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from src.utils.autocrop import Autocropper
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_WEIGHTS_PATH = os.path.join(BASE_PATH, "egopath", "weights")
RNN_WEIGHTS_PATH = os.path.join(BASE_PATH, "egopathrnn", "weights")
DEFAULT_IMAGE = os.path.join(BASE_PATH, "data", "egopath.jpg")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi")
# Images the server can read from a folder directly (no upload).
FOLDER_IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + (".heic", ".heif", ".webp")


def list_folder_images(path):
    """Sorted image filenames in a server-side folder, or None if not a folder."""
    if not path or not os.path.isdir(path):
        return None
    return sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(FOLDER_IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(path, f))
    )

app = Flask(__name__)
# Videos can be large; allow up to 4 GB uploads. (Oversized uploads otherwise
# get the file part silently dropped, surfacing as a confusing "No video
# provided" 400.) Override via WEB_MAX_UPLOAD_MB.
_max_upload_mb = int(os.environ.get("WEB_MAX_UPLOAD_MB", "4096"))
app.config["MAX_CONTENT_LENGTH"] = _max_upload_mb * 1024 * 1024
# Pick up template edits without a server restart (debug mode is off).
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.errorhandler(413)
def _too_large(_e):
    return jsonify(
        {"error": f"Upload exceeds the {_max_upload_mb} MB limit. Increase it with "
                  "WEB_MAX_UPLOAD_MB, or trim/downscale the video."}
    ), 413

# --------------------------------------------------------------------------- #
# Optional HTTP Basic Auth
# --------------------------------------------------------------------------- #
# Credentials come from the environment (or --auth-user/--auth-pass), never
# hardcoded. Auth is enabled only when a non-empty password is configured, so
# trusted LAN use keeps working without credentials while a public Funnel/tunnel
# exposure can be locked down by setting WEB_AUTH_PASS.
AUTH_USER = os.environ.get("WEB_AUTH_USER", "admin")
AUTH_PASS = os.environ.get("WEB_AUTH_PASS", "")


def auth_enabled():
    return bool(AUTH_PASS)


def _credentials_ok(user, password):
    """Constant-time comparison to avoid leaking length/contents via timing."""
    if user is None or password is None:
        return False
    user_ok = hmac.compare_digest(user, AUTH_USER)
    pass_ok = hmac.compare_digest(password, AUTH_PASS)
    return user_ok and pass_ok


@app.before_request
def _require_auth():
    if not auth_enabled():
        return None
    auth = request.authorization
    if auth and _credentials_ok(auth.username, auth.password):
        return None
    return Response(
        "Authentication required.",
        401,
        {"WWW-Authenticate": 'Basic realm="TEP-Net"'},
    )

# Detectors are expensive to construct (model load), so cache them by
# (model_path, device). Crop coordinates and temporal state are reset per
# request, so caching is safe across requests. Guarded by a lock because Flask
# serves requests on multiple threads.
_detector_cache = {}
_detector_lock = threading.Lock()

# Uploaded videos are cached on disk, keyed by a client-supplied file identity
# (name:size:lastModified), so the client can re-run inference (different model/
# crop/device) without re-uploading — and this survives server restarts and
# browser reloads. Bounded to the most recent few; older files are evicted.
import glob as _glob
import hashlib as _hashlib

_VIDEO_CACHE_DIR = os.path.join(BASE_PATH, ".video_cache")
os.makedirs(_VIDEO_CACHE_DIR, exist_ok=True)
_video_cache_lock = threading.Lock()
_VIDEO_CACHE_MAX = int(os.environ.get("WEB_VIDEO_CACHE_MAX", "10"))


def _video_id_for(key):
    """Stable, filesystem-safe id for a client file key (name:size:lastModified)."""
    return _hashlib.sha1(key.encode("utf-8")).hexdigest()


def _cached_video_path(vid):
    """Return the on-disk path for a cached video id, or None."""
    if not vid:
        return None
    matches = _glob.glob(os.path.join(_VIDEO_CACHE_DIR, vid + ".*"))
    return matches[0] if matches else None


def _evict_old_videos():
    """Keep only the most-recently-modified _VIDEO_CACHE_MAX cached videos."""
    files = sorted(
        _glob.glob(os.path.join(_VIDEO_CACHE_DIR, "*")), key=os.path.getmtime
    )
    for old in files[:-_VIDEO_CACHE_MAX] if _VIDEO_CACHE_MAX > 0 else []:
        try:
            os.unlink(old)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Model / device discovery
# --------------------------------------------------------------------------- #
def list_dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def read_model_info(model_path):
    """Return a dict describing a model from its config.yaml, or None."""
    config_file = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_file):
        return None
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
    return {
        "method": config.get("method", "Unknown"),
        "backbone": config.get("backbone", "Unknown"),
        "temporal": bool(config.get("temporal", False)),
        "seq_len": config.get("seq_len"),
    }


def discover_models():
    """Discover available models as base/RNN pairs (mirrors the desktop GUI).

    Returns a list of entries::

        {"name", "base_path", "rnn_path", "base_info", "rnn_info"}

    where either *_path may be None when that variant is missing on disk.
    """
    base_models = [d for d in list_dirs(BASE_WEIGHTS_PATH) if not d.endswith("RNN")]
    rnn_models = {d for d in list_dirs(RNN_WEIGHTS_PATH) if d.endswith("RNN")}

    entries = []
    for name in base_models:
        base_path = os.path.join(BASE_WEIGHTS_PATH, name)
        rnn_name = name + "RNN"
        rnn_path = os.path.join(RNN_WEIGHTS_PATH, rnn_name) if rnn_name in rnn_models else None
        entries.append(
            {
                "name": name,
                "base_path": base_path,
                "rnn_path": rnn_path,
                "base_info": read_model_info(base_path),
                "rnn_info": read_model_info(rnn_path) if rnn_path else None,
            }
        )

    # Standalone RNN models without a matching base model.
    for rnn_name in sorted(rnn_models):
        if rnn_name[:-3] in base_models:
            continue
        rnn_path = os.path.join(RNN_WEIGHTS_PATH, rnn_name)
        entries.append(
            {
                "name": rnn_name,
                "base_path": None,
                "rnn_path": rnn_path,
                "base_info": None,
                "rnn_info": read_model_info(rnn_path),
            }
        )
    return entries


def model_paths_for(name):
    """Resolve (single_frame_path, rnn_path) for a selected model name."""
    for entry in discover_models():
        if entry["name"] == name:
            return entry["base_path"], entry["rnn_path"]
    return None, None


def available_devices():
    devices = ["cpu"]
    try:
        if torch.cuda.is_available():
            devices.append("cuda")
            devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 - CUDA enumeration can fail on a bad env
        pass
    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:  # noqa: BLE001
        pass
    return devices


def device_status():
    try:
        if torch.cuda.is_available():
            return f"✓ CUDA available: {torch.cuda.get_device_name(0)}"
        if torch.backends.mps.is_available():
            return "✓ MPS (Metal) available"
    except Exception as e:  # noqa: BLE001
        return f"CUDA detected but unavailable ({str(e).splitlines()[0]}); using CPU"
    return "Using CPU"


# --------------------------------------------------------------------------- #
# Detector handling
# --------------------------------------------------------------------------- #
def get_detector(model_path, device):
    """Return a cached Detector for (model_path, device), constructing if needed."""
    key = (model_path, device)
    with _detector_lock:
        det = _detector_cache.get(key)
        if det is None:
            det = Detector(
                model_path=model_path,
                crop_coords=None,
                runtime="pytorch",
                device=device,
            )
            _detector_cache[key] = det
        return det


def build_crop_coords(mode, coords, config):
    """Translate a web crop selection into what Detector.crop_coords expects."""
    if mode == "auto":
        return Autocropper(config)
    if mode == "manual" and coords:
        return tuple(int(c) for c in coords)
    return None


def prepare_detector(model_path, device, crop_mode, crop_coords):
    """Fetch a detector and reset its per-inference state (crop + temporal)."""
    det = get_detector(model_path, device)
    det.crop_coords = build_crop_coords(crop_mode, crop_coords, det.config)
    det.reset_temporal()
    return det


def run_both(img, det_single, det_rnn):
    """Compute (single_frame_vis, rnn_vis) PIL images, mirroring the desktop GUI."""
    # Models expect 3-channel RGB; uploads may be RGBA (PNG w/ alpha), L, etc.
    if img.mode != "RGB":
        img = img.convert("RGB")
    single_vis = None
    rnn_vis = None
    if det_single is not None:
        crop_s = det_single.get_crop_coords()
        single_vis = draw_egopath(img, det_single.detect(img), crop_coords=crop_s)
        if det_rnn is not None:
            crop_r = det_rnn.get_crop_coords()
            rnn_vis = draw_egopath(img, det_rnn.detect(img), crop_coords=crop_r)
    elif det_rnn is not None:
        # No standalone single-frame model: derive it from the RNN's base net.
        crop_r = det_rnn.get_crop_coords()
        single_res, rnn_res = det_rnn.detect_pair(img)
        single_vis = draw_egopath(img, single_res, crop_coords=crop_r)
        if rnn_res is not None:
            rnn_vis = draw_egopath(img, rnn_res, crop_coords=crop_r)
    return single_vis, rnn_vis


def detectors_for_request(model_name, device, crop_mode, crop_coords):
    """Build the (single, rnn) detector pair for a request, or raise ValueError."""
    single_path, rnn_path = model_paths_for(model_name)
    if not single_path and not rnn_path:
        raise ValueError(f"Unknown model: {model_name}")

    det_single = None
    det_rnn = None
    if single_path and os.path.exists(single_path):
        det_single = prepare_detector(single_path, device, crop_mode, crop_coords)
    if rnn_path and os.path.exists(rnn_path):
        det_rnn = prepare_detector(rnn_path, device, crop_mode, crop_coords)
    if det_single is None and det_rnn is None:
        raise ValueError("No usable model checkpoint found")
    return det_single, det_rnn


def pil_to_data_uri(img, fmt="JPEG", quality=90):
    """Encode a PIL image as a base64 data URI for inline display."""
    if img is None:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"


def downscale_to_width(img, max_w):
    """Return (image, scale) downscaled so width <= max_w (preserving aspect).

    Cheap when no resize is needed. Cutting the working resolution speeds up
    preprocessing, drawing, JPEG encoding and network transfer all at once —
    the dominant per-frame costs for high-resolution (e.g. 4K) video.
    """
    if not max_w or img.width <= max_w:
        return img, 1.0
    scale = max_w / img.width
    return img.resize((max_w, max(1, round(img.height * scale)))), scale


def decode_video_frames(path):
    """Yield (frame_index, PIL.Image RGB, total_or_None) for each video frame.

    Tries OpenCV first (fast, hardware-friendly). If OpenCV cannot open the file
    or decodes zero frames — typically because its bundled FFmpeg lacks the codec
    (e.g. H.265/HEVC) — falls back to PyAV, which ships a fuller FFmpeg. Raises
    ValueError with actionable guidance when neither decoder can read the file.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), total
            idx += 1
        cap.release()
        if idx > 0:
            return  # OpenCV handled it
    else:
        cap.release()

    # Fallback decoder for codecs OpenCV's FFmpeg can't handle.
    try:
        import av
    except ImportError:
        raise ValueError(
            "Video codec unsupported by OpenCV and the PyAV fallback is not "
            "installed (pip install av). Re-encode to H.264 mp4: "
            "ffmpeg -i input -c:v libx264 -pix_fmt yuv420p out.mp4"
        )

    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            total = stream.frames or None
            idx = 0
            for frame in container.decode(stream):
                yield idx, frame.to_image(), total
                idx += 1
    except (av.error.FFmpegError, IndexError) as e:
        raise ValueError(
            f"Could not decode the video ({e}). The file may be corrupt or use an "
            "unsupported codec. Re-encode to H.264 mp4: "
            "ffmpeg -i input -c:v libx264 -pix_fmt yuv420p out.mp4"
        )
    if idx == 0:
        raise ValueError(
            "No frames could be decoded from the video (file may be corrupt or "
            "an unsupported codec). Re-encode to H.264 mp4: "
            "ffmpeg -i input -c:v libx264 -pix_fmt yuv420p out.mp4"
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    return jsonify(
        {
            "models": discover_models(),
            "devices": available_devices(),
            "device_status": device_status(),
            "default_device": "cuda:0" if "cuda:0" in available_devices() else "cpu",
            "has_default_image": os.path.exists(DEFAULT_IMAGE),
        }
    )


def _parse_crop(form):
    """Pull crop mode + coords from a request form."""
    mode = (form.get("crop_mode") or "none").lower()
    coords = None
    if mode == "manual":
        coords = (
            form.get("crop_left", 0),
            form.get("crop_top", 0),
            form.get("crop_right", 0),
            form.get("crop_bottom", 0),
        )
    return mode, coords


@app.route("/api/infer_image", methods=["POST"])
def api_infer_image():
    """Run inference on a single uploaded image (or the bundled default)."""
    model_name = request.form.get("model")
    device = request.form.get("device", "cpu")
    crop_mode, crop_coords = _parse_crop(request.form)

    try:
        upload = request.files.get("file")
        if upload is not None and upload.filename:
            try:
                img = Image.open(upload.stream)
                img.load()
            except Exception as e:  # noqa: BLE001 - PIL can't identify/decode the file
                ext = os.path.splitext(upload.filename)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTENSIONS or ext in (".mov", ".mkv", ".webm", ".m4v"):
                    return jsonify(
                        {"error": f"'{upload.filename}' looks like a video, not an image. "
                                  "Use a video file (it will be processed frame by frame)."}
                    ), 400
                # A truncated/partial upload of a real image fails here too — give
                # the right hint instead of blaming the format.
                if "truncated" in str(e).lower() or ext in SUPPORTED_IMAGE_EXTENSIONS:
                    return jsonify(
                        {"error": f"'{upload.filename}' did not arrive intact (upload may have "
                                  "been cut off — common on unstable connections). Try uploading "
                                  "it again, ideally on a stable network."}
                    ), 400
                return jsonify(
                    {"error": f"Could not read '{upload.filename}' as an image. Supported: "
                              "JPG, PNG, BMP, TIFF, HEIC. Convert other formats first."}
                ), 400
        elif os.path.exists(DEFAULT_IMAGE):
            img = Image.open(DEFAULT_IMAGE)
            img.load()
        else:
            return jsonify({"error": "No image provided"}), 400

        det_single, det_rnn = detectors_for_request(
            model_name, device, crop_mode, crop_coords
        )
        single_vis, rnn_vis = run_both(img, det_single, det_rnn)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 - surface any inference failure to the UI
        return jsonify({"error": f"Inference error: {e}"}), 500

    return jsonify(
        {
            "original": pil_to_data_uri(img),
            "single": pil_to_data_uri(single_vis),
            "rnn": pil_to_data_uri(rnn_vis),
            "has_rnn": det_rnn is not None,
            "single_is_fallback": det_single is None and det_rnn is not None,
        }
    )


@app.route("/api/video_cached")
def api_video_cached():
    """Tell the client whether a file (by its key) is already cached server-side,
    so it can skip re-uploading."""
    key = (request.args.get("key") or "").strip()
    vid = _video_id_for(key) if key else None
    return jsonify({"cached": bool(_cached_video_path(vid)), "video_id": vid})


@app.route("/api/infer_video", methods=["POST"])
def api_infer_video():
    """Run inference on an uploaded video, streaming per-frame results via SSE."""
    model_name = request.form.get("model")
    device = request.form.get("device", "cpu")
    crop_mode, crop_coords = _parse_crop(request.form)
    stride = max(1, int(request.form.get("stride", 1)))
    # Cap the working resolution for speed (0 = keep original). The client may
    # override; default from WEB_MAX_PROC_WIDTH.
    try:
        proc_width = int(request.form.get("proc_width", os.environ.get("WEB_MAX_PROC_WIDTH", "1280")))
    except ValueError:
        proc_width = 1280

    upload = request.files.get("file")
    video_key = (request.form.get("video_key") or "").strip()
    vid = _video_id_for(video_key) if video_key else None

    if upload is not None and upload.filename:
        # New upload: persist into the on-disk cache keyed by the client's file
        # identity so future runs (even after a restart) can reuse it.
        suffix = os.path.splitext(upload.filename)[1] or ".mp4"
        if vid:
            old = _cached_video_path(vid)
            if old:
                try:
                    os.unlink(old)
                except OSError:
                    pass
            video_path = os.path.join(_VIDEO_CACHE_DIR, vid + suffix)
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=_VIDEO_CACHE_DIR
            )
            video_path = tmp.name
            tmp.close()

        upload.save(video_path)
        if os.path.getsize(video_path) == 0:
            try:
                os.unlink(video_path)
            except OSError:
                pass
            return jsonify(
                {"error": "Upload was empty (0 bytes). The file failed to upload — "
                          "check your connection/file and try again."}
            ), 400
        with _video_cache_lock:
            _evict_old_videos()
        video_id = vid or os.path.basename(video_path)
        reused = False
    elif vid:
        # Re-run without re-uploading: look up the previously cached file.
        video_path = _cached_video_path(vid)
        if not video_path or not os.path.exists(video_path):
            return jsonify(
                {"error": "Cached video is no longer available on the server — "
                          "please upload the file again."}
            ), 410
        video_id = vid
        reused = True
    else:
        return jsonify({"error": "No video provided"}), 400

    saved_mb = os.path.getsize(video_path) / (1024 * 1024)

    def event_stream():
        import json
        import time

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        t0 = time.time()
        try:
            first_msg = (
                f"Using cached video ({saved_mb:.1f} MB) · loading model…"
                if reused else
                f"Received {saved_mb:.1f} MB · loading model…"
            )
            yield sse({"type": "status", "message": first_msg, "video_id": video_id})
            det_single, det_rnn = detectors_for_request(
                model_name, device, crop_mode, crop_coords
            )
            yield sse({"type": "status", "message": "Model ready · decoding video…"})
            emitted = 0
            crop_scaled = False
            for frame_idx, img, total in decode_video_frames(video_path):
                if frame_idx % stride == 0:
                    # Downscale the working frame for speed (preprocessing,
                    # drawing, JPEG encoding and transfer all scale with pixels).
                    proc, scale = downscale_to_width(img, proc_width)
                    # Manual crop coords are in original pixels — scale them once
                    # to match the downscaled frame.
                    if scale != 1.0 and crop_mode == "manual" and not crop_scaled:
                        for det in (det_single, det_rnn):
                            if det is not None and isinstance(det.crop_coords, tuple):
                                det.crop_coords = tuple(round(c * scale) for c in det.crop_coords)
                        crop_scaled = True
                    single_vis, rnn_vis = run_both(proc, det_single, det_rnn)
                    payload = {
                        "type": "frame",
                        "frame": frame_idx,
                        "total": total or 0,
                        "emitted": emitted + 1,
                        "elapsed": round(time.time() - t0, 1),
                        "original": pil_to_data_uri(proc, quality=80),
                        "single": pil_to_data_uri(single_vis, quality=80),
                        "rnn": pil_to_data_uri(rnn_vis, quality=80),
                    }
                    yield sse(payload)
                    emitted += 1
            yield sse({"type": "done", "frames": emitted,
                       "elapsed": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "moov atom not found" in msg:
                msg = (
                    f"The uploaded file is incomplete or truncated (received "
                    f"{saved_mb:.1f} MB; its mp4 index 'moov atom' is missing). "
                    "This means the video was cut off — either the source "
                    "recording was interrupted, or the upload did not finish. "
                    "Check that the received size matches your original file, "
                    "re-upload over a stable connection, or repair/re-export the "
                    "video (e.g. ffmpeg -i input -c copy -movflags faststart out.mp4)."
                )
            yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
            # On decode failure, drop the bad file from the cache so a re-upload
            # is required (a truncated file won't get better on re-run).
            try:
                os.unlink(video_path)
            except OSError:
                pass
        # NOTE: on success the file is intentionally kept in _video_cache so the
        # client can re-run (different model/crop/device) without re-uploading.

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_BROWSE_ROOT = os.environ.get("WEB_BROWSE_ROOT", "/data3/bhkim/datasets")


@app.route("/api/browse")
def api_browse():
    """List subfolders (with image counts) for a server-side folder browser,
    clamped within WEB_BROWSE_ROOT so callers can't escape it."""
    root = os.path.realpath(_BROWSE_ROOT)
    req = (request.args.get("path") or "").strip()
    cur = os.path.realpath(req) if req else root
    if cur != root and not cur.startswith(root + os.sep):
        cur = root  # clamp back into the allowed root
    if not os.path.isdir(cur):
        return jsonify({"ok": False, "error": f"Not a folder: {cur}"}), 400

    dirs = []
    try:
        for name in sorted(os.listdir(cur)):
            full = os.path.join(cur, name)
            if os.path.isdir(full):
                dirs.append({"name": name, "path": full,
                             "images": len(list_folder_images(full) or [])})
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    parent = os.path.dirname(cur)
    if cur == root or not (parent == root or parent.startswith(root + os.sep)):
        parent = None
    return jsonify({
        "ok": True, "path": cur, "parent": parent, "root": root,
        "dirs": dirs, "images": len(list_folder_images(cur) or []),
    })


@app.route("/api/list_server_folder")
def api_list_server_folder():
    """Count image files in a server-side folder (so the client can run inference
    on them without uploading anything)."""
    path = (request.args.get("path") or "").strip()
    files = list_folder_images(path)
    if files is None:
        return jsonify({"ok": False, "error": f"Not a folder: {path or '(empty)'}"}), 400
    return jsonify({"ok": True, "path": path, "count": len(files), "sample": files[:20]})


_UPLOAD_ROOT = os.environ.get("WEB_UPLOAD_ROOT", "/data3/bhkim/datasets")


def _safe_dest_dir(subdir):
    """Resolve a destination folder strictly under _UPLOAD_ROOT, or None.

    Rejects absolute paths, '..', and anything that would escape the root.
    """
    subdir = (subdir or "").strip().strip("/")
    if not subdir:
        return None
    parts = []
    for p in subdir.split("/"):
        p = p.strip()
        if p in ("", ".", "..") or "\\" in p:
            return None
        parts.append(p)
    dest = os.path.realpath(os.path.join(_UPLOAD_ROOT, *parts))
    root = os.path.realpath(_UPLOAD_ROOT)
    if dest != root and not dest.startswith(root + os.sep):
        return None
    return dest


@app.route("/api/upload_to_server", methods=["POST"])
def api_upload_to_server():
    """Save uploaded files into a subfolder under WEB_UPLOAD_ROOT (no SSH needed),
    so they can then be processed with the no-upload server-folder mode."""
    from werkzeug.utils import secure_filename

    dest = _safe_dest_dir(request.form.get("subdir"))
    if dest is None:
        return jsonify({"error": "Invalid destination folder name."}), 400
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided."}), 400

    os.makedirs(dest, exist_ok=True)
    saved, skipped = 0, []
    for f in files:
        if not f.filename:
            continue
        name = secure_filename(os.path.basename(f.filename)) or None
        if not name:
            skipped.append(f.filename)
            continue
        out = os.path.join(dest, name)
        f.save(out)
        if os.path.getsize(out) == 0:
            os.unlink(out)
            skipped.append(name)
            continue
        saved += 1
    return jsonify({"ok": True, "path": dest, "saved": saved, "skipped": len(skipped)})


def _under_allowed_root(path):
    """Realpath of *path* if it sits within the browse or upload root, else None."""
    rp = os.path.realpath(path)
    for r in (os.path.realpath(_BROWSE_ROOT), os.path.realpath(_UPLOAD_ROOT)):
        if rp == r or rp.startswith(r + os.sep):
            return rp
    return None


@app.route("/api/optimize_folder", methods=["POST"])
def api_optimize_folder():
    """Convert a folder of (often huge PNG/RGBA) images to resized JPEGs in a
    sibling '<name>_opt' folder, so later inference reads small files fast.
    Streams per-image progress via SSE."""
    src = (request.form.get("folder") or "").strip()
    try:
        max_w = int(request.form.get("max_width", "1920"))
    except ValueError:
        max_w = 1920

    src_real = _under_allowed_root(src) if src else None
    if src_real is None or not os.path.isdir(src_real):
        return jsonify({"error": "Folder must be inside the allowed data root."}), 400
    files = list_folder_images(src_real)
    if not files:
        return jsonify({"error": "No images found in that folder."}), 400
    dest = src_real.rstrip("/") + "_opt"
    if _under_allowed_root(dest) is None:
        return jsonify({"error": "Destination outside allowed root."}), 400

    def event_stream():
        import json
        import time

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        t0 = time.time()
        try:
            os.makedirs(dest, exist_ok=True)
            yield sse({"type": "status",
                       "message": f"Optimizing {len(files)} images → {os.path.basename(dest)}"})
            done = 0
            for i, name in enumerate(files):
                try:
                    im = Image.open(os.path.join(src_real, name))
                    im.load()
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    if max_w and im.width > max_w:
                        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))))
                    out = os.path.join(dest, os.path.splitext(name)[0] + ".jpg")
                    im.save(out, "JPEG", quality=90)
                    done += 1
                except Exception as e:  # noqa: BLE001
                    yield sse({"type": "image_error", "name": name, "error": str(e)})
                yield sse({"type": "progress", "index": i, "total": len(files),
                           "name": name, "elapsed": round(time.time() - t0, 1)})
            yield sse({"type": "done", "count": done, "total": len(files),
                       "dest": dest, "elapsed": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "error": str(e)})

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/infer_folder", methods=["POST"])
def api_infer_folder():
    """Run inference on every image in a server-side folder, streaming per-image
    results via SSE. No upload — the server reads the files directly."""
    model_name = request.form.get("model")
    device = request.form.get("device", "cpu")
    crop_mode, crop_coords = _parse_crop(request.form)
    folder = (request.form.get("folder") or "").strip()
    try:
        proc_width = int(request.form.get("proc_width", os.environ.get("WEB_MAX_PROC_WIDTH", "1280")))
    except ValueError:
        proc_width = 1280

    files = list_folder_images(folder)
    if files is None:
        return jsonify({"error": f"Not a folder: {folder or '(empty)'}"}), 400
    if not files:
        return jsonify({"error": "No image files found in that folder."}), 400
    workers = max(1, int(os.environ.get("WEB_DECODE_WORKERS", "6")))

    def event_stream():
        import json
        import time
        from concurrent.futures import ThreadPoolExecutor

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        def decode_one(name):
            """Read + decode + downscale a frame (runs in worker threads; PIL
            releases the GIL during decode so this parallelises real work)."""
            img = Image.open(os.path.join(folder, name))
            img.load()
            return downscale_to_width(img, proc_width)

        t0 = time.time()
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            yield sse({"type": "status",
                       "message": f"Found {len(files)} images · loading model…"})
            det_single, det_rnn = detectors_for_request(
                model_name, device, crop_mode, crop_coords
            )
            yield sse({"type": "status",
                       "message": f"Model ready · processing ({workers} decode workers)…"})

            # Decode-ahead: keep a sliding window of images decoding in worker
            # threads while the GPU (serialized here) processes the current one.
            window = workers + 2
            futures = {}
            submitted = 0

            def fill(upto):
                nonlocal submitted
                while submitted < len(files) and submitted <= upto + window:
                    futures[submitted] = ex.submit(decode_one, files[submitted])
                    submitted += 1

            fill(0)
            crop_scaled = False
            done = 0
            for i, name in enumerate(files):
                fut = futures.pop(i)
                fill(i)  # refill the window while we run inference + encode
                try:
                    proc, scale = fut.result()
                    if scale != 1.0 and crop_mode == "manual" and not crop_scaled:
                        for det in (det_single, det_rnn):
                            if det is not None and isinstance(det.crop_coords, tuple):
                                det.crop_coords = tuple(round(c * scale) for c in det.crop_coords)
                        crop_scaled = True
                    for det in (det_single, det_rnn):  # treat each image independently
                        if det is not None:
                            det.reset_temporal()
                    single_vis, rnn_vis = run_both(proc, det_single, det_rnn)
                    yield sse({
                        "type": "image", "index": i, "total": len(files), "name": name,
                        "elapsed": round(time.time() - t0, 1),
                        "original": pil_to_data_uri(proc, quality=80),
                        "single": pil_to_data_uri(single_vis, quality=80),
                        "rnn": pil_to_data_uri(rnn_vis, quality=80),
                        "has_rnn": det_rnn is not None,
                        "single_is_fallback": det_single is None and det_rnn is not None,
                    })
                    done += 1
                except Exception as e:  # noqa: BLE001 - report and continue
                    yield sse({"type": "image_error", "index": i, "total": len(files),
                               "name": name, "error": str(e)})
            yield sse({"type": "done", "count": done, "total": len(files),
                       "elapsed": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "error": str(e)})
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Ego-path labeling (create training annotations in the browser)
# --------------------------------------------------------------------------- #
_LABEL_DISPLAY_W = 1280  # images are sent downscaled to this width for labeling


def _annots_path_ok(path):
    """Annotations JSON must sit under the workspace or a data root (writable)."""
    rp = os.path.realpath(path)
    roots = [os.path.realpath(BASE_PATH),
             os.path.realpath(_BROWSE_ROOT), os.path.realpath(_UPLOAD_ROOT)]
    return any(rp == r or rp.startswith(r + os.sep) for r in roots)


def _load_annotations(path):
    if path and os.path.isfile(path):
        try:
            import json
            with open(path) as f:
                return json.load(f) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


@app.route("/api/label/list")
def api_label_list():
    folder = (request.args.get("folder") or "").strip()
    annots = (request.args.get("annots") or "").strip()
    files = list_folder_images(folder)
    if files is None:
        return jsonify({"ok": False, "error": f"Not a folder: {folder or '(empty)'}"}), 400
    labeled = set(_load_annotations(annots).keys())
    return jsonify({"ok": True, "count": len(files),
                    "images": [{"name": n, "labeled": n in labeled} for n in files]})


@app.route("/api/label/image")
def api_label_image():
    folder = (request.args.get("folder") or "").strip()
    name = os.path.basename((request.args.get("name") or "").strip())
    fp = os.path.join(folder, name)
    if not name or not os.path.isfile(fp):
        return jsonify({"error": "Image not found."}), 400
    try:
        img = Image.open(fp)
        img.load()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Could not read image: {e}"}), 400
    if img.mode != "RGB":
        img = img.convert("RGB")
    ow, oh = img.size
    disp, scale = downscale_to_width(img, _LABEL_DISPLAY_W)
    annot = _load_annotations((request.args.get("annots") or "").strip()).get(name)
    return jsonify({
        "name": name, "orig_w": ow, "orig_h": oh,
        "disp_w": disp.size[0], "disp_h": disp.size[1],
        "data_uri": pil_to_data_uri(disp, quality=85),
        "annotation": annot,
    })


@app.route("/api/label/save", methods=["POST"])
def api_label_save():
    import json
    annots = (request.form.get("annots") or "").strip()
    name = os.path.basename((request.form.get("name") or "").strip())
    if not annots or not name:
        return jsonify({"error": "Missing annotations path or image name."}), 400
    if not annots.lower().endswith(".json") or not _annots_path_ok(annots):
        return jsonify({"error": "Annotations must be a .json under the workspace/data root."}), 400
    try:
        left = json.loads(request.form.get("left_rail") or "[]")
        right = json.loads(request.form.get("right_rail") or "[]")
    except ValueError:
        return jsonify({"error": "Invalid rail coordinates."}), 400
    if len(left) < 2 or len(right) < 2:
        return jsonify({"error": "Each rail needs at least 2 points."}), 400

    data = _load_annotations(annots)
    data[name] = {
        "left_rail": [[int(x), int(y)] for x, y in left],
        "right_rail": [[int(x), int(y)] for x, y in right],
    }
    os.makedirs(os.path.dirname(annots) or ".", exist_ok=True)
    tmp = annots + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, annots)
    return jsonify({"ok": True, "labeled_count": len(data)})


@app.route("/api/label/delete", methods=["POST"])
def api_label_delete():
    import json
    annots = (request.form.get("annots") or "").strip()
    name = os.path.basename((request.form.get("name") or "").strip())
    if not _annots_path_ok(annots):
        return jsonify({"error": "Invalid annotations path."}), 400
    data = _load_annotations(annots)
    if name in data:
        del data[name]
        tmp = annots + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, annots)
    return jsonify({"ok": True, "labeled_count": len(data)})


# --------------------------------------------------------------------------- #
# RNN transfer-learning training
# --------------------------------------------------------------------------- #
_train_lock = threading.Lock()
_train = {"proc": None, "log": None, "args": None, "output": None}


def list_full_gpus():
    """Non-MIG GPUs as [{index, uuid, mem_used}], for the training device picker."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,mig.mode.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[3] != "Enabled":
            gpus.append({"index": int(parts[0]), "uuid": parts[1], "mem_used": int(parts[2])})
    return gpus


@app.route("/api/train/options")
def api_train_options():
    bases = [
        {"name": e["name"], **(e["base_info"] or {})}
        for e in discover_models() if e["base_path"] and e["base_info"]
    ]
    images_path = annotations_path = ""
    try:
        with open(os.path.join(BASE_PATH, "configs", "global.yaml")) as f:
            g = yaml.safe_load(f) or {}
        images_path = g.get("images_path", "")
        annotations_path = g.get("annotations_path", "")
    except Exception:  # noqa: BLE001
        pass
    return jsonify({
        "base_models": bases,
        "gpus": list_full_gpus(),
        "defaults": {
            "epochs": 50, "learning_rate": 0.0001, "batch_size": 32,
            "images_path": images_path, "annotations_path": annotations_path,
        },
    })


@app.route("/api/train/status")
def api_train_status():
    p = _train.get("proc")
    return jsonify({
        "running": bool(p and p.poll() is None),
        "args": _train.get("args"),
        "output": _train.get("output"),
        "returncode": (p.returncode if p and p.poll() is not None else None),
    })


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    with _train_lock:
        p = _train.get("proc")
        if p and p.poll() is None:
            return jsonify({"error": "A training run is already in progress."}), 409

        base = (request.form.get("base_model") or "").strip()
        entry = next((e for e in discover_models()
                      if e["name"] == base and e["base_path"]), None)
        if entry is None:
            return jsonify({"error": f"Unknown base model: {base}"}), 400
        info = entry["base_info"] or {}
        method = info.get("method", "regression")
        backbone = info.get("backbone", "resnet18")
        try:
            epochs = int(request.form.get("epochs", 50))
            lr = float(request.form.get("learning_rate", 0.0001))
            batch = int(request.form.get("batch_size", 32))
        except ValueError:
            return jsonify({"error": "Invalid hyperparameter value."}), 400
        finetune = request.form.get("finetune_base") == "true"
        gpu_pre = request.form.get("gpu_preprocess") == "true"
        gpu_uuid = (request.form.get("gpu_uuid") or "").strip()

        images_path = (request.form.get("images_path") or "").strip()
        annotations_path = (request.form.get("annotations_path") or "").strip()
        if images_path and not os.path.isdir(images_path):
            return jsonify({"error": f"Images folder not found: {images_path}"}), 400
        if annotations_path:
            if not os.path.isfile(annotations_path):
                return jsonify({"error": f"Annotations file not found: {annotations_path}"}), 400
            if not annotations_path.lower().endswith(".json"):
                return jsonify({"error": "Annotations must be a .json file."}), 400

        cmd = [sys.executable, "train.py", method, backbone,
               "--temporal", "--base-model", base, "--device", "cuda:0",
               "--epochs", str(epochs), "--learning-rate", str(lr),
               "--batch-size", str(batch)]
        if images_path:
            cmd += ["--images-path", images_path]
        if annotations_path:
            cmd += ["--annotations-path", annotations_path]
        if finetune:
            cmd.append("--finetune-base")
        # GPU preprocessing is only implemented for the regression method.
        if gpu_pre and method == "regression":
            cmd.append("--gpu-preprocess")

        env = os.environ.copy()
        if gpu_uuid:
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid  # pin training to one full GPU
        log_path = os.path.join(BASE_PATH, "train_web.log")
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=BASE_PATH, stdout=logf,
                                stderr=subprocess.STDOUT, env=env)
        _train.update({
            "proc": proc, "log": log_path, "logf": logf,
            "output": f"{base}RNN",
            "args": {"base": base, "method": method, "backbone": backbone,
                     "epochs": epochs, "lr": lr, "batch": batch,
                     "finetune": finetune, "gpu_preprocess": gpu_pre},
        })
    return jsonify({"ok": True, "output": f"{base}RNN", "cmd": " ".join(cmd)})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    p = _train.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        return jsonify({"ok": True, "stopped": True})
    return jsonify({"ok": True, "stopped": False})


@app.route("/api/train/stream")
def api_train_stream():
    log_path = _train.get("log")
    if not log_path or not os.path.exists(log_path):
        return jsonify({"error": "No training run yet."}), 400

    def event_stream():
        import json
        import re
        import time

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        epoch_re = re.compile(
            r"EPOCH (\d+)/(\d+) \| TRAIN LOSS: ([0-9.]+) \| VAL LOSS: ([0-9.]+)")
        with open(log_path) as f:
            while True:
                line = f.readline()
                if line:
                    m = epoch_re.search(line)
                    if m:
                        yield sse({"type": "epoch", "epoch": int(m.group(1)),
                                   "total": int(m.group(2)),
                                   "train_loss": float(m.group(3)),
                                   "val_loss": float(m.group(4))})
                    elif line.strip():
                        yield sse({"type": "log", "line": line.strip()})
                else:
                    p = _train.get("proc")
                    if p is None or p.poll() is not None:
                        for rl in f.read().splitlines():
                            if rl.strip():
                                yield sse({"type": "log", "line": rl.strip()})
                        yield sse({"type": "done",
                                   "code": p.returncode if p else None})
                        break
                    time.sleep(0.5)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    parser = argparse.ArgumentParser(description="TEP-Net web inference app")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument(
        "--auth-user",
        default=None,
        help="HTTP Basic Auth username (overrides WEB_AUTH_USER; default 'admin')",
    )
    parser.add_argument(
        "--auth-pass",
        default=None,
        help="HTTP Basic Auth password (overrides WEB_AUTH_PASS). "
        "Auth is enabled only when a non-empty password is set.",
    )
    args = parser.parse_args()

    global AUTH_USER, AUTH_PASS
    if args.auth_user is not None:
        AUTH_USER = args.auth_user
    if args.auth_pass is not None:
        AUTH_PASS = args.auth_pass

    if auth_enabled():
        print(f"HTTP Basic Auth ENABLED (user: {AUTH_USER!r})")
    else:
        print(
            "HTTP Basic Auth DISABLED — no password set. "
            "Set WEB_AUTH_PASS (or --auth-pass) before exposing this server publicly."
        )

    # threaded=True so the long-lived SSE video stream doesn't block other requests.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
