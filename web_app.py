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
from flask import Flask, g, jsonify, render_template, request, Response, send_file, stream_with_context
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
AUTH_USER = os.environ.get("WEB_AUTH_USER") or "admin"  # empty env var -> default
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


def _auth_cookie_token():
    """Stable session token derived from the credentials; rotating the password
    invalidates it. Lets the browser authenticate subsequent requests (images,
    SSE, fetches) via cookie so it never re-prompts — iOS Safari in particular
    asks for Basic credentials a second time on some subresource requests."""
    import hashlib

    return hashlib.sha256(f"tepnet-cookie:{AUTH_USER}:{AUTH_PASS}".encode()).hexdigest()


@app.before_request
def _require_auth():
    if not auth_enabled():
        return None
    token = request.cookies.get("tep_auth")
    if token and hmac.compare_digest(token, _auth_cookie_token()):
        return None
    auth = request.authorization
    if auth and _credentials_ok(auth.username, auth.password):
        g.set_auth_cookie = True
        return None
    return Response(
        "Authentication required.",
        401,
        {"WWW-Authenticate": 'Basic realm="TEP-Net"'},
    )


@app.after_request
def _issue_auth_cookie(resp):
    if auth_enabled() and getattr(g, "set_auth_cookie", False):
        resp.set_cookie(
            "tep_auth",
            _auth_cookie_token(),
            max_age=30 * 24 * 3600,
            httponly=True,
            samesite="Lax",
        )
    return resp

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
    """Compute (single_frame_vis, rnn_vis, timing) mirroring the desktop GUI.
    timing = {"single_ms", "rnn_ms"} is the per-method detection wall-time (ms),
    for comparing single-frame vs RNN inference speed."""
    import time
    # Models expect 3-channel RGB; uploads may be RGBA (PNG w/ alpha), L, etc.
    if img.mode != "RGB":
        img = img.convert("RGB")
    single_vis = None
    rnn_vis = None
    timing = {"single_ms": None, "rnn_ms": None}
    if det_single is not None:
        crop_s = det_single.get_crop_coords()
        t = time.perf_counter()
        res_s = det_single.detect(img)
        timing["single_ms"] = round((time.perf_counter() - t) * 1000, 1)
        single_vis = draw_egopath(img, res_s, crop_coords=crop_s)
        if det_rnn is not None:
            crop_r = det_rnn.get_crop_coords()
            t = time.perf_counter()
            res_r = det_rnn.detect(img)
            timing["rnn_ms"] = round((time.perf_counter() - t) * 1000, 1)
            rnn_vis = draw_egopath(img, res_r, crop_coords=crop_r)
    elif det_rnn is not None:
        # No standalone single-frame model: derive it from the RNN's base net.
        crop_r = det_rnn.get_crop_coords()
        t = time.perf_counter()
        single_res, rnn_res = det_rnn.detect_pair(img)
        timing["rnn_ms"] = round((time.perf_counter() - t) * 1000, 1)
        single_vis = draw_egopath(img, single_res, crop_coords=crop_r)
        if rnn_res is not None:
            rnn_vis = draw_egopath(img, rnn_res, crop_coords=crop_r)
    return single_vis, rnn_vis, timing


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


@app.route("/recorder")
def recorder():
    """Bandicam-style screen/window recorder (browser getDisplayMedia)."""
    return render_template("recorder.html")


def _ffmpeg_exe():
    """Static ffmpeg from imageio-ffmpeg, falling back to one on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        import shutil
        return shutil.which("ffmpeg")


@app.route("/api/recorder/save", methods=["POST"])
def api_recorder_save():
    """Save a browser recording into datasets/Recordings, optionally transcoding
    the WebM to MP4 (H.264 + AAC, faststart) with the bundled ffmpeg."""
    import re
    import subprocess

    up = request.files.get("file")
    if up is None or not up.filename:
        return jsonify({"error": "No recording provided."}), 400
    convert = request.form.get("convert") == "true"
    name = os.path.basename((request.form.get("name") or up.filename).strip())
    name = re.sub(r"[^\w.\-가-힣 ]", "_", name) or "recording.webm"

    dest = os.path.join(os.path.realpath(_UPLOAD_ROOT), "Recordings")
    os.makedirs(dest, exist_ok=True)
    stem, ext = os.path.splitext(name)
    webm_path = os.path.join(dest, stem + (ext if ext else ".webm"))
    up.save(webm_path)

    if not convert:
        return jsonify({"ok": True, "saved": webm_path})

    ff = _ffmpeg_exe()
    if not ff:
        return jsonify({"ok": True, "saved": webm_path,
                        "warn": "ffmpeg가 없어 MP4 변환을 건너뛰었습니다."})
    mp4_path = os.path.join(dest, stem + ".mp4")
    cmd = [ff, "-y", "-i", webm_path,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p",
           # H.264 requires even dimensions; browser captures can be odd-sized.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", mp4_path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800)
        if r.returncode != 0 or not os.path.exists(mp4_path):
            detail = r.stderr.decode("utf-8", "replace")[-400:]
            return jsonify({"ok": True, "saved": webm_path,
                            "warn": f"MP4 변환 실패 — WebM 원본만 저장됨: {detail}"})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": True, "saved": webm_path,
                        "warn": "MP4 변환 시간 초과 — WebM 원본만 저장됨."})
    os.unlink(webm_path)  # conversion succeeded; keep only the MP4
    return jsonify({"ok": True, "saved": mp4_path, "mp4": mp4_path})


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
        single_vis, rnn_vis, timing = run_both(img, det_single, det_rnn)
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
            "single_ms": timing["single_ms"],
            "rnn_ms": timing["rnn_ms"],
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
                    single_vis, rnn_vis, timing = run_both(proc, det_single, det_rnn)
                    payload = {
                        "type": "frame",
                        "frame": frame_idx,
                        "total": total or 0,
                        "emitted": emitted + 1,
                        "elapsed": round(time.time() - t0, 1),
                        "original": pil_to_data_uri(proc, quality=80),
                        "single": pil_to_data_uri(single_vis, quality=80),
                        "rnn": pil_to_data_uri(rnn_vis, quality=80),
                        "single_ms": timing["single_ms"],
                        "rnn_ms": timing["rnn_ms"],
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

    dirs, files = [], []
    try:
        for name in sorted(os.listdir(cur)):
            full = os.path.join(cur, name)
            if os.path.isdir(full):
                dirs.append({"name": name, "path": full,
                             "images": len(list_folder_images(full) or [])})
            elif os.path.isfile(full):
                files.append({"name": name, "path": full,
                              "size": os.path.getsize(full)})
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    parent = os.path.dirname(cur)
    if cur == root or not (parent == root or parent.startswith(root + os.sep)):
        parent = None
    return jsonify({
        "ok": True, "path": cur, "parent": parent, "root": root,
        "dirs": dirs, "files": files, "images": len(list_folder_images(cur) or []),
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


def _under_data_root(path):
    """Realpath of *path* strictly inside the upload root (not the root itself)."""
    root = os.path.realpath(_UPLOAD_ROOT)
    rp = os.path.realpath(path or "")
    return rp if rp.startswith(root + os.sep) else None


def _resolve_dest(subdir, path):
    """Destination dir from an absolute *path* (root or under it) or a *subdir*."""
    root = os.path.realpath(_UPLOAD_ROOT)
    if path:
        rp = os.path.realpath(path)
        return rp if (rp == root or rp.startswith(root + os.sep)) else None
    return _safe_dest_dir(subdir)


@app.route("/api/fs/mkdir", methods=["POST"])
def api_fs_mkdir():
    """Create a new folder under the data root (parent = current folder, or root)."""
    parent = (request.form.get("parent") or "").strip()
    name = os.path.basename((request.form.get("name") or "").strip())
    if not name or name in (".", ".."):
        return jsonify({"error": "Invalid folder name."}), 400
    root = os.path.realpath(_UPLOAD_ROOT)
    base = os.path.realpath(parent) if parent else root
    if base != root and not base.startswith(root + os.sep):
        return jsonify({"error": "Parent folder outside data root."}), 400
    target = os.path.realpath(os.path.join(base, name))
    if not target.startswith(root + os.sep):
        return jsonify({"error": "Invalid path."}), 400
    os.makedirs(target, exist_ok=True)
    return jsonify({"ok": True, "path": target})


@app.route("/api/fs/delete", methods=["POST"])
def api_fs_delete():
    """Delete a file or folder (recursively) inside the data root."""
    import shutil

    target = _under_data_root(request.form.get("path"))
    if target is None:
        return jsonify({"error": "Path must be inside the data root."}), 400
    if not os.path.exists(target):
        return jsonify({"error": "Path not found."}), 400
    if os.path.isdir(target):
        shutil.rmtree(target)
    else:
        os.unlink(target)
    return jsonify({"ok": True})


@app.route("/api/upload_to_server", methods=["POST"])
def api_upload_to_server():
    """Save uploaded files into a subfolder under WEB_UPLOAD_ROOT (no SSH needed),
    so they can then be processed with the no-upload server-folder mode."""
    from werkzeug.utils import secure_filename

    dest = _resolve_dest(request.form.get("subdir"), request.form.get("path"))
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


@app.route("/api/upload_zip", methods=["POST"])
def api_upload_zip():
    """Upload a .zip and extract it into a subfolder under WEB_UPLOAD_ROOT.
    Protects against zip-slip (entries escaping the destination)."""
    import shutil
    import tempfile
    import zipfile

    dest = _resolve_dest(request.form.get("subdir"), request.form.get("path"))
    if dest is None:
        return jsonify({"error": "Invalid destination folder name."}), 400
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "No zip file provided."}), 400
    if not upload.filename.lower().endswith(".zip"):
        return jsonify({"error": "File must be a .zip."}), 400

    os.makedirs(dest, exist_ok=True)
    dest_real = os.path.realpath(dest)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=dest)
    upload.save(tmp.name)
    tmp.close()

    extracted, skipped = 0, 0
    try:
        with zipfile.ZipFile(tmp.name) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                target = os.path.realpath(os.path.join(dest, info.filename))
                if target != dest_real and not target.startswith(dest_real + os.sep):
                    skipped += 1  # zip-slip: entry would escape the destination
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                extracted += 1
    except zipfile.BadZipFile:
        os.unlink(tmp.name)
        return jsonify({"error": "Not a valid zip file."}), 400
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)  # remove the uploaded archive after extraction
    return jsonify({"ok": True, "path": dest, "extracted": extracted, "skipped": skipped})


# ---------------------------------------------------------------------------
# File browser tab: move / extract (all formats) / preview / download
# ---------------------------------------------------------------------------
_FB_TEXT_MAX = 1024 * 1024  # 1 MB text-preview cap
_FB_TEXT_EXT = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".sh", ".bash", ".c", ".cpp", ".h", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".html", ".css", ".xml", ".csv",
    ".tsv", ".log", ".sql", ".env", ".gitignore", ".dockerfile", ".m", ".lua",
}
_FB_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico"}
_FB_TABLE_MAX_ROWS = 300   # cap rows shown per sheet in the preview
_FB_TABLE_MAX_COLS = 40     # cap columns shown per sheet


def _fb_sheets_xlsx(src):
    """Rows of every sheet in a .xlsx/.xlsm workbook (capped for preview)."""
    import openpyxl
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    sheets = []
    try:
        for ws in wb.worksheets:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= _FB_TABLE_MAX_ROWS:
                    break
                rows.append(["" if c is None else str(c) for c in row[:_FB_TABLE_MAX_COLS]])
            sheets.append({"name": ws.title, "rows": rows,
                           "truncated": bool(ws.max_row and ws.max_row > _FB_TABLE_MAX_ROWS)})
    finally:
        wb.close()
    return sheets


def _fb_sheets_xls(src):
    """Rows of every sheet in a legacy .xls workbook (capped for preview)."""
    import xlrd
    book = xlrd.open_workbook(src)
    sheets = []
    for sh in book.sheets():
        rows = []
        for r in range(min(sh.nrows, _FB_TABLE_MAX_ROWS)):
            rows.append([str(sh.cell_value(r, c)) for c in range(min(sh.ncols, _FB_TABLE_MAX_COLS))])
        sheets.append({"name": sh.name, "rows": rows,
                       "truncated": sh.nrows > _FB_TABLE_MAX_ROWS})
    return sheets


def _fb_text_hwpx(src):
    """Plain text from a .hwpx (OWPML zip): join <hp:t> runs per <hp:p>."""
    import re
    import zipfile
    from xml.etree import ElementTree as ET
    paras = []
    with zipfile.ZipFile(src) as z:
        names = sorted(n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n))
        for n in names:
            root = ET.fromstring(z.read(n))
            for p in root.iter():
                if p.tag.rsplit("}", 1)[-1] != "p":
                    continue
                txt = "".join(t.text or "" for t in p.iter()
                              if t.tag.rsplit("}", 1)[-1] == "t")
                paras.append(txt)
    return "\n".join(paras).strip() or "(텍스트 없음)"


def _fb_text_docx(src):
    """Plain text from a .docx: join <w:t> runs per <w:p>."""
    import zipfile
    from xml.etree import ElementTree as ET
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(src) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = ["".join(t.text or "" for t in p.iter(W + "t")) for p in root.iter(W + "p")]
    return "\n".join(paras).strip() or "(텍스트 없음)"


# HWP5 PARA_TEXT control chars that occupy a single wchar (others span 8 wchars).
_HWP_CHAR_CTRL = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}


def _fb_hwp_para_text(payload):
    """Decode one HWPTAG_PARA_TEXT record (UTF-16LE with inline control chars)."""
    import struct
    out, i, n = [], 0, len(payload) - (len(payload) % 2)
    while i < n:
        c = struct.unpack_from("<H", payload, i)[0]
        if c in _HWP_CHAR_CTRL:
            if c in (10, 13):
                out.append("\n")
            i += 2
        elif 1 <= c <= 31:
            i += 16  # extended/inline control spans 8 wchars
        else:
            out.append(chr(c)); i += 2
    return "".join(out)


def _fb_hwp_to_html(src):
    """Convert a binary HWP 5.0 file to a self-contained HTML string via hwp5html
    (pyhwp): the stylesheet is inlined and every bindata image becomes a data URI,
    so the result renders standalone in an <iframe> with the real page layout."""
    import base64
    import glob
    import mimetypes
    import shutil
    import subprocess
    import tempfile

    hwp5html = os.path.join(os.path.dirname(sys.executable), "hwp5html")
    tmp = tempfile.mkdtemp(prefix="hwp5_")
    try:
        subprocess.run([hwp5html, "--output", tmp, src],
                       check=True, capture_output=True, timeout=120)
        with open(os.path.join(tmp, "index.xhtml"), encoding="utf-8") as f:
            html = f.read()
        css_path = os.path.join(tmp, "styles.css")
        if os.path.exists(css_path):
            with open(css_path, encoding="utf-8") as f:
                html = html.replace(
                    '<link rel="stylesheet" href="styles.css" type="text/css" />',
                    "<style>%s</style>" % f.read())
        for asset in glob.glob(os.path.join(tmp, "bindata", "*")):
            name = os.path.basename(asset)
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
            with open(asset, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            html = html.replace("bindata/%s" % name, "data:%s;base64,%s" % (mime, b64))
        return html
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _fb_hwp_memos(src):
    """Extract memo (메모) texts from a binary HWP 5.0 file via hwp5proc's XML
    model: each populated <MemoList> holds one memo's paragraphs. Returns a list
    of memo strings (empty when the document has no memos)."""
    import io
    import subprocess
    from xml.etree import ElementTree as ET

    hwp5proc = os.path.join(os.path.dirname(sys.executable), "hwp5proc")
    xml = subprocess.run([hwp5proc, "xml", src],
                         check=True, capture_output=True, timeout=120).stdout
    memos = []
    for _ev, el in ET.iterparse(io.BytesIO(xml), events=("end",)):
        if el.tag == "MemoList":
            txt = "".join(el.itertext()).strip()
            if txt:
                memos.append(txt)
            el.clear()
    return memos


def _fb_text_hwp(src):
    """Plain text from a binary HWP 5.0 file (OLE compound) — best effort."""
    import struct
    import zlib
    import olefile
    if not olefile.isOleFile(src):
        return "(HWP 5.0 형식이 아닙니다. 다운로드해서 여세요.)"
    ole = olefile.OleFileIO(src)
    try:
        compressed = True
        if ole.exists("FileHeader"):
            hdr = ole.openstream("FileHeader").read()
            if len(hdr) > 36:
                compressed = bool(hdr[36] & 1)
        sects = sorted(
            (e for e in ole.listdir()
             if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")),
            key=lambda e: int(e[1][7:] or 0))
        chunks = []
        for entry in sects:
            data = ole.openstream(entry).read()
            if compressed:
                data = zlib.decompress(data, -15)
            i, n = 0, len(data)
            while i + 4 <= n:
                h = struct.unpack_from("<I", data, i)[0]; i += 4
                tag, size = h & 0x3FF, (h >> 20) & 0xFFF
                if size == 0xFFF:
                    size = struct.unpack_from("<I", data, i)[0]; i += 4
                payload = data[i:i + size]; i += size
                if tag == 67:  # HWPTAG_PARA_TEXT
                    chunks.append(_fb_hwp_para_text(payload))
        return "\n".join(c for c in chunks if c).strip() or "(텍스트 없음)"
    finally:
        ole.close()


def _fb_archive_kind(name):
    """Return 'zip'|'tar'|'single'|'7z'|'rar'|None for an archive filename."""
    low = name.lower()
    if low.endswith(".zip"):
        return "zip"
    if low.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
                     ".tar.xz", ".txz")):
        return "tar"
    if low.endswith(".7z"):
        return "7z"
    if low.endswith(".rar"):
        return "rar"
    if low.endswith((".gz", ".bz2", ".xz")):
        return "single"
    return None


def _fb_unrar_tool():
    """Return a path to an unrar binary: the bundled copy first, then PATH."""
    import shutil

    bundled = os.path.join(BASE_PATH, "bin", "unrar")
    if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
        return bundled
    return shutil.which("unrar")


def _fb_no_traversal(members, base):
    """Reject archive members whose extracted path would escape *base*."""
    base_real = os.path.realpath(base)
    for m in members:
        target = os.path.realpath(os.path.join(base, m))
        if target != base_real and not target.startswith(base_real + os.sep):
            raise ValueError(f"Unsafe path inside archive: {m}")


def _fb_extract(src_full, dest_full):
    """Extract *src_full* into *dest_full*. Returns a human-readable message."""
    import bz2
    import gzip
    import lzma
    import shutil
    import tarfile
    import zipfile

    os.makedirs(dest_full, exist_ok=True)
    name = os.path.basename(src_full)
    kind = _fb_archive_kind(name)

    if kind == "zip":
        with zipfile.ZipFile(src_full) as zf:
            _fb_no_traversal(zf.namelist(), dest_full)
            zf.extractall(dest_full)
        return "ZIP 해제 완료"
    if kind == "tar":
        with tarfile.open(src_full) as tf:
            _fb_no_traversal([m.name for m in tf.getmembers()], dest_full)
            try:
                tf.extractall(dest_full, filter="data")
            except TypeError:
                tf.extractall(dest_full)
        return "TAR 해제 완료"
    if kind == "single":
        low = name.lower()
        if low.endswith(".gz"):
            opener, strip = gzip.open, 3
        elif low.endswith(".bz2"):
            opener, strip = bz2.open, 4
        else:
            opener, strip = lzma.open, 3
        out_name = name[:-strip] or (name + ".out")
        with opener(src_full, "rb") as fin, \
                open(os.path.join(dest_full, out_name), "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return f"단일파일 해제 완료 → {out_name}"
    if kind == "7z":
        try:
            import py7zr
            with py7zr.SevenZipFile(src_full, "r") as z:
                z.extractall(path=dest_full)
            return "7z 해제 완료 (py7zr)"
        except ImportError:
            pass
        exe = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
        if exe:
            subprocess.run([exe, "x", "-y", "-o" + dest_full, src_full],
                           check=True, capture_output=True)
            return "7z 해제 완료 (7z CLI)"
        raise RuntimeError("7z 해제 도구가 없습니다 (pip install py7zr 또는 7z CLI 설치).")
    if kind == "rar":
        unrar = _fb_unrar_tool()
        try:
            import rarfile
            if unrar:
                rarfile.UNRAR_TOOL = unrar  # use the bundled binary
            _fb_no_traversal(rarfile.RarFile(src_full).namelist(), dest_full)
            with rarfile.RarFile(src_full) as rf:
                rf.extractall(dest_full)
            tag = "bundled unrar" if unrar and unrar.startswith(BASE_PATH) else "rarfile"
            return f"rar 해제 완료 ({tag})"
        except Exception:
            pass
        if unrar:
            subprocess.run([unrar, "x", "-y", src_full, dest_full + os.sep],
                           check=True, capture_output=True)
            return "rar 해제 완료 (unrar)"
        exe = shutil.which("7z") or shutil.which("7za")
        if exe:
            subprocess.run([exe, "x", "-y", "-o" + dest_full, src_full],
                           check=True, capture_output=True)
            return "rar 해제 완료 (7z CLI)"
        raise RuntimeError("rar 해제 도구가 없습니다 (unrar 또는 7z CLI 설치).")
    raise ValueError(f"지원하지 않는 압축 형식입니다: {name}")


@app.route("/api/fs/move", methods=["POST"])
def api_fs_move():
    """Move or rename a file/folder within the data root."""
    import shutil

    src = _under_data_root(request.form.get("src"))
    if src is None or not os.path.exists(src):
        return jsonify({"error": "Source must be inside the data root."}), 400
    dst_raw = (request.form.get("dst") or "").strip()
    if not dst_raw:
        return jsonify({"error": "Destination required."}), 400
    root = os.path.realpath(_UPLOAD_ROOT)
    dst = os.path.realpath(dst_raw)
    if os.path.isdir(dst):  # moving into an existing folder
        dst = os.path.realpath(os.path.join(dst, os.path.basename(src)))
    if not dst.startswith(root + os.sep):
        return jsonify({"error": "Destination outside data root."}), 400
    if os.path.exists(dst):
        return jsonify({"error": "Destination already exists."}), 400
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return jsonify({"ok": True, "src": src, "dst": dst})


@app.route("/api/fs/copy", methods=["POST"])
def api_fs_copy():
    """Copy a file/folder to another location within the data root."""
    import shutil

    src = _under_data_root(request.form.get("src"))
    if src is None or not os.path.exists(src):
        return jsonify({"error": "Source must be inside the data root."}), 400
    dst_raw = (request.form.get("dst") or "").strip()
    if not dst_raw:
        return jsonify({"error": "Destination required."}), 400
    root = os.path.realpath(_UPLOAD_ROOT)
    dst = os.path.realpath(dst_raw)
    if os.path.isdir(dst):  # copying into an existing folder
        dst = os.path.realpath(os.path.join(dst, os.path.basename(src)))
    if not dst.startswith(root + os.sep):
        return jsonify({"error": "Destination outside data root."}), 400
    if os.path.realpath(src) == dst:
        return jsonify({"error": "Source and destination are the same."}), 400
    if os.path.exists(dst):
        return jsonify({"error": "Destination already exists."}), 400
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return jsonify({"ok": True, "src": src, "dst": dst})


def _fb_extract_dest(src):
    """Compute the extraction destination dir for *src*, validated under root.

    Returns (dest, error). Single-file archives unpack into the same folder;
    container archives unpack into a '<stem>_extracted' sibling folder.
    """
    base = os.path.basename(src)
    kind = _fb_archive_kind(base)
    stem = base
    for suf in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz",
                ".tar", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    if kind == "single":
        dest = os.path.dirname(src)
    else:
        dest = os.path.join(os.path.dirname(src), stem + "_extracted")
    root = os.path.realpath(_UPLOAD_ROOT)
    dest = os.path.realpath(dest)
    if dest != root and not dest.startswith(root + os.sep):
        return None, "Destination outside data root."
    return dest, None


@app.route("/api/fs/extract", methods=["POST"])
def api_fs_extract():
    """Extract an archive already on the server (zip/tar/gz/bz2/xz/7z/rar)."""
    src = _under_data_root(request.form.get("path"))
    if src is None or not os.path.isfile(src):
        return jsonify({"error": "Archive must be inside the data root."}), 400
    if _fb_archive_kind(os.path.basename(src)) is None:
        return jsonify({"error": "Unsupported archive type."}), 400

    dest, err = _fb_extract_dest(src)
    if err:
        return jsonify({"error": err}), 400
    try:
        msg = _fb_extract(src, dest)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", "replace")[:300]
        return jsonify({"error": f"해제 실패: {detail or e}"}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "message": msg, "dest": dest})


def _fb_sse(obj):
    import json as _json
    return f"data: {_json.dumps(obj, ensure_ascii=False)}\n\n"


def _fb_sse_response(gen):
    return Response(stream_with_context(gen),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/fs/copy_stream", methods=["POST"])
def api_fs_copy_stream():
    """Copy one or more files/folders into a destination folder, streaming
    byte-level progress as SSE. Used by the File Browser copy button."""
    import shutil

    root = os.path.realpath(_UPLOAD_ROOT)
    raw_srcs = request.form.getlist("src")
    dst_raw = (request.form.get("dst") or "").strip()
    dst_dir = os.path.realpath(dst_raw) if dst_raw else None
    srcs = []
    for s in raw_srcs:
        rs = _under_data_root(s)
        if rs and os.path.exists(rs):
            srcs.append(rs)

    def gen():
        if dst_dir is None or not (dst_dir == root or dst_dir.startswith(root + os.sep)) \
                or not os.path.isdir(dst_dir):
            yield _fb_sse({"type": "error", "error": "대상 폴더가 올바르지 않습니다."})
            return
        if not srcs:
            yield _fb_sse({"type": "error", "error": "복사할 항목이 없습니다."})
            return

        # Build a flat copy plan (src_file -> dst_file) and total byte count,
        # skipping any item whose top-level target already exists.
        plan, total, skipped = [], 0, []
        for s in srcs:
            base = os.path.basename(s)
            target_top = os.path.join(dst_dir, base)
            if os.path.exists(target_top):
                skipped.append(base)
                continue
            if os.path.realpath(s) == os.path.realpath(target_top):
                skipped.append(base)
                continue
            if os.path.isdir(s):
                for dp, _, fns in os.walk(s):
                    for fn in fns:
                        f = os.path.join(dp, fn)
                        plan.append((f, os.path.join(target_top, os.path.relpath(f, s))))
                        try:
                            total += os.path.getsize(f)
                        except OSError:
                            pass
                if not os.listdir(s):  # preserve empty dirs
                    plan.append((None, target_top))
            else:
                plan.append((s, target_top))
                total += os.path.getsize(s)

        yield _fb_sse({"type": "status", "total": total, "files": len(plan),
                       "message": f"복사 시작: {len(plan)}개 파일"
                                  + (f", {len(skipped)}개 건너뜀" if skipped else "")})
        done = 0
        step = max(total // 200, 4 * 1024 * 1024)  # throttle: ~200 updates max
        next_emit = step
        BUF = 1024 * 1024
        try:
            for sf, df in plan:
                if sf is None:  # empty directory
                    os.makedirs(df, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(df), exist_ok=True)
                with open(sf, "rb") as fi, open(df, "wb") as fo:
                    while True:
                        chunk = fi.read(BUF)
                        if not chunk:
                            break
                        fo.write(chunk)
                        done += len(chunk)
                        if done >= next_emit:
                            next_emit += step
                            yield _fb_sse({"type": "progress", "done": done, "total": total,
                                           "pct": round(done * 100 / total, 1) if total else 100,
                                           "name": os.path.basename(sf)})
                shutil.copystat(sf, df)
            yield _fb_sse({"type": "done", "files": len(plan), "bytes": done,
                           "skipped": skipped})
        except Exception as e:  # noqa: BLE001
            yield _fb_sse({"type": "error", "error": str(e)})

    return _fb_sse_response(gen())


def _fb_extract_one_events(src):
    """Yield event dicts for extracting ONE archive. Progress dicts have
    type='status'|'progress'; the final yield is a sentinel dict carrying
    '_result' = 'ok' (with message/dest) or 'error' (with error)."""
    import bz2
    import gzip
    import lzma
    import tarfile
    import zipfile

    kind = _fb_archive_kind(os.path.basename(src))
    if kind is None:
        yield {"_result": "error", "error": "지원하지 않는 압축 형식입니다."}
        return
    dest, err = _fb_extract_dest(src)
    if err:
        yield {"_result": "error", "error": err}
        return
    os.makedirs(dest, exist_ok=True)
    try:
        if kind == "zip":
            with zipfile.ZipFile(src) as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                _fb_no_traversal([m.filename for m in members], dest)
                total = len(members)
                yield {"type": "status", "total": total, "message": f"ZIP 해제: {total}개 항목"}
                for i, m in enumerate(members):
                    zf.extract(m, dest)
                    yield {"type": "progress", "done": i + 1, "total": total,
                           "pct": round((i + 1) * 100 / total, 1) if total else 100,
                           "name": os.path.basename(m.filename)}
            yield {"_result": "ok", "message": "ZIP 해제 완료", "dest": dest}

        elif kind == "tar":
            with tarfile.open(src) as tf:
                members = tf.getmembers()
                _fb_no_traversal([m.name for m in members], dest)
                files = [m for m in members if m.isfile() or m.isdir()]
                total = len(files)
                yield {"type": "status", "total": total, "message": f"TAR 해제: {total}개 항목"}
                for i, m in enumerate(files):
                    try:
                        tf.extract(m, dest, filter="data")
                    except TypeError:
                        tf.extract(m, dest)
                    yield {"type": "progress", "done": i + 1, "total": total,
                           "pct": round((i + 1) * 100 / total, 1) if total else 100,
                           "name": os.path.basename(m.name)}
            yield {"_result": "ok", "message": "TAR 해제 완료", "dest": dest}

        elif kind == "single":
            name = os.path.basename(src)
            low = name.lower()
            if low.endswith(".gz"):
                opener, strip = gzip.open, 3
            elif low.endswith(".bz2"):
                opener, strip = bz2.open, 4
            else:
                opener, strip = lzma.open, 3
            out_name = name[:-strip] or (name + ".out")
            total = os.path.getsize(src)
            yield {"type": "status", "total": total, "message": f"단일파일 해제 → {out_name}"}
            read = 0
            step = max(total // 200, 4 * 1024 * 1024)
            next_emit = step
            with opener(src, "rb") as fin, \
                    open(os.path.join(dest, out_name), "wb") as fout:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
                    read += len(chunk)
                    if read >= next_emit:
                        next_emit += step
                        yield {"type": "progress", "done": read, "total": 0,
                               "pct": None, "name": out_name}
            yield {"_result": "ok", "message": f"단일파일 해제 완료 → {out_name}", "dest": dest}

        else:  # 7z / rar — robust helper, indeterminate bar
            yield {"type": "status", "total": 0, "pct": None,
                   "message": f"{kind} 해제 중… (진행률 표시 미지원)"}
            msg = _fb_extract(src, dest)
            yield {"_result": "ok", "message": msg, "dest": dest}

    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", "replace")[:300]
        yield {"_result": "error", "error": f"해제 실패: {detail or e}"}
    except Exception as e:  # noqa: BLE001
        yield {"_result": "error", "error": str(e)}


@app.route("/api/fs/extract_stream", methods=["POST"])
def api_fs_extract_stream():
    """Extract one OR MANY archives (one 'path' per archive), streaming progress
    as SSE. Archives are processed sequentially; events carry the current file's
    index/total so the client can show overall batch progress."""
    raw = request.form.getlist("path")
    srcs = [s for s in (_under_data_root(p) for p in raw)
            if s and os.path.isfile(s)]

    def gen():
        if not srcs:
            yield _fb_sse({"type": "error", "error": "압축 파일을 찾을 수 없습니다."})
            return
        total_files = len(srcs)
        ok = fail = 0
        results = []
        yield _fb_sse({"type": "batch", "files": total_files})
        for i, src in enumerate(srcs):
            base = os.path.basename(src)
            idx = i + 1
            yield _fb_sse({"type": "file", "index": idx, "total": total_files, "name": base})
            for ev in _fb_extract_one_events(src):
                if "_result" in ev:
                    if ev["_result"] == "ok":
                        ok += 1
                        results.append({"name": base, "ok": True,
                                        "message": ev.get("message"), "dest": ev.get("dest")})
                        yield _fb_sse({"type": "file_done", "index": idx, "total": total_files,
                                       "name": base, "message": ev.get("message"),
                                       "dest": ev.get("dest")})
                    else:
                        fail += 1
                        results.append({"name": base, "ok": False, "error": ev.get("error")})
                        yield _fb_sse({"type": "file_error", "index": idx, "total": total_files,
                                       "name": base, "error": ev.get("error")})
                else:
                    ev2 = dict(ev)  # forward status/progress with file context
                    ev2["index"] = idx
                    ev2["totalFiles"] = total_files
                    ev2["file"] = base
                    yield _fb_sse(ev2)
        yield _fb_sse({"type": "done", "ok": ok, "fail": fail, "results": results})

    return _fb_sse_response(gen())


@app.route("/api/fs/preview")
def api_fs_preview():
    """Preview a file: image bytes inline, or JSON text for small text files."""
    src = _under_data_root(request.args.get("path"))
    if src is None or not os.path.isfile(src):
        return jsonify({"ok": False, "error": "File not found in data root."}), 400
    ext = os.path.splitext(src)[1].lower()
    if ext in _FB_IMAGE_EXT:
        return send_file(src)
    if ext == ".pdf":
        return send_file(src)  # inline — the browser renders it in an <iframe>
    # Office / Hangul documents: spreadsheets render as tables, docs as text.
    try:
        if ext in (".xlsx", ".xlsm"):
            return jsonify({"ok": True, "type": "table", "sheets": _fb_sheets_xlsx(src)})
        if ext == ".xls":
            return jsonify({"ok": True, "type": "table", "sheets": _fb_sheets_xls(src)})
        if ext == ".hwpx":
            return jsonify({"ok": True, "type": "text", "text": _fb_text_hwpx(src)})
        if ext == ".hwp":
            return jsonify({"ok": True, "type": "text", "text": _fb_text_hwp(src)})
        if ext == ".docx":
            return jsonify({"ok": True, "type": "text", "text": _fb_text_docx(src)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": True, "type": "text",
                        "text": f"(미리보기 실패: {type(e).__name__}: {e})"})
    size = os.path.getsize(src)
    if size > _FB_TEXT_MAX:
        return jsonify({"ok": True, "type": "text",
                        "text": f"(파일이 너무 큽니다: {size} bytes. 다운로드하세요.)"})
    with open(src, "rb") as f:
        raw = f.read(_FB_TEXT_MAX)
    try:
        return jsonify({"ok": True, "type": "text", "text": raw.decode("utf-8")})
    except UnicodeDecodeError:
        return jsonify({"ok": True, "type": "binary",
                        "text": "(미리보기 불가: 바이너리 파일. 다운로드하세요.)"})


@app.route("/api/fs/download")
def api_fs_download():
    """Download a single file from the data root."""
    src = _under_data_root(request.args.get("path"))
    if src is None or not os.path.isfile(src):
        return jsonify({"error": "File not found in data root."}), 400
    return send_file(src, as_attachment=True,
                     download_name=os.path.basename(src))


@app.route("/api/fs/stash", methods=["POST"])
def api_fs_stash():
    """Save an uploaded (client-local) file into a temp cache under the data root
    so the existing path-based preview endpoints can render it (parity with server
    files). The original extension is preserved for type dispatch; stale stashes
    (older than 1h) are pruned."""
    import time

    up = request.files.get("file")
    if up is None or not up.filename:
        return jsonify({"ok": False, "error": "No file."}), 400
    cache = os.path.join(os.path.realpath(_UPLOAD_ROOT), ".preview_cache")
    os.makedirs(cache, exist_ok=True)
    now = time.time()
    for n in os.listdir(cache):  # best-effort prune
        fp = os.path.join(cache, n)
        try:
            if os.path.isfile(fp) and now - os.path.getmtime(fp) > 3600:
                os.unlink(fp)
        except OSError:
            pass
    ext = os.path.splitext(os.path.basename(up.filename))[1]
    dest = os.path.join(cache, f"stash_{int(now * 1000)}{ext}")
    up.save(dest)
    return jsonify({"ok": True, "path": dest})


@app.route("/api/fs/hwp_html")
def api_fs_hwp_html():
    """Render a binary .hwp as full-layout HTML (hwp5html) for the preview iframe.
    Cached on disk per (path, mtime); falls back to text extraction on failure."""
    import hashlib
    import html as _html
    import tempfile

    src = _under_data_root(request.args.get("path"))
    if src is None or not os.path.isfile(src):
        return "File not found", 404

    cache_dir = os.path.join(tempfile.gettempdir(), "tepnet_hwp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(f"{os.path.realpath(src)}:{os.path.getmtime(src)}".encode()).hexdigest()
    cached = os.path.join(cache_dir, key + ".html")
    if os.path.exists(cached):
        return send_file(cached, mimetype="text/html")

    try:
        html = _fb_hwp_to_html(src)
    except Exception as e:  # noqa: BLE001 — fall back to text so the user still sees content
        try:
            txt = _fb_text_hwp(src)
        except Exception:  # noqa: BLE001
            txt = "(미리보기 실패)"
        html = ("<!doctype html><meta charset=utf-8>"
                "<div style='padding:12px;color:#b00;font-family:sans-serif'>"
                f"레이아웃 변환 실패 — 텍스트로 표시합니다 ({_html.escape(str(e))})</div>"
                "<pre style='white-space:pre-wrap;word-break:break-all;"
                f"font-size:13px;padding:12px'>{_html.escape(txt)}</pre>")
    with open(cached, "w", encoding="utf-8") as f:
        f.write(html)
    return send_file(cached, mimetype="text/html")


@app.route("/api/fs/hwp_memos")
def api_fs_hwp_memos():
    """Return the memo (메모) texts of a binary .hwp for the preview's memo toggle."""
    src = _under_data_root(request.args.get("path"))
    if src is None or not os.path.isfile(src):
        return jsonify({"ok": False, "error": "File not found in data root."}), 400
    try:
        return jsonify({"ok": True, "memos": _fb_hwp_memos(src)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/upload_dirs")
def api_upload_dirs():
    """List existing subfolders under WEB_UPLOAD_ROOT (with file counts)."""
    root = os.path.realpath(_UPLOAD_ROOT)
    dirs = []
    if os.path.isdir(root):
        for n in sorted(os.listdir(root)):
            fp = os.path.join(root, n)
            if os.path.isdir(fp):
                cnt = sum(1 for x in os.listdir(fp) if os.path.isfile(os.path.join(fp, x)))
                dirs.append({"name": n, "files": cnt})
    return jsonify({"ok": True, "root": root, "dirs": dirs})


@app.route("/api/upload_list")
def api_upload_list():
    """List files in an uploaded subfolder under WEB_UPLOAD_ROOT (for review/delete)."""
    dest = _safe_dest_dir(request.args.get("subdir"))
    if dest is None or not os.path.isdir(dest):
        return jsonify({"ok": False, "error": "Folder not found."}), 400
    files = []
    for n in sorted(os.listdir(dest)):
        fp = os.path.join(dest, n)
        if os.path.isfile(fp):
            files.append({"name": n, "size": os.path.getsize(fp)})
    return jsonify({"ok": True, "path": dest, "count": len(files), "files": files})


@app.route("/api/delete_file", methods=["POST"])
def api_delete_file():
    """Delete a single uploaded file (restricted to WEB_UPLOAD_ROOT)."""
    dest = _safe_dest_dir(request.form.get("subdir"))
    name = os.path.basename((request.form.get("name") or "").strip())
    if dest is None or not name:
        return jsonify({"error": "Invalid path."}), 400
    fp = os.path.join(dest, name)
    if not os.path.isfile(fp):
        return jsonify({"error": "File not found."}), 400
    os.unlink(fp)
    remaining = sum(1 for f in os.listdir(dest) if os.path.isfile(os.path.join(dest, f)))
    return jsonify({"ok": True, "remaining": remaining})


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
                    single_vis, rnn_vis, timing = run_both(proc, det_single, det_rnn)
                    yield sse({
                        "type": "image", "index": i, "total": len(files), "name": name,
                        "elapsed": round(time.time() - t0, 1),
                        "original": pil_to_data_uri(proc, quality=80),
                        "single": pil_to_data_uri(single_vis, quality=80),
                        "rnn": pil_to_data_uri(rnn_vis, quality=80),
                        "has_rnn": det_rnn is not None,
                        "single_is_fallback": det_single is None and det_rnn is not None,
                        "single_ms": timing["single_ms"],
                        "rnn_ms": timing["rnn_ms"],
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


@app.route("/api/label/autolabel", methods=["POST"])
def api_label_autolabel():
    """Propagate a labeled frame's rails to the following frames via optical flow
    (for continuous/video sequences). Streams progress via SSE."""
    folder = (request.form.get("folder") or "").strip()
    annots = (request.form.get("annots") or "").strip()
    start = os.path.basename((request.form.get("start_name") or "").strip())
    try:
        max_frames = int(request.form.get("max_frames", 0))  # 0 = all remaining
    except ValueError:
        max_frames = 0

    files = list_folder_images(folder)
    if files is None:
        return jsonify({"error": f"Not a folder: {folder}"}), 400
    if not annots.lower().endswith(".json") or not _annots_path_ok(annots):
        return jsonify({"error": "Invalid annotations path."}), 400
    data = _load_annotations(annots)
    if start not in files:
        return jsonify({"error": "Start image not in folder."}), 400
    if start not in data:
        return jsonify({"error": "Label the start frame first."}), 400
    si = files.index(start)

    FLOW_W = 960  # optical flow runs at this width for speed

    def event_stream():
        import json
        import cv2
        import numpy as np

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        def load_gray(name):
            im = Image.open(os.path.join(folder, name))
            im.load()
            ow, oh = im.size
            nw = FLOW_W if ow > FLOW_W else ow
            nh = max(1, round(oh * nw / ow))
            g = np.array(im.convert("L").resize((nw, nh)), dtype=np.uint8)
            # Boost local contrast so optical flow latches onto the rail edges
            # instead of wandering on low-texture steel / repetitive ballast.
            g = clahe.apply(g)
            return g, ow, oh, nw / ow

        def extend_bottom(pts, H, W):
            """Extend the lowest rail point down to y = H-1 (training needs the
            ego-path to reach the frame bottom)."""
            if not pts:
                return pts
            bi = max(range(len(pts)), key=lambda i: pts[i][1])
            b = pts[bi]
            if b[1] >= H - 1:
                return pts
            others = [p for i, p in enumerate(pts) if i != bi]
            x = float(b[0])
            if others:
                o = max(others, key=lambda p: p[1])
                if o[1] != b[1]:
                    x = b[0] + (b[0] - o[0]) / (b[1] - o[1]) * ((H - 1) - b[1])
            nx = int(max(0, min(W - 1, round(x))))
            return ([[nx, H - 1]] + pts) if bi == 0 else (pts + [[nx, H - 1]])

        try:
            seed_gray, ow, oh, sc = load_gray(start)
            seed = data[start]
            H, Wd = seed_gray.shape
            N = 48                                  # y samples per rail
            Wsearch = max(10, int(0.025 * Wd))      # horizontal search half-window

            def grid_and_x(rail):
                """Fixed y grid (flow coords, top→bottom) + rough x(grid) from seed."""
                a = sorted(rail, key=lambda p: p[1])
                ys = np.array([p[1] for p in a], dtype=np.float32) * sc
                xs = np.array([p[0] for p in a], dtype=np.float32) * sc
                gy = np.linspace(ys.min(), H - 1, N)
                return gy, np.interp(gy, ys, xs)

            gyL, gxL = grid_and_x(seed["left_rail"])
            gyR, gxR = grid_and_x(seed["right_rail"])

            def edge_map(gray):
                e = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))   # vertical edges
                return cv2.GaussianBlur(e, (0, 0), 1.5)

            # Narrow search so a row can't latch onto an adjacent track's rail.
            Wsearch = max(8, int(0.018 * Wd))

            def snap_rows(edge, gy, pred):
                """Per row, the strongest vertical edge near the predicted x (a
                Gaussian prior keeps it from jumping to a neighbouring track)."""
                sx = np.empty(len(gy)); ok = np.zeros(len(gy), bool)
                cols = edge.shape[1]
                for i in range(len(gy)):
                    yy = int(round(min(max(gy[i], 0), edge.shape[0] - 1)))
                    x0 = pred[i]
                    lo = int(max(0, x0 - Wsearch)); hi = int(min(cols - 1, x0 + Wsearch))
                    if hi <= lo:
                        sx[i] = x0; continue
                    xs = np.arange(lo, hi + 1)
                    seg = edge[yy, lo:hi + 1]
                    j = int(np.argmax(seg * np.exp(-((xs - x0) ** 2) / (2 * (Wsearch / 2.0) ** 2))))
                    sx[i] = xs[j]
                    ok[i] = seg[j] > np.mean(seg) + 1e-6     # a real edge, not flat region
                return sx, ok

            def fit2(gy, sx, ok, prev_coef):
                """Robust 2nd-order fit (x = a + b*y + c*y^2) to inliers near the
                previous curve, blended with it so the rail stays smooth/stable."""
                pred = np.polyval(prev_coef, gy)
                inl = ok & (np.abs(sx - pred) <= Wsearch)
                if inl.sum() < 5:
                    return prev_coef                          # not enough evidence → hold
                c = np.polyfit(gy[inl], sx[inl], 2)
                r = np.abs(np.polyval(c, gy) - sx)            # reject remaining outliers
                inl2 = inl & (r <= np.median(r[inl]) * 2 + 2)
                if inl2.sum() >= 5:
                    c = np.polyfit(gy[inl2], sx[inl2], 2)
                return 0.6 * c + 0.4 * prev_coef              # temporal smoothing

            cL = np.polyfit(gyL, gxL, 2)
            cR = np.polyfit(gyR, gxR, 2)
            prevXL = np.polyval(cL, gyL); prevXR = np.polyval(cR, gyR)

            # Include the seed frame so the user's rough clicks are snapped too.
            targets = files[si:]
            if max_frames > 0:
                targets = targets[:max_frames + 1]
            yield sse({"type": "status",
                       "message": f"Auto-labeling {len(targets)} frames (rail detect · 2nd-order curve)…"})
            done = 0
            for k, name in enumerate(targets):
                g, ow2, oh2, sc2 = load_gray(name)
                e = edge_map(g)
                cL = fit2(gyL, *snap_rows(e, gyL, prevXL), cL)
                cR = fit2(gyR, *snap_rows(e, gyR, prevXR), cR)
                prevXL = np.clip(np.polyval(cL, gyL), 0, e.shape[1] - 1)
                prevXR = np.clip(np.polyval(cR, gyR), 0, e.shape[1] - 1)
                left = [[int(round(x / sc2)), int(round(y / sc2))] for x, y in zip(prevXL, gyL)]
                right = [[int(round(x / sc2)), int(round(y / sc2))] for x, y in zip(prevXR, gyR)]
                data[name] = {"left_rail": left, "right_rail": right}
                done += 1
                yield sse({"type": "progress", "index": k + 1, "total": len(targets), "name": name})

            tmp = annots + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, annots)
            yield sse({"type": "done", "count": done, "labeled_total": len(data)})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "error": str(e)})

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _rdp_simplify(points, eps=10.0):
    """Ramer-Douglas-Peucker polyline simplification: drop points closer than
    `eps` px to the chord, so nearly-straight rails collapse to a few points
    (endpoints always kept)."""
    if len(points) < 3:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    denom = (dx * dx + dy * dy) ** 0.5
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = (((px - x1) ** 2 + (py - y1) ** 2) ** 0.5 if denom == 0
             else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return _rdp_simplify(points[:idx + 1], eps)[:-1] + _rdp_simplify(points[idx:], eps)
    return [points[0], points[-1]]


def _largest_blob(mask):
    """Keep only the biggest connected component of a boolean mask.

    Segmentation output often carries stray blobs (sky, buildings) far from the
    track; taking the leftmost/rightmost pixel per row across them produces
    wild zig-zags, so everything but the main path is dropped.
    """
    import cv2
    import numpy as np

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:  # background only, or a single component already
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _widest_run(row, min_width=3):
    """Return (start, end) of the widest contiguous True run in a mask row."""
    import numpy as np

    idx = np.where(row)[0]
    if len(idx) == 0:
        return None
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    widest = int(np.argmax(idx[ends] - idx[starts]))
    lo, hi = int(idx[starts[widest]]), int(idx[ends[widest]])
    return (lo, hi) if hi - lo + 1 >= min_width else None


def _drop_center_outliers(ys, lo, hi, tol=3.0):
    """Boolean keep-mask rejecting rows whose path centre leaves a smooth curve.

    The ego-path centre traces a gentle 2nd-order curve down the image; rows
    that jump away from it come from a mis-segmented side track.
    """
    import numpy as np

    if len(ys) < 5:
        return np.ones(len(ys), dtype=bool)
    centers = (lo + hi) / 2.0
    coef = np.polyfit(ys, centers, 2)
    resid = np.abs(np.polyval(coef, ys) - centers)
    med = np.median(resid)
    keep = resid <= max(tol * med, 4.0)
    if keep.sum() >= 5:  # refit without the outliers for a tighter second pass
        coef = np.polyfit(ys[keep], centers[keep], 2)
        resid = np.abs(np.polyval(coef, ys) - centers)
        med = np.median(resid[keep])
        keep = resid <= max(tol * med, 4.0)
    return keep


@app.route("/api/label/detect_model", methods=["POST"])
def api_label_detect_model():
    """Auto-label every frame by running a trained model (default brilliant-horse-15)
    and extracting the left/right rails from its prediction. No seed, no tracking."""
    import numpy as np

    folder = (request.form.get("folder") or "").strip()
    annots = (request.form.get("annots") or "").strip()
    model_name = (request.form.get("model") or "brilliant-horse-15").strip()
    device = (request.form.get("device") or
              ("cuda:0" if "cuda:0" in available_devices() else "cpu"))
    crop_mode, crop_coords = _parse_crop(request.form)

    files = list_folder_images(folder)
    if files is None:
        return jsonify({"error": f"Not a folder: {folder}"}), 400
    if not files:
        return jsonify({"error": "No images in folder."}), 400
    if not annots.lower().endswith(".json") or not _annots_path_ok(annots):
        return jsonify({"error": "Invalid annotations path."}), 400
    try:
        det_single, det_rnn = detectors_for_request(model_name, device, crop_mode, crop_coords)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    det = det_single or det_rnn
    method = det.config.get("method", "segmentation")

    def rails_from_result(res):
        """Return (left_rail, right_rail) point lists from a detector result."""
        if method == "segmentation":
            arr = _largest_blob(np.array(res.convert("L")) > 0)
            rows = np.where(arr.any(axis=1))[0]
            if len(rows) < 2:
                return None, None
            grid = np.linspace(rows.min(), rows.max(), 48)
            ys, lo, hi = [], [], []
            for y in grid:
                yy = int(round(y))
                span = _widest_run(arr[yy])
                if span is not None:
                    ys.append(yy); lo.append(span[0]); hi.append(span[1])
            keep = _drop_center_outliers(np.array(ys), np.array(lo), np.array(hi))
            if keep.sum() < 2:
                return None, None
            left = [[int(lo[i]), int(ys[i])] for i in np.where(keep)[0]]
            right = [[int(hi[i]), int(ys[i])] for i in np.where(keep)[0]]
            return _rdp_simplify(left), _rdp_simplify(right)
        # regression / classification: detect() already returns [left, right]
        if isinstance(res, (list, tuple)) and len(res) == 2:
            L = [[int(x), int(y)] for x, y in res[0]]
            R = [[int(x), int(y)] for x, y in res[1]]
            if len(L) < 2 or len(R) < 2:
                return None, None
            return _rdp_simplify(L), _rdp_simplify(R)
        return None, None

    data = _load_annotations(annots)

    def event_stream():
        import json

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        try:
            yield sse({"type": "status",
                       "message": f"Detecting rails with {model_name} on {len(files)} frames…"})
            done = 0
            for i, name in enumerate(files):
                try:
                    img = Image.open(os.path.join(folder, name))
                    img.load()
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if det.temporal:
                        det.reset_temporal()
                    left, right = rails_from_result(det.detect(img))
                    if left and right:
                        data[name] = {"left_rail": left, "right_rail": right}
                        done += 1
                    else:
                        yield sse({"type": "log", "name": name, "msg": "no rail detected"})
                except Exception as e:  # noqa: BLE001
                    yield sse({"type": "log", "name": name, "msg": str(e)})
                if i % 5 == 0:
                    yield sse({"type": "progress", "index": i + 1, "total": len(files), "name": name})
            tmp = annots + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, annots)
            yield sse({"type": "done", "count": done, "total": len(files), "labeled_total": len(data)})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "error": str(e)})

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# SAM2 video auto-labeling (prompt the ego-track once, propagate across frames)
# --------------------------------------------------------------------------- #
_SAM2 = {}
_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
_SAM2_CKPT = os.path.join(BASE_PATH, "bin", "sam2", "sam2.1_hiera_base_plus.pt")


def _sam2_predictor():
    """Lazily build and cache the SAM2 video predictor (heavy; load once)."""
    if _SAM2.get("pred") is None:
        from sam2.build_sam import build_sam2_video_predictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM2["pred"] = build_sam2_video_predictor(_SAM2_CFG, _SAM2_CKPT, device=dev)
        _SAM2["dev"] = dev
    return _SAM2["pred"], _SAM2["dev"]


def _sam2_rails_from_mask(mask, n=48):
    """left/right boundary of the track mask on an n-row grid, RDP-simplified, with
    the lowest point extended to the frame bottom (training needs full-height rails)."""
    import numpy as np
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) < 2:
        return None, None
    H = mask.shape[0]
    grid = np.linspace(rows.min(), rows.max(), n)
    left, right = [], []
    for y in grid:
        yy = int(round(y)); cols = np.where(mask[yy])[0]
        if len(cols):
            left.append([int(cols[0]), yy]); right.append([int(cols[-1]), yy])
    if len(left) < 2:
        return None, None
    for rail in (left, right):
        if rail[-1][1] < H - 1:
            rail.append([rail[-1][0], H - 1])  # extend down to the last row
    return _rdp_simplify(left), _rdp_simplify(right)


@app.route("/api/label/sam2", methods=["POST"])
def api_label_sam2():
    """Segment the ego-track on the first frame and propagate it across the whole
    folder (a video's frames) with SAM2, then write left/right rails. SSE-streamed."""
    folder = (request.form.get("folder") or "").strip()
    annots = (request.form.get("annots") or "").strip()
    points_str = (request.form.get("points") or "").strip()

    files = list_folder_images(folder)
    if files is None:
        return jsonify({"error": f"Not a folder: {folder}"}), 400
    if not files:
        return jsonify({"error": "No images in folder."}), 400
    if not annots.lower().endswith(".json") or not _annots_path_ok(annots):
        return jsonify({"error": "Invalid annotations path."}), 400
    if not os.path.exists(_SAM2_CKPT):
        return jsonify({"error": "SAM2 checkpoint missing (run setup)."}), 400

    def event_stream():
        import json
        import shutil
        import tempfile
        import numpy as np
        from PIL import Image

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        tmp = None
        try:
            yield sse({"type": "status", "message": "SAM2 모델 로딩…"})
            pred, dev = _sam2_predictor()
            W, H = Image.open(os.path.join(folder, files[0])).size

            tmp = tempfile.mkdtemp(prefix="sam2f_")  # SAM2 needs int-named jpg frames
            for i, name in enumerate(files):
                os.symlink(os.path.abspath(os.path.join(folder, name)),
                           os.path.join(tmp, f"{i:06d}.jpg"))

            if points_str:
                pts = [[float(a) for a in p.split(",")] for p in points_str.split(";")]
            else:  # auto: vertical line of positive points up the bottom-centre
                pts = [[W / 2, H * f] for f in (0.98, 0.9, 0.82, 0.74, 0.66)]
            points = np.array(pts, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)

            yield sse({"type": "status",
                       "message": f"프레임 로딩·전파 중… ({len(files)}프레임)"})
            data = _load_annotations(annots)
            done = 0
            with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
                state = pred.init_state(video_path=tmp)
                pred.add_new_points_or_box(inference_state=state, frame_idx=0,
                                           obj_id=1, points=points, labels=labels)
                for fidx, _obj_ids, logits in pred.propagate_in_video(state):
                    m = (logits[0] > 0.0).cpu().numpy()
                    m = m[0] if m.ndim == 3 else m
                    L, R = _sam2_rails_from_mask(m)
                    if L and R:
                        data[files[fidx]] = {"left_rail": L, "right_rail": R}
                        done += 1
                    if fidx % 3 == 0:
                        yield sse({"type": "progress", "index": fidx + 1,
                                   "total": len(files), "name": files[fidx]})
            tmpj = annots + ".tmp"
            with open(tmpj, "w") as f:
                json.dump(data, f)
            os.replace(tmpj, annots)
            yield sse({"type": "done", "count": done, "total": len(files),
                       "labeled_total": len(data)})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "error": str(e)})
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/label/interpolate", methods=["POST"])
def api_label_interpolate():
    """Fill unlabeled frames by interpolating between manually-labeled keyframes
    (rails as x=f(y) at a fixed y grid). No optical flow → no drift/divergence."""
    import json
    import numpy as np

    folder = (request.form.get("folder") or "").strip()
    annots = (request.form.get("annots") or "").strip()
    files = list_folder_images(folder)
    if files is None:
        return jsonify({"error": f"Not a folder: {folder}"}), 400
    if not annots.lower().endswith(".json") or not _annots_path_ok(annots):
        return jsonify({"error": "Invalid annotations path."}), 400
    data = _load_annotations(annots)
    keys = [i for i, f in enumerate(files) if f in data
            and len(data[f].get("left_rail", [])) >= 2
            and len(data[f].get("right_rail", [])) >= 2]
    if len(keys) < 2:
        return jsonify({"error": "Label at least 2 frames as keyframes first "
                                 "(e.g. first and last), then Interpolate."}), 400

    try:
        from PIL import Image as _Img
        H = _Img.open(os.path.join(folder, files[keys[0]])).size[1]
    except Exception:  # noqa: BLE001
        H = max(max(p[1] for p in data[files[k]]["left_rail"]) for k in keys) + 1
    N = 48  # y samples per rail

    def grid_for(rail_name):
        top = min(min(p[1] for p in data[files[k]][rail_name]) for k in keys)
        return np.linspace(top, H - 1, N)

    gy = {"left_rail": grid_for("left_rail"), "right_rail": grid_for("right_rail")}

    def x_on_grid(rail, grid):
        a = sorted(rail, key=lambda p: p[1])
        return np.interp(grid, [p[1] for p in a], [p[0] for p in a])

    # Precompute each keyframe's x(grid) per rail.
    kx = {r: {k: x_on_grid(data[files[k]][r], gy[r]) for k in keys}
          for r in ("left_rail", "right_rail")}

    def event_stream():
        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        def rail_points(xarr, grid):
            return [[int(round(x)), int(round(y))] for x, y in zip(xarr, grid)]

        try:
            yield sse({"type": "status",
                       "message": f"Interpolating {len(files)} frames between {len(keys)} keyframes…"})
            done = 0
            for i, name in enumerate(files):
                if name in data and i in keys:
                    continue  # keep manual keyframes as-is
                # locate surrounding keyframes
                lo = max([k for k in keys if k <= i], default=keys[0])
                hi = min([k for k in keys if k >= i], default=keys[-1])
                t = 0.0 if hi == lo else (i - lo) / (hi - lo)
                out = {}
                for r in ("left_rail", "right_rail"):
                    xa = kx[r][lo] * (1 - t) + kx[r][hi] * t
                    out[r] = rail_points(xa, gy[r])
                data[name] = out
                done += 1
                if i % 10 == 0:
                    yield sse({"type": "progress", "index": i + 1, "total": len(files), "name": name})
            tmp = annots + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, annots)
            yield sse({"type": "done", "count": done, "labeled_total": len(data)})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "error": str(e)})

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@app.route("/api/label/clear_all", methods=["POST"])
def api_label_clear_all():
    """Remove every label in an annotations file (reset the whole folder)."""
    import json
    annots = (request.form.get("annots") or "").strip()
    if not annots.lower().endswith(".json") or not _annots_path_ok(annots):
        return jsonify({"error": "Invalid annotations path."}), 400
    removed = len(_load_annotations(annots))
    tmp = annots + ".tmp"
    with open(tmp, "w") as f:
        json.dump({}, f)
    os.replace(tmp, annots)
    return jsonify({"ok": True, "removed": removed})


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
        multi_gpu = request.form.get("multi_gpu") == "true"
        full_gpus = list_full_gpus()
        if multi_gpu and len(full_gpus) < 2:
            multi_gpu = False  # nothing to parallelise over

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
        if multi_gpu:
            cmd.append("--multi-gpu")

        env = os.environ.copy()
        if multi_gpu:
            # Make all full (non-MIG) GPUs visible so DataParallel can use them.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in full_gpus)
        elif gpu_uuid:
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
                     "finetune": finetune, "gpu_preprocess": gpu_pre,
                     "multi_gpu": multi_gpu},
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
