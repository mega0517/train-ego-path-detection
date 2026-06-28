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
import threading

import torch
import yaml
from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from PIL import Image

from src.utils.autocrop import Autocropper
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_WEIGHTS_PATH = os.path.join(BASE_PATH, "egopath", "weights")
RNN_WEIGHTS_PATH = os.path.join(BASE_PATH, "egopathrnn", "weights")
DEFAULT_IMAGE = os.path.join(BASE_PATH, "data", "egopath.jpg")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi")

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


def pil_to_data_uri(img, fmt="JPEG"):
    """Encode a PIL image as a base64 data URI for inline display."""
    if img is None:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"


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
        if "file" in request.files and request.files["file"].filename:
            img = Image.open(request.files["file"].stream)
        elif os.path.exists(DEFAULT_IMAGE):
            img = Image.open(DEFAULT_IMAGE)
        else:
            return jsonify({"error": "No image provided"}), 400
        img.load()

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
            for frame_idx, img, total in decode_video_frames(video_path):
                if frame_idx % stride == 0:
                    single_vis, rnn_vis = run_both(img, det_single, det_rnn)
                    payload = {
                        "type": "frame",
                        "frame": frame_idx,
                        "total": total or 0,
                        "emitted": emitted + 1,
                        "elapsed": round(time.time() - t0, 1),
                        "original": pil_to_data_uri(img),
                        "single": pil_to_data_uri(single_vis),
                        "rnn": pil_to_data_uri(rnn_vis),
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
