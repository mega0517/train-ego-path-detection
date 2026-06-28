# TEP-Net Web Application

A browser-based front end for train ego-path detection inference. It mirrors the
desktop [`gui_app.py`](gui_app.py) (model / device / crop selection and the
side-by-side **Original → Single-frame → RNN** comparison) but is served over
HTTP, so it works on a remote/headless machine without an X server.

## Features

- **Model selection** — base models from `egopath/weights/` and their temporal
  counterparts from `egopathrnn/weights/`, shown as base/RNN pairs.
- **Device selection** — CPU, CUDA (per-GPU), or MPS. CUDA enumeration failures
  degrade gracefully to CPU instead of crashing the page.
- **Crop modes** — `None`, `Auto` (autocrop), or `Manual` (explicit
  left/top/right/bottom coordinates).
- **Image inference** — upload an image (or use the bundled `data/egopath.jpg`
  sample) and get the three-panel comparison.
- **Video inference** — upload an MP4/AVI; frames are streamed back live via
  Server-Sent Events with a progress bar. A frame *stride* lets you process
  every N-th frame for speed.
- **Side-by-side comparison** — single-frame vs temporal-RNN result, with the
  single-frame panel falling back to the RNN's base net when no standalone
  single-frame checkpoint exists.

## Installation

```bash
pip install -r requirements.txt   # adds Flask alongside the existing deps
```

## Usage

```bash
./launch_web.sh                    # serves on http://0.0.0.0:5000
# or
python web_app.py --host 0.0.0.0 --port 5000
```

Environment overrides for the launch script:

```bash
HOST=127.0.0.1 PORT=8080 ./launch_web.sh
```

Then open `http://<host>:<port>` in a browser.

## Walkthrough

1. **Input** — choose an image or video file. With nothing selected, the bundled
   sample image is used.
2. **Model** — pick a model from the list. The `RNN` badge is highlighted when a
   temporal checkpoint is available for that model.
3. **Device** — select the compute device (defaults to `cuda:0` when present).
4. **Crop** — `None`, `Auto`, or `Manual` (reveals coordinate inputs).
5. **Video options** — set the frame stride for video inputs.
6. **Run Inference** — results appear in the three preview panels. For video,
   frames stream in with a progress bar; use **Stop** to cancel.

## Notes

- Detectors are cached per `(model, device)` so repeated runs skip model
  reloading; crop and temporal state are reset on every request.
- Uploads are capped at 512 MB (`MAX_CONTENT_LENGTH` in `web_app.py`).
- The web app does not write result files to disk — results are returned to the
  browser. Use [`detect.py`](detect.py) for saved-to-disk batch inference.

## References

- [GUI_README.md](GUI_README.md) — the desktop GUI equivalent
- [README.md](README.md) — main project documentation
- [AGENTS.md](AGENTS.md) — development guidelines
