"""Remove burned-in overlay text (subtitles, watermarks, timestamps) from
switch-event frames.

Overlays sit at fixed pixel positions across a sequence while scene text moves
with the camera, so: detect text boxes on every frame (EasyOCR/CRAFT detector,
script-agnostic), keep only pixels flagged in >= PERSIST of frames, dilate,
and inpaint that static mask on every frame.

Originals are moved to <event>/orig/ before cleaned frames are written in
place (same filenames), so the web app and eval keep working unchanged and
everything is reversible. Already-cleaned events (non-empty orig/) are skipped.

Usage:
  python tools/remove_overlay_text.py --eval-sample      # events in eval_sample.json
  python tools/remove_overlay_text.py --events evtA,evtB # specific events
  python tools/remove_overlay_text.py --all              # every evt* folder
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

SW = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
PREVIEW = os.path.join(SW, "_overlay_clean_preview")
PERSIST = 0.40   # box must appear on >=40% of frames to count as overlay
DILATE = 7       # px, cover anti-aliased edges
INPAINT_R = 4
MAX_MASK_FRAC = 0.25  # refuse to inpaint more than a quarter of the image

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--eval-sample", action="store_true")
group.add_argument("--events", help="comma-separated event folder names")
group.add_argument("--all", action="store_true")
args = parser.parse_args()

if args.eval_sample:
    with open(os.path.join(SW, "eval_sample.json")) as f:
        events = [m["event"] for m in json.load(f)]
elif args.events:
    events = [os.path.basename(e.strip()) for e in args.events.split(",") if e.strip()]
else:
    events = sorted(d for d in os.listdir(SW)
                    if d.startswith("evt") and os.path.isdir(os.path.join(SW, d)))

import easyocr  # noqa: E402  (heavy import after cheap arg parsing)

reader = easyocr.Reader(["en"], gpu=True, verbose=False)

os.makedirs(PREVIEW, exist_ok=True)
report_path = os.path.join(PREVIEW, "report.json")
try:
    with open(report_path) as f:
        report = json.load(f)
except (OSError, ValueError):
    report = []

for i, ev in enumerate(events):
    d = os.path.join(SW, ev)
    if not os.path.isdir(d):
        print(f"[{i + 1}/{len(events)}] {ev}: not a folder, skip", flush=True)
        continue
    orig_dir = os.path.join(d, "orig")
    if os.path.isdir(orig_dir) and os.listdir(orig_dir):
        print(f"[{i + 1}/{len(events)}] {ev}: already cleaned, skip", flush=True)
        continue
    frames = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    if not frames:
        continue

    # pass 1: accumulate text-box hit counts per pixel
    acc = None
    for fn in frames:
        img = cv2.imread(os.path.join(d, fn))
        if img is None:
            continue
        if acc is None:
            acc = np.zeros(img.shape[:2], np.float32)
        boxes, free = reader.detect(img, text_threshold=0.6, low_text=0.3)
        m = np.zeros(acc.shape, np.uint8)
        for x1, x2, y1, y2 in (boxes[0] if boxes else []):
            cv2.rectangle(m, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
        for poly in (free[0] if free else []):
            cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
        acc += m
    if acc is None:
        continue

    static = (acc / len(frames)) >= PERSIST
    mask = static.astype(np.uint8) * 255
    if mask.any():
        mask = cv2.dilate(mask, np.ones((DILATE, DILATE), np.uint8))
    frac = float(static.mean())

    if not mask.any():
        report.append({"event": ev, "overlay_px_frac": 0.0, "cleaned": 0})
        print(f"[{i + 1}/{len(events)}] {ev}: no static overlay found", flush=True)
        continue
    if frac > MAX_MASK_FRAC:
        report.append({"event": ev, "overlay_px_frac": frac, "cleaned": 0,
                       "skipped": "mask too large"})
        print(f"[{i + 1}/{len(events)}] {ev}: mask {frac:.1%} too large, SKIPPED", flush=True)
        continue

    # pass 2: inpaint every frame (originals -> orig/)
    os.makedirs(orig_dir, exist_ok=True)
    mid = frames[len(frames) // 2]
    for fn in frames:
        src = os.path.join(d, fn)
        img = cv2.imread(src)
        out = cv2.inpaint(img, mask, INPAINT_R, cv2.INPAINT_TELEA)
        shutil.move(src, os.path.join(orig_dir, fn))
        cv2.imwrite(src, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if fn == mid:
            before = cv2.imread(os.path.join(orig_dir, fn))
            side = np.concatenate([before, out], axis=1)
            cv2.imwrite(os.path.join(PREVIEW, f"{ev}__{fn}"), side,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    report.append({"event": ev, "overlay_px_frac": frac, "cleaned": len(frames)})
    print(f"[{i + 1}/{len(events)}] {ev}: cleaned {len(frames)} frames "
          f"(overlay {frac:.2%} of image)", flush=True)

with open(report_path, "w") as f:
    json.dump(report, f, indent=1)
print("DONE")
