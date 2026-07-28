"""Label the ambiguous frames of a turnout passage from the frames that follow it.

During a diverging passage the tongue rail is occluded and the image cannot say
which track the train is taking; a few frames later the vehicle has yawed onto
the branch and the answer is obvious again. The ambiguity is therefore only
forward-looking. Reading the sequence backwards from a frame where the model has
recovered removes it: the future decides what the ambiguous frames meant, so the
label is correct by construction rather than by an annotator's judgement of a
frame that genuinely does not contain the answer.

The path is carried back one frame at a time by dense optical flow, the same
mechanism the labelling tool already trusts going forward, over the ~10 frames of
a passage -- short enough that ego-motion stays smooth.

Frames outside the ambiguous window keep the model's own prediction, which was
measured to be right there (2 of 80 transitions carry the failure).

Usage:
    python tools/backprop_labels.py <event-dir> [...] [--validate] [--out-name NAME]
"""

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.autocrop import Autocropper  # noqa: E402
from src.utils.interface import Detector  # noqa: E402

FAR = 0.25
# Measured on the four labelled diverging events: far-field error against the human
# labels peaks 1-4 frames after the manifest's switch-passage frame (0.100 at +2)
# and falls below 0.01 from +5 to +20. So the ambiguity is about a second long and
# the model is clean again well before the clip ends -- which is what makes reading
# it backwards cheap. Errors reappear past +20, but those are a SECOND turnout
# inside the same 20 s window, not this one; labelling them needs its own passage
# frame, so the window stops short of them.
WINDOW_START = 1
WINDOW_END = 6
RECOVERY_GAP = 1    # frames past the window to read the path back from


def frame_index(path):
    m = re.match(r"f(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def to_mask(res, size):
    if isinstance(res, Image.Image):
        return np.array(res.convert("L")) > 127
    img = Image.new("L", size, 0)
    pts = [tuple(p) for p in res[0]] + [tuple(p) for p in reversed(res[1])]
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def far_centre(mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size < 4:
        return None
    y = rows[0] + int(FAR * (rows[-1] - rows[0]))
    xs = np.nonzero(mask[y])[0]
    return float(xs.mean() / mask.shape[1]) if xs.size else None


def mask_to_rails(mask, step=8):
    """Left and right boundary of the filled path, as the label format wants them."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size < 4:
        return None
    left, right = [], []
    for y in range(rows[0], rows[-1] + 1, step):
        xs = np.nonzero(mask[y])[0]
        if xs.size:
            left.append([int(xs[0]), int(y)])
            right.append([int(xs[-1]), int(y)])
    if len(left) < 3:
        return None
    return left, right


def warp_back(mask_next, img_t, img_next):
    """Bring frame t+1's mask into frame t using the flow between them."""
    g0 = cv2.cvtColor(img_t, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_next, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = g0.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    mx = xs + flow[..., 0]
    my = ys + flow[..., 1]
    warped = cv2.remap(mask_next.astype(np.uint8) * 255, mx, my,
                       interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped > 127


def passage_frame(folder):
    """The frame the crawler marked as the switch passage, or None."""
    try:
        name = json.load(open(os.path.join(folder, "manifest.json")))["center_frame"]
        return int(re.match(r"f(\d+)", name).group(1))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def find_window(folder, frames):
    """Positions in ``frames`` of the ambiguous window and the frame to read back from.

    Anchored on the passage frame rather than discovered from the prediction: a
    detector that jumps is exactly what we are trying to correct, so using its
    jumps to decide where to correct it would let a confident wrong model declare
    itself right. The passage frame comes from the crawler, independent of any
    model.
    """
    centre = passage_frame(folder)
    if centre is None:
        return None
    index = {frame_index(f): i for i, f in enumerate(frames)}
    start = index.get(centre + WINDOW_START)
    end = index.get(centre + WINDOW_END)
    carrier = index.get(centre + WINDOW_END + RECOVERY_GAP)
    if start is None or end is None or carrier is None or end <= start:
        return None
    return start, end, carrier


def process(folder, model, det, out_name, validate):
    frames = sorted(glob.glob(os.path.join(folder, "f*.jpg")), key=frame_index)
    if len(frames) < 30:
        return None
    det.crop_coords = Autocropper(det.config)
    det.reset_temporal()
    masks, sizes, centres = {}, {}, {}
    for f in frames:
        with Image.open(f) as im:
            im = im.convert("RGB")
            sizes[f] = im.size
            masks[f] = to_mask(det.detect(im), im.size)
        centres[f] = far_centre(masks[f])

    win = find_window(folder, frames)
    fixed = []
    if win is not None:
        start, end, carrier_i = win
        carrier = masks[frames[carrier_i]]
        for i in range(carrier_i - 1, start - 1, -1):
            img_t = np.array(Image.open(frames[i]).convert("RGB"))
            img_n = np.array(Image.open(frames[i + 1]).convert("RGB"))
            carrier = warp_back(carrier, img_t, img_n)
            if carrier.sum() < 50:
                break
            masks[frames[i]] = carrier
            fixed.append(os.path.basename(frames[i]))

    labels = {}
    for f in frames:
        rails = mask_to_rails(masks[f])
        if rails is not None:
            labels[os.path.basename(f)] = {"left_rail": rails[0], "right_rail": rails[1]}

    out = {"event": os.path.basename(folder), "model": model,
           "window": [os.path.basename(frames[win[0]]), os.path.basename(frames[win[1]])]
           if win else None,
           "relabelled": sorted(fixed), "n_relabelled": len(fixed)}

    if not validate:
        with open(os.path.join(folder, out_name), "w") as fh:
            json.dump(labels, fh)
    else:
        gt_path = os.path.join(folder, "egopath_labels.json")
        gt = json.load(open(gt_path)) if os.path.isfile(gt_path) else {}
        if gt:
            out["validation"] = compare(labels, gt, folder, sizes, frames, fixed)
    return out


def compare(auto, gt, folder, sizes, frames, fixed):
    """How close the produced labels are to the human ones, inside and outside the window."""
    inside, outside = [], []
    for f in frames:
        name = os.path.basename(f)
        if name not in auto or name not in gt:
            continue
        size = sizes[f]
        a = to_mask([auto[name]["left_rail"], auto[name]["right_rail"]], size)
        g = to_mask([gt[name]["left_rail"], gt[name]["right_rail"]], size)
        u = np.count_nonzero(a | g)
        iou = np.count_nonzero(a & g) / u if u else 1.0
        ca, cg = far_centre(a), far_centre(g)
        err = abs(ca - cg) if (ca is not None and cg is not None) else None
        (inside if name in fixed else outside).append((iou, err))

    def summarise(rows):
        if not rows:
            return None
        ious = [r[0] for r in rows]
        errs = [r[1] for r in rows if r[1] is not None]
        return {"frames": len(rows), "iou": float(np.mean(ious)),
                "far_err": float(np.mean(errs)) if errs else None,
                "wrong_branch": int(sum(1 for e in errs if e > 0.05))}

    return {"relabelled": summarise(inside), "kept": summarise(outside)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+", help="event directories")
    ap.add_argument("--model", default="egopath/weights/twinkling-rocket-21")
    ap.add_argument("--out-name", default="egopath_labels_auto.json")
    ap.add_argument("--validate", action="store_true",
                    help="compare against the human labels instead of writing files")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    det = Detector(model_path=args.model, crop_coords=None,
                   runtime="pytorch", device="cuda:0")
    rows = []
    for folder in args.events:
        r = process(folder, os.path.basename(args.model), det, args.out_name, args.validate)
        if r is None:
            continue
        rows.append(r)
        w = f"{r['window'][0]}~{r['window'][1]}" if r["window"] else "없음"
        line = f"{r['event'][:40]:<42} 창={w:<18} 역전파={r['n_relabelled']:>2}프레임"
        v = r.get("validation") or {}
        if v.get("relabelled"):
            b = v["relabelled"]
            line += (f"  [역전파구간 IoU={b['iou']:.3f} 원거리오차={b['far_err']:.3f}"
                     f" 오분기={b['wrong_branch']}/{b['frames']}]")
        print(line, flush=True)

    if args.validate:
        agg = [r["validation"] for r in rows if r.get("validation")]
        for key, label in (("relabelled", "역전파한 프레임"), ("kept", "모델 라벨 유지 프레임")):
            parts = [a[key] for a in agg if a.get(key)]
            if not parts:
                continue
            n = sum(p["frames"] for p in parts)
            iou = sum(p["iou"] * p["frames"] for p in parts) / n
            wrong = sum(p["wrong_branch"] for p in parts)
            print(f"\n{label}: {n}프레임, 사람 라벨 대비 IoU {iou:.4f}, "
                  f"오분기 {wrong} ({100 * wrong / n:.1f}%)")
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        with open(args.summary, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
