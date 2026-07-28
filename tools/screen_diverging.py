"""Find the turnout events where the train changes track, not the ones it runs through.

The branch failure concentrates almost entirely in diverging moves: on the
labelled events, 3 of 4 diverging passages produce a wrong-branch episode against
4 of 24 straight-through ones. Yet only 4 of the 28 labelled events diverge, which
is why training on them carries almost no signal about the phenomenon.

Labels are what is scarce, so the screen has to work without them. It measures how
far the model's own far-field path has moved by the time the passage is well
behind the train -- a window late enough that the ambiguous frames are over and
the model has recovered, so a wrong branch during the passage does not decide the
verdict. A straight-through passage ends where it started; a diverging one does
not.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.autocrop import Autocropper  # noqa: E402
from src.utils.interface import Detector  # noqa: E402

EVENTS = os.environ.get("SWITCH_EVENTS_DIR", "/data3/bhkim/datasets/Rail_switch_crawling/switch_events")


def far_centre(res, size):
    if isinstance(res, Image.Image):
        m = np.array(res.convert("L")) > 127
    else:
        img = Image.new("L", size, 0)
        pts = [tuple(p) for p in res[0]] + [tuple(p) for p in reversed(res[1])]
        if len(pts) < 3:
            return None
        ImageDraw.Draw(img).polygon(pts, fill=1)
        m = np.array(img, dtype=bool)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size < 4:
        return None
    y = rows[0] + int(0.25 * (rows[-1] - rows[0]))
    xs = np.nonzero(m[y])[0]
    return float(xs.mean() / m.shape[1]) if xs.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="egopath/weights/chromatic-laughter-5")
    ap.add_argument("--threshold", type=float, default=0.06,
                    help="far-field shift (fraction of width) above which the event"
                         " counts as diverging. Default 0.06, the value that separates"
                         " the labelled events.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="output/diverging_events.json")
    args = ap.parse_args()

    det = Detector(model_path=args.model, crop_coords=None,
                   runtime="pytorch", device="cuda:0")
    folders = sorted(d for d in glob.glob(os.path.join(EVENTS, "evt*")) if os.path.isdir(d))
    if args.limit:
        folders = folders[: args.limit]

    rows = []
    for n, folder in enumerate(folders):
        frames = sorted(glob.glob(os.path.join(folder, "f*.jpg")))
        if len(frames) < 40:
            continue
        try:
            centre = int(re.match(r"f(\d+)", json.load(
                open(os.path.join(folder, "manifest.json")))["center_frame"]).group(1))
        except (OSError, ValueError, KeyError):
            centre = len(frames) // 2
        det.crop_coords = Autocropper(det.config)
        det.reset_temporal()
        vals = {}
        for f in frames:
            m = re.match(r"f(\d+)", os.path.basename(f))
            if m is None:
                continue
            idx = int(m.group(1))
            with Image.open(f) as im:
                im = im.convert("RGB")
                v = far_centre(det.detect(im), im.size)
            if v is not None:
                vals[idx] = v
        pre = [v for k, v in vals.items() if centre - 12 <= k <= centre - 4]
        post = [v for k, v in vals.items() if centre + 15 <= k <= centre + 25]
        if not pre or not post:
            continue
        shift = float(np.mean(post) - np.mean(pre))
        rows.append({"event": os.path.basename(folder), "shift": shift,
                     "diverging": abs(shift) > args.threshold, "centre": centre})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(folders)} scanned, "
                  f"{sum(r['diverging'] for r in rows)} diverging so far", flush=True)

    rows.sort(key=lambda r: -abs(r["shift"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    div = [r for r in rows if r["diverging"]]
    print(f"\nscanned {len(rows)} events -> {len(div)} diverging "
          f"({100 * len(div) / max(len(rows), 1):.0f}%)")
    for r in div[:20]:
        print(f"  {r['shift']:+.3f}  {r['event']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
