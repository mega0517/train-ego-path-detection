"""Auto-label diverging turnout events and list the frames a human still has to fix.

Auto-labelling these clips is nearly free -- the detector is accurate everywhere
except the second or so where the switch decides the route -- so the useful output
is not the labels but the short list of frames where it is not to be trusted.
Two things put a frame on that list:

- The passage window. Measured against the human labels on the four labelled
  diverging events, far-field error peaks 1-4 frames after the crawler's switch
  passage frame and is negligible from +7 to +20. Roughly one frame in five in
  that window is on the wrong branch, so the whole window is flagged: it is six
  frames, and which of them went wrong cannot be known without looking.

- Jumps elsewhere. A 20 s clip often contains a second turnout, which the crawler
  did not mark; those show up as far-field jumps far above the clip's own noise.
  They are flagged too, so the list is not blind to switches outside the window.

Both automatic alternatives to human review were tried and failed on this data:
carrying the recovered path back with optical flow (33% wrong-branch against 12%
for leaving the prediction alone -- 4 fps is too coarse for dense flow) and
running the sequence in reverse (5 wrong frames to 4, at lower IoU).

Usage:
    python tools/autolabel_review.py --events output/diverging_events.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.autocrop import Autocropper  # noqa: E402
from src.utils.interface import Detector  # noqa: E402
from tools.backprop_labels import (  # noqa: E402
    far_centre, frame_index, mask_to_rails, passage_frame, to_mask,
)

EVENTS = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
WINDOW = (1, 6)      # frames after the passage frame that the measurement flags
JUMP_FACTOR = 8.0    # far-field jump, relative to the clip's own median, that flags a frame
JUMP_FLOOR = 0.04    # ...and an absolute floor, so a very steady clip does not flag noise


def process(folder, det, out_name):
    frames = sorted(glob.glob(os.path.join(folder, "f*.jpg")), key=frame_index)
    if len(frames) < 30:
        return None
    det.crop_coords = Autocropper(det.config)
    det.drop_unsupported_smoothing()
    det.reset_temporal()

    labels, centres = {}, {}
    for f in frames:
        with Image.open(f) as im:
            im = im.convert("RGB")
            mask = to_mask(det.detect(im), im.size)
        rails = mask_to_rails(mask)
        if rails is not None:
            labels[os.path.basename(f)] = {"left_rail": rails[0], "right_rail": rails[1]}
        centres[f] = far_centre(mask)

    review, reasons = set(), {}

    centre = passage_frame(folder)
    if centre is not None:
        for k in range(centre + WINDOW[0], centre + WINDOW[1] + 1):
            for f in frames:
                if frame_index(f) == k:
                    name = os.path.basename(f)
                    review.add(name)
                    reasons[name] = "분기 통과 구간"

    diffs = [(i, abs(centres[frames[i + 1]] - centres[frames[i]]))
             for i in range(len(frames) - 1)
             if centres[frames[i]] is not None and centres[frames[i + 1]] is not None]
    if len(diffs) >= 10:
        med = float(np.median([d for _, d in diffs]))
        for i, d in diffs:
            if d > max(JUMP_FACTOR * med, JUMP_FLOOR):
                for j in (i, i + 1):
                    name = os.path.basename(frames[j])
                    if name not in review:
                        review.add(name)
                        reasons[name] = f"급격한 이동 {d:.3f} (중앙값의 {d / med:.0f}배)"

    with open(os.path.join(folder, out_name), "w") as fh:
        json.dump(labels, fh)
    with open(os.path.join(folder, "review_frames.json"), "w") as fh:
        json.dump({"event": os.path.basename(folder),
                   "passage_frame": centre,
                   "review": [{"frame": n, "reason": reasons[n]} for n in sorted(review)]},
                  fh, indent=1, ensure_ascii=False)
    return {"event": os.path.basename(folder), "frames": len(labels),
            "passage_frame": centre, "n_review": len(review),
            "review": sorted(review)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="output/diverging_events.json",
                    help="screen_diverging.py output; only entries marked diverging are used")
    ap.add_argument("--model", default="egopath/weights/twinkling-rocket-21")
    ap.add_argument("--out-name", default="egopath_labels_auto.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", default="output/review_list.json")
    args = ap.parse_args()

    listed = json.load(open(args.events))
    names = [r["event"] for r in listed if r.get("diverging")]
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} diverging events to auto-label\n")

    det = Detector(model_path=args.model, crop_coords=None,
                   runtime="pytorch", device="cuda:0")
    rows = []
    for i, name in enumerate(names):
        r = process(os.path.join(EVENTS, name), det, args.out_name)
        if r is None:
            continue
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(names)} done", flush=True)

    total_frames = sum(r["frames"] for r in rows)
    total_review = sum(r["n_review"] for r in rows)
    print(f"\n{len(rows)} events, {total_frames} frames auto-labelled")
    print(f"검수 필요: {total_review} 프레임 "
          f"({100 * total_review / max(total_frames, 1):.1f}%), "
          f"이벤트당 평균 {total_review / max(len(rows), 1):.1f}")
    worst = sorted(rows, key=lambda r: -r["n_review"])[:10]
    print("\n검수량이 많은 이벤트:")
    for r in worst:
        print(f"  {r['n_review']:>3} 프레임  {r['event']}")

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {args.summary}")
    print(f"이벤트별 목록은 각 폴더의 review_frames.json")


if __name__ == "__main__":
    main()
