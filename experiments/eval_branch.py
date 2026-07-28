"""Measure branch errors on the labelled turnout events.

IoU alone hides the failure this experiment is about. When the model follows the
wrong track through a switch, the predicted path still overlaps the labelled one
over most of its length -- the two only separate in the far field -- so a whole
wrong-branch episode costs a few IoU points and reads as ordinary degradation.
What actually happened is that the path pointed at the wrong track for two
seconds.

So the headline metric here is where the far field of the prediction sits
relative to the far field of the label. A frame whose far-field centre is off by
more than a threshold is a *wrong-branch frame*; a maximal run of at least two of
them is an *episode*, which is what an operator would notice. IoU is reported
alongside so a method cannot buy branch stability with accuracy.

Usage:
    python experiments/eval_branch.py <weights-dir>[:<label>] [...] [--tau 0.05]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.autocrop import Autocropper  # noqa: E402
from src.utils.interface import Detector  # noqa: E402

EVENTS = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
FAR = 0.25  # sample the far field at 25% down the path's own vertical range


def rails_to_polygon(left, right, size):
    mask = Image.new("L", size, 0)
    pts = [tuple(p) for p in left] + [tuple(p) for p in reversed(right)]
    if len(pts) >= 3:
        ImageDraw.Draw(mask).polygon(pts, fill=1)
    return np.array(mask, dtype=bool)


def far_field_centre(mask):
    """Mean x of the mask at FAR down its vertical extent, as a fraction of width."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size < 4:
        return None
    y = rows[0] + int(FAR * (rows[-1] - rows[0]))
    xs = np.nonzero(mask[y])[0]
    return float(xs.mean() / mask.shape[1]) if xs.size else None


def result_to_mask(res, size):
    if isinstance(res, Image.Image):
        return np.array(res.convert("L")) > 127
    return rails_to_polygon(res[0], res[1], size)


def episodes(flags):
    """Maximal runs of >=2 consecutive wrong-branch frames."""
    out, run = [], 0
    for f in list(flags) + [False]:
        if f:
            run += 1
        else:
            if run >= 2:
                out.append(run)
            run = 0
    return out


def evaluate(weights_dir, tau, only=None):
    det = Detector(model_path=weights_dir, crop_coords=None,
                   runtime="pytorch", device="cuda:0")
    ious, wrong, per_event = [], [], []
    for lbl_path in sorted(glob.glob(os.path.join(EVENTS, "evt*", "egopath_labels.json"))):
        labels = json.load(open(lbl_path))
        if not labels:
            continue
        folder = os.path.dirname(lbl_path)
        if only is not None and os.path.basename(folder) not in only:
            continue
        det.crop_coords = Autocropper(det.config)
        det.drop_unsupported_smoothing()
        det.reset_temporal()          # each event is an independent sequence
        ev_iou, ev_wrong = [], []
        for name in sorted(labels):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            with Image.open(path) as im:
                im = im.convert("RGB")
                pred = result_to_mask(det.detect(im), im.size)
                gt = rails_to_polygon(labels[name]["left_rail"],
                                      labels[name]["right_rail"], im.size)
            union = np.count_nonzero(pred | gt)
            ev_iou.append(np.count_nonzero(pred & gt) / union if union else 0.0)
            cp, cg = far_field_centre(pred), far_field_centre(gt)
            # A frame with no far field to compare is not evidence either way.
            ev_wrong.append(bool(cp is not None and cg is not None and abs(cp - cg) > tau))
        if not ev_iou:
            continue
        ious += ev_iou
        wrong += ev_wrong
        per_event.append({
            "event": os.path.basename(folder),
            "frames": len(ev_iou),
            "iou": float(np.mean(ev_iou)),
            "wrong_frames": int(sum(ev_wrong)),
            "episodes": episodes(ev_wrong),
        })
    return {
        "events": len(per_event),
        "frames": len(ious),
        "iou": float(np.mean(ious)),
        "wrong_rate": float(np.mean(wrong)),
        "episodes": sum(len(e["episodes"]) for e in per_event),
        "longest_episode": max((max(e["episodes"], default=0) for e in per_event), default=0),
        "events_with_episode": sum(1 for e in per_event if e["episodes"]),
        "per_event": per_event,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="<weights-dir>[:<label>]")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="far-field offset (fraction of width) above which a frame counts"
                         " as wrong-branch. Default 0.05.")
    ap.add_argument("--events", default=None,
                    help="JSON list of event names to restrict the evaluation to."
                         " Use the held-out list when the models were fine-tuned on"
                         " these events, or the comparison scores its own training data.")
    ap.add_argument("--out", default=None, help="write the full result as JSON here")
    args = ap.parse_args()

    only = set(json.load(open(args.events))) if args.events else None
    if only:
        print(f"restricted to {len(only)} events\n")
    results = {}
    for spec in args.models:
        d, _, label = spec.partition(":")
        label = label or os.path.basename(d.rstrip("/"))
        results[label] = evaluate(d, args.tau, only)
        r = results[label]
        print(f"{label}: {r['events']} events / {r['frames']} frames")

    print(f"\n{'model':<22}{'IoU':>9}{'오분기 프레임':>16}{'에피소드':>10}"
          f"{'최장':>8}{'해당 이벤트':>12}")
    for label, r in results.items():
        print(f"{label:<22}{r['iou']:>9.4f}{r['wrong_rate']*100:>14.1f}%"
              f"{r['episodes']:>10}{r['longest_episode']:>8}"
              f"{r['events_with_episode']:>10}/{r['events']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tau": args.tau, "results": results}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
