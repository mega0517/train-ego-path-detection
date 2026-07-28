"""Drop the auto-labels that the rest of the passage proves wrong.

Auto-labels are trustworthy everywhere except the second after the switch, which
is exactly the second the prior-path channel has to learn from. Training on them
unfiltered would teach the model the mistake it is meant to fix.

The mistake is detectable without an annotator. Which branch the train actually
took is written plainly in the frames after it has yawed onto it, and a label in
the ambiguous window that has the path swinging the *other* way contradicts that.
Such frames are removed rather than corrected: reconstructing the true geometry
needs an annotator, but recognising that a label disagrees with where the train
demonstrably went does not.

What survives is a set of correct labels inside the ambiguous window -- the
supervision that was missing -- at the cost of the frames where the detector
flipped.

Usage:
    python tools/filter_window_labels.py --validate      # score against human labels
    python tools/filter_window_labels.py --build OUT.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.backprop_labels import far_centre, frame_index, passage_frame, to_mask  # noqa: E402

EVENTS = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
PRE = (-12, -4)      # frames before the passage: the branch has not been taken yet
POST = (7, 20)       # after the yaw: which branch was taken is unambiguous here
WINDOW = (1, 6)      # the ambiguous frames themselves
MARGIN = 0.03        # how far a label may swing against the branch before it is wrong


def centres_of(labels, folder):
    out = {}
    for name, ann in labels.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        with Image.open(path) as im:
            size = im.size
        c = far_centre(to_mask([ann["left_rail"], ann["right_rail"]], size))
        if c is not None:
            out[frame_index(name)] = c
    return out


def classify(folder, labels):
    """-> (kept, dropped, undecidable) frame-name lists for the ambiguous window."""
    centre = passage_frame(folder)
    if centre is None:
        return [], [], []
    c = centres_of(labels, folder)
    pre = [v for k, v in c.items() if centre + PRE[0] <= k <= centre + PRE[1]]
    post = [v for k, v in c.items() if centre + POST[0] <= k <= centre + POST[1]]
    window = {k: v for k, v in c.items() if centre + WINDOW[0] <= k <= centre + WINDOW[1]}
    if not pre or not post or not window:
        return [], [], sorted(window)
    pre_m, post_m = float(np.mean(pre)), float(np.mean(post))
    direction = np.sign(post_m - pre_m)
    if direction == 0 or abs(post_m - pre_m) < MARGIN:
        # a passage the far field never resolves gives no evidence either way
        return [], [], [f"f{k:03d}.jpg" for k in sorted(window)]
    kept, dropped = [], []
    for k, v in sorted(window.items()):
        name = next((n for n in labels if frame_index(n) == k), f"f{k:03d}.jpg")
        (dropped if (v - pre_m) * direction < -MARGIN else kept).append(name)
    return kept, dropped, []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="score the filter against the human labels on the labelled"
                         " diverging events")
    ap.add_argument("--build", default=None,
                    help="write a combined training annotation file to this path")
    ap.add_argument("--events", default="output/diverging_train_events.json")
    ap.add_argument("--auto-name", default="egopath_labels_auto.json")
    args = ap.parse_args()

    if args.validate:
        labelled = [d for d in sorted(glob.glob(os.path.join(EVENTS, "evt*")))
                    if os.path.isfile(os.path.join(d, "egopath_labels.json"))
                    and os.path.isfile(os.path.join(d, args.auto_name))
                    and json.load(open(os.path.join(d, "egopath_labels.json")))]
        tp = fp = tn = fn = 0
        for folder in labelled:
            auto = json.load(open(os.path.join(folder, args.auto_name)))
            gt = json.load(open(os.path.join(folder, "egopath_labels.json")))
            kept, dropped, _ = classify(folder, auto)
            if not kept and not dropped:
                continue
            gc = centres_of(gt, folder)
            ac = centres_of(auto, folder)
            for name, is_dropped in [(n, False) for n in kept] + [(n, True) for n in dropped]:
                k = frame_index(name)
                if k not in gc or k not in ac:
                    continue
                really_wrong = abs(ac[k] - gc[k]) > 0.05
                if is_dropped and really_wrong:
                    tp += 1
                elif is_dropped and not really_wrong:
                    fp += 1
                elif not is_dropped and really_wrong:
                    fn += 1
                else:
                    tn += 1
            print(f"{os.path.basename(folder)[:40]:<42} 유지={len(kept)} 버림={len(dropped)}")
        total = tp + fp + tn + fn
        print(f"\n임계 구간 {total}프레임에서:")
        print(f"  실제 오분기 {tp + fn}개 중 {tp}개를 걸러냄 (재현율 "
              f"{100 * tp / max(tp + fn, 1):.0f}%)")
        print(f"  버린 {tp + fp}개 중 {fp}개는 멀쩡한 프레임 (오폐기 "
              f"{100 * fp / max(tp + fp, 1):.0f}%)")
        print(f"  남긴 {tn + fn}개 중 오분기가 {fn}개 남음 "
              f"({100 * fn / max(tn + fn, 1):.1f}%)")
        return

    names = json.load(open(args.events))
    combined, stats = {}, {"kept": 0, "dropped": 0, "outside": 0, "events": 0}
    for name in names:
        folder = os.path.join(EVENTS, name)
        auto_path = os.path.join(folder, args.auto_name)
        if not os.path.isfile(auto_path):
            continue
        auto = json.load(open(auto_path))
        kept, dropped, undecidable = classify(folder, auto)
        drop = set(dropped) | set(undecidable)
        review = os.path.join(folder, "review_frames.json")
        if os.path.isfile(review):
            # frames flagged for a reason other than this passage are second
            # turnouts the crawler never marked; nothing here can vouch for them
            r = json.load(open(review))
            drop |= {x["frame"] for x in r["review"] if x["reason"] != "분기 통과 구간"}
        for frame, ann in auto.items():
            if frame in drop:
                stats["dropped"] += 1
                continue
            combined[f"{name}/{frame}"] = ann
            stats["kept" if frame in kept else "outside"] += 1
        stats["events"] += 1

    if args.build:
        with open(args.build, "w") as fh:
            json.dump(combined, fh)
        print(f"{stats['events']} events -> {len(combined)} frames")
        print(f"  임계 구간에서 살린 프레임: {stats['kept']}")
        print(f"  임계 구간 밖: {stats['outside']}")
        print(f"  버린 프레임: {stats['dropped']}")
        print(f"wrote {args.build}")


if __name__ == "__main__":
    main()
