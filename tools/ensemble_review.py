"""Rebuild the review list from where the existing models disagree.

The first version flagged the frames just after the crawler's switch-passage
marker, a window measured on the four labelled diverging events. Across all 28
labelled events that window turns out to hold only 4 of the 36 frames the
auto-labelling model actually gets wrong: the errors are mostly elsewhere, at
second turnouts the crawler never marked and at other hard frames.

Disagreement between independently trained models locates them far better --
median spread 0.0476 on wrong frames against 0.0039 on correct ones, a factor of
twelve -- and catches 29 of 36 at a threshold of 0.02, for 18% of frames reviewed
against 8%. It is a detector only: voting the models does not beat the best
single one (2.2% wrong against 1.5%), because per-frame models facing an image
that does not contain the answer are wrong together.

Usage:
    python tools/ensemble_review.py [--events output/diverging_events.json]
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
from tools.backprop_labels import far_centre, frame_index, passage_frame, to_mask  # noqa: E402

EVENTS = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
ALL_MODELS = [
    "brilliant-horse-15", "chromatic-laughter-5", "fortuitous-goat-12",
    "fortuitous-pig-8", "logical-tree-1", "osdar23-seg-r18", "twinkling-rocket-21",
]
# A model that outputs nothing sends the frame to review, so a weak member of the
# panel spends someone's afternoon on frames the rest of the panel agrees about.
# Measured over the 3981 cached frames, osdar23-seg-r18 is silent on 990 (25%) and
# brilliant-horse-15 on 599 (15%); the other five are silent on 4% or less. Those
# two are also the two worst against the 2268 hand-labelled frames (15.1% and 8.5%
# of far centres off by more than 0.05 of the image width, against 2.4-5.2% for the
# rest). Dropping them halves the queue -- 844 frames to 416 over the 52 reliable
# events -- and still flags every frame where the auto-labelling model is off by
# more than 0.10, so nothing gross goes unreviewed.
MODELS = [m for m in ALL_MODELS
          if m not in ("osdar23-seg-r18", "brilliant-horse-15")]
THRESHOLD = 0.02   # measured: catches 81% of the wrong frames for 18% reviewed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="output/diverging_events.json")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated panel. A frame is reviewed when any member"
                         " of the panel gives no path, so adding a model that often"
                         " gives none costs review time rather than buying accuracy.")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--auto-name", default="egopath_labels_auto.json",
                    help="the auto-labels this queue is review for. A frame missing"
                         " from it has no label at all, which the panel cannot notice:"
                         " the models answer the image, not the label file.")
    ap.add_argument("--summary", default="output/review_list_ensemble.json")
    ap.add_argument("--cache", default="output/ensemble_spread_div.json")
    ap.add_argument("--event-factor", type=float, default=4.0,
                    help="flag a frame whose disagreement exceeds this many times its"
                         " own event's median. An absolute threshold cannot work across"
                         " events: the models disagree everywhere on some clips, and"
                         " flagging every frame of those says nothing about which frame"
                         " to look at.")
    ap.add_argument("--event-reject", type=float, default=0.05,
                    help="an event whose median disagreement exceeds this is reported as"
                         " unreliable as a whole rather than frame by frame.")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    names = [r["event"] for r in json.load(open(args.events)) if r.get("diverging")]
    folders = [os.path.join(EVENTS, n) for n in names]
    print(f"{len(folders)} events x {len(models)} models: {', '.join(models)}\n")

    centres = {}
    if os.path.isfile(args.cache):
        centres = json.load(open(args.cache))
        print(f"(reusing {args.cache})")
    for model in [] if centres else models:
        det = Detector(model_path=os.path.join("egopath", "weights", model),
                       crop_coords=None, runtime="pytorch", device="cuda:0")
        for folder in folders:
            det.crop_coords = Autocropper(det.config)
            det.drop_unsupported_smoothing()
            det.reset_temporal()
            for f in sorted(glob.glob(os.path.join(folder, "f*.jpg")), key=frame_index):
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    c = far_centre(to_mask(det.detect(im), im.size))
                if c is not None:
                    centres.setdefault(f, {})[model] = c
        print(f"  {model} done", flush=True)
    if not os.path.isfile(args.cache):
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w") as fh:
            json.dump(centres, fh)

    rows, total_review, total_frames, rejected = [], 0, 0, []
    for folder in folders:
        frames = sorted(glob.glob(os.path.join(folder, "f*.jpg")), key=frame_index)
        centre = passage_frame(folder)
        spreads = {}
        for f in frames:
            # The cache may hold models outside the panel, so read the panel only.
            got = [v for m, v in centres.get(f, {}).items() if m in models]
            spreads[f] = float(np.std(got)) if len(got) == len(models) else None
        known = [v for v in spreads.values() if v is not None]
        med = float(np.median(known)) if known else 0.0
        if med > args.event_reject:
            rejected.append((os.path.basename(folder), med))
        limit = max(args.threshold, args.event_factor * med)
        # A frame the auto-labeller skipped has no label to check, and agreement
        # between the models cannot reveal that: they answer the image, not the
        # label file. Left to the spread rule such a frame passes silently and ends
        # up the one thing worse than a wrong label, which is no label and nobody
        # looking.
        try:
            with open(os.path.join(folder, args.auto_name), encoding="utf-8") as fh:
                auto = json.load(fh)
        except (OSError, ValueError):
            auto = {}
        labelled = {k for k, v in auto.items()
                    if isinstance(v, dict) and v.get("left_rail") and v.get("right_rail")}
        review = []
        for f in frames:
            v = spreads[f]
            if os.path.basename(f) not in labelled:
                review.append({"frame": os.path.basename(f),
                               "reason": "자동 라벨 없음 (처음부터 그려야 함)"})
            elif v is None:
                review.append({"frame": os.path.basename(f),
                               "reason": "일부 모델이 경로를 못 냄"})
            elif v > limit:
                review.append({"frame": os.path.basename(f),
                               "reason": f"모델 불일치 {v:.3f} (이 이벤트 중앙값의 {v / max(med, 1e-6):.0f}배)",
                               "spread": round(v, 4)})
        total_frames += len(frames)
        total_review += len(review)
        with open(os.path.join(folder, "review_frames.json"), "w") as fh:
            json.dump({"event": os.path.basename(folder), "passage_frame": centre,
                       "method": f"ensemble disagreement > {args.threshold}",
                       "models": models,
                       "review": review}, fh, indent=1, ensure_ascii=False)
        rows.append({"event": os.path.basename(folder), "frames": len(frames),
                     "median_spread": round(med, 4), "n_review": len(review),
                     "unreliable": med > args.event_reject,
                     "review": [r["frame"] for r in review]})

    print(f"\n{len(rows)} events, {total_frames} frames")
    print(f"검수 필요: {total_review} 프레임 ({100 * total_review / max(total_frames, 1):.1f}%), "
          f"이벤트당 평균 {total_review / max(len(rows), 1):.1f}")
    if rejected:
        print(f"\n이벤트 전체가 불신 구간 ({len(rejected)}건, 중앙 불일치 > {args.event_reject}):")
        for name, med in sorted(rejected, key=lambda x: -x[1])[:10]:
            print(f"  중앙 불일치 {med:.3f}  {name}")
        print("  → 이 이벤트들은 프레임 검수가 아니라 데이터셋에서 제외를 검토")
    ok = [r for r in rows if not r["unreliable"]]
    if ok:
        n = sum(r["n_review"] for r in ok)
        print(f"\n신뢰 가능한 {len(ok)}개 이벤트: 검수 {n}프레임 "
              f"({100 * n / sum(r['frames'] for r in ok):.1f}%, 이벤트당 {n / len(ok):.1f})")
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
