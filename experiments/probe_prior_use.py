"""Does the trained model actually read its prior channel?

A null result on the branch metric has two very different explanations: the prior
is used but does not help, or the prior is ignored. They are distinguished by
running the same frames twice -- once with the recursive prior the model would
normally see, once with that channel forced empty -- and asking whether the
predictions differ at all. Identical outputs mean the channel was learned away,
which is the input-side analogue of an output-side refiner collapsing to identity.
"""

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
HELD = json.load(open(os.path.join(EVENTS, "held_out_events.json")))


def mask_of(res, size):
    if isinstance(res, Image.Image):
        return np.array(res.convert("L")) > 127
    m = Image.new("L", size, 0)
    pts = [tuple(p) for p in res[0]] + [tuple(p) for p in reversed(res[1])]
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=1)
    return np.array(m, dtype=bool)


def run(weights, blind):
    det = Detector(model_path=weights, crop_coords=None, runtime="pytorch",
                   device="cuda:0")
    out = {}
    for ev in sorted(HELD):
        folder = os.path.join(EVENTS, ev)
        det.crop_coords = Autocropper(det.config)
        det.reset_temporal()
        for f in sorted(glob.glob(os.path.join(folder, "f*.jpg"))):
            with Image.open(f) as im:
                im = im.convert("RGB")
                res = det.detect(im)
                out[f] = mask_of(res, im.size)
            if blind:
                det.prior_state = None   # never let the prior reach the next frame
    return out


w = sys.argv[1]
with_prior = run(w, blind=False)
without = run(w, blind=True)

ious, changed = [], 0
for k in with_prior:
    a, b = with_prior[k], without[k]
    u = np.count_nonzero(a | b)
    iou = np.count_nonzero(a & b) / u if u else 1.0
    ious.append(iou)
    if iou < 0.99:
        changed += 1
ious = np.array(ious)
print(f"{os.path.basename(w)}: {len(ious)} frames")
print(f"  사전정보 있음 vs 없음 예측 일치도 IoU: 평균 {ious.mean():.4f}, "
      f"최소 {ious.min():.4f}")
print(f"  1% 이상 달라진 프레임: {changed} / {len(ious)} ({100*changed/len(ious):.1f}%)")
