"""Assemble the labelled turnout events into a train set and a held-out set.

The events are the evaluation domain, so a model trained on them must never see
the events it is scored on: frames within one event are near-duplicates, and a
frame-level split would put the same switch passage on both sides. The split is
therefore by EVENT.

It is also STRATIFIED by whether the event exhibits the failure under study. Only
7 of the 28 events contain a wrong-branch episode at all, so an arbitrary split
can leave the held-out set with almost none of them -- and a held-out set where
nothing fails cannot show whether anything was fixed. Which events those are is
decided by a third model that takes no part in the comparison and was never
fine-tuned on any event, so the criterion carries no information about either
run being scored.
"""

import glob
import json
import os

from PIL import Image

ROOT = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
OUT_TRAIN = os.path.join(ROOT, "egopath_labels_train.json")
OUT_HELD = os.path.join(ROOT, "held_out_events.json")

events = []
for p in sorted(glob.glob(os.path.join(ROOT, "evt*", "egopath_labels.json"))):
    labels = json.load(open(p))
    if labels:
        events.append((os.path.basename(os.path.dirname(p)), labels))

FAILING = {
    "evt030__KA7Ov8kJsHU__AD_w03_003",
    "evt338__4dgANFlOgYY__sw0338",
    "evt416___OAFRgVNgws__sw0416",
    "evt003__n0UnTeYC9Us__D_gleiswechsel_0040",
    "evt006__ABSAeSrvsG8__AC_w00_019",
    "evt133__bchgiDFFrIs__sw0171",
    "evt047__pwj-8qhmy7o__R_limmyard_0059",
}

names = [e for e, _ in events]
fail = [e for e in names if e in FAILING]
clean = [e for e in names if e not in FAILING]
# Hold out most of the failing events -- they are the only ones carrying signal --
# plus enough clean ones to check the method does not break the easy cases.
held = set(fail[::2] + fail[1::2][:2] + clean[::4])
train, dropped = {}, []
for name, labels in events:
    if name in held:
        continue
    for frame, ann in labels.items():
        path = os.path.join(ROOT, name, frame)
        if not os.path.isfile(path):
            continue
        # The crop-box augmentation reads the bottom row of the rails mask, so a
        # label that stops short of the image bottom crashes it. Three percent of
        # these labels do (the annotator stopped where the rails leave the frame),
        # and dropping them costs less than special-casing the augmentation.
        with Image.open(path) as im:
            height = im.size[1]
        lowest = max(max(y for _, y in ann["left_rail"]),
                     max(y for _, y in ann["right_rail"]))
        if lowest < height - 1:
            dropped.append(f"{name}/{frame}")
            continue
        train[f"{name}/{frame}"] = ann

with open(OUT_TRAIN, "w") as f:
    json.dump(train, f)
with open(OUT_HELD, "w") as f:
    json.dump(sorted(held), f, indent=1)

print(f"events: {len(events)} -> train {len(events) - len(held)}, held-out {len(held)}")
print(f"train frames: {len(train)} (dropped {len(dropped)} whose label stops above the image bottom)")
print(f"wrote {OUT_TRAIN}\nwrote {OUT_HELD}")
