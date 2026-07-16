"""Figure: turnout-occlusion diagnosis & prescription.

Top rows: an OSDaR23 turnout frame with the bottom h% occluded (h = 0/20/40%),
overlaying GT (yellow dashed), the single-frame prediction (red) and the
occlusion-trained RNN prediction (green). Past frames in the RNN window are the
clean preceding frames, as in eval_occlusion.py.
Usage: python experiments/fig_occlusion.py <occ_model_dir> [<seq_dir> <frame_idx>]
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
from PIL import Image, ImageDraw
from torchvision.transforms import v2 as transforms
from src.utils.common import to_scaled_tensor
from src.utils.interface import Detector

OCC = sys.argv[1] if len(sys.argv) > 1 else "egopathrnn/weights/chromatic-laughter-5RNN-occ"
SEQ = sys.argv[2] if len(sys.argv) > 2 else \
    "/data3/bhkim/datasets/OSDaR23_unzip/11_main_station_11.1_extracted/rgb_center"
FIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 4
FRACS = [0.0, 0.3, 0.4]
FILL = (15, 15, 15)
T = 5
device = "cuda"

det_occ = Detector(OCC, None, "pytorch", device)
det_old = Detector("egopathrnn/weights/chromatic-laughter-5RNN", None, "pytorch", device)
resize = transforms.Resize(det_occ.config["input_shape"][1:][::-1])

labels = json.load(open(os.path.join(SEQ, "egopath_labels.json")))
frames = sorted(f for f in os.listdir(SEQ) if f.lower().endswith((".jpg", ".png")))
fn = frames[FIDX]
imgs = []
for j in range(max(0, FIDX - T + 1), FIDX + 1):
    im = Image.open(os.path.join(SEQ, frames[j])); im.load()
    imgs.append(im.convert("RGB") if im.mode != "RGB" else im)
size = imgs[-1].size


def base_out(det, img):
    t = resize(to_scaled_tensor(img).unsqueeze(0)).to(device)
    with torch.no_grad():
        return det.model.base_forward(t).cpu().numpy()[0]


def occlude(img, frac):
    if frac <= 0:
        return img
    img = img.copy()
    img.paste(FILL, (0, int(round(img.height * (1 - frac))), img.width, img.height))
    return img


def refine(det, window):
    seq = torch.from_numpy(np.stack(window)[None]).to(device)
    with torch.no_grad():
        return det.model.refine(seq).cpu().numpy()[0]


def draw_rails(g, rails, color, width=7, dash=None):
    for rail in rails:
        pts = [tuple(p) for p in rail]
        if len(pts) < 2:
            continue
        if dash:
            for i in range(len(pts) - 1):
                if (i // dash) % 2 == 0:
                    g.line([pts[i], pts[i + 1]], fill=color, width=width)
        else:
            g.line(pts, fill=color, width=width)


past_clean = [base_out(det_occ, im) for im in imgs[:-1]]
gt = labels[fn]
panels = []
for frac in FRACS:
    img_o = occlude(imgs[-1], frac)
    cur = base_out(det_occ, img_o)
    window = ([past_clean[0]] * (T - 1 - len(past_clean)) + past_clean + [cur]
              if past_clean else [cur] * T)
    single = det_occ.pred_to_result(cur[None, :], None, size)
    rnn_occ = det_occ.pred_to_result(refine(det_occ, window)[None, :], None, size)
    rnn_old = det_old.pred_to_result(refine(det_old, window)[None, :], None, size)
    canvas = img_o.copy(); g = ImageDraw.Draw(canvas)
    draw_rails(g, [gt["left_rail"], gt["right_rail"]], (255, 220, 0), 5, dash=12)
    draw_rails(g, single, (230, 40, 40), 7)
    draw_rails(g, rnn_old, (60, 120, 255), 7)
    draw_rails(g, rnn_occ, (40, 220, 60), 7)
    g.rectangle([0, 0, 620, 54], fill=(0, 0, 0))
    g.text((12, 6), f"occlusion {int(frac*100)}%  |  GT-- yellow  single red  "
                    f"baseline-RNN blue  occl-RNN green", fill=(255, 255, 255))
    canvas.thumbnail((1100, 1100))
    panels.append(canvas)

W = max(p.width for p in panels); H = sum(p.height + 8 for p in panels)
sheet = Image.new("RGB", (W, H), (20, 20, 20))
y = 0
for p in panels:
    sheet.paste(p, (0, y)); y += p.height + 8
out = "figures/fig_occlusion_verify.jpg"
sheet.save(out, quality=90)
print(f"saved {out} (frame {fn})", flush=True)
