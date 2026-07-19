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
    "/data3/bhkim/datasets/OSDaR23_unzip/9_station_ruebenkamp_9.7_extracted/rgb_center"
FIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 7
FRACS = [0.0, 0.3]
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


from PIL import ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def draw_rails(g, rails, color, width=9, dash=None):
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


def fill_path(img, rails, rgba, outline=None):
    """Fill the ego-path region (between the rails) as a translucent polygon."""
    L, R = rails
    if len(L) < 2 or len(R) < 2:
        return img
    poly = [tuple(p) for p in L] + [tuple(p) for p in R][::-1]
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(ov)
    g.polygon(poly, fill=rgba)
    if outline:
        g.line([tuple(p) for p in L], fill=outline, width=6)
        g.line([tuple(p) for p in R], fill=outline, width=6)
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


past_clean = [base_out(det_occ, im) for im in imgs[:-1]]
gt = labels[fn]
W0, H0 = size
# zoom on the track area so the rails are large and readable
CROP = (int(W0 * 0.10), int(H0 * 0.20), int(W0 * 0.95), H0)
titles = {0.0: "(a) 비폐색", 0.3: "(b) 폐색 30%", 0.4: "(c) 폐색 40%"}
panels = []
from src.utils.evaluate import compute_iou
from src.utils.postprocessing import rails_to_mask as _r2m
gtm = _r2m([gt["left_rail"], gt["right_rail"]], size)
for frac in FRACS:
    img_o = occlude(imgs[-1], frac)
    cur = base_out(det_occ, img_o)
    window = ([past_clean[0]] * (T - 1 - len(past_clean)) + past_clean + [cur]
              if past_clean else [cur] * T)
    single = det_occ.pred_to_result(cur[None, :], None, size)
    rnn_occ = det_occ.pred_to_result(refine(det_occ, window)[None, :], None, size)
    rnn_old = det_old.pred_to_result(refine(det_old, window)[None, :], None, size)
    print(f"occ {int(frac*100):2d}%: single={compute_iou(_r2m(single, size), gtm):.3f} "
          f"oldRNN={compute_iou(_r2m(rnn_old, size), gtm):.3f} "
          f"occRNN={compute_iou(_r2m(rnn_occ, size), gtm):.3f}", flush=True)
    canvas = img_o.copy()
    # filled translucent path regions (like the reference visualization):
    # red = single-frame path (diverges under occlusion), green = occ-RNN path
    if frac > 0:
        canvas = fill_path(canvas, single, (225, 30, 30, 110), outline=(225, 30, 30, 235))
    canvas = fill_path(canvas, rnn_occ, (25, 200, 60, 120), outline=(20, 190, 50, 235))
    g = ImageDraw.Draw(canvas)
    draw_rails(g, [gt["left_rail"], gt["right_rail"]], (255, 220, 0), 7, dash=10)
    if frac > 0:  # label the occluded input band
        f28 = ImageFont.truetype(FONT, 46)
        g.text((CROP[0] + 30, H0 - int(H0 * frac) + 18),
               "전두부 모사 입력 차폐 구간", font=f28, fill=(180, 180, 180))
    panel = canvas.crop(CROP)
    panel.thumbnail((1400, 1400))
    # title strip above each panel
    f_t = ImageFont.truetype(FONT_B, 34)
    strip = Image.new("RGB", (panel.width, 52), (255, 255, 255))
    ImageDraw.Draw(strip).text((10, 4), titles.get(frac, f"폐색 {int(frac*100)}%"),
                               font=f_t, fill=(0, 0, 0))
    panels += [strip, panel]

# legend strip
LEG = [((255, 220, 0), "선", "정답(GT)"),
       ((25, 200, 60), "면", "폐색 증강 RNN 경로(유지)"),
       ((225, 30, 30), "면", "단일 프레임 경로(이탈)")]
leg = Image.new("RGB", (panels[1].width, 60), (255, 255, 255))
g = ImageDraw.Draw(leg)
f_l = ImageFont.truetype(FONT, 30)
x = 14
for color, kind, name in LEG:
    if kind == "면":
        g.rectangle([x, 16, x + 56, 44], fill=tuple(int(c * 0.55 + 255 * 0.45) for c in color),
                    outline=color, width=3)
    else:
        g.line([(x, 30), (x + 56, 30)], fill=color, width=10)
    x += 66
    g.text((x, 10), name, font=f_l, fill=(0, 0, 0))
    x += g.textlength(name, font=f_l) + 40

panels = [leg] + panels
W = max(p.width for p in panels); H = sum(p.height + 4 for p in panels)
sheet = Image.new("RGB", (W, H), (255, 255, 255))
y = 0
for p in panels:
    sheet.paste(p, (0, y)); y += p.height + 4
out = "figures/fig_occlusion_verify.jpg"
sheet.save(out, quality=92)
print(f"saved {out} (frame {fn}, {sheet.size})", flush=True)
