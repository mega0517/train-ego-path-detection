"""SAM2 video auto-labeling: prompt the ego-track on frame 0, propagate through the
frame sequence, extract left/right rails per frame -> egopath_labels.json."""
import argparse, glob, json, os, shutil, tempfile
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_video_predictor

CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT = "bin/sam2/sam2.1_hiera_base_plus.pt"


def rdp(points, eps=10.0):
    if len(points) < 3:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    denom = (dx * dx + dy * dy) ** 0.5
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = (((px - x1) ** 2 + (py - y1) ** 2) ** 0.5 if denom == 0
             else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(points[:idx + 1], eps)[:-1] + rdp(points[idx:], eps)
    return [points[0], points[-1]]


def rails_from_mask(mask, n=48):
    """left/right boundary of the track mask, sampled on an n-row grid, then RDP."""
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) < 2:
        return None, None
    grid = np.linspace(rows.min(), rows.max(), n)
    left, right = [], []
    for y in grid:
        yy = int(round(y)); cols = np.where(mask[yy])[0]
        if len(cols):
            left.append([int(cols[0]), yy]); right.append([int(cols[-1]), yy])
    if len(left) < 2:
        return None, None
    return rdp(left), rdp(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default="", help="x,y;x,y (pixels on frame 0); default = auto bottom-centre")
    ap.add_argument("--preview", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.frames)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise SystemExit("no frames")
    W, H = Image.open(os.path.join(args.frames, names[0])).size
    print(f"{len(names)} frames, {W}x{H}", flush=True)

    # SAM2 wants a dir of jpgs with integer stems -> temp symlink dir
    tmp = tempfile.mkdtemp(prefix="sam2f_")
    for i, n in enumerate(names):
        os.symlink(os.path.abspath(os.path.join(args.frames, n)),
                   os.path.join(tmp, f"{i:06d}.jpg"))

    # prompt on frame 0
    if args.points:
        pts = [[float(a) for a in p.split(",")] for p in args.points.split(";")]
    else:  # auto: a vertical line of positive points up the bottom-centre (ego track)
        cx = W / 2
        pts = [[cx, H * f] for f in (0.98, 0.9, 0.82, 0.74, 0.66)]
    points = np.array(pts, dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    print("prompt points:", points.tolist(), flush=True)

    predictor = build_sam2_video_predictor(CFG, CKPT, device=args.device)
    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=tmp)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        points=points, labels=labels)
        masks = {}
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            m = (logits[0] > 0.0).cpu().numpy()
            masks[fidx] = m[0] if m.ndim == 3 else m

    data, ok = {}, 0
    for i, n in enumerate(names):
        m = masks.get(i)
        if m is None:
            continue
        L, R = rails_from_mask(m)
        if L and R:
            data[n] = {"left_rail": L, "right_rail": R}; ok += 1
    tmpj = args.out + ".tmp"
    json.dump(data, open(tmpj, "w")); os.replace(tmpj, args.out)
    print(f"wrote {ok}/{len(names)} -> {args.out}", flush=True)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for n in [names[0], names[len(names) // 2], names[-1]]:
            if n not in data:
                continue
            im = Image.open(os.path.join(args.frames, n)).convert("RGB")
            d = ImageDraw.Draw(im)
            L = [(x, y) for x, y in data[n]["left_rail"]]
            R = [(x, y) for x, y in data[n]["right_rail"]]
            if len(L) > 1: d.line(L, fill=(0, 255, 0), width=6)
            if len(R) > 1: d.line(R, fill=(255, 40, 40), width=6)
            im.thumbnail((720, 720)); im.save(os.path.join(args.preview, "prev_" + n + ".jpg"), quality=85)
        print("previews saved", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
