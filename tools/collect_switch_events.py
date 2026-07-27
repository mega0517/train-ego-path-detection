"""Mine new switch events from a cab-view video.

Stage 1 (screen): download the video at a sane resolution, sample coarse frames
and lay them out as numbered contact sheets. A human (or a vision model) reads
the sheets and writes down which cells show a facing switch -- one look at a
sheet replaces dozens of single-frame lookups.

Stage 2 (cut): given those cell numbers, cluster nearby ones into events and cut
each event as the dataset's standard window: 20s at 4fps -> 81 frames, 1280px
wide. Frames without enough real detail are dropped, because the point of this
round is that the switch blades must actually be visible.

  python tools/collect_switch_events.py screen VIDEO_ID work_dir [--max-height 1080]
  python tools/collect_switch_events.py cut VIDEO_ID work_dir out_dir --cells 12,48,91
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

COARSE_EVERY = 3.0     # seconds between screening frames
SHEET_COLS, SHEET_ROWS = 10, 8
CELL_W, CELL_H = 320, 180
WINDOW_S = 20.0        # dataset convention: 20s window ...
EVENT_FPS = 4          # ... at 4 fps -> 81 frames
FRAME_WIDTH = 1280
MIN_DETAIL = 3.0
CLUSTER_GAP_S = 25.0


def detail_score(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    r = img[int(0.50 * h):int(0.95 * h), int(0.22 * w):int(0.78 * w)].astype(np.float32)
    if r.size < 1000:
        return 0.0
    small = cv2.resize(r, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_CUBIC)
    return float(np.abs(r - back).mean())


def video_path(work, vid):
    return os.path.join(work, f"{vid}.mp4")


def download(vid, work, max_height):
    out = video_path(work, vid)
    if os.path.exists(out):
        return out
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-N", "4",
           "-f", f"bestvideo[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]",
           "-o", out, f"https://www.youtube.com/watch?v={vid}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
    return out if r.returncode == 0 and os.path.exists(out) else None


def screen(args):
    os.makedirs(args.work, exist_ok=True)
    video = download(args.video_id, args.work, args.max_height)
    if not video:
        print("download failed")
        return 1
    coarse = os.path.join(args.work, "coarse")
    os.makedirs(coarse, exist_ok=True)
    if not os.listdir(coarse):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                        "-vf", f"fps=1/{COARSE_EVERY},scale={CELL_W}:-2", "-q:v", "4",
                        os.path.join(coarse, "c%05d.jpg")], timeout=7200)
    cells = sorted(f for f in os.listdir(coarse) if f.endswith(".jpg"))
    sheets_dir = os.path.join(args.work, "sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    for s in range((len(cells) + per - 1) // per):
        canvas = np.zeros((CELL_H * SHEET_ROWS, CELL_W * SHEET_COLS, 3), np.uint8)
        for i, fn in enumerate(cells[s * per:(s + 1) * per]):
            img = cv2.imread(os.path.join(coarse, fn))
            if img is None:
                continue
            img = cv2.resize(img, (CELL_W, CELL_H))
            n = s * per + i
            cv2.rectangle(img, (0, 0), (46, 16), (0, 0, 0), -1)
            cv2.putText(img, str(n), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
            r, c = divmod(i, SHEET_COLS)
            canvas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = img
        cv2.imwrite(os.path.join(sheets_dir, f"sheet{s:02d}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump({"video_id": args.video_id, "coarse_every": COARSE_EVERY,
               "n_cells": len(cells)}, open(os.path.join(args.work, "screen.json"), "w"))
    print(f"cells={len(cells)} sheets={len(os.listdir(sheets_dir))} -> {sheets_dir}")
    print(f"cell n 의 영상 시각 = n * {COARSE_EVERY}초")
    return 0


def rail_lines(path):
    """Count rail-like line segments in the near field.

    A plain double track shows a couple of long converging lines; a turnout or
    yard shows many more. Ranking coarse frames by this cuts the frames a human
    has to look at from ~900 to one sheetful, without needing to understand the
    scene.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    band = img[int(0.55 * h):h, :]
    band = cv2.GaussianBlur(band, (3, 3), 0)
    edges = cv2.Canny(band, 60, 160)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                           minLineLength=max(12, band.shape[0] // 4), maxLineGap=6)
    if segs is None:
        return 0.0
    total = 0.0
    for x1, y1, x2, y2 in segs[:, 0]:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 20 < ang < 160:          # rails run away from the camera, not across it
            total += np.hypot(x2 - x1, y2 - y1)
    return float(total / band.shape[0])


def rank(args):
    """Shortlist the coarse frames most likely to hold a switch."""
    info = json.load(open(os.path.join(args.work, "screen.json")))
    coarse = os.path.join(args.work, "coarse")
    cells = sorted(f for f in os.listdir(coarse) if f.endswith(".jpg"))
    scored = [(i, rail_lines(os.path.join(coarse, fn))) for i, fn in enumerate(cells)]
    scored.sort(key=lambda t: -t[1])
    # 이웃한 칸이 몰리지 않도록 최소 간격을 두고 고른다
    picked, used = [], set()
    for i, sc in scored:
        if any(abs(i - j) < args.min_gap for j in used):
            continue
        picked.append((i, sc)); used.add(i)
        if len(picked) >= SHEET_COLS * SHEET_ROWS:
            break
    picked.sort()
    canvas = np.zeros((CELL_H * SHEET_ROWS, CELL_W * SHEET_COLS, 3), np.uint8)
    for k, (i, sc) in enumerate(picked):
        img = cv2.imread(os.path.join(coarse, cells[i]))
        if img is None:
            continue
        img = cv2.resize(img, (CELL_W, CELL_H))
        cv2.rectangle(img, (0, 0), (86, 16), (0, 0, 0), -1)
        cv2.putText(img, f"{i} {sc:.0f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(k, SHEET_COLS)
        canvas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = img
    out = os.path.join(args.work, "ranked.jpg")
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump([{"cell": i, "score": round(sc, 1), "t": i * info["coarse_every"]}
               for i, sc in picked], open(os.path.join(args.work, "ranked.json"), "w"), indent=1)
    print(f"{len(cells)} cells -> top {len(picked)} on {out}")
    return 0


def cut(args):
    info = json.load(open(os.path.join(args.work, "screen.json")))
    every = info["coarse_every"]
    cells = sorted({int(c) for c in args.cells.split(",") if c.strip().isdigit()})
    if not cells:
        print("no cells")
        return 1
    groups, cur = [], [cells[0]]
    for c in cells[1:]:
        if (c - cur[-1]) * every <= CLUSTER_GAP_S:
            cur.append(c)
        else:
            groups.append(cur); cur = [c]
    groups.append(cur)
    print(f"{len(cells)} cells -> {len(groups)} events")

    # 이미 데이터셋에 있는 같은 영상의 이벤트와 겹치면 다시 만들지 않는다
    taken = []
    sw = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
    try:
        for e in json.load(open(os.path.join(sw, "index.json"))):
            if e.get("video_id") != args.video_id:
                continue
            m = os.path.join(sw, e["event"], "manifest.json")
            if os.path.exists(m):
                t = json.load(open(m)).get("center_time_s")
                if t is not None:
                    taken.append(float(t))
    except (OSError, ValueError):
        pass
    if taken:
        print(f"  이미 보유한 이벤트 {len(taken)}개와 {args.skip_within:.0f}초 이내면 건너뜀")

    video = video_path(args.work, args.video_id)
    os.makedirs(args.out_dir, exist_ok=True)
    made = []
    for gi, g in enumerate(groups):
        center = (sum(g) / len(g)) * every
        if any(abs(center - t) < args.skip_within for t in taken):
            print(f"  @{center:.0f}s: 기존 이벤트와 겹쳐 건너뜀")
            continue
        start = max(0.0, center - WINDOW_S / 2)
        name = f"new{args.video_id}_{gi:02d}"
        dst = os.path.join(args.out_dir, name)
        os.makedirs(dst, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
                        "-t", str(WINDOW_S + 0.3), "-i", video,
                        "-vf", f"fps={EVENT_FPS},scale={FRAME_WIDTH}:-2", "-q:v", "2",
                        os.path.join(dst, "f%03d.jpg")], timeout=900)
        frames = sorted(f for f in os.listdir(dst) if f.endswith(".jpg"))
        if len(frames) < 40:
            print(f"  {name}: only {len(frames)} frames, dropped")
            continue
        mid = frames[len(frames) // 2]
        hi = detail_score(os.path.join(dst, mid))
        if hi < args.min_detail:
            print(f"  {name}: hi={hi:.2f} too soft, dropped")
            continue
        json.dump({"center_frame": mid, "center_time_s": center, "fps": EVENT_FPS,
                   "n_frames": len(frames), "video_id": args.video_id,
                   "window": [start, start + WINDOW_S + 0.3],
                   "youtube_url": f"https://www.youtube.com/watch?v={args.video_id}",
                   "collected_round": "highres"}, open(os.path.join(dst, "manifest.json"), "w"),
                  ensure_ascii=False, indent=1)
        made.append({"event": name, "hi": round(hi, 2), "center_time_s": center,
                     "frames": len(frames)})
        print(f"  {name}: {len(frames)} frames hi={hi:.2f} @{center:.0f}s")
    json.dump(made, open(os.path.join(args.out_dir, f"_{args.video_id}_events.json"), "w"),
              indent=2)
    print(f"kept {len(made)} events")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen"); s.add_argument("video_id"); s.add_argument("work")
    s.add_argument("--max-height", type=int, default=1080)
    k = sub.add_parser("rank"); k.add_argument("work")
    k.add_argument("--min-gap", type=int, default=4)
    c = sub.add_parser("cut"); c.add_argument("video_id"); c.add_argument("work")
    c.add_argument("out_dir"); c.add_argument("--cells", required=True)
    c.add_argument("--min-detail", type=float, default=MIN_DETAIL)
    c.add_argument("--skip-within", type=float, default=25.0,
                   help="이미 보유한 같은 영상 이벤트와 이 초 이내면 건너뛴다")
    a = ap.parse_args()
    return {"screen": screen, "rank": rank, "cut": cut}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
