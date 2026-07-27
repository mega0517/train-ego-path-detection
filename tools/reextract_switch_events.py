"""Re-extract switch events from their source videos at full resolution.

The first crawl downloaded sources at <=720p, so events cut from them were
effectively upscaled: the switch blades (텅레일) were a smear and a closed
tongue could not be told from an open one. The source videos are mostly 1080p
or 4K, so the fix is not new footage but re-cutting the same windows from a
properly downloaded copy.

Each event's manifest already records video_id and the window, and the
split/merge curation is already done, so re-extraction keeps all of that work.
Frames come out at the dataset's usual 1280px width, which is a real downscale
from a >=1080p source instead of an upscale from 360p.

Usage:
  python tools/reextract_switch_events.py plan.json out_dir [--limit N]

plan.json: {video_id: [{event, window, fps, region, title, url}, ...]}
Events whose re-extracted frames still lack detail are skipped and reported.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

FRAME_WIDTH = 1280
MIN_DETAIL = 3.0   # same "hi" metric that flagged the unusable events


def detail_score(path):
    """High-frequency detail in the central-lower band (see switch_sharpness)."""
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


def download(video_id, dest, max_height=1080):
    """Video-only download, capped so 4K sources stay a manageable size."""
    out = os.path.join(dest, f"{video_id}.mp4")
    if os.path.exists(out):
        return out
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-N", "4",
           "-f", f"bestvideo[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]",
           "-o", out, f"https://www.youtube.com/watch?v={video_id}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print("    download timed out", flush=True)
        return None
    if r.returncode != 0 or not os.path.exists(out):
        print(f"    download failed: {r.stderr.strip()[-200:]}", flush=True)
        return None
    return out


def extract(video, start, duration, out_dir, fps=1):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration),
           "-i", video, "-vf", f"fps={fps},scale={FRAME_WIDTH}:-2",
           "-q:v", "2", os.path.join(out_dir, "f%03d.jpg")]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0:
        print(f"    ffmpeg failed: {r.stderr.strip()[-200:]}", flush=True)
    return sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out_dir")
    ap.add_argument("--work", default="/data3/bhkim/datasets/_reextract_work")
    ap.add_argument("--limit", type=int, default=0, help="process at most N videos")
    ap.add_argument("--min-detail", type=float, default=MIN_DETAIL)
    ap.add_argument("--max-height", type=int, default=1080)
    args = ap.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    videos = list(plan)[: args.limit] if args.limit else list(plan)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for vi, vid in enumerate(videos, 1):
        items = plan[vid]
        print(f"[{vi}/{len(videos)}] {vid}: {len(items)} events", flush=True)
        video = download(vid, args.work, args.max_height)
        if not video:
            results.extend({"event": it["event"], "ok": False, "why": "download"} for it in items)
            continue
        for it in items:
            ev = it["event"]
            win = it.get("window") or []
            if len(win) != 2:
                c = it.get("center_time_s")
                if c is None:
                    results.append({"event": ev, "ok": False, "why": "no window"})
                    continue
                win = [max(0, c - 20), c + 20]
            # 이 데이터셋은 20초 창을 4fps로 잘라 81프레임을 만든다 (1fps가 아니다).
            # 원본 manifest의 fps를 그대로 따라야 프레임 수와 시간 간격이 일치한다.
            fps = it.get("fps") or 4
            tmp = os.path.join(args.work, "_frames", ev)
            shutil.rmtree(tmp, ignore_errors=True)
            frames = extract(video, win[0], win[1] - win[0], tmp, fps=fps)
            want = it.get("n_frames")
            if want and abs(len(frames) - want) > 2:
                print(f"    {ev}: 프레임 {len(frames)} (원본 {want}) — 창/fps 확인 필요", flush=True)
            if len(frames) < 10:
                results.append({"event": ev, "ok": False, "why": f"only {len(frames)} frames"})
                shutil.rmtree(tmp, ignore_errors=True)
                continue
            mid = frames[len(frames) // 2]
            hi = detail_score(os.path.join(tmp, mid))
            if hi < args.min_detail:
                results.append({"event": ev, "ok": False, "why": f"still soft (hi={hi:.2f})"})
                shutil.rmtree(tmp, ignore_errors=True)
                continue
            dst = os.path.join(args.out_dir, ev)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.move(tmp, dst)
            with open(os.path.join(dst, "manifest.json"), "w") as f:
                json.dump({"center_frame": mid, "center_time_s": it.get("center_time_s"),
                           "fps": fps, "n_frames": len(frames), "region": it.get("region", ""),
                           "title": it.get("title", ""), "video_id": vid,
                           "window": win, "youtube_url": it.get("url", ""),
                           "reextracted": True, "source_height": args.max_height}, f,
                          ensure_ascii=False, indent=1)
            results.append({"event": ev, "ok": True, "hi": round(hi, 3), "frames": len(frames)})
            print(f"    {ev}: {len(frames)} frames, hi={hi:.2f}", flush=True)
        try:
            os.remove(video)   # keep the working dir small; one video at a time
        except OSError:
            pass
        with open(os.path.join(args.out_dir, "_reextract_results.json"), "w") as f:
            json.dump(results, f, indent=1)

    ok = [r for r in results if r["ok"]]
    print(f"\nDONE: {len(ok)}/{len(results)} events re-extracted")
    if ok:
        his = sorted(r["hi"] for r in ok)
        print(f"  detail: min={his[0]:.2f} med={his[len(his)//2]:.2f} max={his[-1]:.2f}")
    for r in results:
        if not r["ok"]:
            print(f"  skip {r['event']}: {r['why']}")


if __name__ == "__main__":
    sys.exit(main())
