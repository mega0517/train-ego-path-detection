#!/usr/bin/env python3
"""Run Alpamayo-R1 over a video and record everything it returns.

The model does not produce a video. Per request it takes four consecutive
frames plus an ego history and returns a predicted future trajectory, a
chain-of-causation string, and the inference time. This script walks a video,
asks once per sampled step, and writes the answers to one JSON file so the web
app can replay them against the footage without holding a 10B model open.

The ego history is synthesised as straight-line constant speed. A video alone
does not say how the vehicle was moving, and the alternative -- estimating it
from the footage -- would put a second model's error inside this one's input.
Where a real trajectory log exists, pass it with --history.

Usage:
    python tools/alpamayo_run.py clip.mp4 --out clip.alpamayo.json --fps 2
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ALPA_ROOT = os.environ.get("ALPAMAYO_ROOT", "/data3/bhkim/workspace/alpamayo")
N_FRAMES = 4          # the model's message builder expects four views
N_HISTORY = 16        # history steps, matching the server's own self-test


def sample_frames(video, fps, out_dir):
    """Write frames sampled at *fps* as PNG; return their paths in order."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(0.1, fps))))
    paths, idx, kept = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            p = os.path.join(out_dir, f"s{kept:05d}.png")
            cv2.imwrite(p, frame)
            paths.append((p, idx / src_fps))
            kept += 1
        idx += 1
    cap.release()
    return paths, src_fps


HISTORY_HZ = 10.0     # the server's own self-test spaces history at 1.4 m for 14 m/s


def straight_history(speed_mps):
    """Ego history for constant forward motion: t0 at the origin, past behind.

    Spacing follows HISTORY_HZ, not the video sampling rate. Tying it to the
    sampling rate was wrong: at 2 fps it put 7 m between history steps, the
    model read that as 70 m/s, and predicted 435 m of travel where the
    self-test's 14 m/s history gives 88 m.
    """
    d = speed_mps / HISTORY_HZ
    return [[-(N_HISTORY - 1 - i) * d, 0.0, 0.0] for i in range(N_HISTORY)]


def resolve_gpu(spec):
    """Turn a physical GPU index into its UUID; pass UUIDs through unchanged.

    One of the cards on this host has MIG enabled, and once any card is split
    the numeric CUDA_VISIBLE_DEVICES ordering stops matching nvidia-smi's. Asking
    for "2" landed a run on a 1g.10gb slice of card 0, which died out of memory
    part-way through loading. UUIDs are unambiguous.
    """
    spec = str(spec)
    if spec.startswith(("GPU-", "MIG-")):
        return spec
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return spec                       # no nvidia-smi: leave it to CUDA
    for line in out.splitlines():
        idx, _, uuid = line.partition(",")
        if idx.strip() == spec:
            return uuid.strip()
    raise SystemExit(f"no GPU with index {spec}")


class Server:
    """The alpamayo server speaks JSON lines over stdin/stdout, not HTTP."""

    def __init__(self, gpu, model, root):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=resolve_gpu(gpu))
        cmd = [os.path.join(root, ".venv", "bin", "python"),
               os.path.join(root, "isaac_bridge", "alpamayo_server.py"),
               "--alpamayo-root", root, "--model", model, "--attn", "sdpa"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=None, text=True, bufsize=1, env=env)
        # The readiness banner only appears once the weights are on the GPU,
        # which is minutes on a cold cache. Anything before it is progress noise.
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("server exited before signalling ready")
            try:
                if json.loads(line).get("ready"):
                    return
            except ValueError:
                continue

    def infer(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("server closed stdout (crashed?)")
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="default: <video>.alpamayo.json")
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate (default 2)")
    ap.add_argument("--speed", type=float, default=14.0,
                    help="assumed constant speed in m/s (default 14, about 50 km/h)")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--model", default="nvidia/Alpamayo-R1-10B")
    ap.add_argument("--root", default=ALPA_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="stop after N steps (0 = all)")
    ap.add_argument("--diffusion-steps", type=int, default=5,
                    help="euler steps for the trajectory sampler (default 5; the "
                         "model's own default of 10 costs 220 ms more per request "
                         "and moves the path by 0.18 m)")
    ap.add_argument("--history", default=None,
                    help="JSON file with a real ego history (T,3); overrides --speed")
    args = ap.parse_args()

    out = args.out or (os.path.splitext(args.video)[0] + ".alpamayo.json")
    tmp = tempfile.mkdtemp(prefix="alpamayo_frames_")
    frames, src_fps = sample_frames(args.video, args.fps, tmp)
    if len(frames) < N_FRAMES:
        raise SystemExit(f"need at least {N_FRAMES} sampled frames, got {len(frames)}")

    hist = (json.load(open(args.history)) if args.history
            else straight_history(args.speed))

    print(f"{len(frames)} frames sampled at {args.fps} fps (source {src_fps:.1f} fps)")
    print(f"starting server on GPU {args.gpu} — a cold model cache takes minutes")
    srv = Server(args.gpu, args.model, args.root)
    print("server ready")

    steps, t_start = [], time.time()
    total = len(frames) - N_FRAMES + 1
    if args.limit:
        total = min(total, args.limit)
    for i in range(total):
        window = frames[i:i + N_FRAMES]
        resp = srv.infer({"images": [p for p, _ in window],
                          "ego_history_xyz": hist,
                          "num_traj_samples": 1, "temperature": 0.6,
                          "diffusion_steps": args.diffusion_steps})
        rec = {"step": i, "time": window[-1][1], "ok": bool(resp.get("ok"))}
        if resp.get("ok"):
            rec.update(pred_xyz=resp["pred_xyz"], coc=resp.get("coc"),
                       infer_ms=resp.get("infer_ms"))
        else:
            rec["error"] = resp.get("error")
        steps.append(rec)
        print(f"  step {i + 1}/{total}  t={rec['time']:.2f}s  "
              f"{resp.get('infer_ms', '-')} ms", flush=True)
    srv.close()

    doc = {"video": os.path.abspath(args.video), "sample_fps": args.fps,
           "source_fps": src_fps, "assumed_speed_mps": args.speed,
           "attn": "sdpa", "diffusion_steps": args.diffusion_steps,
           "history": hist, "n_steps": len(steps),
           "wall_s": round(time.time() - t_start, 1), "steps": steps}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print(f"wrote {out}  ({len(steps)} steps, {doc['wall_s']}s)")


if __name__ == "__main__":
    main()
