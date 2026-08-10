#!/usr/bin/env python3
"""Run Alpamayo-R1 over a video and record everything it returns.

The model does not produce a video. Per request it takes four consecutive
frames plus an ego history and returns a predicted trajectory for the next
6.4 s, a chain-of-causation string, and the inference time. This script walks a
video, asks once per sampled step, and writes the answers to one JSON file so
the web app can replay them against the footage without holding a 10B model
open.

Read the chain of causation; do not read the trajectory. On dashcam footage the
predicted path is always straight and exactly assumed_speed x 6.4 s long -- the
kinematic default, carrying nothing from the scene. It does not respond to the
ego history either: bending the history into a 25 m radius turn moves the
prediction less than a metre. See N_FRAMES for why, and for what happens when
you try to fix it. The reasoning text does track the scene, naming level
crossing barriers, lead vehicles and speed limit signs as they appear.

The ego history is synthesised as straight-line constant speed. A video alone
does not say how the vehicle was moving, and the alternative -- estimating it
from the footage -- would put a second model's error inside this one's input.
Where a real trajectory log exists, pass it with --history.

Several videos in one invocation share a server, which matters on short clips:
loading the weights takes about as long as inferring over a 15 s clip.

Usage:
    python tools/alpamayo_run.py clip.mp4 --out clip.alpamayo.json --fps 2
    python tools/alpamayo_run.py dir/*.mp4 --skip-existing
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ALPA_ROOT = os.environ.get("ALPAMAYO_ROOT", "/data3/bhkim/workspace/alpamayo")

# Four frames from the one camera we have. The reference loader sends sixteen
# images -- four instants 0.1 s apart across four cameras (cross left 120,
# front wide 120, cross right 120, front tele 30), stacked camera-major -- and
# create_message imposes no count of its own, so this is short of the protocol.
# It stays this way on purpose. Three ways of filling sixteen slots were tried
# on road_snow_night, six consecutive requests each:
#
#   four frames, one camera   forward 60-62 m, lateral 0.3-0.9 m, jitter 0.2 m
#   sixteen, front duplicated forward 19-59 m, lateral 23-61 m,   jitter 6.9 m
#   sixteen, front + 3 black  forward 38-53 m, lateral 34-47 m,   jitter 3.1 m
#
# The trajectory head reads the four cameras geometrically, so whatever fills
# the three slots we cannot supply becomes false geometry, and the model turns
# into it -- 40 m sideways in 6.4 s on a straight snow-covered road. The wilder
# answers are the worse ones: they look informative and are invented, while the
# straight line is visibly empty. None was checked against ground truth; these
# clips have none, and physical_ai_av is not installed to compare on NVIDIA's.
N_FRAMES = 4
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

# The assumed speed is not a detail: the synthesised history is a straight line,
# so whatever speed goes in comes back out as the length of the predicted path.
# Running every clip at the self-test's 14 m/s put a tram's pace on a motorway.
# These are the speeds each kind of vehicle actually travels at, not a tuning
# knob -- pick the one that matches the footage, or pass --speed outright.
VEHICLE_SPEEDS = {
    "car": 25.0,        # 90 km/h, open road
    "car-city": 11.0,   # 40 km/h, urban streets and slow going in rain or snow
    "train": 14.0,      # 50 km/h, the value every clip used before
    "tram": 8.0,        # 30 km/h, street running
}


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
        # --attn sdpa only takes effect if the alpamayo checkout carries
        # tools/alpamayo_server_sdpa.patch. That tree is not under version
        # control, so the patch lives here. Without it transformers rejects the
        # override on a missing class flag and silently falls back to eager:
        # slower, same answers.
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
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default=None,
                    help="single video only; default: <video>.alpamayo.json")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave videos that already have a result file alone")
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate (default 2)")
    ap.add_argument("--vehicle", choices=sorted(VEHICLE_SPEEDS), default=None,
                    help="assume this vehicle's speed: "
                         + ", ".join(f"{k}={v:g} m/s" for k, v in VEHICLE_SPEEDS.items()))
    ap.add_argument("--speed", type=float, default=None,
                    help="assumed constant speed in m/s; overrides --vehicle "
                         "(default 14, about 50 km/h)")
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

    if args.out and len(args.videos) > 1:
        raise SystemExit("--out names one file; drop it to write <video>.alpamayo.json")

    speed = (args.speed if args.speed is not None
             else VEHICLE_SPEEDS.get(args.vehicle, 14.0))
    hist = (json.load(open(args.history)) if args.history
            else straight_history(speed))
    if not args.history:
        label = args.vehicle or "default"
        print(f"assuming {speed:g} m/s ({speed * 3.6:.0f} km/h, {label})")

    todo = []
    for video in args.videos:
        out = args.out or (os.path.splitext(video)[0] + ".alpamayo.json")
        if args.skip_existing and os.path.exists(out):
            print(f"skip {os.path.basename(video)} — {os.path.basename(out)} exists")
            continue
        todo.append((video, out))
    if not todo:
        return

    # One server for the whole list. Loading the weights costs about 20 s, and
    # paying that per video is most of the wall clock on short clips.
    print(f"starting server on GPU {args.gpu} — a cold model cache takes minutes")
    srv = Server(args.gpu, args.model, args.root)
    print("server ready")

    try:
        for n, (video, out) in enumerate(todo, 1):
            print(f"\n[{n}/{len(todo)}] {os.path.basename(video)}", flush=True)
            tmp = tempfile.mkdtemp(prefix="alpamayo_frames_")
            frames, src_fps = sample_frames(video, args.fps, tmp)
            if len(frames) < N_FRAMES:
                print(f"  skipped: need {N_FRAMES} sampled frames, got {len(frames)}")
                continue
            print(f"{len(frames)} frames sampled at {args.fps} fps "
                  f"(source {src_fps:.1f} fps)")

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

            doc = {"video": os.path.abspath(video), "sample_fps": args.fps,
                   "source_fps": src_fps, "assumed_speed_mps": speed,
                   "vehicle": args.vehicle,
                   "attn": "sdpa", "diffusion_steps": args.diffusion_steps,
                   "history": hist, "n_steps": len(steps),
                   "wall_s": round(time.time() - t_start, 1), "steps": steps}
            with open(out, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            print(f"wrote {out}  ({len(steps)} steps, {doc['wall_s']}s)")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
