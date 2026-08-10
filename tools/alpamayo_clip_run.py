#!/usr/bin/env python3
"""Run Alpamayo-R1 over NVIDIA PhysicalAI-AV clips, with the rig it was built for.

The dashcam runner next door feeds four frames from one camera and a
synthesised straight-line history, and its trajectories are worthless: always
straight, always assumed_speed x 6.4 s long. This one feeds what the model
actually expects -- sixteen images from four cameras 0.1 s apart, plus the
vehicle's real recorded motion -- and the trajectories are good. Measured
against ground truth on nine clips from chunk 3119: median minADE_6 of 1.05 m
with the rig, 5.29 m from the same clips through the dashcam path. NVIDIA
publishes 1.22 m over 937 samples.

Ground truth ships with the data, so every step records it beside the
prediction and the web tab can draw both.

Runs under the alpamayo venv, which holds the model and the dataset kit:

    /data3/bhkim/workspace/alpamayo/.venv/bin/python tools/alpamayo_clip_run.py \
        030c760c-ae38-49aa-9ad8-f5650a545d26 --fps 2

Clips have to be downloaded first, in chunks of about 100:

    avdi.download_clip_features(clip_id, [4 cameras, LABELS.EGOMOTION])
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time
import zipfile

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CACHE = os.environ.get("PAV_CACHE", "/data3/bhkim/datasets/physical_ai_av")
ALPA_SRC = os.environ.get("ALPAMAYO_ROOT", "/data3/bhkim/workspace/alpamayo") + "/src"
sys.path.insert(0, ALPA_SRC)

N_HISTORY, N_FUTURE, STEP_S = 16, 64, 0.1
# t0 needs a full history behind it and a full horizon ahead, and the loader
# rejects anything inside the first 1.6 s outright.
T0_MIN_US = int(N_HISTORY * STEP_S * 1e6) + 100_000
HORIZON_US = int(N_FUTURE * STEP_S * 1e6)

CAM_FOR_PLAYBACK = "camera_front_wide_120fov"


def chunk_zip(camera, chunk):
    hits = glob.glob(f"{CACHE}/**/{camera}.chunk_{chunk:04d}.zip", recursive=True)
    if not hits:
        raise SystemExit(f"missing chunk {chunk} for {camera} under {CACHE}")
    return hits[0]


def clip_timestamps(clip_id, chunk):
    """Frame times of the playback camera, microseconds, as the mp4 records them."""
    import pandas as pd

    with zipfile.ZipFile(chunk_zip(CAM_FOR_PLAYBACK, chunk)) as zf:
        name = f"{clip_id}.{CAM_FOR_PLAYBACK}.timestamps.parquet"
        ts = pd.read_parquet(io.BytesIO(zf.read(name)))
    return ts[ts.columns[0]].to_numpy()


def export_video(clip_id, chunk, out_dir):
    """Put the front-wide view where the tab can play it.

    The dataset ships HEVC, which browsers will not decode in a <video> tag --
    the pane sits at 0:00 with no error. Transcoding to H.264 once here is
    cheaper than discovering that again. The copy is for playback only; the
    model reads the original out of the zip.
    """
    dst = os.path.join(out_dir, f"pav_{clip_id[:8]}.mp4")
    if os.path.exists(dst):
        return dst
    raw = dst + ".hevc.mp4"
    with zipfile.ZipFile(chunk_zip(CAM_FOR_PLAYBACK, chunk)) as zf:
        with zf.open(f"{clip_id}.{CAM_FOR_PLAYBACK}.mp4") as src, open(raw, "wb") as f:
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                f.write(b)
    rc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", raw, "-an",
         "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst, "-y"],
        capture_output=True, text=True)
    if rc.returncode != 0 or not os.path.exists(dst):
        os.replace(raw, dst)          # 재생은 못 해도 결과는 볼 수 있게 둔다
        print(f"  note: transcode failed, leaving HEVC ({rc.stderr.strip()[:80]})")
        return dst
    os.remove(raw)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="PhysicalAI-AV clip ids")
    ap.add_argument("--out-dir", default="/data3/bhkim/datasets/alpamayo/in")
    ap.add_argument("--fps", type=float, default=2.0, help="requests per second of clip")
    ap.add_argument("--samples", type=int, default=1,
                    help="trajectories drawn per request (default 1; the published "
                         "minADE_6 uses 6)")
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="stop after N steps (0 = all)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import physical_ai_av
    import torch
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from transformers import AutoConfig

    os.makedirs(args.out_dir, exist_ok=True)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        cache_dir=CACHE, confirm_download_threshold_gb=1e9)

    todo = []
    for cid in args.clips:
        out = os.path.join(args.out_dir, f"pav_{cid[:8]}.alpamayo.json")
        if args.skip_existing and os.path.exists(out):
            print(f"skip {cid[:8]} — result exists")
            continue
        todo.append((cid, out))
    if not todo:
        return

    # The class declares no attention support and transformers checks the flag
    # before anything else, so sdpa needs opening by hand. See
    # tools/alpamayo_server_sdpa.patch for the same change on the server side.
    AlpamayoR1._supports_sdpa = True
    cfg = AutoConfig.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "sdpa"
    print("loading the model — minutes on a cold cache")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", config=cfg,
                                       attn_implementation="sdpa",
                                       dtype=torch.bfloat16).to("cuda")
    model.eval()
    proc = helper.get_processor(model.tokenizer)
    print("model ready")

    for n, (cid, out) in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {cid}", flush=True)
        chunk = avdi.get_clip_chunk(cid)
        video = export_video(cid, chunk, args.out_dir)
        stamps = clip_timestamps(cid, chunk)
        t_first, t_last = int(stamps.min()), int(stamps.max())

        lo = max(T0_MIN_US, t_first + T0_MIN_US)
        hi = t_last - HORIZON_US
        if hi <= lo:
            print(f"  skipped: clip is {(t_last - t_first) / 1e6:.1f} s, too short for "
                  f"a {HORIZON_US / 1e6:.1f} s horizon plus history")
            continue
        stride = int(1e6 / max(0.1, args.fps))
        t0s = list(range(lo, hi + 1, stride))
        if args.limit:
            t0s = t0s[:args.limit]
        print(f"  chunk {chunk} · clip {(t_last - t_first) / 1e6:.1f} s · "
              f"{len(t0s)} requests from {(lo - t_first) / 1e6:.1f}s "
              f"to {(t0s[-1] - t_first) / 1e6:.1f}s", flush=True)

        steps, t_start = [], time.time()
        for i, t0 in enumerate(t0s):
            rec = {"step": i, "time": (t0 - t_first) / 1e6, "t0_us": t0}
            try:
                data = load_physical_aiavdataset(cid, t0_us=t0, avdi=avdi)
                frames = data["image_frames"]
                msgs = helper.create_message(frames.flatten(0, 1))
                inputs = proc.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False,
                    continue_final_message=True, return_dict=True, return_tensors="pt")
                md = helper.to_device({"tokenized_data": dict(inputs),
                                       "ego_history_xyz": data["ego_history_xyz"],
                                       "ego_history_rot": data["ego_history_rot"]}, "cuda")
                t = time.time()
                torch.cuda.manual_seed_all(42)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    px, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=md, top_p=0.98, temperature=0.6,
                        num_traj_samples=args.samples, num_traj_sets=1,
                        max_generation_length=256, return_extra=True,
                        diffusion_kwargs={"inference_step": args.diffusion_steps})
                torch.cuda.synchronize()
                infer_ms = (time.time() - t) * 1000.0
                pred = px.detach().float().cpu().numpy()[0, 0]        # (samples,64,3)
                gt = data["ego_future_xyz"].cpu().numpy()[0, 0]       # (64,3)
                ade = np.linalg.norm(pred[:, :, :2] - gt[None, :, :2], axis=2).mean(axis=1)
                best = int(ade.argmin())
                c = extra.get("cot")
                while isinstance(c, (list, tuple, np.ndarray)) and len(c):
                    c = c[0]
                rec.update(ok=True, pred_xyz=pred[best].tolist(), gt_xyz=gt.tolist(),
                           min_ade=float(ade.min()), n_samples=int(pred.shape[0]),
                           coc=str(c), infer_ms=round(infer_ms, 1))
                print(f"  step {i + 1}/{len(t0s)}  t={rec['time']:.2f}s  "
                      f"{infer_ms:.0f} ms  ADE {ade.min():.2f} m", flush=True)
            except Exception as exc:
                rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                print(f"  step {i + 1}/{len(t0s)}  failed: {rec['error'][:90]}", flush=True)
            steps.append(rec)

        ok = [s for s in steps if s.get("ok")]
        doc = {"video": os.path.abspath(video), "source": "physical_ai_av",
               "clip_id": cid, "chunk": int(chunk), "rig": "4cam",
               "images_per_request": 16, "frame_hz": 1.0 / STEP_S,
               "history": "recorded", "sample_fps": args.fps,
               "attn": "sdpa", "diffusion_steps": args.diffusion_steps,
               "n_steps": len(steps), "n_ok": len(ok),
               "median_min_ade": (float(np.median([s["min_ade"] for s in ok]))
                                  if ok else None),
               "wall_s": round(time.time() - t_start, 1), "steps": steps}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        print(f"  wrote {out}  ({len(ok)}/{len(steps)} ok, {doc['wall_s']}s, "
              f"median ADE {doc['median_min_ade']})" if ok else f"  wrote {out}")


if __name__ == "__main__":
    main()
