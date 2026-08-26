"""Re-encode an ego-path dataset so the dataloader stops starving the GPU.

Training reads full-resolution PNGs, and zlib decoding of a 2464x1600 frame costs
~170 ms against ~40 ms for the same frame as JPEG. With 16 workers that is the
difference between 11.6 s and 2.7 s of decode per epoch on OSDaR23, on epochs
that measure 22 s -- the A100 sat at 8% utilisation waiting for pixels.

Re-encoding to JPEG is the whole win and it leaves geometry untouched. Downscaling
is offered too (--long-side) but is usually a bad trade: the random crop already
lands around 660 px wide on OSDaR23 against a 512 px model input, so even a 0.75x
downscale starts upsampling the crops.

    python prepare_dataset.py \
        --images-path  /data3/bhkim/datasets/OSDaR23_rgb_center \
        --annotations-path /data3/bhkim/datasets/OSDaR23_rgb_center/egopath_labels.json \
        --output /data3/bhkim/datasets/OSDaR23_rgb_center_jpg
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Re-encode (and optionally downscale) an ego-path dataset."
    )
    parser.add_argument("--images-path", type=str, required=True,
                        help="Directory holding the source images.")
    parser.add_argument("--annotations-path", type=str, required=True,
                        help="Ego-path annotations JSON for those images.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory. Gets the images plus a rewritten "
                             "egopath_labels.json.")
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "png"],
                        help="Output encoding (default: jpeg, ~4x faster to decode).")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (default: 95).")
    parser.add_argument("--long-side", type=int, default=None,
                        help="Downscale so the longer side is at most this many pixels, "
                             "rescaling the annotations to match. Omitted = keep full "
                             "resolution, which is what you usually want.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel encoder processes (default: 16).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode images that already exist in the output.")
    return parser.parse_args()


def out_name(name, fmt):
    return os.path.splitext(name)[0] + (".jpg" if fmt == "jpeg" else ".png")


def convert(job):
    """Encodes one image, returning (source name, output name, scale, size)."""
    name, src_dir, dst_dir, fmt, quality, long_side, overwrite = job
    dst = os.path.join(dst_dir, out_name(name, fmt))
    with Image.open(os.path.join(src_dir, name)) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = 1.0
        if long_side and max(w, h) > long_side:
            scale = long_side / max(w, h)
            # round() rather than int(): the annotation rescale below uses the same
            # ratio, and a truncated size would push bottom-row points out of bounds
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        size = img.size
        if overwrite or not os.path.exists(dst):
            if fmt == "jpeg":
                img.save(dst, "JPEG", quality=quality, subsampling=0, optimize=True)
            else:
                img.save(dst, "PNG", compress_level=1)
    return name, os.path.basename(dst), scale, size


def scale_annotation(annotation, scale, size):
    """Scales rail points, clamping into the image.

    Clamping matters more than it looks: training crops from the bottom row up and
    raises IndexError if no rail pixel lands on it, so a point at y=1599 must come
    out at y=799 in an 800-row image, not 800.
    """
    w, h = size
    out = {}
    for rail, points in annotation.items():
        if not isinstance(points, list):
            out[rail] = points
            continue
        out[rail] = [
            [min(max(round(x * scale), 0), w - 1), min(max(round(y * scale), 0), h - 1)]
            for x, y in points
        ]
    return out


def main(args):
    with open(args.annotations_path) as f:
        annotations = json.load(f)
    names = sorted(
        n for n in os.listdir(args.images_path)
        if n.lower().endswith(IMAGE_EXTENSIONS) and n in annotations
    )
    missing = len(annotations) - len(names)
    if not names:
        raise SystemExit("No annotated images found in --images-path.")
    os.makedirs(args.output, exist_ok=True)
    print(f"{len(names)} annotated images -> {args.output} "
          f"({args.format}{', q' + str(args.quality) if args.format == 'jpeg' else ''}"
          f"{', long side ' + str(args.long_side) if args.long_side else ', full resolution'})")
    if missing:
        print(f"  note: {missing} annotation entries have no image file and are dropped")

    jobs = [(n, args.images_path, args.output, args.format, args.quality,
             args.long_side, args.overwrite) for n in names]
    done, scaled = {}, set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, dst, scale, size) in enumerate(pool.map(convert, jobs, chunksize=8), 1):
            done[name] = (dst, scale, size)
            if scale != 1.0:
                scaled.add(name)
            if i % 200 == 0 or i == len(names):
                print(f"  {i}/{len(names)}")

    out_annotations = {}
    for name, (dst, scale, size) in done.items():
        ann = annotations[name]
        out_annotations[dst] = scale_annotation(ann, scale, size) if scale != 1.0 else ann
    dst_json = os.path.join(args.output, "egopath_labels.json")
    tmp = dst_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out_annotations, f)
    os.replace(tmp, dst_json)

    src_mb = sum(os.path.getsize(os.path.join(args.images_path, n)) for n in names) / 1e6
    dst_mb = sum(os.path.getsize(os.path.join(args.output, d))
                 for d, _, _ in done.values()) / 1e6
    print(f"\nWrote {len(out_annotations)} images and {dst_json}")
    print(f"  {src_mb:.0f} MB -> {dst_mb:.0f} MB ({src_mb / max(dst_mb, 1e-9):.1f}x smaller)"
          + (f", {len(scaled)} images rescaled" if scaled else ""))
    print("\nTrain on it with:")
    print(f"  python train.py segmentation resnet18 --device cuda \\\n"
          f"    --images-path {args.output} \\\n"
          f"    --annotations-path {dst_json}")


if __name__ == "__main__":
    main(parse_arguments())
