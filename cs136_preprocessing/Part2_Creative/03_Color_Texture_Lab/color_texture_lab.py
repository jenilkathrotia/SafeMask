"""Part 2 (Creative #3) — Mean-Shift color segmentation in CIE Lab.

A different, complementary segmentation algorithm from Part 1.6's
Gabor+KMeans approach. Mean-Shift is **nonparametric** — it does not need
a preset cluster count, which is more honest given that fog/night/rain
frames have wildly different intrinsic scene complexity.

We use OpenCV's `pyrMeanShiftFiltering` (the spatial-range mean-shift
that operates directly in colour space), then quantize the smoothed
output and write a colour-coded label map for visual inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG))
from utils.io_utils import (  # noqa: E402
    add_io_args, banner, discover_images, output_stem, read_image, save_image,
)

OUT_DIR = Path(__file__).resolve().parent / "Color_Texture_Images"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mean-Shift colour segmentation in Lab.")
    add_io_args(parser)
    parser.add_argument("--spatial-radius", type=int, default=15)
    parser.add_argument("--color-radius", type=int, default=25)
    parser.add_argument("--max-pyr-level", type=int, default=2)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("color_texture_lab", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, condition in images:
        bgr = read_image(path)
        # OpenCV mean-shift expects 8-bit 3-channel; do it in Lab to match
        # human perceptual differences better than RGB.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ms = cv2.pyrMeanShiftFiltering(
            lab, sp=args.spatial_radius, sr=args.color_radius,
            maxLevel=args.max_pyr_level,
        )
        ms_bgr = cv2.cvtColor(ms, cv2.COLOR_LAB2BGR)

        # Quantize for a clean visualization.
        Z = ms_bgr.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, K=8, bestLabels=None, criteria=criteria,
                                        attempts=3, flags=cv2.KMEANS_PP_CENTERS)
        quant = centers[labels.flatten()].astype(np.uint8).reshape(bgr.shape)

        side = np.hstack([bgr, ms_bgr, quant])
        stem = output_stem(path, condition)
        save_image(OUT_DIR / f"{stem}__source_meanshift_quant.png", side)
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images)} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
