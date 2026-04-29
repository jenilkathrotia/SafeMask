"""Part 1.2 — Sobel edge detector (Project 3 port).

Applies the same 3×3 Sx / Sy kernels as Project 3's ``sobel()`` and saves:
  * |gx|, |gy| (component magnitudes),
  * gradient magnitude (linear stretch to 0–255),
  * a thresholded binary edge map at the 75th-percentile of the magnitude
    (a stable, image-adaptive default).
A sigma=1 Gaussian pre-blur is offered via ``--pre-blur`` since CS 136 cv07
emphasizes pre-smoothing before any first-derivative operator.
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
    add_io_args, banner, discover_images, output_stem, read_image, save_image, to_gray,
)
from utils.algorithms import normalize_to_uint8, sobel_components  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "Sobel_Edge_Images"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sobel edge detection (Project 3 kernels).")
    add_io_args(parser)
    parser.add_argument("--pre-blur", type=float, default=1.0,
                        help="sigma for Gaussian pre-blur (0 disables).")
    parser.add_argument("--threshold-percentile", type=float, default=75.0,
                        help="Percentile of |grad| used as the binary threshold.")
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("sobel_edge", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, condition in images:
        img = read_image(path)
        gray = to_gray(img)
        if args.pre_blur > 0:
            k = max(3, int(round(args.pre_blur * 6)) | 1)
            gray = cv2.GaussianBlur(gray, (k, k), args.pre_blur,
                                    borderType=cv2.BORDER_REPLICATE)

        gx, gy, mag = sobel_components(gray)
        gx_u8 = normalize_to_uint8(np.abs(gx))
        gy_u8 = normalize_to_uint8(np.abs(gy))
        mag_u8 = normalize_to_uint8(mag)
        thresh = np.percentile(mag_u8, args.threshold_percentile)
        binary = ((mag_u8 >= thresh).astype(np.uint8) * 255)

        stem = output_stem(path, condition)
        save_image(OUT_DIR / f"{stem}__00_source.png", gray)
        save_image(OUT_DIR / f"{stem}__01_gx_abs.png", gx_u8)
        save_image(OUT_DIR / f"{stem}__02_gy_abs.png", gy_u8)
        save_image(OUT_DIR / f"{stem}__03_magnitude.png", mag_u8)
        save_image(OUT_DIR / f"{stem}__04_binary_p{int(args.threshold_percentile)}.png", binary)
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images) * 5} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
