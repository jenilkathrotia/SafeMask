"""Part 1.1 — Gaussian filter.

For every input image we save:
  * the grayscale source,
  * Gaussian-smoothed outputs at sigma = 1, 2, 4 (kernel size = 6*sigma+1),
  * the Project 3 fixed 5x5/273 kernel for direct comparison.

Run from the repo root:
    python cs136_preprocessing/Part1_Preprocessing/01_Gaussian_Filter/gaussian_filter.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make ``utils`` importable regardless of CWD.
_PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG))
from utils.io_utils import (  # noqa: E402
    add_io_args, banner, discover_images, output_stem, read_image, save_image, to_gray,
)
from utils.algorithms import GAUSS_5_273  # noqa: E402

SIGMAS = (1.0, 2.0, 4.0)
OUT_DIR = Path(__file__).resolve().parent / "Gaussian_Filter_Images"


def gaussian_with_sigma(gray: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(round(sigma * 6)) | 1)  # ksize must be odd
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REPLICATE)


def project3_5x5(gray: np.ndarray) -> np.ndarray:
    out = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, GAUSS_5_273,
                       borderType=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Gaussian smoothing to driving scenes.")
    add_io_args(parser)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("gaussian_filter", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, condition in images:
        img = read_image(path)
        gray = to_gray(img)
        stem = output_stem(path, condition)

        save_image(OUT_DIR / f"{stem}__00_source.png", gray)
        save_image(OUT_DIR / f"{stem}__01_proj3_5x5_273.png", project3_5x5(gray))
        for sigma in SIGMAS:
            save_image(
                OUT_DIR / f"{stem}__sigma{sigma:0.1f}.png",
                gaussian_with_sigma(gray, sigma),
            )
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images) * (2 + len(SIGMAS))} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
