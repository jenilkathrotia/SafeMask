"""Part 3, Step 1: Make distorted copies.

Generates three corrupted copies of every input image:
  * Gaussian noise (sigma is configurable)
  * Motion blur (kernel length is configurable)
  * Low contrast (linear contrast and brightness rescale)

Outputs go into three folders under ``01_Distortions/``:
  * ``Noisy/``: Gaussian noise (default sigma = 25)
  * ``Blurred/``: horizontal motion blur (default kernel length = 15)
  * ``LowContrast/``: alpha=0.4, beta=40 (range squashed)

These folders are the inputs to ``02_Pipeline_Comparison``.
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

OUT_BASE = Path(__file__).resolve().parent
NOISY_DIR = OUT_BASE / "Noisy"
BLUR_DIR = OUT_BASE / "Blurred"
LOWC_DIR = OUT_BASE / "LowContrast"


def add_gaussian_noise(bgr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = bgr.astype(np.float32) + rng.normal(0.0, sigma, bgr.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(bgr: np.ndarray, length: int) -> np.ndarray:
    if length < 3:
        return bgr.copy()
    kern = np.zeros((length, length), dtype=np.float32)
    kern[length // 2, :] = 1.0 / length  # horizontal motion
    return cv2.filter2D(bgr, ddepth=-1, kernel=kern, borderType=cv2.BORDER_REPLICATE)


def low_contrast(bgr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Gaussian noise / blur / low contrast.")
    add_io_args(parser)
    parser.add_argument("--noise-sigma", type=float, default=25.0)
    parser.add_argument("--blur-length", type=int, default=15)
    parser.add_argument("--contrast-alpha", type=float, default=0.4)
    parser.add_argument("--contrast-beta", type=float, default=40.0)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("apply_distortions", len(images), args.input_dir, OUT_BASE)
    if not images:
        print("No images found.")
        return 1

    for d in (NOISY_DIR, BLUR_DIR, LOWC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for path, condition in images:
        bgr = read_image(path)
        stem = output_stem(path, condition)
        save_image(NOISY_DIR / f"{stem}__noise_sigma{int(args.noise_sigma)}.png",
                   add_gaussian_noise(bgr, args.noise_sigma, rng))
        save_image(BLUR_DIR / f"{stem}__motion_len{args.blur_length}.png",
                   motion_blur(bgr, args.blur_length))
        save_image(LOWC_DIR / f"{stem}__alpha{args.contrast_alpha}_beta{int(args.contrast_beta)}.png",
                   low_contrast(bgr, args.contrast_alpha, args.contrast_beta))
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images) * 3} distorted images across Noisy/ Blurred/ LowContrast/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
