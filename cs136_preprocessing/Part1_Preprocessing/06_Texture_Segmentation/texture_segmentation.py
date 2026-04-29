"""Part 1.6: Texture segmentation.

Steps (texture features, then clustering):
  1. Build a Gabor filter bank: 4 orientations x 3 scales = 12 filters.
  2. For each pixel, take the absolute response of all 12 filters, then
     smooth each response with a small Gaussian. This gives every pixel
     a stable feature vector that describes the local texture.
  3. (Color version only) Add the (a, b) channels from CIE Lab as
     extra color features.
  4. Z-score the features and run K-Means with k=4 by default.
  5. Color in the result by which cluster each pixel belongs to.

Two outputs per image:
  * grayscale-only segmentation in ``Grayscale_Texture_Images/``
  * color-augmented segmentation in ``Color_Texture_Images/``

The assignment says to try grayscale first and then add color, so we
do both.
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

OUT_DIR = Path(__file__).resolve().parent
GRAY_OUT = OUT_DIR / "Grayscale_Texture_Images"
COLOR_OUT = OUT_DIR / "Color_Texture_Images"

GABOR_THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GABOR_LAMBDAS = (6.0, 12.0, 24.0)
GABOR_KSIZE = 21
GABOR_SIGMA = 4.0
GABOR_GAMMA = 0.5

# 8-class colormap so cluster IDs stay legible.
PALETTE = np.array([
    [220,  20,  60],
    [ 30, 144, 255],
    [ 50, 205,  50],
    [255, 165,   0],
    [148,   0, 211],
    [  0, 206, 209],
    [255, 215,   0],
    [105, 105, 105],
], dtype=np.uint8)


def gabor_features(gray: np.ndarray) -> np.ndarray:
    feats: list[np.ndarray] = []
    g = gray.astype(np.float32) / 255.0
    for theta in GABOR_THETAS:
        for lam in GABOR_LAMBDAS:
            kern = cv2.getGaborKernel(
                (GABOR_KSIZE, GABOR_KSIZE), GABOR_SIGMA, theta, lam, GABOR_GAMMA, 0,
                ktype=cv2.CV_32F,
            )
            resp = np.abs(cv2.filter2D(g, cv2.CV_32F, kern))
            resp = cv2.GaussianBlur(resp, (0, 0), sigmaX=lam / 2.0)
            feats.append(resp)
    return np.stack(feats, axis=-1)  # H x W x 12


def kmeans_segment(feats_hwc: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
    """Simple K-Means via cv2.kmeans on standardized features."""
    h, w, c = feats_hwc.shape
    flat = feats_hwc.reshape(-1, c).astype(np.float32)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    flat = (flat - mean) / std

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.5)
    _, labels, _ = cv2.kmeans(flat, k, None, criteria, attempts=3,
                              flags=cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w)


def colorize(labels: np.ndarray) -> np.ndarray:
    return PALETTE[labels % len(PALETTE)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Texture-based segmentation (Gabor + KMeans).")
    add_io_args(parser)
    parser.add_argument("--k", type=int, default=4, help="Number of clusters.")
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("texture_segmentation", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    GRAY_OUT.mkdir(parents=True, exist_ok=True)
    COLOR_OUT.mkdir(parents=True, exist_ok=True)

    for path, condition in images:
        bgr = read_image(path)
        gray = to_gray(bgr)
        stem = output_stem(path, condition)
        gabor = gabor_features(gray)

        # Grayscale-only run
        labels_g = kmeans_segment(gabor, args.k)
        seg_g = colorize(labels_g)
        side_g = np.hstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(seg_g, cv2.COLOR_RGB2BGR)])
        save_image(GRAY_OUT / f"{stem}__seg_k{args.k}.png", side_g)

        # Color run: append Lab a,b channels alongside Gabor energy
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ab = lab[..., 1:] / 255.0  # H x W x 2 in [0,1]
        # Match Gabor's smoothing scale to keep features comparable.
        ab = np.stack([cv2.GaussianBlur(ab[..., i], (0, 0), sigmaX=4.0) for i in range(2)], axis=-1)
        feats_color = np.concatenate([gabor, ab], axis=-1)
        labels_c = kmeans_segment(feats_color, args.k)
        seg_c = colorize(labels_c)
        side_c = np.hstack([bgr, cv2.cvtColor(seg_c, cv2.COLOR_RGB2BGR)])
        save_image(COLOR_OUT / f"{stem}__seg_k{args.k}.png", side_c)

        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images)} grayscale + {len(images)} color segmentations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
