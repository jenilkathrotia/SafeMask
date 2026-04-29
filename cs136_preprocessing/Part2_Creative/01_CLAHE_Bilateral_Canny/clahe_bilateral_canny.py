"""Part 2 (Creative #1) — CLAHE + Bilateral + Canny pipeline.

Motivation: ACDC frames in fog/night have very compressed luminance
histograms. Plain Canny under-detects faint edges. We combine:
  1. **CLAHE** (contrast-limited adaptive histogram equalization) on the
     L-channel of CIE Lab — boosts local contrast without blowing highlights.
  2. **Bilateral filter** — denoises while preserving edges (ordinary
     Gaussian blur would erase the contrast we just added).
  3. **Canny** — same Project 3 thresholds, now operating on a much
     more discriminative input.

We save the intermediate stages so the contribution of each step is visible.
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
from utils.algorithms import canny_project3  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "CLAHE_Canny_Images"


def clahe_l(bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLAHE + bilateral + Canny.")
    add_io_args(parser)
    parser.add_argument("--clip-limit", type=float, default=3.0)
    parser.add_argument("--tile-grid", type=int, default=8)
    parser.add_argument("--bilateral-d", type=int, default=9)
    parser.add_argument("--bilateral-sigma-color", type=float, default=75.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=75.0)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("clahe_bilateral_canny", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, condition in images:
        bgr = read_image(path)
        stem = output_stem(path, condition)

        clahe_bgr = clahe_l(bgr, args.clip_limit, args.tile_grid)
        bilat = cv2.bilateralFilter(
            clahe_bgr, d=args.bilateral_d,
            sigmaColor=args.bilateral_sigma_color,
            sigmaSpace=args.bilateral_sigma_space,
        )
        edges_baseline = canny_project3(to_gray(bgr))
        edges_pipeline = canny_project3(to_gray(bilat))

        save_image(OUT_DIR / f"{stem}__01_source.png", bgr)
        save_image(OUT_DIR / f"{stem}__02_clahe.png", clahe_bgr)
        save_image(OUT_DIR / f"{stem}__03_clahe_bilateral.png", bilat)
        save_image(OUT_DIR / f"{stem}__04_canny_baseline.png", edges_baseline)
        save_image(OUT_DIR / f"{stem}__05_canny_pipeline.png", edges_pipeline)
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images) * 5} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
