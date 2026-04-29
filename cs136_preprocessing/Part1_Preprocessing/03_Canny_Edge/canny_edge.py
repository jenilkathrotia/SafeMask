"""Part 1.3 — Canny edge detector (Project 3 port + OpenCV reference).

For each input we run two implementations side-by-side:
  * ``project3``  — exact port of Project 3's canny() (5×5/273 Gaussian, 2×2
    P/Q gradient, sector NMS, connected hysteresis).
  * ``cv2.Canny`` — at three (low, high) preset pairs to illustrate parameter
    sensitivity.

This makes Part 1.5 (edge-detector evaluation) easy: it just consumes these
binary maps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG))
from utils.io_utils import (  # noqa: E402
    add_io_args, banner, discover_images, output_stem, read_image, save_image, to_gray,
)
from utils.algorithms import canny_project3  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "Canny_Edge_Images"

# Three (low, high) presets sweep the typical CS 136 cv08 demo range.
CV2_PRESETS = (
    ("low_50_150", 50, 150),
    ("mid_75_200", 75, 200),
    ("high_100_250", 100, 250),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canny edge detection — Project 3 port + cv2 reference.")
    add_io_args(parser)
    parser.add_argument("--theta-low-frac", type=float, default=0.05,
                        help="theta_low = frac * max(E) for the Project 3 port.")
    parser.add_argument("--theta-high-mult", type=float, default=2.5,
                        help="theta_high = mult * theta_low for the Project 3 port.")
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("canny_edge", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, condition in images:
        img = read_image(path)
        gray = to_gray(img)
        stem = output_stem(path, condition)

        proj3 = canny_project3(gray, args.theta_low_frac, args.theta_high_mult)
        save_image(OUT_DIR / f"{stem}__00_source.png", gray)
        save_image(OUT_DIR / f"{stem}__01_project3.png", proj3)
        for name, lo, hi in CV2_PRESETS:
            save_image(OUT_DIR / f"{stem}__cv2_{name}.png",
                       cv2.Canny(gray, lo, hi, L2gradient=True))
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images) * (2 + len(CV2_PRESETS))} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
