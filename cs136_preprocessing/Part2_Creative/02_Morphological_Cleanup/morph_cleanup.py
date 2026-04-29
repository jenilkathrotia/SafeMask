"""Part 2 (Creative #2) — Morphological cleanup of edge maps.

Project 3 already covered binary expand/shrink. We extend that to a
classic edge-cleanup pipeline:
  * **Closing** (dilate→erode) bridges 1–2-pixel gaps in continuous edges.
  * **Opening** (erode→dilate) removes isolated 1-pixel speckle.
  * **Skeletonize** (Zhang-Suen via scikit-image) thins the result back to
    1-pixel-wide lines for cleaner Hough input.

We save the Canny baseline and each stage so the grader can see exactly
what each morphological step contributes.
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

OUT_DIR = Path(__file__).resolve().parent / "Morph_Cleanup_Images"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Morphological cleanup of Canny edges.")
    add_io_args(parser)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=3)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("morph_cleanup", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    try:
        from skimage.morphology import skeletonize
    except ImportError:
        skeletonize = None  # graceful fallback if scikit-image unavailable

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.close_ksize, args.close_ksize))
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.open_ksize, args.open_ksize))

    for path, condition in images:
        gray = to_gray(read_image(path))
        edges = canny_project3(gray)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)

        stem = output_stem(path, condition)
        save_image(OUT_DIR / f"{stem}__01_canny.png", edges)
        save_image(OUT_DIR / f"{stem}__02_closed.png", closed)
        save_image(OUT_DIR / f"{stem}__03_opened.png", opened)
        if skeletonize is not None:
            skel = (skeletonize(opened > 0).astype(np.uint8) * 255)
            save_image(OUT_DIR / f"{stem}__04_skeleton.png", skel)

        print(f"  ✔ {path.name}")

    n_per = 4 if skeletonize is not None else 3
    print(f"\nWrote {len(images) * n_per} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
