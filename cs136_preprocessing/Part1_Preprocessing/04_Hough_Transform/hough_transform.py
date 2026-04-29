"""Part 1.4 — Hough transform.

Two passes per image, both visualized as overlays on the source:
  * **Lines** — `cv2.HoughLinesP` on the Project 3 Canny output. Useful for
    lane markings in driving scenes.
  * **Circles** — *both* `cv2.HoughCircles` and the Project 4 NumPy port of
    ``houghTransformCircles`` + 3D local-maxima extraction. The Project 4 port
    runs on a downsampled image so vote casting stays tractable.

Outputs go to ``Hough_Images/`` next to this script.
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
from utils.algorithms import (  # noqa: E402
    canny_project3, find_local_maxima_3d, hough_circles_project4,
)

OUT_DIR = Path(__file__).resolve().parent / "Hough_Images"


def draw_lines(bgr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=80, minLineLength=40, maxLineGap=10)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def draw_circles_cv2(bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=120, param2=40, minRadius=10, maxRadius=80,
    )
    if circles is not None:
        for cx, cy, r in np.uint16(np.around(circles[0])):
            cv2.circle(out, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
            cv2.circle(out, (int(cx), int(cy)), 2, (0, 255, 255), 2)
    return out


def project4_circles(
    bgr: np.ndarray,
    edges: np.ndarray,
    min_r: int = 10,
    max_r: int = 40,
    n_circles: int = 8,
    target_long_side: int = 200,
) -> np.ndarray:
    """Run the Project 4 circle Hough on a downsampled edge map.

    Voting in a 3D parameter space is O(edges · radii · 360); we downsample
    to keep the demo runtime under a second per image. Detected centers are
    rescaled back to the source resolution before drawing.
    """
    h, w = edges.shape
    scale = target_long_side / max(h, w)
    scale = min(scale, 1.0)  # never upsample
    if scale < 1.0:
        small = cv2.resize(edges, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_NEAREST)
        small_min_r = max(2, int(round(min_r * scale)))
        small_max_r = max(small_min_r + 1, int(round(max_r * scale)))
    else:
        small = edges
        small_min_r, small_max_r = min_r, max_r

    acc = hough_circles_project4(small, small_min_r, small_max_r)
    sep = max(3.0, (small_max_r - small_min_r) * 0.5)
    maxima = find_local_maxima_3d(acc, n_max=n_circles, min_separation=sep,
                                  threshold=acc.max() * 0.4 if acc.size else 0.0)

    out = bgr.copy()
    for y_s, x_s, r_idx, _ in maxima:
        # invert downsampling
        cx = int(x_s / scale) if scale < 1.0 else int(x_s)
        cy = int(y_s / scale) if scale < 1.0 else int(y_s)
        radius = int((small_min_r + r_idx) / scale) if scale < 1.0 else int(small_min_r + r_idx)
        cv2.circle(out, (cx, cy), radius, (255, 128, 0), 2)
        cv2.circle(out, (cx, cy), 2, (255, 255, 255), 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hough transforms (lines + circles).")
    add_io_args(parser)
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--max-radius", type=int, default=40)
    parser.add_argument("--n-circles", type=int, default=8)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("hough_transform", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, condition in images:
        bgr = read_image(path)
        gray = to_gray(bgr)
        edges = canny_project3(gray)
        stem = output_stem(path, condition)

        save_image(OUT_DIR / f"{stem}__00_edges.png", edges)
        save_image(OUT_DIR / f"{stem}__01_lines_overlay.png", draw_lines(bgr, edges))
        save_image(OUT_DIR / f"{stem}__02_circles_cv2.png", draw_circles_cv2(bgr, gray))
        save_image(
            OUT_DIR / f"{stem}__03_circles_project4_port.png",
            project4_circles(bgr, edges,
                             min_r=args.min_radius, max_r=args.max_radius,
                             n_circles=args.n_circles),
        )
        print(f"  ✔ {path.name}")

    print(f"\nWrote {len(images) * 4} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
