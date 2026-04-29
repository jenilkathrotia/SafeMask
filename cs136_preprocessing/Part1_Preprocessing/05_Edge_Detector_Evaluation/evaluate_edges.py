"""Part 1.5: Compare edge detectors with numbers (cv08).

For each input image we run four detectors:
  * Sobel with the top 25% of gradients as edges (P75)
  * Sobel with the top 10% (P90)
  * Project 3 Canny (our Python port)
  * cv2.Canny at (75, 200)

Numbers we record per detector per image:
  * edge_density: percent of pixels marked as edges
  * runtime_ms: how long the detector took
  * iou_vs_proj3_canny: how much the output overlaps with our
    Project 3 Canny. We use Project 3 Canny as the reference because
    cv08 treats Canny as the standard, and we do not have hand-drawn
    ground truth for ACDC images.

Outputs:
  * Evaluation_Results/per_image_metrics.csv
  * Evaluation_Results/aggregate_metrics.csv (mean and std per detector)
  * Evaluation_Results/edge_density_by_detector.png
  * Evaluation_Results/iou_by_detector.png
  * REPORT.md is the write-up. This script only generates the numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG))
from utils.io_utils import (  # noqa: E402
    add_io_args, banner, discover_images, read_image, to_gray,
)
from utils.algorithms import canny_project3, sobel_components  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "Evaluation_Results"
DETECTORS = ("sobel_p75", "sobel_p90", "project3_canny", "cv2_canny_75_200")


def detect_all(gray: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    edges: Dict[str, np.ndarray] = {}
    runtimes: Dict[str, float] = {}

    t = time.perf_counter()
    _, _, mag = sobel_components(gray)
    mag_u8 = ((mag - mag.min()) / max(mag.max() - mag.min(), 1e-12) * 255).astype(np.uint8)
    edges["sobel_p75"] = (mag_u8 >= np.percentile(mag_u8, 75)).astype(np.uint8) * 255
    edges["sobel_p90"] = (mag_u8 >= np.percentile(mag_u8, 90)).astype(np.uint8) * 255
    runtimes["sobel_p75"] = runtimes["sobel_p90"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    edges["project3_canny"] = canny_project3(gray)
    runtimes["project3_canny"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    edges["cv2_canny_75_200"] = cv2.Canny(gray, 75, 200, L2gradient=True)
    runtimes["cv2_canny_75_200"] = (time.perf_counter() - t) * 1000.0

    return edges, runtimes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = (a > 0)
    bb = (b > 0)
    inter = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return inter / union if union > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edge detector quantitative evaluation.")
    add_io_args(parser)
    args = parser.parse_args(argv)

    images = discover_images(args.input_dir, args.limit, args.per_condition, args.seed, split=args.split, include_refs=args.include_refs)
    banner("evaluate_edges", len(images), args.input_dir, OUT_DIR)
    if not images:
        print("No images found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[List[str]] = []
    aggregates: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for path, condition in images:
        gray = to_gray(read_image(path))
        edges, runtimes = detect_all(gray)
        ref = edges["project3_canny"]
        for det in DETECTORS:
            e = edges[det]
            density = float((e > 0).mean())
            jacc = iou(e, ref) if det != "project3_canny" else 1.0
            rows.append([path.name, condition, det,
                         f"{density:.6f}", f"{runtimes[det]:.3f}", f"{jacc:.4f}"])
            aggregates[det]["density"].append(density)
            aggregates[det]["runtime"].append(runtimes[det])
            aggregates[det]["iou"].append(jacc)
        print(f"  ✔ {path.name}")

    per_image_csv = OUT_DIR / "per_image_metrics.csv"
    with per_image_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "condition", "detector", "edge_density", "runtime_ms", "iou_vs_proj3_canny"])
        w.writerows(rows)

    agg_csv = OUT_DIR / "aggregate_metrics.csv"
    with agg_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "density_mean", "density_std",
                    "runtime_ms_mean", "runtime_ms_std",
                    "iou_mean", "iou_std", "n"])
        for det in DETECTORS:
            d = np.array(aggregates[det]["density"])
            r = np.array(aggregates[det]["runtime"])
            j = np.array(aggregates[det]["iou"])
            w.writerow([det,
                        f"{d.mean():.6f}", f"{d.std():.6f}",
                        f"{r.mean():.3f}",  f"{r.std():.3f}",
                        f"{j.mean():.4f}",  f"{j.std():.4f}",
                        len(d)])

    # bar charts
    detectors = list(DETECTORS)

    def _bar(metric: str, ylabel: str, fname: str) -> None:
        means = [np.mean(aggregates[d][metric]) for d in detectors]
        stds  = [np.std(aggregates[d][metric])  for d in detectors]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(detectors, means, yerr=stds, capsize=4, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} across detectors (n={len(images)})")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=120)
        plt.close(fig)

    _bar("density", "Edge density (fraction of pixels)", "edge_density_by_detector.png")
    _bar("iou",     "IoU vs. Project 3 Canny",            "iou_by_detector.png")
    _bar("runtime", "Runtime (ms)",                       "runtime_by_detector.png")

    print(f"\nPer-image rows: {len(rows)}")
    print(f"Wrote {per_image_csv}, {agg_csv}, and 3 PNG plots to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
