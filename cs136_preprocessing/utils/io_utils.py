"""Shared I/O helpers for the cs136_preprocessing pipeline.

Centralizes input discovery so every algorithm script accepts the same
``--input-dir``, ``--limit``, and ``--per-condition`` options. Defaults to the
``My_Test/`` driving samples shipped in the SafeMask repo, but is wired so it
can also walk an ACDC-style ``rgb_anon/{fog,night,rain,snow}/train`` tree once
those images are downloaded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ACDC adverse-condition tags — detected from path components.
ACDC_CONDITIONS = ("fog", "night", "rain", "snow")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Default input dir: the 6 driving samples bundled in My_Test/.
# Resolved relative to this file so scripts can run from anywhere.
_THIS = Path(__file__).resolve()
DEFAULT_INPUT_DIR = (_THIS.parents[2] / "My_Test").resolve()


def add_io_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared --input-dir / --limit / --per-condition / --seed flags."""
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder of input images (recursive). Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum total images to process (0 = no cap). Useful for ACDC's 4006-image set.",
    )
    parser.add_argument(
        "--per-condition",
        type=int,
        default=0,
        help="When inputs sit under fog/night/rain/snow folders, cap per condition.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Filter ACDC splits by path component. Default 'all'.",
    )
    parser.add_argument(
        "--include-refs",
        action="store_true",
        help="Include ACDC clear-weather references (*_rgb_ref_anon.png). "
             "Off by default — adverse frames only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling when --limit/--per-condition truncate.",
    )


def _condition_of(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    for cond in ACDC_CONDITIONS:
        if cond in parts:
            return cond
    return "general"


def discover_images(
    root: Path,
    limit: int = 0,
    per_condition: int = 0,
    seed: int = 42,
    split: str = "all",
    include_refs: bool = False,
) -> List[Tuple[Path, str]]:
    """Walk *root* recursively and return ``[(image_path, condition_tag), ...]``.

    The ``condition_tag`` is one of fog/night/rain/snow when the path lives
    under that folder (ACDC layout); otherwise ``"general"``.

    ``per_condition`` caps how many images are kept per tag (random sample
    when the bucket is larger). ``limit`` then caps the final total.

    ``split`` filters ACDC train/val/test by path component (``"all"`` = no
    filter). ``include_refs`` controls whether ACDC clear-weather reference
    frames (``*_rgb_ref_anon.png``) are kept; default skips them.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    rng = random.Random(seed)
    split_filter = split if split != "all" else None

    by_cond: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        # ACDC reference frames are not adverse-weather inputs.
        if not include_refs and path.name.endswith("_rgb_ref_anon.png"):
            continue
        # Optional split filter (ACDC layout: .../rgb_anon/<cond>/<split>/<seq>/...)
        if split_filter is not None:
            parts = {p.lower() for p in path.parts}
            if split_filter not in parts:
                continue
        cond = _condition_of(path)
        by_cond.setdefault(cond, []).append(path)

    if per_condition > 0:
        for cond, paths in by_cond.items():
            if len(paths) > per_condition:
                by_cond[cond] = rng.sample(paths, per_condition)

    flat: List[Tuple[Path, str]] = []
    for cond, paths in by_cond.items():
        flat.extend((p, cond) for p in paths)
    flat.sort(key=lambda pc: (pc[1], str(pc[0])))

    if limit > 0 and len(flat) > limit:
        flat = rng.sample(flat, limit)
        flat.sort(key=lambda pc: (pc[1], str(pc[0])))

    return flat


def read_image(path: Path) -> np.ndarray:
    """Read an image in BGR uint8. Handles WebP files with .png extension."""
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        # cv2 occasionally chokes on WebP-disguised-as-png; fall back via numpy.
        with open(path, "rb") as fh:
            buf = np.frombuffer(fh.read(), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Could not decode image: {path}")
    return arr


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → single-channel uint8."""
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def save_image(out_path: Path, image: np.ndarray) -> None:
    """Save *image* as PNG, creating parent dirs as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(out_path), image)
    if not ok:
        raise IOError(f"Failed to write {out_path}")


def output_stem(path: Path, condition: str) -> str:
    """Build a flat output filename that keeps the condition prefix and source name."""
    return f"{condition}__{path.stem}"


def banner(script_name: str, n_inputs: int, input_dir: Path, output_dir: Path) -> None:
    """Print a uniform start banner for every script."""
    print(f"[{script_name}] {n_inputs} input images")
    print(f"[{script_name}]   input : {input_dir}")
    print(f"[{script_name}]   output: {output_dir}")
    print(f"[{script_name}]   cv2={cv2.__version__}, numpy={np.__version__}, py={sys.version.split()[0]}")
