"""NumPy ports of the C algorithms from CS 136 Project 3 / Project 4.

Kernels and pipeline stages match the reference C implementations under
``CS 136/Project3/netpbm/main.c`` and ``CS 136/Project4/netpbm_hough.c`` so
results are directly comparable. OpenCV is used only for convolution and the
final image I/O — all decision logic (thresholds, NMS, hysteresis, Hough
voting) is hand-rolled to mirror the lecture pseudocode (cv07/cv08).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# ----- Project 3: Sobel ------------------------------------------------------

# Same kernels as Project3/main.c::sobel
SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float64)


def sobel_components(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gx, gy, magnitude) in float64, mirroring the Project 3 sobel()."""
    g = gray.astype(np.float64)
    gx = cv2.filter2D(g, ddepth=cv2.CV_64F, kernel=SOBEL_X, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(g, ddepth=cv2.CV_64F, kernel=SOBEL_Y, borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linear stretch to 0–255 like Project 3's matrix2Image(scale=1)."""
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


# ----- Project 3: Canny ------------------------------------------------------

# 5x5 Gaussian / 273 (the exact kernel used in Project 3 canny())
GAUSS_5_273 = np.array([[1,  4,  7,  4, 1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1,  4,  7,  4, 1]], dtype=np.float64) / 273.0

# 2x2 P/Q gradient kernels from Project 3 canny()
P_KERNEL = np.array([[ 0.5,  0.5],
                     [-0.5, -0.5]], dtype=np.float64)
Q_KERNEL = np.array([[ 0.5, -0.5],
                     [ 0.5, -0.5]], dtype=np.float64)


def canny_project3(
    gray: np.ndarray,
    theta_low_frac: float = 0.05,
    theta_high_mult: float = 2.5,
) -> np.ndarray:
    """Reproduce Project 3's canny() pipeline in NumPy.

    Stages:
      1. 5x5 Gaussian smoothing (kernel/273).
      2. 2x2 P/Q gradient → magnitude m, orientation alpha.
      3. Sector assignment (zeta ∈ {0,1,2,3}).
      4. Non-maxima suppression along the sector direction.
      5. Hysteresis with theta_low = theta_low_frac * m.max() and
         theta_high = theta_high_mult * theta_low. Candidates that touch a
         strong edge through any 8-connected candidate chain are promoted.
    Returns a uint8 binary edge map (0 / 255).
    """
    g = gray.astype(np.float64)
    s = cv2.filter2D(g, cv2.CV_64F, GAUSS_5_273, borderType=cv2.BORDER_REPLICATE)
    p = cv2.filter2D(s, cv2.CV_64F, P_KERNEL, borderType=cv2.BORDER_REPLICATE)
    q = cv2.filter2D(s, cv2.CV_64F, Q_KERNEL, borderType=cv2.BORDER_REPLICATE)

    m = np.sqrt(p * p + q * q)
    alpha = np.degrees(np.arctan2(q, p))
    alpha[alpha < 0] += 360.0

    # Sectors: 0=vertical edge, 1=diag NE-SW, 2=horizontal edge, 3=diag NW-SE
    zeta = np.full(m.shape, -1, dtype=np.int8)
    zeta[((alpha >= 0)     & (alpha <  22.5)) |
         ((alpha >= 157.5) & (alpha < 202.5)) |
         ((alpha >= 337.5) & (alpha <= 360.0))] = 0
    zeta[((alpha >=  22.5) & (alpha <  67.5)) |
         ((alpha >= 202.5) & (alpha < 247.5))] = 1
    zeta[((alpha >=  67.5) & (alpha < 112.5)) |
         ((alpha >= 247.5) & (alpha < 292.5))] = 2
    zeta[zeta == -1] = 3

    # Non-maxima suppression
    h, w = m.shape
    pad = np.pad(m, 1, mode="edge")
    n1 = np.zeros_like(m)
    n2 = np.zeros_like(m)
    s0 = (zeta == 0)
    s1 = (zeta == 1)
    s2 = (zeta == 2)
    s3 = (zeta == 3)
    n1[s0] = pad[:-2, 1:-1][s0]; n2[s0] = pad[2:,  1:-1][s0]
    n1[s1] = pad[:-2, :-2][s1];  n2[s1] = pad[2:,  2:][s1]
    n1[s2] = pad[1:-1, :-2][s2]; n2[s2] = pad[1:-1, 2:][s2]
    n1[s3] = pad[:-2, 2:][s3];   n2[s3] = pad[2:,  :-2][s3]

    e = np.where((m >= n1) & (m >= n2), m, 0.0)
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = 0.0  # ignore borders

    # Hysteresis
    max_e = float(e.max())
    if max_e <= 0.0:
        return np.zeros_like(gray, dtype=np.uint8)
    theta_l = max_e * theta_low_frac
    theta_h = theta_l * theta_high_mult

    strong = (e > theta_h).astype(np.uint8)
    candidate = ((e >= theta_l) & (e <= theta_h)).astype(np.uint8)

    # Connected-component growth: any candidate 8-connected to a strong pixel
    # through other candidates becomes an edge. Use cv2.connectedComponents on
    # (strong | candidate); keep components that contain any strong pixel.
    union = (strong | candidate).astype(np.uint8)
    n_comp, labels = cv2.connectedComponents(union, connectivity=8)
    keep = np.zeros(n_comp, dtype=bool)
    keep[np.unique(labels[strong.astype(bool)])] = True
    keep[0] = False  # background
    final = keep[labels]
    return (final.astype(np.uint8) * 255)


# ----- Project 4: Hough circles ---------------------------------------------

def hough_circles_project4(
    edges: np.ndarray,
    min_r: int,
    max_r: int,
    map_h: int = 0,
    map_w: int = 0,
    map_d: int = 0,
    n_angles: int = 360,
) -> np.ndarray:
    """Compute the 3D Hough accumulator for circles on a binary edge map.

    Mirrors Project4/netpbm_hough.c::houghTransformCircles. Edge pixels each
    cast votes for every (cy, cx, r) circle they could lie on.

    Returns the accumulator with shape (map_h, map_w, map_d). Pass the return
    value to ``find_local_maxima_3d`` to extract circle centers.
    """
    img_h, img_w = edges.shape
    if map_h == 0:
        map_h = img_h
    if map_w == 0:
        map_w = img_w
    if map_d == 0:
        map_d = max(1, max_r - min_r + 1)

    radii = np.arange(min_r, max_r + 1, dtype=np.int32)
    angles = 2.0 * np.pi * np.arange(n_angles) / n_angles
    sin_t = np.sin(angles)
    cos_t = np.cos(angles)

    ys, xs = np.where(edges > 0)
    if ys.size == 0:
        return np.zeros((map_h, map_w, map_d), dtype=np.float64)

    scale_y = map_h / img_h
    scale_x = map_w / img_w
    scale_r = map_d / float(max_r - min_r + 1)

    acc = np.zeros((map_h, map_w, map_d), dtype=np.float64)
    weights = edges[ys, xs].astype(np.float64)
    for ri, r in enumerate(radii):
        cy = ys[:, None] - (r * sin_t)[None, :]
        cx = xs[:, None] - (r * cos_t)[None, :]
        hy = (cy * scale_y).astype(np.int32)
        hx = (cx * scale_x).astype(np.int32)
        hr = int((r - min_r) * scale_r)
        valid = (hy >= 0) & (hy < map_h) & (hx >= 0) & (hx < map_w) & (hr >= 0) & (hr < map_d)
        flat_idx = hy[valid] * map_w + hx[valid]
        plane = acc[:, :, hr].reshape(-1)
        np.add.at(plane, flat_idx, np.broadcast_to(weights[:, None], hy.shape)[valid])
        acc[:, :, hr] = plane.reshape(map_h, map_w)
    return acc


def find_local_maxima_3d(
    volume: np.ndarray,
    n_max: int,
    min_separation: float,
    threshold: float = 0.0,
) -> np.ndarray:
    """Extract up to *n_max* local maxima from a 3D Hough volume.

    Each candidate must:
      * exceed *threshold*,
      * be ≥ all 26 neighbours (true 3D local max),
      * be at least *min_separation* (Euclidean) from already-accepted maxima.

    Returns ``[(y, x, z, strength), ...]`` sorted by descending strength.
    """
    if volume.size == 0 or n_max <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    # 3D non-max suppression via dilation
    from scipy.ndimage import maximum_filter
    nbr_max = maximum_filter(volume, size=3, mode="constant", cval=-np.inf)
    is_local_max = (volume == nbr_max) & (volume > threshold)
    coords = np.argwhere(is_local_max)
    if coords.size == 0:
        return np.zeros((0, 4), dtype=np.float64)

    strengths = volume[is_local_max]
    order = np.argsort(-strengths)
    coords = coords[order]
    strengths = strengths[order]

    accepted: list[tuple[int, int, int, float]] = []
    sep2 = min_separation * min_separation
    for (y, x, z), s in zip(coords, strengths):
        ok = True
        for ay, ax, az, _ in accepted:
            dy, dx, dz = y - ay, x - ax, z - az
            if dy * dy + dx * dx + dz * dz < sep2:
                ok = False
                break
        if ok:
            accepted.append((int(y), int(x), int(z), float(s)))
            if len(accepted) >= n_max:
                break
    return np.asarray(accepted, dtype=np.float64) if accepted else np.zeros((0, 4), dtype=np.float64)
