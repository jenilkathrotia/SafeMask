# CS 136 preprocessing for the SafeMask data loader.
# Applies the four findings from the term-project robustness study to every
# training image: sigma=1 Gaussian, CLAHE on Lab L (fog/night only), Canny
# edges as a 4th channel, and morphological cleanup on those edges.
# All four steps are configurable in configs/config.yaml.

import cv2
import numpy as np

# 5x5 Gaussian / 273 (same as Project 3)
GAUSS = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
], dtype=np.float64) / 273.0

P = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float64)
Q = np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float64)


def project3_canny(gray, theta_low_frac=0.05, theta_high_mult=2.5):
    g = gray.astype(np.float64)
    s = cv2.filter2D(g, cv2.CV_64F, GAUSS, borderType=cv2.BORDER_REPLICATE)
    p = cv2.filter2D(s, cv2.CV_64F, P, borderType=cv2.BORDER_REPLICATE)
    q = cv2.filter2D(s, cv2.CV_64F, Q, borderType=cv2.BORDER_REPLICATE)
    m = np.sqrt(p * p + q * q)
    alpha = np.degrees(np.arctan2(q, p))
    alpha[alpha < 0] += 360

    zeta = np.full(m.shape, -1, dtype=np.int8)
    zeta[((alpha >= 0) & (alpha < 22.5)) | ((alpha >= 157.5) & (alpha < 202.5)) |
         ((alpha >= 337.5) & (alpha <= 360))] = 0
    zeta[((alpha >= 22.5) & (alpha < 67.5)) | ((alpha >= 202.5) & (alpha < 247.5))] = 1
    zeta[((alpha >= 67.5) & (alpha < 112.5)) | ((alpha >= 247.5) & (alpha < 292.5))] = 2
    zeta[zeta == -1] = 3

    pad = np.pad(m, 1, mode="edge")
    n1 = np.zeros_like(m); n2 = np.zeros_like(m)
    s0 = (zeta == 0); s1 = (zeta == 1); s2 = (zeta == 2); s3 = (zeta == 3)
    n1[s0] = pad[:-2, 1:-1][s0]; n2[s0] = pad[2:, 1:-1][s0]
    n1[s1] = pad[:-2, :-2][s1];  n2[s1] = pad[2:, 2:][s1]
    n1[s2] = pad[1:-1, :-2][s2]; n2[s2] = pad[1:-1, 2:][s2]
    n1[s3] = pad[:-2, 2:][s3];   n2[s3] = pad[2:, :-2][s3]

    e = np.where((m >= n1) & (m >= n2), m, 0.0)
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = 0
    max_e = float(e.max())
    if max_e <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    theta_l = max_e * theta_low_frac
    theta_h = theta_l * theta_high_mult
    strong = (e > theta_h).astype(np.uint8)
    candidate = ((e >= theta_l) & (e <= theta_h)).astype(np.uint8)
    union = (strong | candidate).astype(np.uint8)
    n_comp, labels = cv2.connectedComponents(union, connectivity=8)
    keep = np.zeros(n_comp, dtype=bool)
    keep[np.unique(labels[strong.astype(bool)])] = True
    keep[0] = False
    return (keep[labels].astype(np.uint8) * 255)


def gaussian_blur(image, sigma=1.0):
    if sigma <= 0:
        return image
    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma,
                            borderType=cv2.BORDER_REPLICATE)


def apply_clahe_lab(image, clip_limit=3.0, tile_grid=8):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def compute_canny_channel(image, theta_low_frac=0.05, theta_high_mult=2.5,
                          morph_cleanup=True):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = project3_canny(gray, theta_low_frac, theta_high_mult)
    if morph_cleanup:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k)
    return edges


def detect_acdc_condition(path):
    # find fog/night/rain/snow in the path, otherwise None
    parts = {p.lower() for p in path.replace("\\", "/").split("/")}
    for cond in ("fog", "night", "rain", "snow"):
        if cond in parts:
            return cond
    return None


def apply_cs136_preprocessing(image, condition, config):
    # config is the cs136_preprocessing dict from configs/config.yaml.
    # Returns (H, W, 3) or (H, W, 4) depending on whether canny_channel is on.
    if not config.get("enabled", False):
        return image

    out = image
    if config.get("gaussian", {}).get("enabled", True):
        out = gaussian_blur(out, sigma=float(config["gaussian"].get("sigma", 1.0)))

    clahe_cfg = config.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        active = set(clahe_cfg.get("conditions", ["fog", "night"]))
        if condition is None or condition in active:
            out = apply_clahe_lab(out,
                                  clip_limit=float(clahe_cfg.get("clip_limit", 3.0)),
                                  tile_grid=int(clahe_cfg.get("tile_grid", 8)))

    canny_cfg = config.get("canny_channel", {})
    if canny_cfg.get("enabled", True):
        edges = compute_canny_channel(
            out,
            theta_low_frac=float(canny_cfg.get("theta_low_frac", 0.05)),
            theta_high_mult=float(canny_cfg.get("theta_high_mult", 2.5)),
            morph_cleanup=bool(canny_cfg.get("morph_cleanup", True)),
        )
        out = np.dstack([out, edges])  # (H, W, 4)

    return out
