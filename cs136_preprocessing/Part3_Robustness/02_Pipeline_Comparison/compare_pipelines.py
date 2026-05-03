# Part 3 / Step 2 - Compare clean vs distorted outputs
#
# We run four methods on the clean image and on each distorted version,
# then measure how much the output changed:
#   - Sobel at P75
#   - Project 3 Canny (no pre-blur)
#   - Project 3 Canny WITH a sigma=1 pre-blur (the cv08 recommendation)
#   - Texture segmentation (Gabor + K-Means)
# Score: IoU for edge detectors, ARI for the segmentation (label-permutation invariant).

import os, sys, glob, csv
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 20

OUT_DIR = os.path.join(os.path.dirname(__file__), "Results")
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_SIGMA = 25
BLUR_LEN = 15
ALPHA = 0.4
BETA = 40

SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
GAUSS = np.array([
    [1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]
], dtype=np.float64) / 273.0
P = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float64)
Q = np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float64)


def canny_proj3(gray, theta_low_frac=0.05, theta_high_mult=2.5):
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


def sobel_p75(gray):
    gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, SOBEL_X,
                      borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, SOBEL_Y,
                      borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx * gx + gy * gy)
    lo, hi = mag.min(), mag.max()
    if hi - lo < 1e-9:
        return np.zeros_like(mag, dtype=np.uint8)
    mag_u8 = ((mag - lo) / (hi - lo) * 255).astype(np.uint8)
    return ((mag_u8 >= np.percentile(mag_u8, 75)).astype(np.uint8) * 255)


def canny_with_preblur(gray):
    sm = cv2.GaussianBlur(gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)
    return canny_proj3(sm)


def texture_seg(gray, k=4):
    feats = []
    g = gray.astype(np.float32) / 255
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for lam in [6, 12, 24]:
            kern = cv2.getGaborKernel((21, 21), 4, theta, lam, 0.5, 0,
                                      ktype=cv2.CV_32F)
            r = np.abs(cv2.filter2D(g, cv2.CV_32F, kern))
            r = cv2.GaussianBlur(r, (0, 0), sigmaX=lam / 2)
            feats.append(r)
    feats = np.stack(feats, axis=-1)
    h, w, c = feats.shape
    flat = feats.reshape(-1, c)
    flat = (flat - flat.mean(0, keepdims=True)) / (flat.std(0, keepdims=True) + 1e-6)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(flat.astype(np.float32), k, None, crit,
                              attempts=3, flags=cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w)


def add_noise(bgr, sigma, rng):
    return np.clip(bgr.astype(np.float32) + rng.normal(0, sigma, bgr.shape),
                   0, 255).astype(np.uint8)

def motion_blur(bgr, length):
    kern = np.zeros((length, length), dtype=np.float32)
    kern[length // 2, :] = 1.0 / length
    return cv2.filter2D(bgr, -1, kern, borderType=cv2.BORDER_REPLICATE)

def low_contrast(bgr, alpha, beta):
    return np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def iou(a, b):
    ab = a > 0; bb = b > 0
    inter = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return inter / union if union > 0 else 1.0


def ari(a, b):
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(a.flatten(), b.flatten()))
    except ImportError:
        return float((a.flatten() == b.flatten()).mean())


def run_all(gray):
    return {
        "sobel_p75": sobel_p75(gray),
        "canny_proj3": canny_proj3(gray),
        "canny_proj3_preblur": canny_with_preblur(gray),
        "texture_seg": texture_seg(gray),
    }


def find_images():
    images = []
    for cond in ["fog", "night", "rain", "snow"]:
        files = sorted(glob.glob(f"{INPUT_DIR}/{cond}/train/*/*_rgb_anon.png"))
        for f in files[:PER_WEATHER]:
            images.append((f, cond))
    if images:
        return images
    my_test = os.path.join(os.path.dirname(__file__), "..", "..", "..", "My_Test")
    return [(f, "test") for f in sorted(glob.glob(os.path.join(my_test, "*.png")))]


images = find_images()
print(f"Comparing on {len(images)} images")

DETECTORS = ["sobel_p75", "canny_proj3", "canny_proj3_preblur", "texture_seg"]
DISTORTIONS = ["noisy", "blurred", "lowcontrast"]
rng = np.random.default_rng(42)

rows = []
agg = {d: {dn: [] for dn in DISTORTIONS} for d in DETECTORS}

for path, cond in images:
    name = os.path.basename(path)
    bgr = cv2.imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clean = run_all(gray)

    variants = {
        "noisy": add_noise(bgr, NOISE_SIGMA, rng),
        "blurred": motion_blur(bgr, BLUR_LEN),
        "lowcontrast": low_contrast(bgr, ALPHA, BETA),
    }
    for dname, d_bgr in variants.items():
        d_gray = cv2.cvtColor(d_bgr, cv2.COLOR_BGR2GRAY)
        distorted = run_all(d_gray)
        for det in DETECTORS:
            score = ari(clean[det], distorted[det]) if det == "texture_seg" \
                    else iou(clean[det], distorted[det])
            rows.append([name, cond, dname, det, f"{score:.4f}"])
            agg[det][dname].append(score)
    print(f"  done: {name}")

# diagnostic strip on the first image: clean vs noisy vs preblurred Canny
diag_dir = os.path.join(OUT_DIR, "diagnostic_strips")
os.makedirs(diag_dir, exist_ok=True)
if images:
    path0, cond0 = images[0]
    bgr0 = cv2.imread(path0)
    name0 = cond0 + "__" + os.path.splitext(os.path.basename(path0))[0]
    for dname, fn in [("noisy", lambda x: add_noise(x, NOISE_SIGMA, rng)),
                      ("blurred", lambda x: motion_blur(x, BLUR_LEN)),
                      ("lowcontrast", lambda x: low_contrast(x, ALPHA, BETA))]:
        d_bgr = fn(bgr0)
        clean_e = canny_proj3(cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY))
        dist_e = canny_proj3(cv2.cvtColor(d_bgr, cv2.COLOR_BGR2GRAY))
        dist_pre = canny_with_preblur(cv2.cvtColor(d_bgr, cv2.COLOR_BGR2GRAY))
        strip = np.hstack([
            cv2.cvtColor(clean_e, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(dist_e, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(dist_pre, cv2.COLOR_GRAY2BGR),
        ])
        cv2.imwrite(f"{diag_dir}/{name0}__{dname}__clean_vs_distorted_vs_preblur.png", strip)

# write CSVs
with open(os.path.join(OUT_DIR, "per_image_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "condition", "distortion", "detector", "score_iou_or_ari"])
    w.writerows(rows)

with open(os.path.join(OUT_DIR, "aggregate_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["detector", "distortion", "score_mean", "score_std", "n"])
    for det in DETECTORS:
        for dn in DISTORTIONS:
            arr = np.array(agg[det][dn])
            if arr.size:
                w.writerow([det, dn, f"{arr.mean():.4f}", f"{arr.std():.4f}", arr.size])

# Heatmap
grid = np.zeros((len(DETECTORS), len(DISTORTIONS)))
for i, det in enumerate(DETECTORS):
    for j, dn in enumerate(DISTORTIONS):
        arr = agg[det][dn]
        grid[i, j] = float(np.mean(arr)) if arr else 0
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(DISTORTIONS))); ax.set_xticklabels(DISTORTIONS, rotation=20, ha="right")
ax.set_yticks(range(len(DETECTORS))); ax.set_yticklabels(DETECTORS)
for i in range(len(DETECTORS)):
    for j in range(len(DISTORTIONS)):
        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                color="white" if grid[i, j] < 0.5 else "black")
fig.colorbar(im, ax=ax, label="Mean IoU (edges) / ARI (segmentation)")
ax.set_title(f"Distortion robustness (n={len(images)})")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "iou_heatmap.png"), dpi=120)
plt.close(fig)

print(f"\nSaved CSVs and heatmap to {OUT_DIR}")
