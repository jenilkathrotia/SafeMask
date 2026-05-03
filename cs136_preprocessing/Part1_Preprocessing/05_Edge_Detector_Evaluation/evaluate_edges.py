# Part 1.5 - Compare edge detectors with numbers
# Run all four detectors on each image, measure:
#   - edge density (% of pixels marked as edges)
#   - runtime in ms
#   - IoU vs the Project 3 Canny (treated as the reference)
# Save per-image and aggregate CSVs + bar charts.

import os, sys, glob, csv, time
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 20  # 80 total images keeps the report numbers stable

OUT_DIR = os.path.join(os.path.dirname(__file__), "Evaluation_Results")
os.makedirs(OUT_DIR, exist_ok=True)

# Sobel kernels
SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

# Canny (5x5 / 273) Gaussian
GAUSS = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
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
    for sec, (a_idx, b_idx) in enumerate([
        ((slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1))),
        ((slice(None, -2), slice(None, -2)), (slice(2, None), slice(2, None))),
        ((slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None))),
        ((slice(None, -2), slice(2, None)), (slice(2, None), slice(None, -2))),
    ]):
        mask = (zeta == sec)
        n1[mask] = pad[a_idx][mask]
        n2[mask] = pad[b_idx][mask]

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


def sobel_mag(gray):
    gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, SOBEL_X,
                      borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, SOBEL_Y,
                      borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx * gx + gy * gy)
    lo, hi = mag.min(), mag.max()
    if hi - lo < 1e-9:
        return np.zeros_like(mag, dtype=np.uint8)
    return ((mag - lo) / (hi - lo) * 255).astype(np.uint8)


def iou(a, b):
    ab = a > 0; bb = b > 0
    inter = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return inter / union if union > 0 else 0


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
print(f"Evaluating on {len(images)} images")

DETECTORS = ["sobel_p75", "sobel_p90", "project3_canny", "cv2_canny_75_200"]
rows = []
agg = {d: {"density": [], "runtime": [], "iou": []} for d in DETECTORS}

for path, cond in images:
    name = os.path.basename(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel (P75 and P90)
    t = time.perf_counter()
    mag_u8 = sobel_mag(gray)
    sobel_time = (time.perf_counter() - t) * 1000
    sobel_p75 = (mag_u8 >= np.percentile(mag_u8, 75)).astype(np.uint8) * 255
    sobel_p90 = (mag_u8 >= np.percentile(mag_u8, 90)).astype(np.uint8) * 255

    # Project 3 Canny (the reference)
    t = time.perf_counter()
    canny = canny_proj3(gray)
    canny_time = (time.perf_counter() - t) * 1000

    # OpenCV Canny
    t = time.perf_counter()
    cv_canny = cv2.Canny(gray, 75, 200, L2gradient=True)
    cv_canny_time = (time.perf_counter() - t) * 1000

    edges = {"sobel_p75": sobel_p75, "sobel_p90": sobel_p90,
             "project3_canny": canny, "cv2_canny_75_200": cv_canny}
    times = {"sobel_p75": sobel_time, "sobel_p90": sobel_time,
             "project3_canny": canny_time, "cv2_canny_75_200": cv_canny_time}

    for d in DETECTORS:
        density = float((edges[d] > 0).mean())
        score = 1.0 if d == "project3_canny" else iou(edges[d], canny)
        rows.append([name, cond, d, f"{density:.6f}", f"{times[d]:.3f}", f"{score:.4f}"])
        agg[d]["density"].append(density)
        agg[d]["runtime"].append(times[d])
        agg[d]["iou"].append(score)

    print(f"  done: {name}")

# write CSVs
per_csv = os.path.join(OUT_DIR, "per_image_metrics.csv")
with open(per_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "condition", "detector", "edge_density", "runtime_ms", "iou_vs_proj3_canny"])
    w.writerows(rows)

agg_csv = os.path.join(OUT_DIR, "aggregate_metrics.csv")
with open(agg_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["detector", "density_mean", "density_std",
                "runtime_ms_mean", "runtime_ms_std",
                "iou_mean", "iou_std", "n"])
    for d in DETECTORS:
        dens = np.array(agg[d]["density"])
        run = np.array(agg[d]["runtime"])
        ious = np.array(agg[d]["iou"])
        w.writerow([d, f"{dens.mean():.6f}", f"{dens.std():.6f}",
                    f"{run.mean():.3f}", f"{run.std():.3f}",
                    f"{ious.mean():.4f}", f"{ious.std():.4f}", len(dens)])

# bar charts
def make_bar(metric, ylabel, fname):
    means = [np.mean(agg[d][metric]) for d in DETECTORS]
    stds = [np.std(agg[d][metric]) for d in DETECTORS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(DETECTORS, means, yerr=stds, capsize=4,
           color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (n={len(images)})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)

make_bar("density", "Edge density (fraction of pixels)", "edge_density_by_detector.png")
make_bar("iou", "IoU vs. Project 3 Canny", "iou_by_detector.png")
make_bar("runtime", "Runtime (ms)", "runtime_by_detector.png")

print(f"\nSaved CSVs and 3 PNG plots to {OUT_DIR}")
