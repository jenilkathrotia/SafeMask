# Part 1.3 - Canny edge detector
# Our Python version of the Project 3 canny() pipeline:
#   1. 5x5 Gaussian smoothing (kernel/273)
#   2. 2x2 P/Q gradient kernels -> magnitude m, orientation alpha
#   3. 4-sector non-maxima suppression
#   4. Hysteresis (theta_low = 5% of max, theta_high = 2.5 * theta_low)
#
# Also save cv2.Canny() output at 3 different threshold pairs for comparison.

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 1

OUT_DIR = os.path.join(os.path.dirname(__file__), "Canny_Edge_Images")
os.makedirs(OUT_DIR, exist_ok=True)

# 5x5 Gaussian / 273
GAUSS = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
], dtype=np.float64) / 273.0

# 2x2 P/Q gradient kernels
P = np.array([[0.5,  0.5], [-0.5, -0.5]], dtype=np.float64)
Q = np.array([[0.5, -0.5], [ 0.5, -0.5]], dtype=np.float64)


def canny_from_scratch(gray, theta_low_frac=0.05, theta_high_mult=2.5):
    # step 1: smooth
    g = gray.astype(np.float64)
    s = cv2.filter2D(g, cv2.CV_64F, GAUSS, borderType=cv2.BORDER_REPLICATE)

    # step 2: gradient
    p = cv2.filter2D(s, cv2.CV_64F, P, borderType=cv2.BORDER_REPLICATE)
    q = cv2.filter2D(s, cv2.CV_64F, Q, borderType=cv2.BORDER_REPLICATE)
    m = np.sqrt(p * p + q * q)
    alpha = np.degrees(np.arctan2(q, p))
    alpha[alpha < 0] += 360

    # step 3: 4-sector NMS
    # sector 0=horizontal edge, 1=NE-SW diag, 2=vertical edge, 3=NW-SE diag
    zeta = np.full(m.shape, -1, dtype=np.int8)
    zeta[((alpha >= 0) & (alpha < 22.5)) | ((alpha >= 157.5) & (alpha < 202.5)) |
         ((alpha >= 337.5) & (alpha <= 360))] = 0
    zeta[((alpha >= 22.5) & (alpha < 67.5)) | ((alpha >= 202.5) & (alpha < 247.5))] = 1
    zeta[((alpha >= 67.5) & (alpha < 112.5)) | ((alpha >= 247.5) & (alpha < 292.5))] = 2
    zeta[zeta == -1] = 3

    pad = np.pad(m, 1, mode="edge")
    n1 = np.zeros_like(m)
    n2 = np.zeros_like(m)
    s0 = (zeta == 0); s1 = (zeta == 1); s2 = (zeta == 2); s3 = (zeta == 3)
    n1[s0] = pad[:-2, 1:-1][s0]; n2[s0] = pad[2:, 1:-1][s0]
    n1[s1] = pad[:-2, :-2][s1];  n2[s1] = pad[2:,  2:][s1]
    n1[s2] = pad[1:-1, :-2][s2]; n2[s2] = pad[1:-1, 2:][s2]
    n1[s3] = pad[:-2, 2:][s3];   n2[s3] = pad[2:,  :-2][s3]

    e = np.where((m >= n1) & (m >= n2), m, 0.0)
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = 0  # border pixels = no edge

    # step 4: hysteresis
    max_e = float(e.max())
    if max_e <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    theta_l = max_e * theta_low_frac
    theta_h = theta_l * theta_high_mult

    strong = (e > theta_h).astype(np.uint8)
    candidate = ((e >= theta_l) & (e <= theta_h)).astype(np.uint8)
    union = (strong | candidate).astype(np.uint8)

    # candidates connected to a strong edge through other candidates become edges
    n_comp, labels = cv2.connectedComponents(union, connectivity=8)
    keep = np.zeros(n_comp, dtype=bool)
    keep[np.unique(labels[strong.astype(bool)])] = True
    keep[0] = False  # background
    return (keep[labels].astype(np.uint8) * 255)


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
print(f"Processing {len(images)} images")

cv2_presets = [("low_50_150", 50, 150), ("mid_75_200", 75, 200), ("high_100_250", 100, 250)]

for path, cond in images:
    name = cond + "__" + os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"{OUT_DIR}/{name}__00_source.png", gray)
    cv2.imwrite(f"{OUT_DIR}/{name}__01_project3.png", canny_from_scratch(gray))

    for label, lo, hi in cv2_presets:
        cv2.imwrite(f"{OUT_DIR}/{name}__cv2_{label}.png",
                    cv2.Canny(gray, lo, hi, L2gradient=True))

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
