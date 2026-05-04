# Part 1.6 - Texture segmentation with Gabor filters + K-Means
#
# Steps:
#   1. Build a Gabor filter bank (4 orientations x 3 wavelengths = 12 filters)
#   2. For each pixel, compute |response| of all 12 filters and smooth them
#   3. (Color version) add Lab a,b channels as extra features
#   4. K-Means with k=4
#   5. Color the output by cluster

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 1
K = 4  # number of clusters

GRAY_OUT = os.path.join(os.path.dirname(__file__), "Grayscale_Texture_Images")
COLOR_OUT = os.path.join(os.path.dirname(__file__), "Color_Texture_Images")
os.makedirs(GRAY_OUT, exist_ok=True)
os.makedirs(COLOR_OUT, exist_ok=True)

THETAS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
LAMBDAS = [6.0, 12.0, 24.0]
KSIZE = 21

# 8-color palette so cluster colors are easy to tell apart
PALETTE = np.array([
    [220,  20,  60], [ 30, 144, 255], [ 50, 205,  50], [255, 165,   0],
    [148,   0, 211], [  0, 206, 209], [255, 215,   0], [105, 105, 105],
], dtype=np.uint8)


def gabor_features(gray):
    feats = []
    g = gray.astype(np.float32) / 255
    for theta in THETAS:
        for lam in LAMBDAS:
            kern = cv2.getGaborKernel((KSIZE, KSIZE), 4.0, theta, lam, 0.5, 0,
                                      ktype=cv2.CV_32F)
            r = np.abs(cv2.filter2D(g, cv2.CV_32F, kern))
            r = cv2.GaussianBlur(r, (0, 0), sigmaX=lam / 2)
            feats.append(r)
    return np.stack(feats, axis=-1)  # H x W x 12


def kmeans_segment(feats):
    h, w, c = feats.shape
    flat = feats.reshape(-1, c).astype(np.float32)
    # z-score
    flat = (flat - flat.mean(0, keepdims=True)) / (flat.std(0, keepdims=True) + 1e-6)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(flat, K, None, crit, attempts=3,
                              flags=cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w)


def colorize(labels):
    return PALETTE[labels % len(PALETTE)]


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

for path, cond in images:
    name = cond + "__" + os.path.splitext(os.path.basename(path))[0]
    bgr = cv2.imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # grayscale-only segmentation
    gab = gabor_features(gray)
    labels_g = kmeans_segment(gab)
    seg_g = colorize(labels_g)
    side_g = np.hstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(seg_g, cv2.COLOR_RGB2BGR)])
    cv2.imwrite(f"{GRAY_OUT}/{name}__seg_k{K}.png", side_g)

    # color version: add Lab a,b channels
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[..., 1:] / 255  # H x W x 2
    ab = np.stack([cv2.GaussianBlur(ab[..., i], (0, 0), sigmaX=4) for i in range(2)],
                  axis=-1)
    feats_color = np.concatenate([gab, ab], axis=-1)
    labels_c = kmeans_segment(feats_color)
    seg_c = colorize(labels_c)
    side_c = np.hstack([bgr, cv2.cvtColor(seg_c, cv2.COLOR_RGB2BGR)])
    cv2.imwrite(f"{COLOR_OUT}/{name}__seg_k{K}.png", side_c)

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved grayscale to {GRAY_OUT}")
print(f"Saved color to {COLOR_OUT}")
