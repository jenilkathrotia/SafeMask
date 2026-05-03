# Part 3 / Step 1 - Make distorted copies of every input image
# Three distortions:
#   - Gaussian noise (sigma 25)
#   - Motion blur (15-pixel horizontal kernel)
#   - Low contrast (alpha 0.4, beta 40)
# These feed into 02_Pipeline_Comparison.

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 5

NOISY_DIR = os.path.join(os.path.dirname(__file__), "Noisy")
BLUR_DIR = os.path.join(os.path.dirname(__file__), "Blurred")
LOWC_DIR = os.path.join(os.path.dirname(__file__), "LowContrast")
for d in (NOISY_DIR, BLUR_DIR, LOWC_DIR):
    os.makedirs(d, exist_ok=True)

NOISE_SIGMA = 25
BLUR_LEN = 15
ALPHA = 0.4  # contrast multiplier
BETA = 40    # brightness offset


def add_noise(bgr, sigma, rng):
    return np.clip(bgr.astype(np.float32) + rng.normal(0, sigma, bgr.shape),
                   0, 255).astype(np.uint8)


def motion_blur(bgr, length):
    if length < 3:
        return bgr.copy()
    kern = np.zeros((length, length), dtype=np.float32)
    kern[length // 2, :] = 1.0 / length
    return cv2.filter2D(bgr, ddepth=-1, kernel=kern,
                        borderType=cv2.BORDER_REPLICATE)


def low_contrast(bgr, alpha, beta):
    return np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


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
rng = np.random.default_rng(42)
print(f"Processing {len(images)} images")

for path, cond in images:
    name = cond + "__" + os.path.splitext(os.path.basename(path))[0]
    bgr = cv2.imread(path)
    cv2.imwrite(f"{NOISY_DIR}/{name}__noise_sigma{NOISE_SIGMA}.png",
                add_noise(bgr, NOISE_SIGMA, rng))
    cv2.imwrite(f"{BLUR_DIR}/{name}__motion_len{BLUR_LEN}.png",
                motion_blur(bgr, BLUR_LEN))
    cv2.imwrite(f"{LOWC_DIR}/{name}__alpha{ALPHA}_beta{BETA}.png",
                low_contrast(bgr, ALPHA, BETA))
    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved into Noisy/, Blurred/, LowContrast/")
