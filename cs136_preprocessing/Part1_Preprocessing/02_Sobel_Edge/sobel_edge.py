# Part 1.2 - Sobel edge detector
# Same 3x3 kernels as our Project 3 sobel().
# Pre-blur with sigma=1 first, then compute Sx, Sy, magnitude, and binary edges.

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 5

OUT_DIR = os.path.join(os.path.dirname(__file__), "Sobel_Edge_Images")
os.makedirs(OUT_DIR, exist_ok=True)

# Sobel kernels
SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


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


def stretch(arr):
    # rescale to 0..255
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


images = find_images()
print(f"Processing {len(images)} images")

for path, cond in images:
    name = cond + "__" + os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pre-blur (sigma 1, 7x7 kernel) - reduces noise before edge detection
    gray = cv2.GaussianBlur(gray, (7, 7), 1)

    # Compute gradients
    gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, SOBEL_X,
                      borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, SOBEL_Y,
                      borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx * gx + gy * gy)

    gx_u8 = stretch(np.abs(gx))
    gy_u8 = stretch(np.abs(gy))
    mag_u8 = stretch(mag)

    # Binary edges = pixels in the top 25% of magnitude
    thresh = np.percentile(mag_u8, 75)
    binary = ((mag_u8 >= thresh).astype(np.uint8) * 255)

    cv2.imwrite(f"{OUT_DIR}/{name}__00_source.png", gray)
    cv2.imwrite(f"{OUT_DIR}/{name}__01_gx_abs.png", gx_u8)
    cv2.imwrite(f"{OUT_DIR}/{name}__02_gy_abs.png", gy_u8)
    cv2.imwrite(f"{OUT_DIR}/{name}__03_magnitude.png", mag_u8)
    cv2.imwrite(f"{OUT_DIR}/{name}__04_binary_p75.png", binary)

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
