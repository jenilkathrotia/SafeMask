# Part 1.1 - Gaussian filter
# Smooth each image with Gaussian at sigma 1, 2, and 4.
# Also save the 5x5 kernel from Project 3 (for comparison).

import os, glob, cv2, numpy as np

# CHANGE THIS to where you put the ACDC training images
INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 1  # how many random images to pick from each weather

# folder where outputs go
OUT_DIR = os.path.join(os.path.dirname(__file__), "Gaussian_Filter_Images")
os.makedirs(OUT_DIR, exist_ok=True)

# Project 3 fixed 5x5 Gaussian (divided by 273)
GAUSS_5 = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
], dtype=np.float64) / 273.0


def find_images():
    # try ACDC first (5 images per weather)
    images = []
    for cond in ["fog", "night", "rain", "snow"]:
        files = sorted(glob.glob(f"{INPUT_DIR}/{cond}/train/*/*_rgb_anon.png"))
        for f in files[:PER_WEATHER]:
            images.append((f, cond))
    if images:
        return images
    # fallback: My_Test images
    my_test = os.path.join(os.path.dirname(__file__), "..", "..", "..", "My_Test")
    return [(f, "test") for f in sorted(glob.glob(os.path.join(my_test, "*.png")))]


images = find_images()
print(f"Processing {len(images)} images")

for path, cond in images:
    name = cond + "__" + os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save the source
    cv2.imwrite(f"{OUT_DIR}/{name}__00_source.png", gray)

    # Project 3 5x5/273 kernel
    proj3 = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, GAUSS_5,
                         borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite(f"{OUT_DIR}/{name}__01_proj3_5x5_273.png",
                np.clip(proj3, 0, 255).astype(np.uint8))

    # OpenCV Gaussian at sigma 1, 2, 4
    for sigma in [1, 2, 4]:
        ksize = max(3, int(round(sigma * 6)) | 1)  # odd kernel size
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma,
                                   borderType=cv2.BORDER_REPLICATE)
        cv2.imwrite(f"{OUT_DIR}/{name}__sigma{sigma}.png", blurred)

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
