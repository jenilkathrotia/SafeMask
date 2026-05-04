# Part 2 / Creative idea 1 - CLAHE + bilateral filter + Canny
#
# Why: in fog and night images, edges get washed out. So we boost the contrast
# first with CLAHE, denoise without blurring edges with a bilateral filter,
# then run Canny on the cleaner image. We compare against plain Canny.

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 1

OUT_DIR = os.path.join(os.path.dirname(__file__), "CLAHE_Canny_Images")
os.makedirs(OUT_DIR, exist_ok=True)


def clahe_on_L(bgr, clip=3.0, tile=8):
    # Apply CLAHE only to the L channel of CIE Lab
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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

    clahe_bgr = clahe_on_L(bgr, clip=3.0, tile=8)
    bilat = cv2.bilateralFilter(clahe_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Canny on raw vs Canny on the cleaned-up image
    raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bilat_gray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
    edges_baseline = cv2.Canny(raw_gray, 75, 200, L2gradient=True)
    edges_pipeline = cv2.Canny(bilat_gray, 75, 200, L2gradient=True)

    cv2.imwrite(f"{OUT_DIR}/{name}__01_source.png", bgr)
    cv2.imwrite(f"{OUT_DIR}/{name}__02_clahe.png", clahe_bgr)
    cv2.imwrite(f"{OUT_DIR}/{name}__03_clahe_bilateral.png", bilat)
    cv2.imwrite(f"{OUT_DIR}/{name}__04_canny_baseline.png", edges_baseline)
    cv2.imwrite(f"{OUT_DIR}/{name}__05_canny_pipeline.png", edges_pipeline)

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
