# Part 2 / Creative idea 3 - Mean-Shift segmentation in CIE Lab
#
# Different from Part 1.6 (Gabor + K-Means). Mean-Shift doesn't need a
# preset cluster count - it finds them automatically. We do this in Lab
# color space so distances match how the eye sees colors. Then we
# quantize with K-Means just for clean visualization.

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 5

OUT_DIR = os.path.join(os.path.dirname(__file__), "Color_Texture_Images")
os.makedirs(OUT_DIR, exist_ok=True)


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

    # Mean-shift in Lab space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ms = cv2.pyrMeanShiftFiltering(lab, sp=15, sr=25, maxLevel=2)
    ms_bgr = cv2.cvtColor(ms, cv2.COLOR_LAB2BGR)

    # Quantize the result down to 8 colors so regions are clear
    Z = ms_bgr.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, 8, None, crit, attempts=3,
                                    flags=cv2.KMEANS_PP_CENTERS)
    quant = centers[labels.flatten()].astype(np.uint8).reshape(bgr.shape)

    side = np.hstack([bgr, ms_bgr, quant])
    cv2.imwrite(f"{OUT_DIR}/{name}__source_meanshift_quant.png", side)

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
