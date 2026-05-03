# Part 2 / Creative idea 2 - Morphological cleanup of Canny edges
#
# Project 3 covered binary expand/shrink. We extend that:
#   closing  (dilate then erode) -> fills small gaps in edges
#   opening  (erode then dilate) -> removes single noisy pixels
#   skeleton (Zhang-Suen)        -> thins edges to 1 pixel wide
# Cleaner edges => cleaner Hough voting later.

import os, glob, cv2, numpy as np

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 5

OUT_DIR = os.path.join(os.path.dirname(__file__), "Morph_Cleanup_Images")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None  # if not installed, just skip the skeleton step


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

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

for path, cond in images:
    name = cond + "__" + os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 75, 200, L2gradient=True)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(f"{OUT_DIR}/{name}__01_canny.png", edges)
    cv2.imwrite(f"{OUT_DIR}/{name}__02_closed.png", closed)
    cv2.imwrite(f"{OUT_DIR}/{name}__03_opened.png", opened)
    if skeletonize is not None:
        skel = (skeletonize(opened > 0).astype(np.uint8) * 255)
        cv2.imwrite(f"{OUT_DIR}/{name}__04_skeleton.png", skel)

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
