# Part 1.4 - Hough transform
# Lines: cv2.HoughLinesP for lane markings.
# Circles: both cv2.HoughCircles AND our Project 4 port (3D voting + local maxima).

import os, glob, cv2, numpy as np
from scipy.ndimage import maximum_filter

INPUT_DIR = "/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon"
PER_WEATHER = 1

OUT_DIR = os.path.join(os.path.dirname(__file__), "Hough_Images")
os.makedirs(OUT_DIR, exist_ok=True)


def get_edges(gray):
    # cv2.Canny is fast and good enough as input to Hough
    return cv2.Canny(gray, 75, 200, L2gradient=True)


def draw_lines(bgr, edges):
    out = bgr.copy()
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=80, minLineLength=40, maxLineGap=10)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def draw_circles_cv2(bgr, gray):
    out = bgr.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=120, param2=40, minRadius=10, maxRadius=80)
    if circles is not None:
        for cx, cy, r in np.uint16(np.around(circles[0])):
            cv2.circle(out, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
            cv2.circle(out, (int(cx), int(cy)), 2, (0, 255, 255), 2)
    return out


def hough_circles_proj4(edges, min_r, max_r, n_angles=360):
    # Project 4 port: 3D voting accumulator (cy, cx, r).
    # Uses sin/cos lookup tables for speed.
    h, w = edges.shape
    angles = 2 * np.pi * np.arange(n_angles) / n_angles
    sin_t = np.sin(angles)
    cos_t = np.cos(angles)

    ys, xs = np.where(edges > 0)
    if ys.size == 0:
        return np.zeros((h, w, max_r - min_r + 1), dtype=np.float64)

    acc = np.zeros((h, w, max_r - min_r + 1), dtype=np.float64)
    weights = edges[ys, xs].astype(np.float64)
    for r in range(min_r, max_r + 1):
        cy = ys[:, None] - (r * sin_t)[None, :]
        cx = xs[:, None] - (r * cos_t)[None, :]
        hy = cy.astype(np.int32)
        hx = cx.astype(np.int32)
        valid = (hy >= 0) & (hy < h) & (hx >= 0) & (hx < w)
        flat_idx = hy[valid] * w + hx[valid]
        plane = acc[:, :, r - min_r].reshape(-1)
        np.add.at(plane, flat_idx, np.broadcast_to(weights[:, None], hy.shape)[valid])
        acc[:, :, r - min_r] = plane.reshape(h, w)
    return acc


def find_circle_maxima(volume, n_max, min_sep, threshold):
    # 3D non-max suppression then greedy pick top peaks
    nbr_max = maximum_filter(volume, size=3, mode="constant", cval=-np.inf)
    is_local = (volume == nbr_max) & (volume > threshold)
    coords = np.argwhere(is_local)
    if coords.size == 0:
        return []
    strengths = volume[is_local]
    order = np.argsort(-strengths)

    accepted = []
    sep2 = min_sep * min_sep
    for idx in order:
        y, x, r = coords[idx]
        ok = True
        for ay, ax, ar in accepted:
            dy, dx, dr = y - ay, x - ax, r - ar
            if dy * dy + dx * dx + dr * dr < sep2:
                ok = False; break
        if ok:
            accepted.append((int(y), int(x), int(r)))
            if len(accepted) >= n_max:
                break
    return accepted


def draw_circles_proj4(bgr, edges, min_r=10, max_r=40, n_circles=8):
    # Voting on full resolution is slow in Python.
    # Shrink the edge map first, then scale results back up.
    h, w = edges.shape
    target_long = 200
    scale = min(target_long / max(h, w), 1.0)
    if scale < 1.0:
        small = cv2.resize(edges, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_NEAREST)
        small_min_r = max(2, int(round(min_r * scale)))
        small_max_r = max(small_min_r + 1, int(round(max_r * scale)))
    else:
        small = edges
        small_min_r, small_max_r = min_r, max_r

    acc = hough_circles_proj4(small, small_min_r, small_max_r)
    sep = max(3, (small_max_r - small_min_r) * 0.5)
    threshold = acc.max() * 0.4 if acc.size else 0
    maxima = find_circle_maxima(acc, n_circles, sep, threshold)

    out = bgr.copy()
    for y, x, r_idx in maxima:
        cx = int(x / scale) if scale < 1.0 else x
        cy = int(y / scale) if scale < 1.0 else y
        radius = int((small_min_r + r_idx) / scale) if scale < 1.0 else (small_min_r + r_idx)
        cv2.circle(out, (cx, cy), radius, (255, 128, 0), 2)
        cv2.circle(out, (cx, cy), 2, (255, 255, 255), 2)
    return out


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
    edges = get_edges(gray)

    cv2.imwrite(f"{OUT_DIR}/{name}__00_edges.png", edges)
    cv2.imwrite(f"{OUT_DIR}/{name}__01_lines_overlay.png", draw_lines(bgr, edges))
    cv2.imwrite(f"{OUT_DIR}/{name}__02_circles_cv2.png", draw_circles_cv2(bgr, gray))
    cv2.imwrite(f"{OUT_DIR}/{name}__03_circles_project4_port.png",
                draw_circles_proj4(bgr, edges))

    print(f"  done: {os.path.basename(path)}")

print(f"\nSaved to {OUT_DIR}")
