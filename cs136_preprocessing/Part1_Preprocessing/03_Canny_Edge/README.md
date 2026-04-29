# Part 1.3: Canny Edge Detector

Two versions of Canny per image, side by side.

## What we run

1. **`__01_project3.png`**, our Python version of Project 3 `canny()`:
   1. 5x5 Gaussian smoothing (kernel divided by 273)
   2. 2x2 P and Q gradient kernels, then magnitude `m` and angle `alpha`
   3. Pick a sector 0 to 3 from the angle
   4. Non-maxima suppression along the sector direction
   5. Hysteresis with `theta_low = 0.05 * max(E)` and `theta_high = 2.5 * theta_low`

   You can change the thresholds with `--theta-low-frac` and
   `--theta-high-mult`.

2. **`__cv2_low_50_150.png`**, **`__cv2_mid_75_200.png`**,
   **`__cv2_high_100_250.png`**, OpenCV's Canny at three threshold
   pairs. This shows how much the thresholds change the output.

The binary maps from this folder feed into Part 1.5.
