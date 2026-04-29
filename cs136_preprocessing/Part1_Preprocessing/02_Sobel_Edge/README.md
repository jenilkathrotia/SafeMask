# Part 1.2 — Sobel Edge Detector

This is the same Sobel from Project 3, just rewritten in Python.

## The kernels

```
Sx = [[ 1, 0, -1],     Sy = [[-1, -2, -1],
      [ 2, 0, -2],            [ 0,  0,  0],
      [ 1, 0, -1]]            [ 1,  2,  1]]
```

## What we save per image

| Suffix                | What it is                                              |
| --------------------- | ------------------------------------------------------- |
| `__01_gx_abs.png`     | horizontal gradient (size only, no sign)                |
| `__02_gy_abs.png`     | vertical gradient                                       |
| `__03_magnitude.png`  | sqrt(gx^2 + gy^2), scaled to 0 to 255                   |
| `__04_binary_p75.png` | edges = pixels in the top 25% of the magnitude image    |

## Flags

`--pre-blur SIGMA` runs a Gaussian first. We default to sigma=1 because
cv07 says you should blur before any first-derivative operator. Pass 0
to skip. `--threshold-percentile` lets you change the binarization
cutoff from the command line instead of editing the script.
