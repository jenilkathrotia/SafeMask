# Part 2 / Creative #1 — CLAHE + Bilateral + Canny

## The idea

ACDC fog and night images have a very compressed range of brightness.
Plain Canny misses faint edges in those images. So we chain three
things together:

1. **CLAHE**, a smarter version of histogram equalization. It boosts
   contrast in small tiles of the image so the bright spots do not
   wash out.
2. **Bilateral filter**, a denoiser that does not blur edges. We need
   this because CLAHE also makes noise stronger, and a regular
   Gaussian would erase the contrast we just added.
3. **Canny** with the same Project 3 thresholds, now running on a
   much cleaner input.

## What we save per image

We save every step so you can see what each one does:

| Suffix                          | Step                              |
| ------------------------------- | --------------------------------- |
| `__01_source.png`               | original BGR image                |
| `__02_clahe.png`                | after CLAHE on the L channel      |
| `__03_clahe_bilateral.png`      | after the bilateral filter        |
| `__04_canny_baseline.png`       | Canny on the raw image            |
| `__05_canny_pipeline.png`       | Canny on the CLAHE+bilateral image|

The headline comparison is `__04_canny_baseline.png` vs.
`__05_canny_pipeline.png`.

## Flags you can change

| Flag                       | Default | What it does                                          |
| -------------------------- | ------- | ----------------------------------------------------- |
| `--clip-limit`             | 3.0     | higher = stronger contrast boost                      |
| `--tile-grid`              | 8       | bigger tiles = more like global histogram equalize    |
| `--bilateral-sigma-color`  | 75      | how different two pixel values can be and still mix   |
| `--bilateral-sigma-space`  | 75      | how far apart two pixels can be and still mix         |

## What we found

Just doing CLAHE on its own makes noise look worse, especially in flat
sky areas. The bilateral filter cleans up that noise without blurring
the new edges. On fog images we recovered guard rails and lane
markings that the plain Canny missed completely. On night images
with strong street lights, CLAHE can push the lights into a glare,
and `--clip-limit 2.0` works better there.
