# Part 1.1: Gaussian Filter

This folder runs Gaussian smoothing on each input image at a few
different sigma values.

## What it does

For every input image we save:

- the grayscale source
- the same Project 3 kernel from `canny()` (5x5, divided by 273)
- a sigma=1 Gaussian (kernel size 7)
- a sigma=2 Gaussian (kernel size 13)
- a sigma=4 Gaussian (kernel size 25)

## How to run

```bash
python gaussian_filter.py                       # uses My_Test by default
python gaussian_filter.py --input-dir <PATH>    # ACDC train images
python gaussian_filter.py --per-condition 5     # 5 random images per weather
```

## What the outputs are called

For each image, files are named `<weather>__<name>__<suffix>.png`:

| Suffix                    | What it is                                |
| ------------------------- | ----------------------------------------- |
| `__00_source.png`         | grayscale source image                    |
| `__01_proj3_5x5_273.png`  | the Project 3 fixed kernel                |
| `__sigma1.0.png`          | sigma=1 (light smoothing)                 |
| `__sigma2.0.png`          | sigma=2 (medium smoothing)                |
| `__sigma4.0.png`          | sigma=4 (heavy smoothing, often too much) |

## What we noticed

Sigma=1 keeps lane markings and signs sharp. Sigma=2 cleans up noise on
snow and rain images, but starts blurring text. Sigma=4 is too strong
for Canny later. The Project 3 kernel acts like a sigma somewhere
between 1 and 1.5, which is why it pairs well with the rest of the
Project 3 Canny stages.
