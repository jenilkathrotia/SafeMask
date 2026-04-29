# Part 2, Creative Idea 3: Mean-Shift in Lab Color Space

This is a different way to do segmentation than Part 1.6. Part 1.6 used
Gabor + K-Means, which needs you to pick the number of clusters up
front. Mean-Shift does not need that. It just looks for clusters that
naturally exist in color space.

## What we do

1. Convert BGR to CIE Lab. Lab is a color space where two colors that
   look different to the eye are actually far apart in the numbers.
2. Run `cv2.pyrMeanShiftFiltering` with `sp=15`, `sr=25`,
   `maxLevel=2`. Each pixel walks toward the nearest local color mode.
3. Convert back to BGR.
4. Run K-Means with k=8 just for visualization, so the regions show
   up in clearly different colors.

Each output PNG is three images side by side: original, mean-shift
filtered, then quantized to 8 colors.

## Flags

| Flag                | Default | What it does                                  |
| ------------------- | ------- | --------------------------------------------- |
| `--spatial-radius`  | 15      | bigger = more pixels get joined into a region |
| `--color-radius`    | 25      | bigger = more color variation per region      |
| `--max-pyr-level`   | 2       | how many image-pyramid levels (2 is fast)     |
