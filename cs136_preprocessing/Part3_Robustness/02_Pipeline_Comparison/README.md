# Part 3 / Step 2 — Compare Clean vs. Distorted

This script runs four methods (Sobel, Canny, Canny with pre-blur,
texture segmentation) on every image and on every distorted copy of
that image. Then it compares the clean output to the distorted output
to see how much each method broke. See `../REPORT.md` for the
write-up.

## Outputs (in `Results/`)

| File                                  | What it has                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| `per_image_metrics.csv`               | one row per (image, distortion, method) with the score            |
| `aggregate_metrics.csv`               | average and standard deviation per method per distortion          |
| `iou_heatmap.png`                     | 4x3 heatmap (methods on the rows, distortions on the columns)     |
| `diagnostic_strips/*.png`             | clean Canny / distorted Canny / pre-blurred Canny side by side    |

## How to run

```bash
python compare_pipelines.py                   # uses My_Test by default
python compare_pipelines.py --per-condition 5 # 5 images per weather
python compare_pipelines.py --input-dir <ACDC RGB> --split train --per-condition 20
```
