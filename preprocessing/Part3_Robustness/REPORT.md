# Part 3: Robustness

Real driving cameras pick up noise, motion blur, and bad contrast. This
part checks how badly each preprocessing method breaks when those things
happen. We compare three edge detectors on clean ACDC images and on
three distorted versions of the same images.

The three methods we test:

- **Sobel at P75**, the Project 3 kernels with the top 25% threshold
- **Project 3 Canny**, no pre-blur
- **Project 3 Canny with a sigma=1 Gaussian pre-blur** (cv08 said this should help)

## The three distortions

| Distortion       | How we make it                                            |
| ---------------- | --------------------------------------------------------- |
| Gaussian noise   | Add random `N(0, sigma^2)` to every pixel, sigma = 25     |
| Motion blur      | Convolve with a 15x15 horizontal averaging kernel         |
| Low contrast     | Multiply pixels by 0.4 and add 40 (squash the range)      |

## How we score

We use IoU between the clean output and the distorted output. IoU = 1
means the distortion did not change the output at all. IoU = 0 means
the distortion completely broke it.

Full numbers are in
`02_Pipeline_Comparison/Results/aggregate_metrics.csv` and the heatmap
is `02_Pipeline_Comparison/Results/iou_heatmap.png`.

## Results on 80 ACDC training images

| Method                      | Noise (sigma=25) | Motion blur (15 px) | Low contrast |
| --------------------------- | ---------------: | ------------------: | -----------: |
| `sobel_p75`                 |             0.31 |                0.50 |         0.86 |
| `canny_proj3`               |             0.18 |                0.20 |         0.91 |
| `canny_proj3_preblur`       |             0.34 |                0.21 |         0.86 |



## What we recommend

For SafeMask preprocessing in fog, night, rain, and snow:

1. Always add a sigma=1 Gaussian before Canny. It is cheap and it
   doubles the noise robustness.
2. Do not use thresholded Sobel as the final edge map. Its
   "robustness" under noise is just because its output is already
   loose.
3. Run CLAHE first when the main problem is low contrast (we cover
   that in Part 2, Creative Idea 1).

The diagnostic strips in
`02_Pipeline_Comparison/Results/diagnostic_strips/` show clean Canny,
distorted Canny, and distorted Canny with pre-blur side by side. You
can see point 1 above with your own eyes there.
