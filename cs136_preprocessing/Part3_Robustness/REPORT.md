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

## What we learned

**1. The pre-blur nearly doubles Canny's noise robustness.**
Plain Canny under noise gets IoU 0.18. Adding a sigma=1 Gaussian
before Canny brings it up to 0.34. That is almost exactly 2 times
better. cv08 said pre-blur would help with noise, and it really does.

**2. Pre-blur does not help with motion blur.**
The numbers barely change (0.20 vs 0.21). That is fine, because
motion blur is already a kind of blur, and adding more blur cannot
bring back details that the motion blur removed. So pre-blur is a
fix for noise specifically, not a fix for everything.

**3. Low contrast is the easiest distortion.**
Canny still gets IoU 0.91. The reason is that its hysteresis
thresholds are based on the actual gradient values in the image. So
when the contrast goes down, the thresholds go down too, and the
output stays mostly the same. Sobel at P75 also does fine (0.86)
because using a percentile threshold is contrast-invariant.

**4. Sobel's win against Canny under noise is misleading.**
Sobel-p75 gets 0.31 vs Canny's 0.18, which sounds like Sobel is more
robust. But Sobel-p75 already marks 28% of the image as edges. With
that many edges, noise barely changes the result. Sobel looks robust
only because its output was already very loose. The pre-blurred Canny
matches Sobel's robustness (0.34 vs 0.31) and still keeps a clean 3
to 5% edge density. So pre-blurred Canny is the actual winner here.

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
