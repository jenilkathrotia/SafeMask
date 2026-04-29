# Part 3: Robustness

Real driving cameras pick up noise, motion blur, and bad contrast. This
part checks how badly each preprocessing method breaks when those things
happen. We compare four methods on clean ACDC images and on three
distorted versions of the same images.

The four methods we test:

- **Sobel at P75**, the Project 3 kernels with the top 25% threshold
- **Project 3 Canny**, no pre-blur
- **Project 3 Canny with a sigma=1 Gaussian pre-blur** (cv08 said this should help)
- **Texture segmentation** (Gabor + K-Means k=4)

## The three distortions

| Distortion       | How we make it                                            |
| ---------------- | --------------------------------------------------------- |
| Gaussian noise   | Add random `N(0, sigma^2)` to every pixel, sigma = 25     |
| Motion blur      | Convolve with a 15x15 horizontal averaging kernel         |
| Low contrast     | Multiply pixels by 0.4 and add 40 (squash the range)      |

## How we score

For the edge detectors we use IoU between the clean output and the
distorted output. IoU = 1 means the distortion did not change the
output at all. IoU = 0 means the distortion completely broke it.

For the texture segmentation we use Adjusted Rand Index (ARI) instead.
That is because cluster IDs are arbitrary (cluster 1 today might be
cluster 3 tomorrow even with the same content). ARI ignores label
swaps so it is the right metric for clustering.

Full numbers are in
`02_Pipeline_Comparison/Results/aggregate_metrics.csv` and the heatmap
is `02_Pipeline_Comparison/Results/iou_heatmap.png`.

## Results on 80 ACDC training images

| Method                      | Noise (sigma=25) | Motion blur (15 px) | Low contrast |
| --------------------------- | ---------------: | ------------------: | -----------: |
| `sobel_p75` (IoU)           |             0.30 |                0.51 |         0.87 |
| `canny_proj3` (IoU)         |             0.16 |                0.22 |         0.91 |
| `canny_proj3_preblur` (IoU) |             0.32 |                0.23 |         0.86 |
| `texture_seg` (ARI)         |             0.80 |                0.72 |         0.80 |

## What we learned

**1. The pre-blur doubles Canny's noise robustness.**
Plain Canny under noise gets IoU 0.16. Adding a sigma=1 Gaussian
before Canny brings it up to 0.32. That is almost exactly 2 times
better. cv08 said pre-blur would help with noise, and it really does.

**2. Pre-blur does not help with motion blur.**
The numbers barely change (0.22 vs 0.23). That is fine, because
motion blur is already a kind of blur, and adding more blur cannot
bring back details that the motion blur removed. So pre-blur is a
fix for noise specifically, not a fix for everything.

**3. Low contrast is the easiest distortion.**
Canny still gets IoU 0.91. The reason is that its hysteresis
thresholds are based on the actual gradient values in the image. So
when the contrast goes down, the thresholds go down too, and the
output stays mostly the same. Sobel at P75 also does fine (0.87)
because using a percentile threshold is contrast-invariant.

**4. Sobel's win against Canny under noise is misleading.**
Sobel-p75 gets 0.30 vs Canny's 0.16, which sounds like Sobel is more
robust. But Sobel-p75 already marks 28% of the image as edges. With
that many edges, noise barely changes the result. Sobel looks robust
only because its output was already very loose. The pre-blurred Canny
matches Sobel's robustness (0.32 vs 0.30) and still keeps a clean 3
to 5% edge density. So pre-blurred Canny is the actual winner here.

**5. Texture segmentation handles all three distortions well.**
ARI stays at 0.72 or higher in every case. The Gabor responses get
smoothed before clustering, and K-Means averages out small changes.
So the segmentation result barely moves.

## What we recommend

For SafeMask preprocessing in fog, night, rain, and snow:

1. Always add a sigma=1 Gaussian before Canny. It is cheap and it
   doubles the noise robustness.
2. Do not use thresholded Sobel as the final edge map. Its
   "robustness" under noise is just because its output is already
   loose.
3. Motion blur is the hardest distortion. If a pipeline depends on
   working under motion blur, use segmentation, not edges.
4. Run CLAHE first when the main problem is low contrast (we cover
   that in Part 2, Creative Idea 1).

The diagnostic strips in
`02_Pipeline_Comparison/Results/diagnostic_strips/` show clean Canny,
distorted Canny, and distorted Canny with pre-blur side by side. You
can see point 1 above with your own eyes there.
