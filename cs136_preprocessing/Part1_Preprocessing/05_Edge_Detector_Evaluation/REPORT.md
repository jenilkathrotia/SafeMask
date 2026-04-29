# Part 1.5 — Comparing Edge Detectors

We test four edge detectors on the SafeMask training images.

| Detector              | What it is                                          |
| --------------------- | --------------------------------------------------- |
| `sobel_p75`           | Project 3 Sobel, threshold = top 25% of gradients   |
| `sobel_p90`           | Project 3 Sobel, threshold = top 10% of gradients   |
| `project3_canny`      | Our Python port of Project 3 Canny (the reference)  |
| `cv2_canny_75_200`    | OpenCV Canny with thresholds 75 and 200             |

## How we judge them

Lecture cv08 says a good edge detector should:

1. Find the real edges (not miss them).
2. Put each edge in the right place.
3. Give one response per edge (not double or triple lines).

We do not have hand-drawn ground truth for ACDC images. So like cv08
shows in its demos, we use a strong Canny as a stand-in. For each
detector we record:

- **Edge density**, the percent of pixels marked as edges. Too low
  means we missed edges. Too high means we are letting noise through.
- **Runtime**, in milliseconds per image.
- **IoU vs. our Project 3 Canny**, which tells us how much each method
  agrees with the reference.

Full numbers are in `Evaluation_Results/per_image_metrics.csv` and
`Evaluation_Results/aggregate_metrics.csv`.

## Results on 80 ACDC training images (20 per weather)

| Detector              | Edge density | IoU vs. P3 Canny | Runtime (ms) |
| --------------------- | -----------: | ---------------: | -----------: |
| `sobel_p75`           |       27.7 % |             0.18 |        50.8  |
| `sobel_p90`           |       10.6 % |             0.27 |        50.8  |
| `project3_canny`      |        5.1 % |       1.00 (ref) |       254.9  |
| `cv2_canny_75_200`    |        3.2 % |             0.25 |          2.1 |

## What the plots show

`edge_density_by_detector.png`. Sobel at P75 marks about 28% of pixels
as edges, which is too many. You can see the noise in the output, mostly
in fog and snow images. Sobel at P90 drops to 11% but cuts long edges
into pieces. Both Cannys land at 3 to 5%, which is what cv08 says a
clean edge map should look like.

`iou_by_detector.png`. The two Cannys overlap each other at IoU around
0.25. That sounds low, but the reason is just that cv2's thresholds
(75, 200) are stricter than the Project 3 port. So cv2 is producing a
smaller set of edges, almost all of which the Project 3 version also
finds. The IoU for both Sobels is also low, but for a different reason:
their edges are thick (multiple pixels wide) so they cover much more
area than the thin Canny edges.

`runtime_by_detector.png`. cv2.Canny is about 120 times faster than our
Python port (2 ms vs 255 ms per image). For running on all 4006 ACDC
images, we should use cv2.

## What we picked

We are going with Canny for SafeMask. We use the Project 3 port to
check our work matches the lecture, and cv2.Canny for the full 4006
image batch because it is so much faster. We are not using thresholded
Sobel as our final edge layer. Sobel still helps as a rough gradient
map, but its edges are too thick and too noisy to be the final answer.
