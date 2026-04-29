# Part 2 / Creative #2 — Morphological Cleanup

Project 3 already had expand and shrink. We extend that to the standard
edge-cleanup steps.

## What we do

1. **Closing** (dilate then erode). Fills 1 to 2 pixel gaps in real
   edges that Canny sometimes thins too much.
2. **Opening** (erode then dilate). Removes single noisy pixels that
   slipped through Canny's hysteresis.
3. **Skeletonize** (Zhang-Suen, from scikit-image). Shrinks the edges
   back to 1 pixel wide, which makes the next step (Hough) cleaner.

Each output PNG shows: Canny baseline, then closed, then opened, then
skeletonized.

The skeletonize step needs scikit-image. If it is not installed, we
just save 3 outputs instead of 4.

## Why this is worth doing

Lecture cv08 says good edges should be one pixel wide and not have
extra speckle. Canny mostly does that, but on snow and rain images,
some noise still leaks through. The opening step takes those out
without hurting long real edges. That makes the Hough voting in Part
1.4 give cleaner results.
