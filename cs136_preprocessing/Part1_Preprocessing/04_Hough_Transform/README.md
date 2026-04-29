# Part 1.4: Hough Transform

Lines and circles, both shown as overlays on the original image.

## What we save per image

| Suffix                              | What it shows                                                 |
| ----------------------------------- | ------------------------------------------------------------- |
| `__00_edges.png`                    | Project 3 Canny output (the input to Hough)                   |
| `__01_lines_overlay.png`            | `cv2.HoughLinesP` lines drawn in green (good for lanes)       |
| `__02_circles_cv2.png`              | `cv2.HoughCircles` circles drawn in red                       |
| `__03_circles_project4_port.png`    | Our Python version of the Project 4 circle Hough              |

The Project 4 port is the same algorithm we wrote in C: build a sin/cos
lookup table, have each edge pixel vote for circles it could belong to,
find local maxima in 3D. Voting is slow in Python on a full-resolution
image, so we shrink the edge map to about 200 pixels on its long side
first, then scale the detected centers back up before drawing.

## Flags

`--min-radius` and `--max-radius` set the radius range to search.
`--n-circles` sets how many circles to draw. The defaults (10 to 40
pixels, up to 8 circles) work for traffic lights and wheels in
1080x1920 ACDC images.
