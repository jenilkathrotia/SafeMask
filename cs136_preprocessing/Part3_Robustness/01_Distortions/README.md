# Part 3, Step 1: Make Distorted Copies

This script makes three corrupted copies of every input image. The next
step (`02_Pipeline_Comparison`) runs the algorithms on these to see how
much they break.

## The three distortions

| Subfolder       | Distortion              | Default settings           |
| --------------- | ----------------------- | -------------------------- |
| `Noisy/`        | Add Gaussian noise      | sigma = 25                 |
| `Blurred/`      | Horizontal motion blur  | kernel length = 15 pixels  |
| `LowContrast/`  | Squash brightness range | alpha = 0.4, beta = 40     |

## Flags

You can change every distortion's settings from the command line:

- `--noise-sigma` (how strong the noise is)
- `--blur-length` (how long the motion-blur kernel is)
- `--contrast-alpha` (multiplier for pixel value)
- `--contrast-beta` (constant added after multiplying)

The output goes into the three subfolders here. The next script reads
those subfolders.
