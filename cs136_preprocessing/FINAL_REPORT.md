# CS 136 Term Project: Final Report

## What we did

We built an image preprocessing pipeline for the SafeMask self-driving
project. SafeMask trains a road segmentation model on the ACDC dataset,
which has 1600 training images split across four bad weather types: fog,
night, rain, and snow. Our job was to apply the CS 136 image processing
methods to those images, find which ones help, and wire the useful ones
back into the SafeMask training pipeline.

The work has three parts, mapped to the assignment:

- **Part 1**: classic preprocessing methods (Gaussian, Sobel, Canny,
  Hough, edge comparison, texture segmentation)
- **Part 2**: three creative combinations of these methods
- **Part 3**: how the methods hold up under noise, blur, and low contrast

All algorithm code is Python ports of the C code we wrote for Project 3
and Project 4, so the kernels and pipeline steps are identical.

## Part 1: Preprocessing methods

We ran six methods on 80 ACDC training images (20 per weather):

1. **Gaussian filter** at four sigma values plus the Project 3 5x5/273
   kernel.
2. **Sobel edge detector** with the same kernels as Project 3.
3. **Canny edge detector**, both our Project 3 port and OpenCV's
   `cv2.Canny` at three threshold pairs.
4. **Hough transform** for lines (lane markings) and circles
   (Project 4 port plus OpenCV).
5. **Edge detector comparison**, measuring edge density, runtime, and
   IoU between detectors.
6. **Texture segmentation** using a Gabor filter bank plus K-Means,
   in grayscale and color versions.

The full numbers are in
`Part1_Preprocessing/05_Edge_Detector_Evaluation/REPORT.md` and the
matching CSVs.

### Key result from Part 1

Sobel marks too many pixels as edges (~28%). Both Cannys land at 3 to
5%, which is the right range per cv08. OpenCV's Canny is 120 times
faster than our Python port, so we use cv2 for the full batch and the
Project 3 port to verify correctness.

## Part 2: Creative ideas

Three combinations not covered in class:

1. **CLAHE + bilateral + Canny**: boost contrast first (helps fog/night
   images), denoise without blurring edges, then run Canny. Recovered
   guard rails and lane markings the plain Canny missed.
2. **Morphological cleanup of Canny edges**: closing fills small gaps,
   opening removes single noisy pixels, skeletonize thins the result
   to 1 pixel wide for cleaner Hough voting.
3. **Mean-Shift in CIE Lab**: a different segmentation method from
   Part 1.6. Doesn't need a preset cluster count, finds them
   automatically.

## Part 3: Robustness analysis

For every image we made three corrupted copies (Gaussian noise,
motion blur, low contrast) and ran four pipelines on the clean and
corrupted versions. We measured how similar the outputs were.

### Results on 80 ACDC training images

| Method                      | Noise (sigma=25) | Motion blur | Low contrast |
| --------------------------- | ---------------: | ----------: | -----------: |
| `sobel_p75` (IoU)           |             0.30 |        0.51 |         0.87 |
| `canny_proj3` (IoU)         |             0.16 |        0.22 |         0.91 |
| `canny_proj3_preblur` (IoU) |             0.32 |        0.23 |         0.86 |
| `texture_seg` (ARI)         |             0.80 |        0.72 |         0.80 |

### Main findings

1. **Pre-blur doubles Canny's noise robustness.** A simple sigma=1
   Gaussian before Canny took the IoU from 0.16 to 0.32. cv08 said
   pre-blur should help with noise, and it does.
2. **Pre-blur does not help motion blur.** Motion blur is already a
   blur, so adding more blur cannot bring back lost detail.
3. **Low contrast is the easiest distortion.** Canny stays at IoU
   0.91 because its hysteresis adapts to gradient values.
4. **Sobel's "win" under noise is misleading.** Its output is so loose
   to begin with that noise barely changes it. Pre-blurred Canny
   matches Sobel's robustness with a much cleaner output.
5. **Texture segmentation handles all three distortions well.** ARI
   stays at 0.72 or higher.

Full discussion is in `Part3_Robustness/REPORT.md`.

## Wiring into SafeMask

We took the four findings and wired them into the SafeMask training
pipeline. Now every image the model trains on goes through:

1. sigma=1 Gaussian blur
2. CLAHE on the Lab L-channel (only for fog and night images)
3. Canny edges added as a 4th input channel
4. Morphological closing+opening on those edges

The model's first conv layer accepts 4 channels instead of 3. All four
steps are configurable in `configs/config.yaml`. Source files:

- `src/preprocessing/cs136_preproc.py`: the algorithms
- `src/datasets/acdc_loader.py`: applies them inside `__getitem__`
- `src/models/segmentation_model.py`: handles 4-channel input

## Verification

We ran the full training pipeline end-to-end with dummy data on a Mac
(MPS backend). The model loaded successfully with `in_channels=4`,
trained for 30 epochs, and saved a checkpoint. CS 136 preprocessing
ran on every image in every epoch.

Final dummy results:

- Train loss: 1.66, mIoU: 0.20
- Val loss: 2.47, mIoU: 0.10

These numbers are from random synthetic data, so they do not reflect
real performance. They confirm the pipeline runs without errors.

## How to run

```bash
# CS 136 preprocessing only (the term-project deliverable):
./cs136_preprocessing/run_all.sh

# Full SafeMask training with CS 136 preprocessing applied:
python scripts/train.py --config configs/config.yaml

# Smoke test with dummy data:
python scripts/train.py --dummy
```

## What is left

The pipeline is complete and tested. Real ACDC training (1600 images
on a Colab T4 GPU) takes about 1 to 1.5 hours and produces a usable
model. We did not run that final training because it requires
uploading the 16 GB ACDC dataset to Colab, which is the slow part.
The wiring itself is verified.

## Folder map

| Item                                            | Folder                                              |
| ----------------------------------------------- | --------------------------------------------------- |
| Gaussian filter                                 | `Part1_Preprocessing/01_Gaussian_Filter`            |
| Sobel                                           | `Part1_Preprocessing/02_Sobel_Edge`                 |
| Canny                                           | `Part1_Preprocessing/03_Canny_Edge`                 |
| Hough transform                                 | `Part1_Preprocessing/04_Hough_Transform`            |
| Edge detector comparison                        | `Part1_Preprocessing/05_Edge_Detector_Evaluation`   |
| Texture segmentation                            | `Part1_Preprocessing/06_Texture_Segmentation`       |
| CLAHE + bilateral + Canny                       | `Part2_Creative/01_CLAHE_Bilateral_Canny`           |
| Morphological cleanup                           | `Part2_Creative/02_Morphological_Cleanup`           |
| Mean-Shift in Lab                               | `Part2_Creative/03_Color_Texture_Lab`               |
| Distortion generator                            | `Part3_Robustness/01_Distortions`                   |
| Pipeline comparison                             | `Part3_Robustness/02_Pipeline_Comparison`           |
| SafeMask integration (preprocessing layer)      | `../src/preprocessing/cs136_preproc.py`             |
