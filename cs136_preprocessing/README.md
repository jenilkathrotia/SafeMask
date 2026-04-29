# SafeMask Term Project — CS 136 Preprocessing

This folder is our CS 136 term project. We rewrote the C code from
Project 3 (smoothing, Sobel, Canny) and Project 4 (Hough circles) in
Python. The kernels and steps are the same, so the answers should match.

We use the SafeMask training images as our input. SafeMask is the
self-driving project our team is building, and the training images come
from the ACDC dataset (fog, night, rain, snow). We are not using the
Marine, Geology, or Anthropology samples from Canvas.

## Folders

```
cs136_preprocessing/
├── requirements.txt
├── run_all.sh                # runs everything in one go
├── utils/                    # shared helpers
├── Part1_Preprocessing/
│   ├── 01_Gaussian_Filter/
│   ├── 02_Sobel_Edge/
│   ├── 03_Canny_Edge/
│   ├── 04_Hough_Transform/
│   ├── 05_Edge_Detector_Evaluation/
│   └── 06_Texture_Segmentation/
├── Part2_Creative/
│   ├── 01_CLAHE_Bilateral_Canny/
│   ├── 02_Morphological_Cleanup/
│   └── 03_Color_Texture_Lab/
└── Part3_Robustness/
    ├── 01_Distortions/
    └── 02_Pipeline_Comparison/
```

Each folder has its own script and its own output images, like the
assignment says. You can run any one folder by itself.

## Setup

```bash
cd "/Users/jenilkathrotiya/Downloads/CS 136/TermProject/SafeMask"
python3 -m venv .venv && source .venv/bin/activate
pip install -r cs136_preprocessing/requirements.txt
```

## How to run one folder

Every script takes the same flags:

```bash
python cs136_preprocessing/Part1_Preprocessing/01_Gaussian_Filter/gaussian_filter.py \
  --input-dir /path/to/ACDC/rgb_anon \
  --split train \
  --per-condition 20
```

`--per-condition 20` picks 20 random images from each weather
(fog, night, rain, snow). That keeps the run short. Drop the flag to
use every image. `--split train` only uses the training split. ACDC
also ships clear-weather reference images named `*_rgb_ref_anon.png`.
We skip those by default. Pass `--include-refs` if you want them.

## How to run everything

```bash
./cs136_preprocessing/run_all.sh
```

About 20 minutes. Makes around 2 GB of output across all 11 steps.

## Where each part lives

| Assignment item                                | Folder                                              |
| ---------------------------------------------- | --------------------------------------------------- |
| Part 1 Gaussian filter                         | `Part1_Preprocessing/01_Gaussian_Filter`            |
| Part 1 Sobel                                   | `Part1_Preprocessing/02_Sobel_Edge`                 |
| Part 1 Canny                                   | `Part1_Preprocessing/03_Canny_Edge`                 |
| Part 1 Hough transform                         | `Part1_Preprocessing/04_Hough_Transform`            |
| Part 1 edge detector comparison                | `Part1_Preprocessing/05_Edge_Detector_Evaluation`   |
| Part 1 texture segmentation                    | `Part1_Preprocessing/06_Texture_Segmentation`       |
| Part 2 creative ideas                          | `Part2_Creative/01_*` through `03_*`                |
| Part 3 robustness                              | `Part3_Robustness/`                                 |
