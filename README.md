# SafeMask: Road Segmentation That Knows When It Is Unsure

SafeMask is a self-driving road-segmentation project. The model labels
every pixel in a driving image (road, car, sky, etc.) and at the same
time tells you how confident it is about each label. The goal is to
flag the parts of the image where the model is probably wrong, so a
self-driving system can be careful around them. This is mostly useful
in fog, rain, snow, and night, where regular models tend to be
confidently wrong.

## What it does

- **Segmentation**: We use DeepLabV3+ from `segmentation_models_pytorch`
  with a ResNet backbone. This is a standard semantic segmentation
  model.
- **Uncertainty**: For every pixel we compute Shannon entropy on the
  model's output probabilities. Low entropy means the model is sure,
  high entropy means it is guessing. We normalize this to a 0..1 score.
- **Warning overlay**: We draw the high-uncertainty pixels on top of
  the predicted image as red warning regions. The threshold is
  adjustable.
- **Evaluation**: We measure mean IoU and pixel accuracy on clear vs.
  adverse weather, and we check whether the uncertainty score actually
  matches where the model gets things wrong.
- **Live demo**: A Streamlit app lets you upload an image and see the
  segmentation and uncertainty overlays in real time, with a slider
  for the threshold.

## Folders

```
SafeMask/
├── app/                  # Streamlit demo app
├── configs/              # YAML config files
├── cs136_preprocessing/  # CS 136 term project (see its own README)
├── outputs/              # Trained weights and saved visualizations
├── scripts/              # CLI scripts: train, evaluate, infer
├── src/                  # Main code
│   ├── datasets/         # PyTorch data loaders
│   ├── models/           # Model definitions
│   ├── training/         # Training loop
│   ├── uncertainty/      # Entropy and warning regions
│   ├── evaluation/       # Validation metrics
│   └── visualization/    # Plotting helpers
├── tests/                # Tests
└── README.md
```

## Setup

```bash
git clone https://github.com/jenilkathrotia/SafeMask.git
cd SafeMask
pip install -r requirements.txt    # Python 3.8 or newer
```

## How it works

A normal segmentation model just outputs a label for each pixel. The
problem is that in bad weather the model often gives a confident wrong
answer, which is dangerous for self-driving.

SafeMask adds an uncertainty layer on top. The model still outputs
probabilities for every class. We compute the Shannon entropy of those
probabilities, `H = -sum(p * log(p))`. If the model is sure (one class
has probability close to 1), entropy is near 0. If the model is
unsure (probabilities spread across classes), entropy is high. We
normalize this to a 0..1 score, then mark anything above a threshold
as a "warning region". A self-driving system can then choose to slow
down or ask for human help in those areas.

## How to use it

### Train

Edit `configs/config.yaml` to point at your ACDC or Cityscapes images,
then run:

```bash
python scripts/train.py --config configs/config.yaml
```

To smoke-test the pipeline with no real data:

```bash
python scripts/train.py --dummy
```

### Inference on one image

```bash
python scripts/infer.py --image path/to/image.jpg --output outputs/visualizations/res.png
```

### Evaluate by weather

```bash
python scripts/evaluate.py --condition fog
```

### Live demo

```bash
streamlit run app/app.py
```

## CS 136 term project

The folder `cs136_preprocessing/` is our CS 136 term project. It is a
separate Python pipeline that runs classic image-processing methods
(Gaussian, Sobel, Canny, Hough, texture segmentation, etc.) on the
SafeMask training images and writes a report on which methods help
under noise, blur, and low contrast. It does not change anything about
the SafeMask model. See `cs136_preprocessing/README.md` for details.

## What does not work yet, and what is next

- **Calibration**: Shannon entropy is a good baseline, but if the
  model is poorly calibrated it can still be confidently wrong on
  out-of-distribution images. We are aware of this.
- **Next**: We want to try Monte Carlo Dropout (running the model
  several times with dropout enabled and averaging) for a stronger
  uncertainty signal. We also want to check whether uncertainty stays
  consistent across short video clips, since a single noisy frame
  should not flip the prediction.
