# SafeMask: Uncertainty-Aware Road Segmentation for Low-Visibility Driving

SafeMask is an end-to-end Python project designed to perform robust road-scene semantic segmentation while simultaneously estimating pixel-level uncertainty. This helps flag unreliable predictions in adverse driving conditions such as fog, rain, snow, and night time.

## 🚀 Features
- **Semantic Segmentation Baseline:** Employs DeepLabV3+ (via `segmentation_models_pytorch`) with a ResNet-50 backbone.
- **Uncertainty Estimation:** Calculates pixel-wise Shannon entropy from model logits to provide normalized confidence scores.
- **Warning Overlay:** Interactively visualizes high-uncertainty regions with tunable thresholds.
- **Evaluation Pipeline:** Measures performance across clear vs. adverse conditions (mIoU, Pixel Accuracy, Error-Uncertainty correlation).
- **Interactive App:** Streamlit dashboard for real-time visualization and threshold tuning.

## 📁 Directory Structure
```
SafeMask/
├── app/             # Streamlit application
├── configs/         # YAML configurations
├── outputs/         # Saved model weights and visualizations
├── scripts/         # CLI execution scripts (train, evaluate, infer)
├── src/             # Core library
│   ├── datasets/    # PyTorch dataset loaders
│   ├── models/      # Model architectures
│   ├── training/    # PyTorch trainer loop
│   ├── uncertainty/ # Softmax entropy and warning region tools
│   ├── evaluation/  # Validation metrics (IoU, accuracy)
│   └── visualization/
├── tests/           # Automated workflow testing
└── README.md
```

## 🛠️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/jenilkathrotia/SafeMask.git
cd SafeMask

# Install dependencies (requires Python 3.8+)
pip install -r requirements.txt
```

## 🧠 Methodology
Standard semantic segmentation maps inputs directly to labels, but deep neural networks are prone to overconfident misclassifications, especially in out-of-distribution (OoD) scenarios like dense fog or heavy rain.
SafeMask utilizes **Softmax Entropy** to quantify predictive uncertainty. By passing the output probability distribution `p` through Shannon's entropy `H = -sum(p * log(p))` and normalizing it, SafeMask creates a heatmap identifying regions the model is unsure of. These are thresholded to produce explicit **Warning Regions**.

## 🚀 Usage

### 1. Training
Update `configs/config.yaml` with your dataset paths (ACDC/Cityscapes format) and run:
```bash
python scripts/train.py --config configs/config.yaml
```
*To test the pipeline without a dataset, use:*
```bash
python scripts/train.py --dummy
```

### 2. Inference
Run inference on a single image and save the side-by-side visualization:
```bash
python scripts/infer.py --image path/to/image.jpg --output outputs/visualizations/res.png
```

### 3. Evaluation
Evaluate metrics across normal and adverse conditions:
```bash
python scripts/evaluate.py --condition fog
```

### 4. Interactive Demo (Streamlit)
To visualize the uncertainty overlays in real time:
```bash
streamlit run app/app.py
```

## ⚠️ Limitations & Future Work
- **Current Limitation:** Softmax entropy is a strong baseline but can suffer from poor model calibration (i.e., overconfident predictions on heavily corrupted data).
- **Future Scope:** Integrating multi-pass **Monte Carlo (MC) Dropout** for more robust epistemic uncertainty estimation, and investigating temporal consistency over short video clips.
