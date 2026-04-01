import os
import sys
import yaml
import torch
import cv2
import numpy as np
import streamlit as st
import albumentations as A
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.segmentation_model import create_model
from src.uncertainty.entropy import compute_entropy, get_warning_regions

# --- Styling & Setup ---
st.set_page_config(page_title="SafeMask Demo", layout="wide", page_icon="🚗")
st.title("SafeMask: Uncertainty-Aware Road Segmentation")
st.markdown("""
This application demonstrates **SafeMask**, a deep learning system for road scene semantic segmentation 
that also estimates model uncertainty. The system identifies regions where predictions are unreliable 
(e.g., due to fog, rain, blur, or night driving).
""")

@st.cache_resource
def load_safemask_model(config_path="configs/config.yaml", weights_path="outputs/checkpoints/best_model.pth"):
    if not os.path.exists(config_path):
        return None, None, None
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = create_model(config)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    model.to(device)
    model.eval()
    
    return model, config, device

model, config, device = load_safemask_model()

if model is None:
    st.error("Configuration file not found. Ensure you run this from the project root.")
    st.stop()

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload a Driving Scene Image", type=["jpg", "jpeg", "png"])

with st.sidebar:
    st.header("Settings")
    entropy_threshold = st.slider("Uncertainty Threshold", 0.0, 1.0, 0.5, 0.05,
                                  help="Pixels with normalized entropy above this value are flagged as uncertain.")

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = image.shape[:2]
    
    # Preprocess
    img_size = config['dataset']['image_size']
    transform = A.Compose([A.Resize(height=img_size[0], width=img_size[1])])
    
    augmented = transform(image=image)
    img_tensor = augmented['image']
    img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Segmentation
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
            # Uncertainty
            entropy_tensor = compute_entropy(outputs)
            entropy = entropy_tensor.squeeze(0).cpu().numpy()
            
            warning_tensor = get_warning_regions(entropy_tensor, entropy_threshold)
            warning_mask = warning_tensor.squeeze(0).cpu().numpy()
            
    # Resize back to original
    preds_resized = cv2.resize(preds.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    entropy_resized = cv2.resize(entropy, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    warning_resized = cv2.resize(warning_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Create colormapped mask
    cm = plt.get_cmap('tab20')
    mask_colored = cm(preds_resized / config['dataset']['num_classes'])[:, :, :3]
    mask_colored = (mask_colored * 255).astype(np.uint8)
    
    # Create entropy heatmap
    cm_jet = plt.get_cmap('jet')
    entropy_colored = cm_jet(entropy_resized)[:, :, :3]
    entropy_colored = (entropy_colored * 255).astype(np.uint8)
    
    # Create warning overlay
    overlay = image.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255  # Red channel
    
    alpha = 0.5
    overlay[warning_resized == 1] = cv2.addWeighted(
        overlay[warning_resized == 1], 1 - alpha,
        red_mask[warning_resized == 1], alpha, 0
    )
    
    # --- UI Grid layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        st.subheader("Uncertainty Heatmap")
        st.image(entropy_colored, use_container_width=True, caption="Jet colormap: Red = High Uncertainty")
        
    with col2:
        st.subheader("Segmentation Mask")
        st.image(mask_colored, use_container_width=True)
        
        st.subheader(f"Warning Regions (Threshold: {entropy_threshold})")
        st.image(overlay, use_container_width=True, caption="Highlighted in red")
        
    # --- Uncertainty Statistics ---
    st.subheader("Analysis Summary")
    total_pixels = orig_w * orig_h
    uncertain_pixels = np.sum(warning_resized)
    percent_uncertain = (uncertain_pixels / total_pixels) * 100
    
    st.metric(label="% of Over-threshold Uncertain Pixels", value=f"{percent_uncertain:.2f}%")
