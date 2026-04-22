import streamlit as st
import os
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

# Config
NUM_CLASSES = 23
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "."
MASK_DIR = os.path.join(DATA_DIR, "CameraMask")

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()
    return model

# Define a fixed color map for 23 classes
np.random.seed(42)
COLOR_MAP = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color_mask[mask == c] = COLOR_MAP[c]
    return color_mask

st.set_page_config(page_title="CityScape Segmentation", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Page 1: Training Metrics", "Page 2: Inference"])

if page == "Page 1: Training Metrics":
    st.title("Training Metrics")
    
    if os.path.exists("metrics.txt"):
        with open("metrics.txt", "r") as f:
            lines = f.readlines()
            st.metric("Test mIOU", lines[0].strip())
            st.metric("Test mDice", lines[1].strip())
    else:
        st.warning("metrics.txt not found. Please train the model first.")
        
    st.subheader("Training Plots")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("loss_curve.png"):
            st.image("loss_curve.png", caption="Training Loss", use_column_width=True)
    with col2:
        if os.path.exists("metrics_curve.png"):
            st.image("metrics_curve.png", caption="mIOU & mDice", use_column_width=True)

elif page == "Page 2: Inference":
    st.title("Segmentation Inference")
    st.write("Upload 4 input images from the test set.")
    
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) != 4:
            st.warning(f"Please upload exactly 4 images. You uploaded {len(uploaded_files)}.")
            
        model = load_model()
        
        for file in uploaded_files:
            st.write(f"### Image: {file.name}")
            
            # Read image
            image = Image.open(file).convert("RGB")
            img_np = np.array(image)
            
            # Predict
            img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(DEVICE)
            
            with torch.no_grad():
                pred = model(img_tensor)
                pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
                
            pred_color = colorize_mask(pred_mask)
            
            # Load GT mask
            gt_mask_path = os.path.join(MASK_DIR, file.name)
            if os.path.exists(gt_mask_path):
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[:, :, 2]
                gt_mask = np.clip(gt_mask, 0, NUM_CLASSES - 1)
                gt_color = colorize_mask(gt_mask)
            else:
                gt_color = np.zeros_like(pred_color)
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image, caption="Original RGB", use_column_width=True)
            with col2:
                if os.path.exists(gt_mask_path):
                    st.image(gt_color, caption="Ground Truth Mask", use_column_width=True)
                else:
                    st.write("GT Mask not found.")
            with col3:
                st.image(pred_color, caption="Predicted Mask", use_column_width=True)
            
            st.markdown("---")
