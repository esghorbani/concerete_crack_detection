import base64
import os
import sys
from io import BytesIO

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import streamlit as st
import torch

# Add project root to Python path so local imports work reliably
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Avenum Solutions | Damage Detection",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------
# Important paths used by the app
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "best_model.pth")
LOGO_PATH = os.path.join(BASE_DIR, "app", "logo.png")


# ---------------------------------------------------------
# Custom CSS styling for the page
# ---------------------------------------------------------
def apply_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 1.5rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 1400px;
        }

        .hero-row {
            display: flex;
            align-items: center;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }

        .hero-row img {
            max-width: 240px;
            height: auto;
        }

        .top-banner {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0 0 1.2rem 0;
            margin-bottom: 0;
            box-shadow: none;
        }

        .top-title {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.25rem;
        }

        .top-subtitle {
            font-size: 1rem;
            color: #4b5563;
            line-height: 1.5;
        }

        .section-wrapper {
            background: transparent;
            border-radius: 22px;
            padding-bottom: 1rem;
        }

        .section-header {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            min-height: 48px;
            display: flex;
            align-items: center;
            padding: 0 18px;
            font-size: 1rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.85rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.04);
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.5rem;
        }

        .subtle-text {
            color: #6b7280;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .result-box {
            background: transparent;
            border-radius: 22px;
            padding: 0 0 1rem 0;
        }

        .result-header {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            min-height: 48px;
            display: flex;
            align-items: center;
            padding: 0 18px;
            font-size: 1rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.85rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }

        .status-pill-detected {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: #fff7ed;
            color: #c2410c;
            border: 1px solid #fdba74;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .status-pill-clear {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #86efac;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .footer-note {
            color: #6b7280;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_logo_img_tag():
    """Return an inline base64 <img> tag for the logo if available."""
    if not os.path.exists(LOGO_PATH):
        return ""

    with open(LOGO_PATH, "rb") as logo_file:
        encoded = base64.b64encode(logo_file.read()).decode("utf-8")

    return f'<img src="data:image/png;base64,{encoded}" alt="Avenum Solutions logo" />'


# ---------------------------------------------------------
# Load trained model once and cache it
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


# ---------------------------------------------------------
# Image preprocessing used before inference
# Must match the baseline training pipeline
# ---------------------------------------------------------
def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------
# Run prediction on one uploaded image
# ---------------------------------------------------------
def predict(model, device, image_pil, threshold=0.3):
    image_rgb = np.array(image_pil.convert("RGB"))
    original_h, original_w = image_rgb.shape[:2]

    transform = get_transform()
    transformed = transform(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (probs > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    overlay = image_rgb.copy()
    overlay[mask > 0] = [255, 153, 51]
    blended = cv2.addWeighted(image_rgb, 0.75, overlay, 0.25, 0)

    crack_pixels = int(np.sum(mask > 0))
    total_pixels = mask.shape[0] * mask.shape[1]
    crack_ratio = (crack_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    crack_detected = crack_pixels > 0

    return image_rgb, mask, blended, crack_pixels, crack_ratio, crack_detected


# ---------------------------------------------------------
# Convert a PIL image to PNG bytes for downloading
# ---------------------------------------------------------
def pil_to_png_bytes(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------
# Main app UI
# ---------------------------------------------------------
def main():
    apply_custom_style()

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at: {MODEL_PATH}. Train the model first and make sure best_model.pth exists."
        )
        st.stop()

    model, device = load_model()

    hero_html = f"""
        <div class="hero-row">
            {get_logo_img_tag()}
            <div class="top-banner">
                <div class="top-title">Avenum Solutions | Concrete Damage Detection</div>
                <div class="top-subtitle">
                    Upload a concrete image to run crack segmentation and review the detected damage region,
                    predicted mask, and overlay result.
                </div>
            </div>
        </div>
    """

    st.markdown(hero_html, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 1.4])

    with left_col:
        st.markdown(
            """
            <div class="section-wrapper">
                <div class="section-header">Upload & Settings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )

        threshold = st.slider(
            "Detection threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
        )

        st.markdown(
            f"""
            <div class="subtle-text">
                <strong>Runtime device:</strong> {device}<br>
                <strong>Model:</strong> U-Net / ResNet34<br>
                <strong>Weights:</strong> src/models/best_model.pth<br>
                <strong>Input size:</strong> 256 x 256
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="card-title">How to use</div>
                <div class="subtle-text">
                    1. Upload a concrete surface image.<br>
                    2. Adjust the detection threshold if needed.<br>
                    3. Review the predicted mask and overlay.<br>
                    4. Download the outputs if needed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div class="section-wrapper">
                <div class="section-header">Detection Workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_file is None:
            st.info("Upload an image to run detection.")
            return

        image_pil = Image.open(uploaded_file)

        with st.spinner("Running damage detection..."):
            image_rgb, mask, blended, crack_pixels, crack_ratio, crack_detected = predict(
                model=model,
                device=device,
                image_pil=image_pil,
                threshold=threshold,
            )

        status_html = (
            '<span class="status-pill-detected">Crack Detected</span>'
            if crack_detected
            else '<span class="status-pill-clear">No Crack Detected</span>'
        )
        st.markdown(status_html, unsafe_allow_html=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Threshold", f"{threshold:.2f}")
        metric_col2.metric("Crack Pixels", f"{crack_pixels:,}")
        metric_col3.metric("Estimated Area", f"{crack_ratio:.2f}%")


    st.markdown("## Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-header">Original Image</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image_rgb, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-header">Predicted Mask</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(mask, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-header">Overlay Result</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(blended, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Export Results")
    mask_pil = Image.fromarray(mask)
    overlay_pil = Image.fromarray(blended)

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            label="Download Mask (PNG)",
            data=pil_to_png_bytes(mask_pil),
            file_name="predicted_mask.png",
            mime="image/png",
            width="stretch",
        )

    with dl2:
        st.download_button(
            label="Download Overlay (PNG)",
            data=pil_to_png_bytes(overlay_pil),
            file_name="overlay_result.png",
            mime="image/png",
            width="stretch",
        )

    st.markdown(
        """
        <div class="footer-note">
            This interface is intended for demonstration and internal evaluation.
            Detection quality depends on training data quality, image quality, lighting, and surface condition.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
