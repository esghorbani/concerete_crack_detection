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
try:
    from streamlit_image_select import image_select
except ImportError:  # pragma: no cover - optional dependency
    image_select = None

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
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test_samples")
# Number of curated samples exposed inside the modal gallery
MAX_TEST_IMAGES = None  # None keeps the gallery synced with the full folder
GALLERY_INDEX_KEY = "test_gallery_index"
RESULT_IMAGE_WIDTH = 360
GALLERY_MODAL_FLAG = "sample_gallery_modal_open"
HAS_STREAMLIT_MODAL = hasattr(st, "modal")
HAS_STREAMLIT_POPOVER = hasattr(st, "popover")


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

        [data-testid="stFooter"] {
            display: none;
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

        :root {
            --box-radius: 18px;
            --box-bg: #ffffff;
            --box-border: #e2e8f0;
            --box-padding: 18px;
            --box-header-height: 56px;
            --box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }

        .section-wrapper {
            background: transparent;
            border-radius: var(--box-radius);
            padding-bottom: 1rem;
        }

        .section-header {
            background: var(--box-bg);
            border: 1px solid var(--box-border);
            border-radius: var(--box-radius);
            min-height: var(--box-header-height);
            display: flex;
            align-items: center;
            padding: 0 var(--box-padding);
            font-size: 1rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.85rem;
            box-shadow: var(--box-shadow);
        }

        .section-card {
            background: var(--box-bg);
            border: 1px solid var(--box-border);
            border-radius: var(--box-radius);
            padding: var(--box-padding);
            box-shadow: var(--box-shadow);
            margin-bottom: 1rem;
        }

        .card-title {
            background: transparent;
            border: none;
            border-radius: 0;
            min-height: auto;
            display: block;
            padding: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.75rem;
            box-shadow: none;
        }

        .instructions-card {
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .subtle-text {
            color: #6b7280;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .sample-gallery-status {
            background: #f8fafc;
            border: 1px dashed #cbd5f5;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 0.65rem;
        }

        .gallery-modal-heading {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #0f172a;
        }

        .gallery-modal-subtext {
            font-size: 0.9rem;
            color: #475569;
            margin-bottom: 0.85rem;
        }

        .result-box {
            background: transparent;
            border-radius: var(--box-radius);
            padding: 0 0 1rem 0;
        }

        .result-header {
            background: var(--box-bg);
            border: 1px solid var(--box-border);
            border-radius: var(--box-radius);
            min-height: var(--box-header-height);
            display: flex;
            align-items: center;
            padding: 0 var(--box-padding);
            font-size: 1rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.85rem;
            box-shadow: var(--box-shadow);
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


def get_test_gallery(limit=MAX_TEST_IMAGES):
    """Return absolute paths for curated test images, optionally capped by `limit`."""
    if not os.path.isdir(TEST_IMAGE_DIR):
        return []

    gallery = []
    for filename in sorted(os.listdir(TEST_IMAGE_DIR)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            gallery.append(os.path.join(TEST_IMAGE_DIR, filename))
        if limit is not None and len(gallery) >= limit:
            break

    return gallery


def render_gallery_selector(paths):
    """Render a clickable gallery using streamlit-image-select and return the chosen path."""
    if not paths:
        return None

    captions = [os.path.basename(path) for path in paths]
    stored_index = st.session_state.get(GALLERY_INDEX_KEY)
    stored_index = (
        min(max(stored_index, 0), len(paths) - 1) if stored_index is not None else None
    )

    if image_select is None:
        placeholder_label = "\u200b"
        options = [placeholder_label] + captions
        st.warning(
            "Install the optional dependency `streamlit-image-select` to enable clickable thumbnails."
        )
        selected_label = st.radio(
            "",
            options=options,
            index=(stored_index + 1) if stored_index is not None else 0,
            horizontal=True,
            label_visibility="collapsed",
        )
        if selected_label == placeholder_label:
            selected_index = None
        else:
            selected_index = captions.index(selected_label)
    else:
        selected_index = image_select(
            label="",
            images=paths,
            captions=captions,
            use_container_width=True,
            return_value="index",
            index=stored_index if stored_index is not None else -1,
            key="gallery_image_select",
        )

    if selected_index is None or selected_index < 0:
        return None

    st.session_state[GALLERY_INDEX_KEY] = selected_index
    active_path = paths[selected_index]

    return active_path


def render_gallery_modal(paths):
    """Wrap the gallery selector inside a modal-like popover trigger."""
    if not paths:
        return None

    if GALLERY_MODAL_FLAG not in st.session_state:
        st.session_state[GALLERY_MODAL_FLAG] = False

    stored_index = st.session_state.get(GALLERY_INDEX_KEY)
    selected_label = None
    if stored_index is not None and 0 <= stored_index < len(paths):
        selected_label = os.path.basename(paths[stored_index])

    selected_path = None
    button_label = "Browse sample gallery"

    if HAS_STREAMLIT_MODAL:
        if st.button(button_label, use_container_width=True):
            st.session_state[GALLERY_MODAL_FLAG] = True

        if st.session_state.get(GALLERY_MODAL_FLAG):
            with st.modal("Sample test images", use_container_width=True):
                st.markdown(
                    """
                    <div class="gallery-modal-heading">Sample test images</div>
                    <div class="gallery-modal-subtext">Pick an example image to quickly preview the detection workflow.</div>
                    """,
                    unsafe_allow_html=True,
                )
                selected_path = render_gallery_selector(paths)

                if st.button("Close gallery", use_container_width=True):
                    st.session_state[GALLERY_MODAL_FLAG] = False
    elif HAS_STREAMLIT_POPOVER:
        with st.popover(button_label, use_container_width=True):
            st.markdown(
                """
                <div class="gallery-modal-heading">Sample test images</div>
                <div class="gallery-modal-subtext">Pick an example image to quickly preview the detection workflow.</div>
                """,
                unsafe_allow_html=True,
            )
            selected_path = render_gallery_selector(paths)
    else:
        st.warning(
            "Upgrade Streamlit to access modal or popover components. Falling back to inline picker."
        )
        selected_path = render_gallery_selector(paths)

    if selected_path is None and selected_label:
        selected_path = paths[stored_index]

    return selected_path


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
    test_gallery_paths = get_test_gallery()
    upload_option = "Upload image"
    gallery_option = "Sample test images"
    image_source = upload_option
    selected_sample_path = None

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
    instructions_html = """
        <div class="section-card instructions-card">
            <div class="card-title">How to use</div>
            <div class="subtle-text">
                1. Pick an input image (upload or sample test images).<br>
                2. Adjust the detection threshold if needed.<br>
                3. Review the prediction mask and overlay.<br>
                4. Download the outputs for reporting.
            </div>
        </div>
    """
    st.markdown(instructions_html, unsafe_allow_html=True)

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

        image_source = st.radio(
            "Image source",
            options=[gallery_option, upload_option],
            horizontal=True,
            label_visibility="collapsed",
        )

        if image_source == upload_option:
            uploaded_file = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        else:
            uploaded_file = None

        if image_source == gallery_option:
            if not test_gallery_paths:
                st.warning(
                    "No test images detected. Add PNG/JPEG images inside the test_samples/ folder."
                )
            else:
                selected_sample_path = render_gallery_modal(test_gallery_paths)

        threshold = st.slider(
            "Detection threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
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

        image_pil = None
        image_name = ""
        if image_source == upload_option and uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
            image_name = uploaded_file.name
        elif image_source == gallery_option and selected_sample_path:
            image_pil = Image.open(selected_sample_path)
            image_name = os.path.basename(selected_sample_path)

        if image_pil is None:
            if image_source == upload_option:
                st.info("Upload an image or switch to the sample test images to continue.")
            else:
                st.info("Choose a test image to continue.")
            return

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
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-header">Original Image</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image_rgb, width=RESULT_IMAGE_WIDTH)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-header">Predicted Mask</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(mask, width=RESULT_IMAGE_WIDTH)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-header">Overlay Result</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(blended, width=RESULT_IMAGE_WIDTH)
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
