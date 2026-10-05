import os
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b0
import segmentation_models_pytorch as smp

st.set_page_config(page_title="Building Footprint Detector", layout="wide")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
eval_transform = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

CLS_CKPT_PATH = "checkpoints/classifier_best.pt"
SEG_CKPT_PATH = "checkpoints/segmenter_best.pt"
FIGURES_DIR = "figures"


def build_classifier():
    model = efficientnet_b0(weights=None)  # weights=None: we load our own trained weights next
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def build_segmenter():
    return smp.DeepLabV3Plus(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = build_classifier()
    classifier.load_state_dict(torch.load(CLS_CKPT_PATH, map_location=device))
    classifier.to(device).eval()

    segmenter = build_segmenter()
    segmenter.load_state_dict(torch.load(SEG_CKPT_PATH, map_location=device))
    segmenter.to(device).eval()

    return classifier, segmenter, device


def preprocess_image(pil_image, size=256):
    pil_image = pil_image.convert("RGB").resize((size, size))
    arr = np.array(pil_image)
    augmented = eval_transform(image=arr)
    return augmented["image"].unsqueeze(0), arr  # (tensor for model, array for display)


def predict(img_tensor, classifier, segmenter, device, cls_threshold, seg_threshold):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        cls_prob = torch.sigmoid(classifier(img_tensor)).item()

        has_building = cls_prob > cls_threshold
        mask = None
        if has_building:
            seg_prob = torch.sigmoid(segmenter(img_tensor))
            mask = (seg_prob > seg_threshold).float().cpu().squeeze().numpy()

    return cls_prob, has_building, mask


def compute_iou_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    return iou, dice


def render_result_image(display_arr, mask=None, has_building=False):
    if has_building and mask is not None:
        overlay = display_arr.copy().astype(float)
        overlay[..., 0] = np.where(mask > 0, 255, overlay[..., 0])
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return (0.6 * display_arr + 0.4 * overlay).astype(np.uint8)
    return display_arr


def image_to_downloadable_bytes(image_array):
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 24%),
            linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.3), rgba(37, 99, 235, 0.28));
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 24px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 45px rgba(2, 6, 23, 0.35);
        animation: fadeIn 0.8s ease;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.6rem;
    }
    .hero-card h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .hero-card p {
        margin-bottom: 0;
        color: #dbeafe;
        font-size: 1rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin: 0.2rem 0 1.1rem 0;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
        color: #e2e8f0;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .feature-card strong {
        color: #ffffff;
    }
    .result-shell {
        background: rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(2, 6, 23, 0.24);
    }
    .image-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 20px;
        padding: 0.7rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .image-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 26px 70px rgba(14, 165, 233, 0.24);
    }
    .image-card .image-caption {
        color: #cbd5e1;
        margin-top: 0.55rem;
        font-size: 0.95rem;
    }
    .stImage {
        border-radius: 22px !important;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 22px 55px rgba(8, 17, 40, 0.28);
    }
    .stImage img {
        border-radius: 22px !important;
        transition: transform 0.2s ease;
    }
    .stImage:hover img {
        transform: scale(1.01);
    }
    .status-badge {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.18);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #bbf7d0;
        margin-bottom: 0.9rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 10px 25px rgba(2, 6, 23, 0.22);
        margin-bottom: 0.65rem;
    }
    .metric-card .label {
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .metric-card .value {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.25rem;
    }
    div[data-testid="stFileUploader"] > section {
        background: rgba(255,255,255,0.06);
        border: 2px dashed rgba(125, 211, 252, 0.4);
        border-radius: 16px;
        padding: 0.75rem;
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #0f766e, #2563eb);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.55rem 1rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 0.9rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #0f766e, #2563eb);
        color: white;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
cls_threshold = st.sidebar.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
seg_threshold = st.sidebar.slider("Segmentation threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline:**\n"
    "1. EfficientNet-B0 classifies: building present?\n"
    "2. If yes, DeepLabV3+ (MobileNetV2 encoder) segments the exact footprint."
)

# ---------------- Main tabs ----------------
tab_demo, tab_training = st.tabs(["🏗️ Live Demo", "📈 Training Results"])

with tab_demo:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">✨ AI-powered aerial analysis</div>
            <h1>🏗️ Building Footprint Detection</h1>
            <p>Upload an aerial image patch and inspect a polished, easy-to-read classify → segment workflow with clear result highlights.</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card"><strong>🧠 Classify</strong><br>Checks whether a building is present.</div>
            <div class="feature-card"><strong>🎯 Segment</strong><br>Outlines the predicted footprint precisely.</div>
            <div class="feature-card"><strong>⬇️ Export</strong><br>Download the generated result image instantly.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="result-shell">
            <h3 style="margin:0 0 0.3rem 0; color:#f8fafc;">Upload an image to get started</h3>
            <p style="margin:0; color:#cbd5e1;">The analysis logic stays the same, but the interface now looks more modern and engaging.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    gt_mask_file = st.file_uploader(
        "Optional: upload a ground-truth mask to compute IoU/Dice", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        classifier, segmenter, device = load_models()

        with st.spinner("Analyzing the image..."):
            pil_image = Image.open(uploaded_file)
            img_tensor, display_arr = preprocess_image(pil_image)
            cls_prob, has_building, mask = predict(
                img_tensor, classifier, segmenter, device, cls_threshold, seg_threshold
            )

        st.success("Analysis complete.")
        st.markdown(
            f"<div class='status-badge'>{'🏢 Building detected' if has_building else '🚫 No building detected'}</div>",
            unsafe_allow_html=True,
        )

        result_image = render_result_image(display_arr, mask=mask, has_building=has_building)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='image-card'><h4 style='margin:0 0 0.6rem 0; color:#f8fafc;'>Input preview</h4></div>", unsafe_allow_html=True)
            st.image(display_arr, use_container_width=True)
            st.markdown("<div class='image-card'><div class='image-caption'>Original input image used for prediction.</div></div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='image-card'><h4 style='margin:0 0 0.6rem 0; color:#f8fafc;'>Prediction preview</h4></div>", unsafe_allow_html=True)
            st.image(result_image, use_container_width=True, caption="Predicted footprint overlay")
            st.markdown("<div class='image-card'><div class='image-caption'>Overlay highlights predicted building footprints in red.</div></div>", unsafe_allow_html=True)

        metric_items = [
            ("Status", "Building detected" if has_building else "No building found"),
            ("Classification probability", f"{cls_prob:.3f}"),
            ("Segmentation mask", "Present" if mask is not None else "Not available"),
        ]
        cols = st.columns(3)
        for col, (label, value) in zip(cols, metric_items):
            with col:
                st.markdown(
                    f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:0.6rem;'><strong>Confidence</strong></div>", unsafe_allow_html=True)
        st.progress(min(max(cls_prob, 0.0), 1.0))

        download_bytes = image_to_downloadable_bytes(result_image)
        st.download_button(
            label="⬇️ Download result image",
            data=download_bytes,
            file_name="building_result.png",
            mime="image/png",
        )

        if gt_mask_file is not None and mask is not None:
            gt_pil = Image.open(gt_mask_file).convert("L").resize((256, 256))
            gt_arr = (np.array(gt_pil) > 127).astype(np.uint8)
            iou, dice = compute_iou_dice(mask, gt_arr)
            c1, c2 = st.columns(2)
            c1.metric("IoU vs. ground truth", f"{iou:.3f}")
            c2.metric("Dice vs. ground truth", f"{dice:.3f}")
        elif gt_mask_file is not None and mask is None:
            st.info("No building was detected, so there's no predicted mask to compare against ground truth.")

with tab_training:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">📊 Training insights</div>
            <h1>📈 Training Results</h1>
            <p>These figures are the static outputs from the training workflow and remain unchanged from the original experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Review the charts and sample predictions generated during training in a cleaner gallery view.")

    for fname, caption in [
        ("classifier_curves.png", "Classifier: train/val loss, accuracy, F1"),
        ("segmenter_curves.png", "Segmenter: train/val loss, IoU, Dice"),
        ("sample_predictions.png", "Sample test-set predictions"),
    ]:
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f"{fname} not found in `{FIGURES_DIR}/` — copy it from your Drive to show it here.")