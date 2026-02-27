"""
Resonair AI - Cough Sound Analysis for Lung Cancer Detection
================================================================
Streamlit MVP: Record or upload a cough sound, convert it to a
spectrogram, and get a lung cancer risk assessment using an
augmented MobileNetV3-Large model with temperature calibration.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from supabase import create_client, Client

# Import model builder from the augmented ensemble script
from augmented_ensemble import build_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results_augmented" / "best_augmented_model.pth"
SAMPLE_CANCER_DIR = BASE_DIR / "2. Lungs Cancer" / "CSI"
SAMPLE_NORMAL_DIR = BASE_DIR / "9. Normal" / "CSI"

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resonair AI - Cough Analysis",
    page_icon="https://em-content.zobj.net/source/twitter/376/lungs_1fac1.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Supabase Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client | None:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.warning("⚠️ Supabase credentials not found in `.streamlit/secrets.toml`. Database saving disabled.")
        return None

supabase_client = init_supabase()

# ---------------------------------------------------------------------------
# Custom CSS - White + Blue Glassmorphism Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ──────── Animated Orb Background ──────── */
    .stApp {
        background: #f0f4ff;
        font-family: 'Inter', sans-serif;
        position: relative;
        overflow-x: hidden;
    }
    .stApp::before,
    .stApp::after {
        content: '';
        position: fixed;
        border-radius: 50%;
        z-index: -1;
        filter: blur(100px);
        opacity: 0.5;
        pointer-events: none;
    }
    .stApp::before {
        width: 500px; height: 500px;
        background: radial-gradient(circle, #3b82f6 0%, transparent 70%);
        top: -100px; left: -80px;
        animation: float1 18s ease-in-out infinite;
    }
    .stApp::after {
        width: 450px; height: 450px;
        background: radial-gradient(circle, #818cf8 0%, transparent 70%);
        bottom: -80px; right: -60px;
        animation: float2 22s ease-in-out infinite;
    }

    /* Third orb - separate element injected via HTML instead */

    @keyframes float1 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(60px, 80px) scale(1.1); }
        66% { transform: translate(-40px, 40px) scale(0.95); }
    }
    @keyframes float2 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(-50px, -60px) scale(1.08); }
        66% { transform: translate(30px, -30px) scale(0.92); }
    }
    @keyframes float3 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-80px, 60px) scale(1.15); }
    }

    /* Orb injected via HTML */
    .orb-3 {
        position: fixed;
        width: 350px; height: 350px;
        background: radial-gradient(circle, #06b6d4 0%, transparent 70%);
        top: 40%; left: 55%;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.35;
        pointer-events: none;
        z-index: -1;
        animation: float3 25s ease-in-out infinite;
    }

    /* Ensure ALL content sits above orbs */
    .stApp > header,
    .stApp > section,
    .stApp > div {
        position: relative;
        z-index: 1;
    }
    .block-container {
        position: relative;
        z-index: 2;
    }
    .stMainBlockContainer {
        position: relative;
        z-index: 2;
    }

    /* ──────── Hero ──────── */
    .hero-title {
        font-size: 3.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e40af, #3b82f6, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
        text-shadow: 0 0 40px rgba(59, 130, 246, 0.15);
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ──────── Glass Cards ──────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 24px;
        padding: 1.75rem;
        backdrop-filter: blur(20px) saturate(1.8);
        -webkit-backdrop-filter: blur(20px) saturate(1.8);
        box-shadow:
            0 8px 32px rgba(31, 38, 135, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        margin-bottom: 1.25rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 40px rgba(31, 38, 135, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .glass-card h3 {
        color: #1e3a5f;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.15rem;
    }

    /* ──────── Risk Badges ──────── */
    .risk-low {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(16px) saturate(1.6);
        -webkit-backdrop-filter: blur(16px) saturate(1.6);
        border: 2px solid rgba(59, 130, 246, 0.4);
        color: #1a56db;
        padding: 1rem 2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        box-shadow:
            0 4px 24px rgba(59, 130, 246, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }
    .risk-medium {
        background: rgba(59, 130, 246, 0.12);
        backdrop-filter: blur(16px) saturate(1.6);
        -webkit-backdrop-filter: blur(16px) saturate(1.6);
        border: 2px solid rgba(59, 130, 246, 0.5);
        color: #1e40af;
        padding: 1rem 2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        box-shadow:
            0 4px 24px rgba(59, 130, 246, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.85), rgba(59, 130, 246, 0.75));
        backdrop-filter: blur(16px);
        color: #ffffff;
        padding: 1rem 2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow:
            0 8px 32px rgba(30, 64, 175, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    /* ──────── Confidence Bar ──────── */
    .conf-container {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 14px;
        overflow: hidden;
        height: 32px;
        margin: 0.5rem 0 1rem 0;
    }
    .conf-bar {
        height: 100%;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: #fff;
        transition: width 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    /* ──────── Stat Boxes ──────── */
    .stat-box {
        background: rgba(255, 255, 255, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 20px;
        padding: 1.5rem 1.25rem;
        text-align: center;
        backdrop-filter: blur(20px) saturate(1.8);
        -webkit-backdrop-filter: blur(20px) saturate(1.8);
        box-shadow:
            0 8px 24px rgba(31, 38, 135, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stat-box:hover {
        transform: translateY(-3px);
        box-shadow:
            0 12px 36px rgba(31, 38, 135, 0.14),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* ──────── Disclaimer ──────── */
    .disclaimer {
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 1.15rem 1.5rem;
        color: #475569;
        font-size: 0.85rem;
        margin-top: 2rem;
        line-height: 1.6;
        backdrop-filter: blur(16px) saturate(1.5);
        -webkit-backdrop-filter: blur(16px) saturate(1.5);
        box-shadow:
            0 4px 16px rgba(31, 38, 135, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    /* ──────── Compare Headers ──────── */
    .compare-header {
        color: #1e3a5f;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ──────── Section Divider ──────── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.25), transparent);
        margin: 2.5rem 0;
    }

    /* ──────── Step Badges ──────── */
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: #fff;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        text-align: center;
        line-height: 34px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.65rem;
        box-shadow: 0 4px 12px rgba(26, 86, 219, 0.35);
    }
    .step-text {
        color: #1e3a5f;
        font-weight: 500;
        font-size: 1rem;
    }

    /* ──────── Override Streamlit defaults ──────── */
    .stMarkdown, .stText, p, span, label {
        color: #1e3a5f !important;
    }
    h1, h2, h3, h4 {
        color: #1e3a5f !important;
    }

    /* ──────── Primary Button ──────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        backdrop-filter: blur(8px) !important;
        box-shadow:
            0 4px 20px rgba(26, 86, 219, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow:
            0 8px 30px rgba(26, 86, 219, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* ──────── File Uploader ──────── */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* ──────── Audio Input ──────── */
    .stAudioInput > div {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
    }
    /* Make mic button big and visible */
    .stAudioInput button {
        width: 80px !important;
        height: 80px !important;
        min-width: 80px !important;
        min-height: 80px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #1a56db, #3b82f6) !important;
        border: 3px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 6px 24px rgba(26, 86, 219, 0.35) !important;
        transition: all 0.3s ease !important;
        animation: mic-pulse 2.5s ease-in-out infinite;
    }
    .stAudioInput button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 8px 32px rgba(26, 86, 219, 0.5) !important;
    }
    .stAudioInput button svg {
        width: 36px !important;
        height: 36px !important;
        color: white !important;
        fill: white !important;
    }
    @keyframes mic-pulse {
        0%, 100% { box-shadow: 0 6px 24px rgba(26, 86, 219, 0.35); }
        50% { box-shadow: 0 6px 32px rgba(26, 86, 219, 0.55), 0 0 0 8px rgba(59, 130, 246, 0.15); }
    }

    /* ──────── Hide Streamlit chrome ──────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------
# Temperature for probability calibration (learned on validation set)
CALIBRATION_TEMPERATURE = 4.95

@st.cache_resource
def load_model():
    """Load the augmented MobileNetV3-Large model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("mobilenet_v3_large")
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    else:
        st.error(
            f"Model weights not found at `{MODEL_PATH}`. "
            "Please run `python augmented_ensemble.py` first to train the models."
        )
        st.stop()
    model.to(device)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Audio-to-Spectrogram
# ---------------------------------------------------------------------------
def audio_to_spectrogram(audio_bytes: bytes) -> Image.Image:
    """Convert audio bytes to a mel-spectrogram image matching training data style."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=22050, mono=True)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000,
        n_fft=2048, hop_length=512,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(4.32, 2.88), dpi=100)
    ax.imshow(S_dB, aspect="auto", origin="lower", cmap="inferno",
              interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    Path(tmp_path).unlink(missing_ok=True)
    return img


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(model, device, spectrogram_img: Image.Image):
    """Run inference with temperature-calibrated probability."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(spectrogram_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(img_tensor).squeeze()
        # Apply temperature scaling for better calibration
        calibrated_logit = logit / CALIBRATION_TEMPERATURE
        prob = torch.sigmoid(calibrated_logit).item()
    return prob


# ---------------------------------------------------------------------------
# PLCOm2012 Risk Calculator
# Reference: Tammemägi et al., NEJM 2013;368:728-36 (Table 2)
# Uses PLCOm2012noRace coefficients for broader applicability.
# ---------------------------------------------------------------------------
import math

def calculate_plcom2012(
    age, education, bmi, copd, personal_cancer,
    family_lung_cancer, smoking_status, cigs_per_day,
    duration_smoked, quit_years,
):
    """
    Calculate the 6-year lung cancer risk per the PLCOm2012 model.
    Returns probability (0 to 1).

    Parameters
    ----------
    age : int              — Age in years (centered around 62)
    education : int        — 1-6 scale (referent = 4 / Some college)
    bmi : float            — kg/m² (centered around 27)
    copd : int             — 0 = No, 1 = Yes
    personal_cancer : int  — 0 = No, 1 = Yes
    family_lung_cancer : int — 0 = No, 1 = Yes
    smoking_status : int   — 0 = Former, 1 = Current
    cigs_per_day : int     — Average cigarettes per day
    duration_smoked : int  — Years smoked (centered around 27)
    quit_years : int       — Years since quitting (centered around 10; 0 for current)
    """
    # Model coefficients (PLCOm2012noRace — Brock University calculator)
    INTERCEPT = -4.536696
    B_AGE = 0.0778895
    B_EDUCATION = -0.0811569
    B_BMI = -0.0251066
    B_COPD = 0.3606082
    B_PERSONAL_CANCER = 0.4683545
    B_FAMILY_LUNG_CANCER = 0.584541
    B_SMOKING_STATUS = 0.2675539
    B_CIGS_PER_DAY = -1.767578        # nonlinear: applied to (cigs/10)^(-1)
    B_DURATION = 0.031949
    B_QUIT_YEARS = -0.0312719

    # Centering values
    AGE_CENTER = 62
    BMI_CENTER = 27
    DURATION_CENTER = 27
    QUIT_CENTER = 10
    EDUCATION_CENTER = 4

    # Nonlinear cigarettes-per-day transformation: (cigs_per_day / 10)^(-1)
    # Guard against division by zero
    cigs_term = max(cigs_per_day, 0.5)
    cigs_transformed = (cigs_term / 10.0) ** (-1)

    # Linear predictor
    xb = (
        INTERCEPT
        + B_AGE * (age - AGE_CENTER)
        + B_EDUCATION * (education - EDUCATION_CENTER)
        + B_BMI * (bmi - BMI_CENTER)
        + B_COPD * copd
        + B_PERSONAL_CANCER * personal_cancer
        + B_FAMILY_LUNG_CANCER * family_lung_cancer
        + B_SMOKING_STATUS * smoking_status
        + B_CIGS_PER_DAY * cigs_transformed
        + B_DURATION * (duration_smoked - DURATION_CENTER)
        + B_QUIT_YEARS * (quit_years - QUIT_CENTER)
    )

    # Logistic transformation → 6-year probability
    probability = 1.0 / (1.0 + math.exp(-xb))
    return probability

# ---------------------------------------------------------------------------
# Load sample spectrograms
# ---------------------------------------------------------------------------
@st.cache_data
def load_sample_spectrograms():
    cancer_samples, normal_samples = [], []
    if SAMPLE_CANCER_DIR.exists():
        for f in sorted(SAMPLE_CANCER_DIR.glob("*.png"))[:3]:
            cancer_samples.append(Image.open(f).convert("RGB"))
    if SAMPLE_NORMAL_DIR.exists():
        for f in sorted(SAMPLE_NORMAL_DIR.glob("*.png"))[:3]:
            normal_samples.append(Image.open(f).convert("RGB"))
    return cancer_samples, normal_samples


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main():
    # ---- Floating Orb (third) ----
    st.markdown('<div class="orb-3"></div>', unsafe_allow_html=True)

    # ---- Hero ----
    st.markdown('<h1 class="hero-title">Resonair AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Advanced cough sound analysis &amp; PLCOm2012 lung cancer risk assessment'
        '</p>',
        unsafe_allow_html=True,
    )

    # ---- Disclaimer ----
    st.markdown(
        '<div class="disclaimer">'
        '<strong>&#9888; Medical Disclaimer:</strong> '
        'This tool is a research prototype / MVP and is <strong>NOT</strong> a substitute '
        'for professional medical diagnosis. Results are for educational and demonstration '
        'purposes only. Always consult a qualified healthcare provider for medical concerns.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- How it works ----
    st.markdown(
        '<div class="glass-card">'
        '<h3>How It Works</h3>'
        '<p style="color:#3b6b9c;">'
        '<span class="step-badge">1</span><span class="step-text">Record or upload a cough sound</span><br><br>'
        '<span class="step-badge">2</span><span class="step-text">AI converts your cough into a spectrogram & analyzes it</span><br><br>'
        '<span class="step-badge">3</span><span class="step-text">Fill in PLCOm2012 clinical risk factors for a comprehensive assessment</span>'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # PLCOm2012 Risk Calculator Section
    # ──────────────────────────────────────────
    st.markdown(
        '<div class="glass-card">'
        '<h3>&#128202; PLCOm2012 Lung Cancer Risk Calculator</h3>'
        '<p style="color:#64748b; font-size:0.9rem;">'
        'Based on Tammem&auml;gi et al., NEJM 2013. Estimates your 6-year lung cancer risk.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    plco_col1, plco_col2, plco_col3 = st.columns(3, gap="medium")

    with plco_col1:
        age = st.number_input("Age (years)", min_value=40, max_value=90, value=55, step=1)
        education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Less than high school",
                2: "High school graduate",
                3: "Post high school training",
                4: "Some college",
                5: "College graduate",
                6: "Postgraduate / professional",
            }[x],
            index=3,  # default: Some college
        )
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0, step=0.5)

    with plco_col2:
        copd = st.selectbox("COPD Diagnosis?", ["No", "Yes"])
        personal_cancer = st.selectbox("Personal History of Cancer?", ["No", "Yes"])
        family_lung_cancer = st.selectbox("Family History of Lung Cancer?", ["No", "Yes"])

    with plco_col3:
        smoking_status = st.selectbox("Smoking Status", ["Never smoker", "Former smoker", "Current smoker"])
        if smoking_status != "Never smoker":
            cigs_per_day = st.number_input("Cigarettes per Day (avg)", min_value=0, max_value=100, value=20, step=1)
            duration_smoked = st.number_input("Years Smoked", min_value=0, max_value=70, value=20, step=1)
            if smoking_status == "Former smoker":
                quit_years = st.number_input("Years Since Quitting", min_value=0, max_value=50, value=5, step=1)
            else:
                quit_years = 0
        else:
            cigs_per_day = 0
            duration_smoked = 0
            quit_years = 0
            st.info("PLCOm2012 is designed for ever-smokers. As a never smoker, your baseline risk is very low.")

    # Calculate PLCOm2012 risk
    if smoking_status == "Never smoker":
        # PLCOm2012 is not validated for never-smokers; return near-zero
        plco_risk = 0.0
    else:
        plco_risk = calculate_plcom2012(
            age=age,
            education=education,
            bmi=bmi,
            copd=1 if copd == "Yes" else 0,
            personal_cancer=1 if personal_cancer == "Yes" else 0,
            family_lung_cancer=1 if family_lung_cancer == "Yes" else 0,
            smoking_status=1 if smoking_status == "Current smoker" else 0,
            cigs_per_day=cigs_per_day,
            duration_smoked=duration_smoked,
            quit_years=quit_years,
        )

    # Display PLCOm2012 result
    plco_pct = plco_risk * 100
    st.markdown(
        '<div class="glass-card">'
        '<h3>&#128200; Your PLCOm2012 6-Year Lung Cancer Risk</h3>'
        '</div>',
        unsafe_allow_html=True,
    )

    risk_cols = st.columns([1, 1], gap="large")
    with risk_cols[0]:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-value">{plco_pct:.2f}%</div>'
            f'<div class="stat-label">6-Year Lung Cancer Risk</div></div>',
            unsafe_allow_html=True,
        )
    with risk_cols[1]:
        if plco_risk >= 0.0151:
            st.markdown(
                '<div class="stat-box" style="border-color: rgba(59, 130, 246, 0.5);">'
                '<div class="stat-value" style="font-size:1.2rem;">Screening Recommended</div>'
                '<div class="stat-label">Risk &ge; 1.51% threshold (USPSTF guideline)</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="stat-box">'
                '<div class="stat-value" style="font-size:1.2rem; color: #64748b;">Below Threshold</div>'
                '<div class="stat-label">Risk &lt; 1.51% screening threshold</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # Cough Analysis Section
    # ──────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")
    audio_bytes = None

    with col_left:
        st.markdown(
            '<div class="glass-card">'
            '<h3>&#127908; Record Your Cough</h3>'
            '</div>',
            unsafe_allow_html=True,
        )
        recorded_audio = st.audio_input(
            "Press the microphone button below and cough clearly",
            key="mic_recorder",
        )
        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()

    with col_right:
        st.markdown(
            '<div class="glass-card">'
            '<h3>&#128193; Upload Audio File</h3>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a WAV, MP3, or other audio file of your cough",
            type=["wav", "mp3", "ogg", "m4a", "flac", "webm"],
            key="file_uploader",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes)

    # ---- Analyze Button ----
    if audio_bytes is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        analyze_btn = st.button(
            "Analyze My Cough",
            type="primary",
            use_container_width=True,
        )

        if analyze_btn:
            with st.spinner("Analyzing your cough sound..."):
                model, device = load_model()
                spectrogram_img = audio_to_spectrogram(audio_bytes)
                probability = predict(model, device, spectrogram_img)

            st.session_state["spectrogram"] = spectrogram_img
            st.session_state["probability"] = probability

    # ── Results ──
    if "probability" in st.session_state:
        spectrogram_img = st.session_state["spectrogram"]
        probability = st.session_state["probability"]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<h2 style="color:#1e3a5f !important; text-align:center; font-weight:700; '
            'margin-bottom:1.5rem;">Cough Analysis Results</h2>',
            unsafe_allow_html=True,
        )

        # Show spectrogram
        res_left, res_right = st.columns([1, 1], gap="large")

        with res_left:
            st.markdown(
                '<div class="glass-card">'
                '<h3>&#127912; Your Cough Spectrogram</h3>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.image(spectrogram_img, use_container_width=True)

        with res_right:
            st.markdown(
                '<div class="glass-card">'
                '<h3>&#128202; AI Assessment</h3>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Only show warning if probability > 70%
            if probability >= 0.70:
                st.markdown(
                    '<div class="risk-high">'
                    '&#9888; Lung Cancer Screening Recommended'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.error(
                    "**Our AI analysis of your cough pattern suggests you may benefit from "
                    "a lung cancer screening.** Please consult a pulmonologist or your "
                    "primary care physician for a comprehensive evaluation including "
                    "low-dose CT scan. Early detection saves lives."
                )
            else:
                st.markdown(
                    '<div class="risk-low">'
                    '&#10003; No Immediate Concern Detected'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.success(
                    "**Our AI did not detect strong lung cancer indicators in your cough pattern.** "
                    "This does not replace a medical diagnosis. If you have persistent symptoms "
                    "such as chronic cough, shortness of breath, or chest pain, please consult "
                    "a healthcare professional."
                )

        # ---- Comparison ----
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<h2 style="color:#1e3a5f !important; text-align:center; font-weight:700; '
            'margin-bottom:1.5rem;">Spectrogram Comparison</h2>',
            unsafe_allow_html=True,
        )

        cancer_samples, normal_samples = load_sample_spectrograms()

        comp_cols = st.columns(3, gap="medium")
        with comp_cols[0]:
            st.markdown('<p class="compare-header">Your Cough</p>', unsafe_allow_html=True)
            st.image(spectrogram_img, use_container_width=True)
        with comp_cols[1]:
            st.markdown('<p class="compare-header">Lung Cancer Sample</p>', unsafe_allow_html=True)
            if cancer_samples:
                st.image(cancer_samples[0], use_container_width=True)
            else:
                st.info("No cancer samples available")
        with comp_cols[2]:
            st.markdown('<p class="compare-header">Normal Sample</p>', unsafe_allow_html=True)
            if normal_samples:
                st.image(normal_samples[0], use_container_width=True)
            else:
                st.info("No normal samples available")

        # ---------------------------------------------------------------------------
        # Save Results to Supabase
        # ---------------------------------------------------------------------------
        if supabase_client is not None:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="glass-card" style="text-align: center;">'
                '<h3>&#128190; Save Your Assessment</h3>'
                '<p style="color:#64748b; font-size:0.9rem;">'
                'Store your demographic information, PLCOm2012 risk score, and Cough AI probability '
                'securely in our database for future reference.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            
            save_cols = st.columns([1, 2, 1])
            with save_cols[1]:
                if st.button("Save Results to Supabase", type="primary", use_container_width=True):
                    with st.spinner("Saving to database..."):
                        # Prepare payload
                        payload = {
                            "age": age,
                            "education": education,
                            "bmi": float(bmi),
                            "copd": True if copd == "Yes" else False,
                            "personal_cancer": True if personal_cancer == "Yes" else False,
                            "family_lung_cancer": True if family_lung_cancer == "Yes" else False,
                            "smoking_status": smoking_status,
                            "cigs_per_day": cigs_per_day,
                            "duration_smoked": duration_smoked,
                            "quit_years": quit_years,
                            "plco_risk_score": float(plco_pct),        # Store as percentage
                            "cough_ai_probability": float(probability) # Store as raw probability
                        }
                        
                        try:
                            # Insert into 'assessments' table
                            res = supabase_client.table("assessments").insert(payload).execute()
                            st.success("✅ Assessment saved successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Failed to save to Supabase: {str(e)}")


    # ---- Footer ----
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    footer_cols = st.columns(4)
    with footer_cols[0]:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-value" style="font-size:1.3rem;">MobileNetV3</div>'
            '<div class="stat-label">Cough Model</div></div>',
            unsafe_allow_html=True,
        )
    with footer_cols[1]:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-value" style="font-size:1.3rem;">PLCOm2012</div>'
            '<div class="stat-label">Risk Model</div></div>',
            unsafe_allow_html=True,
        )
    with footer_cols[2]:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-value" style="font-size:1.3rem;">715</div>'
            '<div class="stat-label">Training Spectrograms</div></div>',
            unsafe_allow_html=True,
        )
    with footer_cols[3]:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-value" style="font-size:1.3rem;">0.94</div>'
            '<div class="stat-label">Cough AUC Score</div></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

