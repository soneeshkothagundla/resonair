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
        st.warning("⚠️ Supabase credentials not found. Database saving disabled.")
        return None

supabase_client = init_supabase()

# ---------------------------------------------------------------------------
# CSS — COAL-style deep blue glassmorphism
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

/* ── Background & font ───────────────────────────────── */
.stApp {
    background: #EEF4FF;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    position: relative;
    overflow-x: hidden;
}
.stApp * { font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* Animated orb 1 */
.stApp::before {
    content: '';
    position: fixed;
    width: 700px; height: 700px;
    background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 65%);
    top: -220px; left: -200px;
    border-radius: 50%;
    z-index: 0; pointer-events: none;
    animation: orb1 22s ease-in-out infinite;
}
/* Animated orb 2 */
.stApp::after {
    content: '';
    position: fixed;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 65%);
    bottom: -150px; right: -150px;
    border-radius: 50%;
    z-index: 0; pointer-events: none;
    animation: orb2 28s ease-in-out infinite;
}
/* Animated orb 3 */
.orb-3 {
    position: fixed;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(6,182,212,0.07) 0%, transparent 65%);
    top: 35%; left: 50%;
    border-radius: 50%;
    z-index: 0; pointer-events: none;
    animation: orb3 30s ease-in-out infinite;
}
@keyframes orb1 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(80px,60px) scale(1.1);} }
@keyframes orb2 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(-60px,-80px) scale(1.08);} }
@keyframes orb3 { 0%,100%{transform:translate(-50%,0) scale(1);} 50%{transform:translate(-50%,40px) scale(1.1);} }

/* Ensure content is above orbs */
.stApp > *, .block-container, .stMainBlockContainer { position: relative; z-index: 1; }

/* Block container sizing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1150px !important;
}

/* ── HERO ─────────────────────────────────────────────── */
.hero-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(59,130,246,0.20);
    border-radius: 100px;
    padding: 6px 18px 6px 9px;
    font-size: 12.5px; font-weight: 500; color: #1E3A5F;
    backdrop-filter: blur(12px);
    box-shadow: 0 2px 12px rgba(59,130,246,0.08);
}
.hero-pill-dot {
    width: 21px; height: 21px;
    background: linear-gradient(135deg, #1D4ED8, #60A5FA);
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.hero-plain {
    font-size: clamp(2.4rem,4.5vw,3.5rem);
    font-weight: 900; letter-spacing: -1.5px; line-height: 1.08;
    color: #0B1120; display: inline;
}
.hero-accent {
    font-size: clamp(2.4rem,4.5vw,3.5rem);
    font-weight: 900; letter-spacing: -1.5px; line-height: 1.08;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: inline;
}
.hero-sub {
    font-size: 1rem; color: #64748B; font-weight: 400; line-height: 1.75;
}

/* ── Label tag (pill above section headings) ──────────── */
.label-tag {
    display: inline-block;
    background: rgba(59,130,246,0.09);
    color: #1D4ED8;
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.8px;
    padding: 5px 14px; border-radius: 100px;
    border: 1px solid rgba(59,130,246,0.14);
    margin-bottom: 10px;
}

/* ── Section headings ─────────────────────────────────── */
.sec-h {
    font-size: 1.75rem; font-weight: 800; letter-spacing: -0.6px;
    color: #0B1120; line-height: 1.15; margin-bottom: 6px;
}
.sec-sub {
    font-size: 0.93rem; color: #64748B; line-height: 1.75;
}

/* ── Glass Cards ──────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(59,130,246,0.13);
    border-radius: 22px;
    padding: 1.5rem 1.75rem;
    backdrop-filter: blur(20px) saturate(1.6);
    -webkit-backdrop-filter: blur(20px) saturate(1.6);
    box-shadow: 0 4px 20px rgba(29,78,216,0.07),
                inset 0 1px 0 rgba(255,255,255,0.80);
    margin-bottom: 1.1rem;
    transition: all 0.28s ease;
    position: relative; overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1D4ED8, #3B82F6);
    opacity: 0; transition: opacity 0.28s;
}
.glass-card:hover::before { opacity: 1; }
.glass-card:hover {
    border-color: rgba(59,130,246,0.26);
    box-shadow: 0 10px 36px rgba(29,78,216,0.11),
                inset 0 1px 0 rgba(255,255,255,0.90);
    transform: translateY(-2px);
}
.glass-card h3 {
    font-size: 1rem; font-weight: 700;
    color: #0B1120 !important; -webkit-text-fill-color: #0B1120 !important;
    margin-bottom: 0.6rem; letter-spacing: -0.2px;
}

/* ── Step cards (How It Works) ────────────────────────── */
.step-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 22px; padding: 2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 20px rgba(29,78,216,0.07);
    transition: all 0.28s ease; position: relative; overflow: hidden;
}
.step-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1D4ED8, #3B82F6);
    opacity: 0; transition: opacity 0.28s;
}
.step-card:hover::before { opacity: 1; }
.step-card:hover {
    border-color: rgba(59,130,246,0.26);
    box-shadow: 0 12px 36px rgba(29,78,216,0.12);
    transform: translateY(-5px);
}
.step-num {
    width: 46px; height: 46px;
    background: linear-gradient(140deg, #1D4ED8, #3B82F6);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 800; color: white;
    margin: 0 auto 1rem;
    box-shadow: 0 4px 16px rgba(29,78,216,0.30);
}
.step-title { font-weight: 700; font-size: 0.95rem; color: #0B1120; margin-bottom: 0.5rem; letter-spacing: -0.2px; }
.step-desc  { font-size: 0.84rem; color: #64748B; line-height: 1.65; }

/* ── Stat boxes ───────────────────────────────────────── */
.stat-box {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 20px; padding: 1.5rem 1rem;
    text-align: center; backdrop-filter: blur(20px);
    box-shadow: 0 4px 20px rgba(29,78,216,0.07);
    transition: all 0.28s ease; height: 100%;
}
.stat-box:hover {
    border-color: rgba(59,130,246,0.26);
    box-shadow: 0 10px 32px rgba(29,78,216,0.12);
    transform: translateY(-3px);
}
.stat-value {
    font-size: 1.55rem; font-weight: 900; letter-spacing: -0.8px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2;
}
.stat-label { font-size: 0.79rem; color: #64748B; margin-top: 0.3rem; font-weight: 500; }

/* ── Risk badges ──────────────────────────────────────── */
.risk-low {
    background: rgba(255,255,255,0.85);
    border: 2px solid rgba(59,130,246,0.28); color: #1D4ED8;
    padding: 1.25rem 2rem; border-radius: 18px;
    font-size: 1.15rem; font-weight: 700; text-align: center;
    margin: 1rem 0; box-shadow: 0 4px 20px rgba(59,130,246,0.10);
}
.risk-high {
    background: linear-gradient(135deg, rgba(29,78,216,0.92), rgba(59,130,246,0.82));
    color: #fff; padding: 1.25rem 2rem; border-radius: 18px;
    font-size: 1.15rem; font-weight: 700; text-align: center; margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(29,78,216,0.35);
}

/* ── Compare headers ──────────────────────────────────── */
.compare-header {
    color: #1D4ED8; font-weight: 700; font-size: 0.78rem;
    text-align: center; margin-bottom: 0.5rem;
    text-transform: uppercase; letter-spacing: 1.2px;
}

/* ── Section divider ──────────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.18), transparent);
    margin: 2rem 0;
}

/* ── Disclaimer ───────────────────────────────────────── */
.disclaimer {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(59,130,246,0.12); border-radius: 16px;
    padding: 1rem 1.5rem; color: #475569; font-size: 0.84rem;
    line-height: 1.65; backdrop-filter: blur(16px);
    box-shadow: 0 2px 12px rgba(29,78,216,0.05);
}

/* ── Streamlit text overrides ─────────────────────────── */
.stMarkdown p, label, .stText { color: #1E3A5F !important; }
h1, h2, h3, h4 { color: #0B1120 !important; }

/* ── Primary button ───────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(140deg, #1D4ED8, #2563EB) !important;
    color: white !important; border: none !important;
    border-radius: 100px !important;
    padding: 0.75rem 2.5rem !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 6px 24px rgba(29,78,216,0.35) !important;
    transition: all 0.28s !important; letter-spacing: 0.1px !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 36px rgba(29,78,216,0.50) !important;
    transform: translateY(-2px) !important;
}

/* ── File uploader ────────────────────────────────────── */
.stFileUploader {
    background: rgba(255,255,255,0.60) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
}

/* ── Audio input ──────────────────────────────────────── */
.stAudioInput > div {
    background: rgba(255,255,255,0.60) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
}
.stAudioInput button {
    width: 80px !important; height: 80px !important;
    min-width: 80px !important; min-height: 80px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    border: 3px solid rgba(255,255,255,0.6) !important;
    box-shadow: 0 6px 24px rgba(29,78,216,0.35) !important;
    animation: mic-pulse 2.5s ease-in-out infinite;
}
.stAudioInput button:hover { transform: scale(1.1) !important; }
.stAudioInput button svg {
    width: 36px !important; height: 36px !important;
    color: white !important; fill: white !important;
}
@keyframes mic-pulse {
    0%,100% { box-shadow: 0 6px 24px rgba(29,78,216,0.35); }
    50%      { box-shadow: 0 6px 32px rgba(29,78,216,0.55), 0 0 0 10px rgba(59,130,246,0.12); }
}

/* ── Hide Streamlit chrome ────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------
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
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(4.32, 2.88), dpi=100)
    ax.imshow(S_dB, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(spectrogram_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(img_tensor).squeeze()
        calibrated_logit = logit / CALIBRATION_TEMPERATURE
        prob = torch.sigmoid(calibrated_logit).item()
    return prob


# ---------------------------------------------------------------------------
# PLCOm2012 Risk Calculator
# ---------------------------------------------------------------------------
import math

def calculate_plcom2012(age, education, bmi, copd, personal_cancer,
                         family_lung_cancer, smoking_status, cigs_per_day,
                         duration_smoked, quit_years):
    INTERCEPT = -4.536696
    B_AGE = 0.0778895;       B_EDUCATION = -0.0811569; B_BMI = -0.0251066
    B_COPD = 0.3606082;      B_PERSONAL_CANCER = 0.4683545
    B_FAMILY_LUNG_CANCER = 0.584541; B_SMOKING_STATUS = 0.2675539
    B_CIGS_PER_DAY = -1.767578; B_DURATION = 0.031949; B_QUIT_YEARS = -0.0312719
    AGE_CENTER = 62; BMI_CENTER = 27; DURATION_CENTER = 27
    QUIT_CENTER = 10; EDUCATION_CENTER = 4
    cigs_term = max(cigs_per_day, 0.5)
    cigs_transformed = (cigs_term / 10.0) ** (-1)
    xb = (INTERCEPT
          + B_AGE * (age - AGE_CENTER)
          + B_EDUCATION * (education - EDUCATION_CENTER)
          + B_BMI * (bmi - BMI_CENTER)
          + B_COPD * copd
          + B_PERSONAL_CANCER * personal_cancer
          + B_FAMILY_LUNG_CANCER * family_lung_cancer
          + B_SMOKING_STATUS * smoking_status
          + B_CIGS_PER_DAY * cigs_transformed
          + B_DURATION * (duration_smoked - DURATION_CENTER)
          + B_QUIT_YEARS * (quit_years - QUIT_CENTER))
    return 1.0 / (1.0 + math.exp(-xb))


# ---------------------------------------------------------------------------
# Sample spectrograms
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
    st.markdown('<div class="orb-3"></div>', unsafe_allow_html=True)

    # ── HERO ──────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        <div style="display:flex; justify-content:center; margin-bottom:18px;">
            <div class="hero-pill">
                <div class="hero-pill-dot">
                    <svg width="8" height="8" viewBox="0 0 10 10" fill="white"><circle cx="5" cy="5" r="3"/></svg>
                </div>
                &nbsp;Pioneering Respiratory Analytics
            </div>
        </div>
        <div style="margin-bottom:14px; line-height:1.08;">
            <span class="hero-plain">Resonair&nbsp;</span><span class="hero-accent">AI</span>
        </div>
        <p class="hero-sub">Advanced cough sound analysis &amp; PLCOm2012 lung cancer risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1.25rem;"></div>', unsafe_allow_html=True)

    # ── DISCLAIMER ────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>&#9888; Medical Disclaimer:</strong>&nbsp;This tool is a research prototype / MVP
        and is <strong>NOT</strong> a substitute for professional medical diagnosis. Results are for
        educational and demonstration purposes only. Always consult a qualified healthcare provider
        for medical concerns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── STATS STRIP ───────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    with s1:
        st.markdown('<div class="stat-box"><div class="stat-value" style="font-size:1.1rem;">MobileNetV3</div><div class="stat-label">Cough AI Model</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat-box"><div class="stat-value" style="font-size:1.1rem;">PLCOm2012</div><div class="stat-label">Clinical Risk Model</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="stat-box"><div class="stat-value">715</div><div class="stat-label">Training Spectrograms</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="stat-box"><div class="stat-value">0.94</div><div class="stat-label">Cough AUC Score</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────────────────
    st.markdown('<div style="text-align:center;"><div class="label-tag">How It Works</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-h" style="text-align:center;">Three Steps to Your Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub" style="text-align:center; max-width:520px; margin:0 auto 1.75rem;">Our AI combines acoustic biomarkers with clinical risk factors for a comprehensive lung cancer analysis.</p>', unsafe_allow_html=True)

    hw1, hw2, hw3 = st.columns(3, gap="medium")
    with hw1:
        st.markdown("""<div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Record or Upload</div>
            <div class="step-desc">Cough clearly into your microphone, or upload an existing audio file for analysis.</div>
        </div>""", unsafe_allow_html=True)
    with hw2:
        st.markdown("""<div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">AI Spectrogram Analysis</div>
            <div class="step-desc">Your cough is converted to a mel-spectrogram and analyzed by our MobileNetV3 model.</div>
        </div>""", unsafe_allow_html=True)
    with hw3:
        st.markdown("""<div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Clinical Risk Score</div>
            <div class="step-desc">Fill in PLCOm2012 clinical factors for a validated 6-year lung cancer risk estimate.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── COUGH ANALYSIS ────────────────────────────────────
    st.markdown('<div style="text-align:center;"><div class="label-tag">Cough Analysis</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-h" style="text-align:center;">Analyze Your Cough</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub" style="text-align:center; max-width:500px; margin:0 auto 1.75rem;">Record a cough or upload audio. Our AI converts it to a mel-spectrogram and detects patterns associated with lung cancer.</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")
    audio_bytes = None

    with col_left:
        st.markdown('<div class="glass-card"><h3>&#127908; Record Your Cough</h3></div>', unsafe_allow_html=True)
        recorded_audio = st.audio_input(
            "Press the microphone button below and cough clearly",
            key="mic_recorder",
        )
        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()

    with col_right:
        st.markdown('<div class="glass-card"><h3>&#128193; Upload Audio File</h3></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a WAV, MP3, or other audio file of your cough",
            type=["wav", "mp3", "ogg", "m4a", "flac", "webm"],
            key="file_uploader",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes)

    if audio_bytes is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        analyze_btn = st.button("Analyze My Cough →", type="primary", use_container_width=True)
        if analyze_btn:
            with st.spinner("Analyzing your cough sound..."):
                model, device = load_model()
                spectrogram_img = audio_to_spectrogram(audio_bytes)
                probability = predict(model, device, spectrogram_img)
            st.session_state["spectrogram"] = spectrogram_img
            st.session_state["probability"] = probability

    # ── RESULTS ───────────────────────────────────────────
    if "probability" in st.session_state:
        spectrogram_img = st.session_state["spectrogram"]
        probability     = st.session_state["probability"]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;"><div class="label-tag">Results</div></div>', unsafe_allow_html=True)
        st.markdown('<p class="sec-h" style="text-align:center; margin-bottom:1.5rem;">Cough Analysis Results</p>', unsafe_allow_html=True)

        res_left, res_right = st.columns([1, 1], gap="large")
        with res_left:
            st.markdown('<div class="glass-card"><h3>&#127912; Your Cough Spectrogram</h3></div>', unsafe_allow_html=True)
            st.image(spectrogram_img, use_container_width=True)

        with res_right:
            st.markdown('<div class="glass-card"><h3>&#128202; AI Assessment</h3></div>', unsafe_allow_html=True)
            if probability >= 0.70:
                st.markdown('<div class="risk-high">&#9888; Lung Cancer Screening Recommended</div>', unsafe_allow_html=True)
                st.error(
                    "**Our AI analysis of your cough pattern suggests you may benefit from "
                    "a lung cancer screening.** Please consult a pulmonologist or your "
                    "primary care physician for a comprehensive evaluation including "
                    "low-dose CT scan. Early detection saves lives."
                )
            else:
                st.markdown('<div class="risk-low">&#10003; No Immediate Concern Detected</div>', unsafe_allow_html=True)
                st.success(
                    "**Our AI did not detect strong lung cancer indicators in your cough pattern.** "
                    "This does not replace a medical diagnosis. If you have persistent symptoms "
                    "such as chronic cough, shortness of breath, or chest pain, please consult "
                    "a healthcare professional."
                )

        # Spectrogram comparison
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;"><div class="label-tag">Comparison</div></div>', unsafe_allow_html=True)
        st.markdown('<p class="sec-h" style="text-align:center; margin-bottom:1.5rem;">Spectrogram Comparison</p>', unsafe_allow_html=True)

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

        # Save to Supabase
        if supabase_client is not None:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <h3>&#128190; Save Your Assessment</h3>
                <p style="color:#64748B; font-size:0.88rem; margin:0;">
                    Store your PLCOm2012 risk score and Cough AI probability securely for future reference.
                </p>
            </div>""", unsafe_allow_html=True)
            save_cols = st.columns([1, 2, 1])
            with save_cols[1]:
                if st.button("Save Results to Supabase", type="primary", use_container_width=True):
                    with st.spinner("Saving to database..."):
                        payload = {
                            "age": age, "education": education, "bmi": float(bmi),
                            "copd": True if copd == "Yes" else False,
                            "personal_cancer": True if personal_cancer == "Yes" else False,
                            "family_lung_cancer": True if family_lung_cancer == "Yes" else False,
                            "smoking_status": smoking_status,
                            "cigs_per_day": cigs_per_day, "duration_smoked": duration_smoked,
                            "quit_years": quit_years,
                            "plco_risk_score": float(plco_pct),
                            "cough_ai_probability": float(probability)
                        }
                        try:
                            supabase_client.table("assessments").insert(payload).execute()
                            st.success("✅ Assessment saved successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Failed to save to Supabase: {str(e)}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── PLCO RISK CALCULATOR ──────────────────────────────
    st.markdown('<div style="text-align:center;"><div class="label-tag">Clinical Risk Calculator</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-h" style="text-align:center;">PLCOm2012 Risk Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub" style="text-align:center; max-width:560px; margin:0 auto 1.75rem;">Based on Tammem&auml;gi et al., NEJM 2013. Estimates your 6-year lung cancer risk from clinical factors.</p>', unsafe_allow_html=True)

    plco_col1, plco_col2, plco_col3 = st.columns(3, gap="medium")

    with plco_col1:
        st.markdown('<div class="glass-card"><h3>&#128100; Demographics</h3></div>', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=40, max_value=90, value=55, step=1)
        education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Less than high school", 2: "High school graduate",
                3: "Post high school training", 4: "Some college",
                5: "College graduate", 6: "Postgraduate / professional",
            }[x],
            index=3,
        )
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0, step=0.5)

    with plco_col2:
        st.markdown('<div class="glass-card"><h3>&#129657; Medical History</h3></div>', unsafe_allow_html=True)
        copd = st.selectbox("COPD Diagnosis?", ["No", "Yes"])
        personal_cancer = st.selectbox("Personal History of Cancer?", ["No", "Yes"])
        family_lung_cancer = st.selectbox("Family History of Lung Cancer?", ["No", "Yes"])

    with plco_col3:
        st.markdown('<div class="glass-card"><h3>&#128684; Smoking History</h3></div>', unsafe_allow_html=True)
        smoking_status = st.selectbox("Smoking Status", ["Never smoker", "Former smoker", "Current smoker"])
        if smoking_status != "Never smoker":
            cigs_per_day = st.number_input("Cigarettes per Day (avg)", min_value=0, max_value=100, value=20, step=1)
            duration_smoked = st.number_input("Years Smoked", min_value=0, max_value=70, value=20, step=1)
            quit_years = st.number_input("Years Since Quitting", min_value=0, max_value=50, value=5, step=1) if smoking_status == "Former smoker" else 0
        else:
            cigs_per_day = duration_smoked = quit_years = 0
            st.info("PLCOm2012 is designed for ever-smokers. As a never smoker, your baseline risk is very low.")

    # Calculate
    if smoking_status == "Never smoker":
        plco_risk = 0.0
    else:
        plco_risk = calculate_plcom2012(
            age=age, education=education, bmi=bmi,
            copd=1 if copd == "Yes" else 0,
            personal_cancer=1 if personal_cancer == "Yes" else 0,
            family_lung_cancer=1 if family_lung_cancer == "Yes" else 0,
            smoking_status=1 if smoking_status == "Current smoker" else 0,
            cigs_per_day=cigs_per_day, duration_smoked=duration_smoked, quit_years=quit_years,
        )
    plco_pct = plco_risk * 100

    risk_r1, risk_r2 = st.columns([1, 1], gap="large")
    with risk_r1:
        st.markdown(
            f'<div class="stat-box" style="padding:1.8rem;">'
            f'<div class="stat-value" style="font-size:2.2rem;">{plco_pct:.2f}%</div>'
            f'<div class="stat-label" style="font-size:0.88rem;">6-Year Lung Cancer Risk (PLCOm2012)</div></div>',
            unsafe_allow_html=True,
        )
    with risk_r2:
        if plco_risk >= 0.0151:
            st.markdown(
                '<div class="stat-box" style="padding:1.8rem; border-color:rgba(59,130,246,0.40);">'
                '<div class="stat-value" style="font-size:1.1rem;">Screening Recommended</div>'
                '<div class="stat-label">Risk &ge; 1.51% threshold (USPSTF guideline)</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="stat-box" style="padding:1.8rem;">'
                '<div class="stat-value" style="font-size:1.1rem; background:none; -webkit-text-fill-color:#64748B; color:#64748B;">Below Threshold</div>'
                '<div class="stat-label">Risk &lt; 1.51% screening threshold</div></div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
