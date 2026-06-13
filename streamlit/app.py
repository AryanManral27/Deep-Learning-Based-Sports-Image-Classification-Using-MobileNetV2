import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
from utils import predict_label, classes
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.h5"
SAMPLE_IMAGE_PATH = APP_DIR / "test_f1_racing.jpg"

st.set_page_config(
    page_title="DEEP LEARNING-BASED SPORTS IMAGE CLASSIFICATION USING MOBILENETV2",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .hero-section {
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stMarkdown"] .hero-section h1.hero-title,
    .hero-section h1.hero-title {
        font-size: clamp(1.8rem, 4vw, 3rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin-top: 0 !important;
        margin-bottom: 0.75rem !important;
        line-height: 1.2 !important;
        background: linear-gradient(135deg, #00D4AA 0%, #5B8DEF 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        color: transparent !important;
    }

    div[data-testid="stMarkdown"] .hero-section p.hero-subtitle,
    .hero-section p.hero-subtitle {
        font-size: 1.2rem !important;
        color: #8B9BB4 !important;
        margin-bottom: 2rem !important;
        line-height: 1.6 !important;
    }

    .metric-card {
        background: linear-gradient(145deg, #151B24 0%, #1A2230 100%);
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8B9BB4;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }

    .result-card {
        background: linear-gradient(145deg, #12201C 0%, #151B24 100%);
        border: 1px solid #1F4D3F;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }

    .result-sport {
        font-size: 1.75rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0.25rem 0 0.5rem 0;
    }

    .result-confidence {
        font-size: 0.95rem;
        color: #8B9BB4;
    }

    .confidence-bar-bg {
        background: #243044;
        border-radius: 999px;
        height: 8px;
        margin-top: 0.75rem;
        overflow: hidden;
    }

    .confidence-bar-fill {
        background: linear-gradient(90deg, #00D4AA, #5B8DEF);
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }

    .image-frame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #243044;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #8B9BB4;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stFileUploader"] section {
        border: 1px dashed #2D3A4F;
        border-radius: 12px;
        background: #111820;
        padding: 0.5rem;
    }

    div[data-testid="stForm"] {
        background: #151B24;
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 1.25rem;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #2D3A4F;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #00D4AA;
        color: #00D4AA;
    }

    div[data-testid="stSidebar"] {
        background: #0E1319;
        border-right: 1px solid #1E2736;
    }

    .sidebar-stat {
        background: #151B24;
        border: 1px solid #243044;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MobileNetV2 model…")
def load_sports_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Place `best_model.h5` in the `streamlit` folder and refresh."
        )
        st.stop()
    return load_model(str(MODEL_PATH))


def render_prediction(image: Image.Image, model) -> None:
    with st.spinner("Analyzing image…"):
        label, confidence = predict_label(image, model)

    if label is None:
        st.warning(
            f"Could not classify this image confidently ({confidence:.1f}% confidence). "
            "Try a clearer sports photo."
        )
        return

    st.markdown(
        f"""
        <div class="result-card">
            <div class="section-label">Prediction</div>
            <p class="result-sport">{label}</p>
            <p class="result-confidence">Confidence: {confidence:.1f}%</p>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: {confidence:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    model = load_sports_model()

    st.markdown(
        """
        <div class="hero-section">
            <h1 class="hero-title" style="
                font-size: clamp(2.25rem, 5.5vw, 3.75rem);
                font-weight: 700;
                line-height: 1.2;
                margin: 0 0 0.75rem 0;
                background: linear-gradient(135deg, #00D4AA 0%, #5B8DEF 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">DEEP LEARNING-BASED SPORTS IMAGE CLASSIFICATION USING MOBILENETV2</h1>
            <p class="hero-subtitle">Upload a sports image and let the model accurately classify and predict the sport it represents in real time!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><p class="metric-value">35</p>'
            '<p class="metric-label">Sport Classes</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><p class="metric-value">MobileNetV2</p>'
            '<p class="metric-label">Architecture</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><p class="metric-value">224×224</p>'
            '<p class="metric-label">Input Size</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "This deep learning model is trained on diverse sports images for accurate classification. "
            "Supported formats: **JPG**, **JPEG**, **PNG**."
        )
        st.markdown("---")
        st.markdown("**Supported sports**")
        with st.expander(f"View all {len(classes)} classes"):
            for sport in classes:
                st.caption(f"• {sport.title()}")

    upload_col, result_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.markdown('<p class="section-label">Upload Image</p>', unsafe_allow_html=True)

        with st.form("classification_form", clear_on_submit=False):
            uploaded_file = st.file_uploader(
                "Choose a sports image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Classify Sport", use_container_width=True)

        st.markdown('<p class="section-label" style="margin-top: 1.5rem;">Quick Demo</p>', unsafe_allow_html=True)
        use_sample = st.button("Use Sample Image (F1 Racing)", use_container_width=True)

        active_image = None
        trigger_prediction = False

        if submitted:
            if uploaded_file is not None:
                active_image = Image.open(uploaded_file)
                trigger_prediction = True
            else:
                st.warning("Please upload an image before submitting.")

        if use_sample:
            if SAMPLE_IMAGE_PATH.exists():
                active_image = Image.open(SAMPLE_IMAGE_PATH)
                trigger_prediction = True
            else:
                st.warning(
                    f"Sample image not found at `{SAMPLE_IMAGE_PATH}`. "
                    "Add `test_f1_racing.jpg` to the `streamlit` folder."
                )

    with result_col:
        st.markdown('<p class="section-label">Preview & Result</p>', unsafe_allow_html=True)

        if active_image is not None:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(active_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if trigger_prediction:
                render_prediction(active_image, model)
        else:
            st.markdown(
                """
                <div class="metric-card" style="padding: 3rem 1.5rem;">
                    <p style="font-size: 2.5rem; margin: 0;">🏟️</p>
                    <p style="color: #8B9BB4; margin-top: 0.75rem;">
                        Upload an image or use the sample to see a prediction here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
