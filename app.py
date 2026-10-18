
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #151922;
        border-right: 1px solid #303642;
    }

    .main-title {
        font-size: 2.65rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        color: #b4bdca;
        font-size: 1rem;
        line-height: 1.8;
    }

    .result-card {
        background: #182330;
        border: 1px solid #35465a;
        border-radius: 14px;
        padding: 22px;
        margin-top: 10px;
    }

    .result-label {
        color: #b9c5d3;
        font-size: 0.95rem;
    }

    .result-value {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 750;
        margin: 10px 0;
    }

    .small-note {
        color: #9ca9b9;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    div.stButton > button {
        background: #ef4444;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 650;
        min-height: 44px;
    }

    div.stButton > button:hover {
        background: #dc2626;
        color: white;
        border: none;
    }

    [data-testid="stMetric"] {
        background: #182330;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid #303d4d;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND PREPROCESSING FILES
# ==========================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    features = joblib.load(ARTIFACT_DIR / "features.pkl")
    medians = joblib.load(ARTIFACT_DIR / "medians.pkl")
    min_values = joblib.load(ARTIFACT_DIR / "min_values.pkl")
    max_values = joblib.load(ARTIFACT_DIR / "max_values.pkl")

    return (
        model,
        scaler,
        features,
        medians,
        min_values,
        max_values
    )


try:
    (
        model,
        scaler,
        features,
        medians,
        min_values,
        max_values
    ) = load_artifacts()

except Exception as error:
    st.error(
        "Unable to load the model. Please run the "
        "model-saving cell in your Jupyter Notebook first."
    )
    st.exception(error)
    st.stop()


# ==========================================
# FEATURE GROUPS FOR THE RADAR CHART
# ==========================================
radar_features = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave points",
    "symmetry",
    "fractal dimension"
]

# Keep only features that exist in the trained model.
available_radar_features = [
    name for name in radar_features
    if all(
        f"{name}_{suffix}" in features
        for suffix in ["mean", "se", "worst"]
    )
]


# ==========================================
# SIDEBAR: CELL NUCLEI DETAILS
# ==========================================
st.sidebar.title("🧬 Cell Nuclei Details")
st.sidebar.caption(
    "Adjust the measurements using the sliders below."
)

input_values = {}


# ==========================================
# SIDEBAR: LIVE FEATURE SLIDERS
# ==========================================
st.sidebar.title("🧬 Cell Nuclei Details")
st.sidebar.caption("Adjust measurements to update the chart live.")

input_values = {}

for feature in features:
    low = float(min_values[feature])
    high = float(max_values[feature])
    default = float(medians[feature])

    if low >= high:
        input_values[feature] = default
        continue

    step = max((high - low) / 1000, 0.00001)

    input_values[feature] = st.sidebar.slider(
        label=feature.replace("_", " ").title(),
        min_value=low,
        max_value=high,
        value=min(max(default, low), high),
        step=step,
        format="%.5f",
        key=f"slider_{feature}"
    )


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown(
    '<div class="main-title">Breast Cancer Diagnosis</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="subtitle">
Enter measurements from your dataset or laboratory workflow.
The trained machine learning model evaluates the measurements
and predicts a benign or malignant class.
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational demonstration only. This model is not a "
    "medical diagnostic tool and must not be used to make "
    "clinical decisions."
)

st.divider()


# ==========================================
# PREPARE INPUT DATA
# ==========================================
input_df = pd.DataFrame(
    [input_values],
    columns=features
)

# Match the training feature order exactly.
scaled_input = scaler.transform(input_df)


# ==========================================
# MAIN DASHBOARD LAYOUT
# ==========================================
chart_col, result_col = st.columns(
    [1.75, 1],
    gap="large"
)


# ==========================================
# RADAR CHART
# ==========================================
with chart_col:
    st.subheader("Cell Measurement Analysis")

    if available_radar_features:

        suffixes = {
            "Mean": "mean",
            "Standard Error": "se",
            "Worst": "worst"
        }

        radar_values = {}

        # Normalize each measurement using its own
        # observed dataset range for visual comparison.
        for display_name, suffix in suffixes.items():
            values = []

            for base_name in available_radar_features:
                feature_name = f"{base_name}_{suffix}"

                value = float(input_values[feature_name])
                low = float(min_values[feature_name])
                high = float(max_values[feature_name])

                if high > low:
                    normalized = (value - low) / (high - low)
                else:
                    normalized = 0.0

                values.append(normalized)

            radar_values[display_name] = values

        fig = go.Figure()

        trace_styles = [
            ("Mean", "#60a5fa", "rgba(96,165,250,0.18)"),
            (
                "Standard Error",
                "#2563eb",
                "rgba(37,99,235,0.12)"
            ),
            ("Worst", "#f87171", "rgba(248,113,113,0.15)")
        ]

        for name, line_color, fill_color in trace_styles:
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_values[name],
                    theta=[
                        name.replace("_", " ").title()
                        for name in available_radar_features
                    ],
                    fill="toself",
                    name=name,
                    line=dict(color=line_color, width=2),
                    fillcolor=fill_color
                )
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
            polar=dict(
                bgcolor="#111722",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.25, 0.5, 0.75, 1],
                    gridcolor="#3a4350"
                ),
                angularaxis=dict(
                    gridcolor="#3a4350"
                )
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=45, r=45, t=70, b=45),
            height=510
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Measurements are normalized against their individual "
            "dataset ranges. The radar chart is for visual comparison, "
            "not a measure of cancer risk."
        )

    else:
        st.info(
            "The radar chart requires mean, se, and worst versions "
            "of the selected features."
        )


# ==========================================
# PREDICTION RESULT PANEL - LIVE UPDATES
# ==========================================
with result_col:
    st.subheader("Prediction Result")

    # Make prediction using the latest slider values
    prediction = int(model.predict(scaled_input)[0])

    # Get model output probabilities
    probabilities = model.predict_proba(scaled_input)[0]

    probability_by_class = dict(
        zip(model.classes_, probabilities)
    )

    benign_probability = float(
        probability_by_class.get(0, 0)
    )

    malignant_probability = float(
        probability_by_class.get(1, 0)
    )

    # ------------------------------------------
    # DISPLAY PREDICTION
    # ------------------------------------------
    if prediction == 0:
        result_text = "Benign"
        st.success("🟢 Model Prediction: Benign")
    else:
        result_text = "Malignant"
        st.error("🔴 Model Prediction: Malignant")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Current classification</div>
            <div class="result-value">{result_text}</div>
            <div class="small-note">
                Updated automatically using the current feature values.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------
    # DISPLAY PROBABILITIES
    # ------------------------------------------
    st.write("### Model Output Probabilities")

    st.metric(
        label="Benign",
        value=f"{benign_probability * 100:.2f}%"
    )

    st.progress(benign_probability)

    st.metric(
        label="Malignant",
        value=f"{malignant_probability * 100:.2f}%"
    )

    st.progress(malignant_probability)

    # ------------------------------------------
    # INTERPRETATION NOTE
    # ------------------------------------------
    st.caption(
        "Probabilities represent the model's outputs, "
        "not clinically validated estimates of cancer risk. "
        "This application is for educational purposes only "
        "and must not be used for medical diagnosis."
    )


# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption(
    "Built with Python, Streamlit, Plotly and Scikit-learn | "
    "Breast Cancer Prediction ML Project"
)
