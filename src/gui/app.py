import sys
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.optimization.recommender import PromptOptimizer
from src.prediction.estimator import predict_energy
from src.anomaly.detector import AnomalyDetector
from src.nlp.complexity_score import extract_features

# --- Page Config ---
st.set_page_config(page_title="GreenMind", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://res.cloudinary.com/dykxtkzm8/image/upload/v1754630959/WhatsApp_Image_2025-08-08_at_10.49.50_AM_gv5cm2.jpg');
            background-size: cover;
            background-attachment: fixed;
        }
        textarea {
            background-color: #f59e0b !important;
            color: black !important;
            font-weight: 500 !important;
        }
        .stButton > button {
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 24px;
            margin: 10px 10px 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.markdown("## GreenMind: Energy-Efficient Prompt and Context Engineering")
st.caption("Check the predicted Energy Consumption and hit **Improve** to see a more energy-efficient Prompt.")

# --- Layout ---
col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("### Enter prompt here:")
    prompt = st.text_area(" ", placeholder="""Role -----------------------
I am... Lorem ipsum dolor sit amet
You are... Consectetur adipiscing elit

Context --------------
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

Expectations-------
Ut enim ad minim veniam""", height=220, label_visibility='collapsed')
    
    from src.nlp.complexity_score import compute_token_count, compute_readability



    # Parameter Inputs
    st.markdown("### Set parameters:")
    num_layers = st.number_input("# Layers", min_value=1, value=4)
    training_hours = st.number_input("Training time (hrs)", min_value=1, value=2)
    flops_per_hour = st.number_input("FLOPs/hr.", min_value=1e5, value=1e20, format="%.2e")

# Submit/Improve Buttons
col_submit, col_improve = st.columns([1, 1])
submit_clicked = col_submit.button("Submit")
improve_clicked = col_improve.button("Improve", use_container_width=True)

# --- Prediction Logic ---
if submit_clicked:
    st.subheader("🔋 Estimated Energy Consumption")

        # --- NLP Analysis ---
    tokens = compute_token_count(prompt)
    readability = compute_readability(prompt)

    st.subheader(" NLP Analysis")
    st.markdown(f"- **Token Count:** {tokens}")
    st.markdown(f"- **readability_score:** {readability:.2f}")

    input_data = {
        "num_layers": num_layers,
        "training_hours": training_hours,
        "flops_per_hour": flops_per_hour,
        "token_count": tokens,
        "readability_score": readability
    }

    model_path = os.path.join("model", "energy_predictor.pkl")

    try:
        prediction = predict_energy(model_path, input_data)
        st.success(f"⚡ Estimated Energy Consumption: **{prediction:.2f} kWh**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

    # Optional Mock Graph
    st.markdown("### 📈 Energy Prediction vs Actual (Mocked)")
    actual = np.array([1, 2, 3, 4, 5, 6, 7])
    predicted = actual + np.random.normal(0, 0.3, size=actual.shape)
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual", marker='o')
    ax.plot(predicted, label="Predicted", marker='D', linestyle='--')
    ax.annotate("Prediction", xy=(len(predicted)-1, predicted[-1]),
                xytext=(len(predicted)-2, predicted[-1]+0.5),
                arrowprops=dict(facecolor='cyan', shrink=0.05),
                fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.set_xlabel("Prompt Iteration")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.legend()
    st.pyplot(fig)

            # --- Anomaly Detection ---
    features = extract_features(prompt)
    flops_per_layer = flops_per_hour / num_layers
    sample = [prediction, features["token_count"], flops_per_layer]
        # Load and run anomaly detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit([
            [0.3, 45, 1e17],
            [0.5, 55, 2e17],
            [0.7, 60, 3e17],
            [0.4, 50, 1.5e17],
            [0.6, 58, 2.5e17],
            ])
    is_anomaly, reason = anomaly_detector.detect(sample)

    if is_anomaly:
                
                st.error("🕵️ Anomaly Detected: This prompt may consume excessive energy.")
                st.info(f"Reason: {reason}")

# --- Prompt Optimization ---
if improve_clicked:
    if prompt.strip() == "":
        st.warning("Please enter a prompt before clicking Improve.")
    else:
        st.info("Optimizing your prompt... please wait ⏳")
        optimizer = PromptOptimizer()
        suggestions = optimizer.suggest_prompts(prompt, num_variants=10, top_k=5)

        st.success("🔧 Here are improved, energy-efficient prompts:")
        for idx, (variant, score) in enumerate(suggestions, 1):
            st.markdown(f"**{idx}.** `{variant}`  &nbsp; _(Similarity: {score.item():.4f})_")
