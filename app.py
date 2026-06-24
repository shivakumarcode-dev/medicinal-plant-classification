import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
import time
import urllib.parse
from chatbot.ollama_chatbot import ask_tinyllama

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Medicinal Plant AI Classifier", layout="wide")

# -----------------------------
# CUSTOM UI (GREEN THEME)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(90deg,#d8f3dc,#b7e4c7);
}

h1, h2, h3, h4, h5 {
    color: #1b4332;
}

.stButton>button {
    background: #2d6a4f;
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
}

section[data-testid="stExpander"] {
    background: #edf6f9;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🌿 Medicinal Plant AI Classifier + Chatbot")

# -----------------------------
# FILE PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_model.h5")
PLANT_DB_PATH = os.path.join(BASE_DIR, "plant_info.json")

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(PLANT_DB_PATH, "r") as f:
    plant_info = json.load(f)

CLASS_NAMES = list(plant_info.keys())

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def confidence_color(conf):
    if conf >= 80:
        return "#2d6a4f"
    elif conf >= 50:
        return "#f4a261"
    else:
        return "#e63946"

def find_plant_key(name):
    name_clean = name.lower().strip()
    for key in plant_info.keys():
        if name_clean in key.lower():
            return key
    return None

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["📷 Image Classification", "🤖 AI Chatbot"])

# =================================================
# TAB 1: IMAGE CLASSIFICATION
# =================================================
with tab1:

    st.subheader("Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Prediction
        img_array = preprocess_image(image)
        start = time.time()
        predictions = model.predict(img_array, verbose=0)[0]  # ✅ Fix TF warning
        end = time.time()

        top5_idx = predictions.argsort()[-5:][::-1]

        top_predictions = [
            {"name": CLASS_NAMES[i], "confidence": float(predictions[i]*100)}
            for i in top5_idx
        ]

        best_index = top5_idx[0]
        plant_name = CLASS_NAMES[best_index]
        confidence = predictions[best_index]*100
        prediction_time = end - start

        st.session_state["plant"] = plant_name

        st.markdown("---")

        col1, col2 = st.columns([1, 1.2])

        # IMAGE LEFT
        with col1:
            st.image(image, width='stretch')  # ✅ permanently fixes use_column_width warning

        # RESULT CARD (SMALLER PLANT NAME)
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg,#2d6a4f,#40916c);
                padding:15px;
                border-radius:12px;
                color:white;
                text-align:center;">
                <h4 style="margin-bottom:5px; font-size:20px;">{plant_name}</h4>
                <h5 style="margin:3px 0; font-size:16px;">{confidence:.2f}% Confidence</h5>
                <p style="font-size:12px;">⏱ {prediction_time:.2f} sec</p>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # SHARE BUTTON
        # -----------------------------
        share_text = f"🌿 I identified a medicinal plant!\n\nPlant: {plant_name}\nConfidence: {confidence:.2f}%"
        encoded_text = urllib.parse.quote(share_text)

        st.markdown("### 🔗 Share Result")

        st.markdown(f"""
        <a href="https://wa.me/?text={encoded_text}" target="_blank">
            <button style="
                background:#25D366;
                color:white;
                border:none;
                padding:10px 20px;
                border-radius:8px;
                cursor:pointer;">
                📤 Share on WhatsApp
            </button>
        </a>
        """, unsafe_allow_html=True)

        # -----------------------------
        # TOP PREDICTIONS
        # -----------------------------
        st.markdown("## 🔝 Top Predictions")

        for p in top_predictions:
            bar_color = confidence_color(p["confidence"])

            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between;">
                    <span>{p["name"]}</span>
                    <span>{p["confidence"]:.2f}%</span>
                </div>
                <div style="background:#d8f3dc; border-radius:10px;">
                    <div style="
                        width:{p['confidence']}%;
                        background:{bar_color};
                        padding:6px;
                        border-radius:10px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # PLANT DETAILS
        # -----------------------------
        plant_key = find_plant_key(plant_name)

        if plant_key:
            info = plant_info[plant_key]

            st.markdown("## 🌿 Complete Information")

            with st.expander("📌 Basic Info"):
                st.write("Scientific Name:", info.get("scientific_name"))
                st.write("Family:", info.get("family"))

            with st.expander("📖 Description"):
                st.write(info.get("description"))

            with st.expander("💊 Uses & Benefits"):
                for u in info.get("uses", []):
                    st.write("✔", u)

            with st.expander("🧪 Parts Used"):
                st.write(info.get("parts_used"))

            with st.expander("⚠️ Dosage"):
                st.write(info.get("dosage"))

            with st.expander("🚫 Precautions"):
                st.write(info.get("precautions"))

            # Download
            result_text = f"""
Plant: {plant_name}
Confidence: {confidence:.2f}%
Time: {prediction_time:.2f}s
"""
            st.download_button("📥 Download Result", result_text)

# =================================================
# TAB 2: CHATBOT
# =================================================
with tab2:

    st.subheader("Ask About Plants 🌿")
    question = st.text_input("Type your question")

    if st.button("Ask AI"):
        if "plant" not in st.session_state:
            st.warning("Please classify a plant first.")
        else:

            plant_key = find_plant_key(st.session_state["plant"])
            info = plant_info[plant_key]

            with st.spinner("🤖 Thinking with tinyllama..."):
                answer = ask_tinyllama(st.session_state["plant"], info, question)

            st.markdown("### 🤖 AI Answer")
            st.success(answer)