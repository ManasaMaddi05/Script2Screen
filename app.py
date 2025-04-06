import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer

# Set Gemini API Key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load models and vectorizer
revenue_model = joblib.load("revenue_model.pkl")
roi_model = joblib.load("roi_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ROI label mapping
roi_map = {0: "Flop", 1: "Hit", 2: "Superhit", 3: "Blockbuster"}

# Streamlit UI
st.set_page_config(page_title="Script2Screen 🎬", layout="centered")
st.title("🎬 Script2Screen")
st.markdown("**Predict your movie's success & generate a poster based on its synopsis.**")

synopsis = st.text_area("📜 Enter your movie synopsis below:", height=250)

if st.button("🚀 Predict & Generate"):
    if synopsis.strip() == "":
        st.warning("Please enter a synopsis to continue.")
    else:
        # --- Predict Success ---
        X = tfidf.transform([synopsis])
        log_revenue = revenue_model.predict(X)[0]
        predicted_revenue = np.expm1(log_revenue)
        roi_pred = roi_model.predict(X)[0]
        roi_label = roi_map.get(roi_pred, "Unknown")

        # --- Show Predictions ---
        st.subheader("📊 Predictions")
        st.success(f"💰 **Predicted Revenue**: ${predicted_revenue:,.2f}")
        st.info(f"🏆 **Predicted Success Metric**: {roi_label}")

        # --- Generate Image with Gemini ---
        st.subheader("🎨 AI-Generated Poster")
        with st.spinner("Generating visual..."):
            try:
                model = genai.GenerativeModel("models/gemini-pro-vision")
                response = model.generate_content(f"Create a cinematic movie poster based on this synopsis: {synopsis}")
                st.image(response._result.candidates[0].content.parts[0].inline_data.data, caption="AI-generated poster", use_column_width=True)
            except Exception as e:
                st.error(f"❌ Failed to generate image: {e}")
