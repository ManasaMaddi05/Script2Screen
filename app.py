import streamlit as st
import numpy as np
import joblib
import openai

# Set OpenAI Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load models
revenue_model = joblib.load("revenue_model.pkl")
roi_model = joblib.load("roi_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# UI
st.set_page_config(page_title="Script2Screen 🎬", layout="centered")
st.title("🎬 Script2Screen")
st.markdown("**Predict your movie's box office success and see its poster come to life!**")

synopsis = st.text_area("✏️ Enter your movie synopsis:", height=250)

if st.button("🚀 Predict & Generate"):
    if synopsis.strip() == "":
        st.warning("Please enter a synopsis.")
    else:
        # Preprocess
        X = tfidf.transform([synopsis])

        # Revenue Prediction
        log_revenue = revenue_model.predict(X)[0]
        revenue = np.expm1(log_revenue)

        # ROI Classification
        roi_pred = roi_model.predict(X)[0]
        roi_map = {0: "Flop", 1: "Hit", 2: "Superhit", 3: "Blockbuster"}
        roi_label = roi_map.get(roi_pred, "Unknown")

        # Output
        st.subheader("📊 Predictions")
        st.success(f"💰 **Estimated Revenue**: ${revenue:,.2f}")
        st.info(f"🏆 **Predicted Success Level**: {roi_label}")

        # AI Poster Generation
        st.subheader("🎨 AI-Generated Poster")
        with st.spinner("Generating image..."):
            try:
                image_response = openai.Image.create(
                    prompt=synopsis,
                    n=1,
                    size="512x512"
                )
                image_url = image_response["data"][0]["url"]
                st.image(image_url, caption="AI-generated movie poster")
            except Exception as e:
                st.error(f"Image generation failed: {e}")
