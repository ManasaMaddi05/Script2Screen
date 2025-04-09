# from flask import Flask, render_template, request
# import numpy as np
# import joblib
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Load saved models and tools
# genre_model = joblib.load("models/genre_model.pkl")  # MultiOutputClassifier(LogReg)
# revenue_model = joblib.load("models/revenue_model.pkl")  # XGBRegressor
# tfidf = joblib.load("models/tfidf_vectorizer.pkl")
# scaler = joblib.load("models/scaler.pkl")
# mlb = joblib.load("models/mlb.pkl")

# # GPT2 for title generation
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# title_gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
# title_gen_model.eval()

# # Sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()

# # Flask app
# app = Flask(__name__)

# def suggest_title(overview, max_length=12):
#     prompt = f"Movie Overview: {overview}\nSuggested Title:"
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"]

#     with torch.no_grad():
#         outputs = title_gen_model.generate(
#             input_ids,
#             max_length=input_ids.shape[1] + max_length,
#             temperature=0.9,
#             top_k=50,
#             top_p=0.95,
#             do_sample=True,
#             num_return_sequences=1,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     title = generated_text.split("Suggested Title:")[-1].strip().split("\n")[0]
#     return title

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST" and request.form.get("form_type") == "main_predictor":
#         overview = request.form["overview"]
#         budget = float(request.form["budget"])
#         release_year = int(request.form["release_year"])

#         # Sentiment
#         sentiment = analyzer.polarity_scores(overview)['compound']

#         # TF-IDF
#         overview_vector = tfidf.transform([overview]).toarray()

#         # Genre prediction
#         genre_preds = genre_model.predict(overview_vector)

#         # Metadata
#         metadata = scaler.transform([[budget, release_year]])
#         emotion = np.array([[sentiment]])
#         combined = np.hstack([genre_preds, emotion, overview_vector, metadata])

#         # Revenue prediction
#         log_pred = revenue_model.predict(combined)[0]
#         revenue_pred = np.expm1(log_pred)

#         # Title suggestion
#         title = suggest_title(overview)

#         return render_template("index.html", prediction=True, revenue=f"${revenue_pred:,.0f}", suggested_title=title)

#     return render_template("index.html", prediction=False)

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import torch
import cloudpickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Load saved models and tools for Model 1 ---
genre_model = joblib.load("models/genre_model.pkl")
revenue_model = joblib.load("models/revenue_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")
mlb = joblib.load("models/mlb.pkl")

# --- GPT2 for title generation ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
title_gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
title_gen_model.eval()

# --- Sentiment Analyzer ---
analyzer = SentimentIntensityAnalyzer()

# --- Load Model 2: GloVe + XGBoost Classification Pipeline ---
with open("models/final_model.pkl", "rb") as f:
    revenue_category_model = cloudpickle.load(f)

# --- Flask app ---
app = Flask(__name__)

# --- Title Generation Function ---
def suggest_title(overview, max_length=12):
    prompt = f"Movie Overview: {overview}\nSuggested Title:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = title_gen_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    title = generated_text.split("Suggested Title:")[-1].strip().split("\n")[0]
    return title

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = False
    prediction2 = False
    revenue = None
    suggested_title = None
    revenue_category = None

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "main_predictor":
            overview = request.form["overview"]
            budget = float(request.form["budget"])
            release_year = int(request.form["release_year"])

            # Sentiment
            sentiment = analyzer.polarity_scores(overview)['compound']

            # TF-IDF
            overview_vector = tfidf.transform([overview]).toarray()

            # Genre prediction
            genre_preds = genre_model.predict(overview_vector)

            # Metadata
            metadata = scaler.transform([[budget, release_year]])
            emotion = np.array([[sentiment]])
            combined = np.hstack([genre_preds, emotion, overview_vector, metadata])

            # Revenue prediction
            log_pred = revenue_model.predict(combined)[0]
            revenue_pred = np.expm1(log_pred)

            # Title suggestion
            title = suggest_title(overview)

            prediction = True
            revenue = f"${revenue_pred:,.0f}"
            suggested_title = title

        elif form_type == "glove_predictor":
            overview2 = request.form["overview2"]
            budget2 = float(request.form["budget2"])
            runtime2 = float(request.form["runtime2"])

            # Create input DataFrame for Model 2
            input_df = pd.DataFrame([{
                "overview": overview2,
                "budget": budget2,
                "runtime": runtime2,
                "adult": False,
                "genre_list": [],
                "production_companies": [],
                "production_countries": []
            }])

            pred_class = revenue_category_model.predict(input_df)[0]
            class_map = {0: "Flop", 1: "Average", 2: "Hit", 3: "Blockbuster"}
            revenue_category = class_map[int(pred_class)]
            prediction2 = True

    return render_template(
        "index.html",
        prediction=prediction,
        revenue=revenue,
        suggested_title=suggested_title,
        prediction2=prediction2,
        revenue_category=revenue_category
    )

if __name__ == "__main__":
    app.run(debug=True)
