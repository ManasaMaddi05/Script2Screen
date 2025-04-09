from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize app
app = Flask(__name__)

# Load models and tools
with open("models/genre_model.pkl", "rb") as f:
    genre_model = pickle.load(f)
with open("models/revenue_model.pkl", "rb") as f:
    revenue_model = pickle.load(f)
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("models/bert_model.pkl", "rb") as f:
    bert = pickle.load(f)
with open("models/final_model.pkl", "rb") as f:
    final_model = pickle.load(f)

title_df = pd.read_csv("models/title_reference.csv")
title_overview_embeddings = np.load("models/overview_embeddings.npy")

analyzer = SentimentIntensityAnalyzer()

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         form_type = request.form.get("form_type")

#         if form_type == "main_predictor":
#             overview = request.form.get("overview", "")
#             budget = float(request.form.get("budget", 0))
#             release_year = int(request.form.get("release_year", 0))

#             sentiment = analyzer.polarity_scores(overview)['compound']
#             synopsis_vec = tfidf.transform([overview]).toarray()
#             genre_preds = genre_model.predict(synopsis_vec)
#             genre_display = [g.title() for g in mlb.inverse_transform(genre_preds)[0]]

#             metadata = scaler.transform([[budget, release_year]])
#             full_input = np.hstack([genre_preds, [[sentiment]], synopsis_vec, metadata])
#             log_revenue = revenue_model.predict(full_input)[0]
#             revenue = round(np.expm1(log_revenue))

#             input_embedding = bert.encode([overview])
#             similarities = cosine_similarity(input_embedding, title_overview_embeddings)[0]
#             most_similar_idx = np.argmax(similarities)
#             suggested_title = title_df.iloc[most_similar_idx]['title']

#             return render_template("index.html",
#                                    prediction=True,
#                                    revenue=f"${revenue:,.0f}",
#                                    genres=genre_display,
#                                    suggested_title=suggested_title.title())

#         elif form_type == "glove_predictor":
# In app.py, modify the prediction route:
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "main_predictor":
            overview = request.form.get("overview", "")
            budget = float(request.form.get("budget", 0))
            release_year = int(request.form.get("release_year", 0))

            # Simplified prediction using BERT embeddings only
            input_embedding = bert.encode([overview])
            metadata = np.array([[budget, release_year]])
            full_input = np.hstack([input_embedding, metadata])
            
            try:
                log_revenue = revenue_model.predict(full_input)[0]
                revenue = round(np.expm1(log_revenue))
                
                # Get similar title
                similarities = cosine_similarity(input_embedding, title_overview_embeddings)[0]
                most_similar_idx = np.argmax(similarities)
                suggested_title = title_df.iloc[most_similar_idx]['title']

                return render_template("index.html",
                                   prediction=True,
                                   revenue=f"${revenue:,.0f}",
                                   suggested_title=suggested_title.title())
            except Exception as e:
                return render_template("index.html",
                                   error=f"Prediction failed: {str(e)}")

        elif form_type == "glove_predictor":
            overview2 = request.form.get("overview2", "")
            budget2 = float(request.form.get("budget2", 0))
            runtime2 = float(request.form.get("runtime2", 100))

            X_input = pd.DataFrame([{
                'overview': overview2,
                'budget': budget2,
                'runtime': runtime2,
                'adult': False,
                'genre_list': [],
                'production_companies': [],
                'production_countries': []
            }])

            rev_class = final_model.predict(X_input)[0]
            rev_label_map = {0: "Flop", 1: "Average", 2: "Hit", 3: "Blockbuster"}
            revenue_category = rev_label_map[rev_class]

            return render_template("index.html",
                                   prediction2=True,
                                   revenue_category=revenue_category)

    return render_template("index.html", prediction=False, prediction2=False)

if __name__ == "__main__":
    app.run(debug=True)
