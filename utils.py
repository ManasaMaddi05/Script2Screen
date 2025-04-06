# utils.py
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Load All Saved Models & Tools
# -----------------------
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

title_df = pd.read_csv("models/title_reference.csv")
stored_embeddings = np.load("models/overview_embeddings.npy")

# -----------------------
# Vader Sentiment
# -----------------------
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# -----------------------
# Prediction Function
# -----------------------
def predict_revenue_and_title(overview, budget, release_year):
    # Step 1: Sentiment
    sentiment = analyzer.polarity_scores(overview)['compound']

    # Step 2: TF-IDF of overview
    synopsis_vector = tfidf.transform([overview]).toarray()

    # Step 3: Predict genres
    predicted_genres = genre_model.predict(synopsis_vector)

    # Step 4: Metadata
    metadata = scaler.transform([[budget, release_year]])

    # Step 5: Combine all features
    final_features = np.hstack([predicted_genres, [[sentiment]], synopsis_vector, metadata])

    # Step 6: Predict revenue
    log_revenue = revenue_model.predict(final_features)[0]
    revenue = np.expm1(log_revenue)  # convert back from log scale

    # Step 7: Suggest title using cosine similarity
    embed = bert.encode([overview])
    similarities = cosine_similarity(embed, stored_embeddings)
    most_similar_idx = similarities.argmax()
    suggested_title = title_df.iloc[most_similar_idx]['title']

    return revenue, suggested_title, predicted_genres








# # utils.py
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity

# # Load saved components
# with open("revenue_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("bert_model.pkl", "rb") as f:
#     bert = pickle.load(f)

# title_df = pd.read_csv("title_reference.csv")
# stored_embeddings = np.load("overview_embeddings.npy")

# def predict_revenue_and_title(overview, budget, release_year):
#     embed = bert.encode([overview])
#     X = np.hstack([embed, [[budget, release_year]]])
    
#     # Predict revenue
#     revenue = model.predict(X)[0]
    
#     # Suggest title using cosine similarity
#     similarities = cosine_similarity(embed, stored_embeddings)
#     most_similar_idx = similarities.argmax()
#     suggested_title = title_df.iloc[most_similar_idx]['title']
    
#     return revenue, suggested_title
