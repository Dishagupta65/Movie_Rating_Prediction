import joblib
import pandas as pd
import os

# ===== Paths =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(project_root, "models", "best_movie_rating_model.pkl")

# ===== Load the trained model =====
xgb_model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# ===== Load all saved LabelEncoders =====
encoders = {}
encoder_files = ['Genre', 'Director', 'Actor_1', 'Actor_2', 'Actor_3', 'Name', 'Year', 'Duration']
for name in encoder_files:
    file_path = os.path.join(project_root, "models", f"le_{name}.pkl")
    if os.path.exists(file_path):
        encoders[name] = joblib.load(file_path)

# ===== Safe transform function for unseen labels =====
def safe_transform(le, values):
    transformed = []
    for v in values:
        if v in le.classes_:
            transformed.append(le.transform([v])[0])
        else:
            transformed.append(-1)  # default for unseen values
    return transformed

# ===== New movie data =====
new_movie = pd.DataFrame({
    'Name': ['Example Movie'],
    'Year': [2025],
    'Duration': [120],
    'Genre': ['Action'],
    'Votes': [1000],
    'Director': ['Some Director'],
    'Actor 1': ['Actor A'],
    'Actor 2': ['Actor B'],
    'Actor 3': ['Actor C'],
})

# ===== Encode categorical columns safely =====
new_movie['Genre'] = safe_transform(encoders['Genre'], new_movie['Genre'])
new_movie['Director'] = safe_transform(encoders['Director'], new_movie['Director'])
new_movie['Actor 1'] = safe_transform(encoders['Actor_1'], new_movie['Actor 1'])
new_movie['Actor 2'] = safe_transform(encoders['Actor_2'], new_movie['Actor 2'])
new_movie['Actor 3'] = safe_transform(encoders['Actor_3'], new_movie['Actor 3'])
new_movie['Name'] = safe_transform(encoders['Name'], new_movie['Name'])
new_movie['Year'] = safe_transform(encoders['Year'], new_movie['Year'])
new_movie['Duration'] = safe_transform(encoders['Duration'], new_movie['Duration'])

# ===== Predict =====
predicted_rating = xgb_model.predict(new_movie)
print("Predicted Rating:", predicted_rating[0])
