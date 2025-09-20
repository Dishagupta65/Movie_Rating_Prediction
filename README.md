# Movie Rating Prediction

This project predicts movie ratings using machine learning models (Linear Regression, Random Forest, XGBoost).

## Folder Structure

- `data/` : Contains the CSV dataset
- `models/` : Contains trained models and label encoders
- `src/` : Python scripts for training and prediction
- `README.md` : Project description

## How to Run

1. Make sure you have Python and required packages installed.
2. Run `main.py` to train models and generate best model.
3. Run `src/predict.py` to predict rating for a new movie.

## Example

```python
import pandas as pd
import joblib

# Load model
xgb_model = joblib.load("models/best_movie_rating_model.pkl")

# Predict rating for a new movie
new_movie = pd.DataFrame({
    'Genre': ['Action'],
    'Director': ['Some Director'],
    'Actor 1': ['Actor A'],
    'Actor 2': ['Actor B'],
    'Actor 3': ['Actor C'],
    # Include other required features if any
})

predicted_rating = xgb_model.predict(new_movie)
print("Predicted Rating:", predicted_rating[0])
