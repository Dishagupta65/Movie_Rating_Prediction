# ===== Step 1: Import Required Libraries =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ===== Step 2: Load Dataset =====
DATA_PATH = "data/imdb_movies_india.csv"

print("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH, encoding='latin1')
    print("✅ CSV loaded successfully!")
    print("Shape of dataset:", df.shape)
    print("First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"❌ Error: File not found! Make sure '{DATA_PATH}' exists.")
    exit()
except Exception as e:
    print("❌ An error occurred while reading the CSV:", e)
    exit()

# ===== Step 3: Handle 'Votes' Column =====
if 'Votes' in df.columns:
    df['Votes'] = df['Votes'].replace(',', '', regex=True)
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)

# ===== Step 4: Handle Missing Values =====
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ===== Step 5: Encode Categorical Columns =====
print("\nEncoding categorical columns...")
label_encoders = {}
for col in df.select_dtypes(include=[object]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded: {col}")

# ===== Step 6: Save LabelEncoders =====
os.makedirs("models", exist_ok=True)
for col, le in label_encoders.items():
    filename = f"models/le_{col.replace(' ', '_')}.pkl"
    joblib.dump(le, filename)
    print(f"✅ Saved LabelEncoder for {col} -> {filename}")

# ===== Step 7: Define Features and Target =====
if 'Rating' not in df.columns:
    print("❌ Error: 'Rating' column not found in dataset.")
    exit()

X = df.drop(columns=['Rating'])
y = df['Rating']

# ===== Step 8: Split Data =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nData split into training and testing sets:")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# ===== Step 9: Model Evaluation Function =====
def evaluate_model(model, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"\n===== {model_name} Performance =====")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return mae, rmse, r2

# ===== Step 10: Train Models =====
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, "Linear Regression")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, "Random Forest Regressor")

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, "XGBoost Regressor")

# ===== Step 11: Feature Importance (Random Forest) =====
print("\nPlotting feature importance...")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ===== Step 12: Save Best Model =====
joblib.dump(xgb_model, "models/best_movie_rating_model.pkl")
print("\n✅ Best model saved as 'models/best_movie_rating_model.pkl'")
