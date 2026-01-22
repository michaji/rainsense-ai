import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# -------------------------------------------------------------------------
# 1. Load Artifacts
# -------------------------------------------------------------------------
MODEL_PATH = "models/rainfall_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------------------------------------------------------
# 2. FastAPI App
# -------------------------------------------------------------------------
app = FastAPI(
    title="RainSense AI",
    description="Indigenous Rainfall Forecasting API",
    version="1.0.0"
)

# -------------------------------------------------------------------------
# 3. Request Schema
# -------------------------------------------------------------------------
class RainfallRequest(BaseModel):
    community: str
    district: str
    indicator: str
    confidence: float
    predicted_intensity: int
    forecast_length: int
    prediction_time: str  # ISO format datetime string


# -------------------------------------------------------------------------
# 4. Feature Engineering (MUST MATCH train.py)
# -------------------------------------------------------------------------
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])

    df["month"] = df["prediction_time"].dt.month
    df["day_of_year"] = df["prediction_time"].dt.dayofyear
    df["hour"] = df["prediction_time"].dt.hour
    df["day_of_week"] = df["prediction_time"].dt.dayofweek

    return df


# -------------------------------------------------------------------------
# 5. Prediction Endpoint
# -------------------------------------------------------------------------
@app.post("/predict")
def predict_rainfall(request: RainfallRequest):
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])

    # Feature engineering
    data = extract_date_features(data)

    # Make prediction
    encoded_pred = model.predict(data)
    decoded_pred = label_encoder.inverse_transform(encoded_pred)

    return {
        "rainfall_prediction": decoded_pred[0]
    }


# -------------------------------------------------------------------------
# 6. Health Check
# -------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "RainSense AI is running"}
