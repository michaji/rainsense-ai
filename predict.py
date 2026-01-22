# import pickle
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from datetime import datetime

# # -------------------------------------------------------------------------
# # 1. Load Artifacts
# # -------------------------------------------------------------------------
# MODEL_PATH = "models/rainfall_model.pkl"
# ENCODER_PATH = "models/label_encoder.pkl"

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# with open(ENCODER_PATH, "rb") as f:
#     label_encoder = pickle.load(f)

# # -------------------------------------------------------------------------
# # 2. FastAPI App
# # -------------------------------------------------------------------------
# app = FastAPI(
#     title="RainSense AI",
#     description="Indigenous Rainfall Forecasting API",
#     version="1.0.0"
# )

# # -------------------------------------------------------------------------
# # 3. Request Schema
# # -------------------------------------------------------------------------
# class RainfallRequest(BaseModel):
#     community: str
#     district: str
#     indicator: str
#     confidence: float
#     predicted_intensity: int
#     forecast_length: int
#     prediction_time: str  # ISO format datetime string


# # -------------------------------------------------------------------------
# # 4. Feature Engineering (MUST MATCH train.py)
# # -------------------------------------------------------------------------
# def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
#     df["prediction_time"] = pd.to_datetime(df["prediction_time"])

#     df["month"] = df["prediction_time"].dt.month
#     df["day_of_year"] = df["prediction_time"].dt.dayofyear
#     df["hour"] = df["prediction_time"].dt.hour
#     df["day_of_week"] = df["prediction_time"].dt.dayofweek

#     return df


# # -------------------------------------------------------------------------
# # 5. Prediction Endpoint
# # -------------------------------------------------------------------------
# @app.post("/predict")
# def predict_rainfall(request: RainfallRequest):
#     # Convert request to DataFrame
#     data = pd.DataFrame([request.dict()])

#     # Feature engineering
#     data = extract_date_features(data)

#     # Make prediction
#     encoded_pred = model.predict(data)
#     decoded_pred = label_encoder.inverse_transform(encoded_pred)

#     return {
#         "rainfall_prediction": decoded_pred[0]
#     }


# # -------------------------------------------------------------------------
# # 6. Health Check
# # -------------------------------------------------------------------------
# @app.get("/")
# def health_check():
#     return {"status": "RainSense AI is running"}




import pickle
import logging
from datetime import datetime
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------
app = FastAPI(
    title="RainSense AI API",
    description="Rainfall prediction service using indigenous indicators",
    version="1.0.0"
)

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------------------
# Load Model & Encoder
# ---------------------------------------------------------------------
with open("models/rainfall_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------
class RainfallRequest(BaseModel):
    community: str
    district: str
    indicator: str
    confidence: float
    predicted_intensity: int
    forecast_length: int
    prediction_time: str

# -------------------------------------------------------------------------
#  Feature Engineering (MUST MATCH train.py)
# -------------------------------------------------------------------------
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])

    df["month"] = df["prediction_time"].dt.month
    df["day_of_year"] = df["prediction_time"].dt.dayofyear
    df["hour"] = df["prediction_time"].dt.hour
    df["day_of_week"] = df["prediction_time"].dt.dayofweek

    return df


# ---------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------
@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring and orchestration systems.
    """
    return {
        "status": "RainSense AI is running",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


# ---------------------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------------------
@app.post("/predict")
def predict_rainfall(request: RainfallRequest):
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])
    #data["prediction_time"] = pd.to_datetime(data["prediction_time"])
    # Feature engineering
    data = extract_date_features(data)
    # Generate prediction
    prediction_encoded = model.predict(data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Log prediction for monitoring
    logging.info(
        f"Community={request.community} | "
        f"District={request.district} | "
        f"Indicator={request.indicator} | "
        f"Prediction={prediction_label}"
    )

    return {
        "predicted_class": prediction_label
        # "class_id": int(prediction_encoded)
    }
