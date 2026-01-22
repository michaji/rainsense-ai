# RainSense AI: Machine Learning for Indigenous Rainfall Forecasting

📌 **Project Overview**  
Accurate short-term rainfall prediction is critical for agricultural planning, food security, and climate resilience—especially in regions where access to modern meteorological infrastructure is limited. In many rural communities across Ghana, farmers rely on indigenous ecological knowledge such as cloud patterns, wind direction, and other environmental cues to anticipate rainfall. While valuable, these predictions can be subjective and inconsistent.

**RainSense AI** is a machine learning project that bridges indigenous knowledge and modern AI by building a data-driven rainfall forecasting system. Using real-world data collected from Ghanaian farmers, this project predicts rainfall intensity categories (e.g., *No Rain, Light, Medium, Heavy*) for short-term forecasts, enabling more reliable and explainable decision support.

This project was developed as **Capstone Project 2** for the **ML Zoomcamp**, following industry-standard practices including data exploration, model selection, reproducible training, and deployment as a web service.

---

## 🎯 Problem Statement
Smallholder farmers face significant risks due to unpredictable rainfall patterns, which can negatively impact crop yields and livelihoods. Traditional weather forecasts are often unavailable, inaccessible, or unreliable at local community scales.

**The goal of this project is to:**
- Build a machine learning model that predicts rainfall intensity using indigenous ecological indicators and observational data.
- Package the solution as a reproducible, production-ready machine learning service.

---

## 🤖 Machine Learning Solution
This project frames rainfall prediction as a **multi-class classification problem** using structured tabular data. The solution includes:

- Data cleaning and preprocessing of real-world, noisy survey data  
- Exploratory Data Analysis (EDA) to understand feature distributions and class imbalance  
- Feature importance and explainability analysis  
- Training and tuning multiple machine learning models  
- Selecting the best-performing model based on evaluation metrics  
- Exporting the trained model for inference  
- Deploying the model as a RESTful API using **FastAPI** and **Docker**

---

## 🌍 Societal Impact
RainSense AI demonstrates how machine learning can be applied responsibly to support:

- Climate-resilient agriculture  
- Food security and rural livelihoods  
- Integration of indigenous knowledge into modern decision systems  
- Low-infrastructure, scalable forecasting solutions  

By emphasizing explainability and real-world deployment, this project highlights the role of AI in solving socially impactful problems beyond purely commercial use cases.

---

## 📊 Dataset
The dataset used in this project comes from the **Ghana’s Indigenous Intel Challenge** hosted on Zindi Africa.

- **Source:** Zindi Africa  
- **Competition:** Ghana’s Indigenous Intel Challenge  
- **Link:** https://zindi.africa/competitions/ghana-indigenous-intel-challenge  
- **Description:** Indigenous weather indicators reported by farmers, confidence levels, forecast horizons, and observed rainfall outcomes.

> 📌 **Note:** Due to licensing restrictions, the dataset is not included directly in this repository.  
Please download it from Zindi and place it in the `data/` directory as described below.

---

## 🧱 Project Structure
```
rainsense-ai/
│
├── data/
│   ├── Train.csv
│
├── models/
│   ├── rainfall_model.pkl   # Trained ML model
│   └── label_encoder.pkl  # Label encoder for target classes
│
├── notebook.ipynb # EDA, feature engineering
├── train.py             # Training script for final model
├── predict.py           # FastAPI inference service
│
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Dependency management with uv
├── Dockerfile           # Containerization for inference
├── README.md            # Project documentation
│
└── screenshots/ # or video for deployment proof

````


---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/rainsense-ai.git
cd rainsense-ai
````

---

### 2️⃣ Download the Dataset

1. Visit: [https://zindi.africa/competitions/ghana-indigenous-intel-challenge](https://zindi.africa/competitions/ghana-indigenous-intel-challenge)
2. Download the competition dataset
3. Place the data files inside:

```bash
data/
```

---

### 3️⃣ Install Dependencies (Using `uv`)

This project uses **uv** for fast and modern dependency management.

```bash
pip install uv
uv pip install -r pyproject.toml
```

Alternatively, you can install directly:

```bash
uv pip install fastapi uvicorn pandas numpy scikit-learn xgboost
```

---

### 4️⃣ Train the Model

Run the training script to generate the trained model artifacts:

```bash
python train.py
```

This will create:

* `models/rainfall_model.pkl` - The trained machine learning pipeline
* `models/label_encoder.pkl` - The tool to convert numbers (0, 1, 2) back to text ("NORAIN", "HEAVYRAIN")

---

### 5️⃣ Run the Prediction API Locally

Start the FastAPI service using Uvicorn:

```bash
uvicorn predict:app --reload --host 0.0.0.0 --port 8000
```

Access:

* **Health check:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 6️⃣ Run with Docker (Optional)

Build and run the containerized service:

```bash
docker build -t rainsense-ai .
docker run -p 8000:8000 rainsense-ai
```

Then open:
- API - http://localhost:8000
- Docs - http://localhost:8000/docs


---
## 🔮 Sample API Request & Response

### 📍 Endpoint
`POST /predict`

---

### 📤 Sample Request (JSON)
```json
{
  "community": "Kintampo",
  "district": "Kintampo North",
  "indicator": "Cloud Cover",
  "confidence": 0.82,
  "predicted_intensity": 2,
  "forecast_length": 6,
  "prediction_time": "2023-01-15 14:00:00"
}
```
### 🔎 Field Descriptions

| Field               | Description                                      |
|---------------------|--------------------------------------------------|
| community           | Name of the local community                      |
| district            | Administrative district                          |
| indicator           | Indigenous weather indicator (e.g., clouds, wind, humidity) |
| confidence          | Confidence score provided by the observer        |
| predicted_intensity | Numeric estimate from indigenous forecast        |
| forecast_length     | Forecast horizon in hours                        |
| prediction_time     | Time the forecast was made                       |
---
### Sample Response (JSON)
```json
{
  "predicted_class": "HEAVYRAIN"
}

```

---

## 📦 Deployment

* The model is deployed locally using Docker
* Cloud deployment (e.g., Render, Railway, or Fly.io) can be added as an extension

---
### 🩺 Health Check & Monitoring
---
- **Health Endpoint:** `/health`
- **Purpose:** Service readiness and monitoring
- **Prediction Logs:** All predictions are logged for auditing and monitoring

Logs are stored locally in: `prediction_logs.log`


This enables traceability, debugging, and future monitoring extensions.

---

## 🧠 Key Skills Demonstrated

* End-to-end machine learning project design
* Feature engineering and model selection
* Production-grade inference with FastAPI
* Dependency management with `uv`
* Containerization with Docker
* Social-impact–driven ML system design

---

## 📌 Author

**Michael Ajiboye**
Machine Learning Engineer | Data Scientist

---

## 📜 License

This project is for educational and portfolio purposes. Dataset usage is subject to Zindi’s competition terms.
