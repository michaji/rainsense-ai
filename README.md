# RainSense AI: Machine Learning for Indigenous Rainfall Forecasting in Ghana

📌 **Project Overview**
Accurate short-term rainfall prediction is critical for agricultural planning, food security, and climate resilience—especially in regions where access to modern meteorological infrastructure is limited. In many rural communities across Ghana, farmers rely on indigenous ecological knowledge such as cloud patterns, wind direction, and other environmental cues to anticipate rainfall. While valuable, these predictions can be subjective and inconsistent.

**RainSense AI** is a machine learning project that bridges indigenous knowledge and modern AI by building a data-driven rainfall forecasting system. Using real-world data collected from Ghanaian farmers, this project predicts rainfall intensity categories (e.g., No Rain, Light, Medium, Heavy) for short-term forecasts, enabling more reliable and explainable decision support.

This project was developed as **Capstone Project 2** for the ML Zoomcamp, following industry-standard practices including data exploration, model selection, explainability, and deployment as a web service.

---

## 🎯 Problem Statement
Smallholder farmers face significant risks due to unpredictable rainfall patterns, which can negatively impact crop yields and livelihoods. Traditional weather forecasts are often unavailable, inaccessible, or unreliable at local scales.

**The goal of this project is to:**
- Build a machine learning model that predicts rainfall intensity using indigenous ecological indicators and observational data.
- Deploy it as a reproducible, production-ready service.

---

## 🤖 Machine Learning Solution
This project frames rainfall prediction as a **multi-class classification problem** using structured tabular data. The solution includes:

- **Data cleaning and preprocessing** of real-world, noisy survey data.
- **Exploratory Data Analysis (EDA)** to understand feature distributions and class imbalance.
- **Feature importance and explainability analysis**.
- **Training and tuning multiple machine learning models**.
- **Selecting the best-performing model** based on evaluation metrics.
- **Exporting the trained model** for inference.
- **Deploying the model as a RESTful API** using Docker.

---

## 🌍 Societal Impact
RainSense AI demonstrates how machine learning can be used responsibly to support:

- **Climate-resilient agriculture**
- **Food security and rural livelihoods**
- **Integration of indigenous knowledge into modern decision systems**
- **Low-infrastructure, scalable forecasting solutions**

By emphasizing explainability and real-world deployment, this project showcases how AI can be applied to socially relevant problems beyond purely commercial use cases.

---

## 📊 Dataset
The dataset used in this project comes from the **Ghana’s Indigenous Intel Challenge** hosted on [Zindi Africa](https://zindi.africa/competitions/ghana-indigenous-intel-challenge).

- **Source:** Zindi Africa
- **Competition:** [Ghana’s Indigenous Intel Challenge](https://zindi.africa/competitions/ghana-indigenous-intel-challenge)
- **Data includes:** Indigenous weather indicators reported by farmers, confidence levels, and corresponding observed rainfall outcomes.

---

## 🧱 Project Scope
This repository contains:

- **Data preparation and exploratory analysis**
- **Model training and evaluation**
- **Model explainability and feature importance analysis**
- **A training pipeline for reproducibility**
- **A prediction service deployed locally using Docker**

**Optional extensions include:**
- Cloud deployment for public access

---
