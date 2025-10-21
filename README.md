# sensex-prediction
Production-level Sensex prediction project using LSTM and XGBoost with Streamlit, Airflow, MLflow, and GCP deployment
# 📈 Sensex Prediction Project — LSTM + XGBoost | Streamlit | Airflow | MLflow | GCP

A **production-level stock market prediction pipeline** built to forecast **Sensex index prices** using **LSTM** and **XGBoost models**.  
This project demonstrates a **complete MLOps workflow** — from **data ingestion to cloud deployment** — with **Airflow orchestration**, **MLflow tracking**, and a **Streamlit web app** for real-time visualization.

---

## 🚀 **Project Overview**

This project automates the entire **machine learning lifecycle** for Sensex prediction:

1. **ETL Pipeline** – Ingests, cleans, and stores Sensex data in a PostgreSQL database.  
2. **Feature Engineering** – Generates over 20 quantitative indicators and temporal features.  
3. **Model Training** – Trains and compares **LSTM** (for time-series trends) and **XGBoost** (for tabular insights).  
4. **Experiment Tracking** – Logs metrics, parameters, and artifacts in **MLflow**.  
5. **Automation** – Uses **Apache Airflow** to schedule daily data updates and retraining.  
6. **Deployment** – Exposes predictions through a **Streamlit dashboard**, containerized with **Docker** and deployed on **Google Cloud Platform (GCP)**.

---

## 🧠 **Key Features**

✅ Automated ETL with **Airflow DAGs**  
✅ Dual-model approach — **LSTM + XGBoost**  
✅ Real-time prediction through **Streamlit UI**  
✅ **MLflow** for experiment tracking & model registry  
✅ Containerized using **Docker**  
✅ Cloud deployment on **Google Cloud Run**  
✅ Modular, scalable, and production-ready architecture  

---

## 🏗️ **Architecture**

      ┌──────────────────────────────┐
      │     Data Source (NSE API)    │
      └────────────┬─────────────────┘
                   │
         ┌─────────▼──────────┐
         │   Airflow DAGs     │
         │  (ETL + Training)  │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  PostgreSQL DB     │
         │ (Cleaned Dataset)  │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Model Training     │
         │ (LSTM + XGBoost)    │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │     MLflow          │
         │ (Metrics & Models)  │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   Streamlit UI      │
         │ (Visualization)     │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │    Docker + GCP     │
         │ (Cloud Deployment)  │
         └─────────────────────┘



---

## ⚙️ **Tech Stack**

| Category | Tools / Technologies |
|-----------|----------------------|
| Programming | Python, Pandas, NumPy |
| ML Models | LSTM (TensorFlow/Keras), XGBoost |
| Data Storage | PostgreSQL |
| Workflow Automation | Apache Airflow |
| Experiment Tracking | MLflow |
| Frontend | Streamlit |
| Containerization | Docker |
| Cloud Platform | Google Cloud Run (GCP) |

---

## 📂 **Project Structure**



📊 Streamlit Dashboard Highlights

✅ Interactive visualization of actual vs. predicted Sensex prices
✅ Historical data trends and feature correlations
✅ Model performance metrics (RMSE, MAE, R²)
✅ Option to switch between LSTM and XGBoost predictions🧾 MLflow Tracking Example

All experiments are logged via MLflow:

Model parameters (epochs, learning rate, etc.)

Training/validation metrics

Stored model artifacts for versioning

Model registry for production deployment



