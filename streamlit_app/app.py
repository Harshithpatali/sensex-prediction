import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import joblib
import logging
from datetime import datetime

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path.cwd()  # assumes Streamlit is run from project root
FEATURES_DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "models" / "scaler.save"

# ---------------------------
# Load model and scaler
# ---------------------------
@st.cache_resource
def load_lstm_model(model_path, scaler_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_lstm_model(MODEL_PATH, SCALER_PATH)

# ---------------------------
# Load features CSV
# ---------------------------
@st.cache_data
def load_features(csv_path):
    df = pd.read_csv(csv_path)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df_features = load_features(FEATURES_DATA_PATH)
feature_cols = [col for col in df_features.columns if col not in ["date", "close"]]

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Sensex LSTM Prediction", layout="wide")
st.title("📈 Sensex Next-Day Closing Price Prediction (LSTM)")

# Show historical chart
st.subheader("Historical Sensex Close Prices")
st.line_chart(df_features.set_index("date")["close"])

st.subheader("Predict Next-Day Close")

# Option to pick a date
date_input = st.date_input(
    "Select date to base prediction on",
    value=datetime.strptime(df_features['date'].iloc[-1], "%Y-%m-%d"),
    min_value=datetime.strptime(df_features['date'].min(), "%Y-%m-%d"),
    max_value=datetime.strptime(df_features['date'].max(), "%Y-%m-%d")
)

# Get the last SEQ_LENGTH days before the selected date
SEQ_LENGTH = 20
try:
    idx = df_features.index[df_features['date'] == pd.Timestamp(date_input)][0]
except IndexError:
    st.error("Selected date not found in dataset.")
    st.stop()

if idx < SEQ_LENGTH:
    st.warning(f"Not enough previous data for SEQ_LENGTH={SEQ_LENGTH}. Using first {SEQ_LENGTH} rows.")
    input_seq = df_features[feature_cols].iloc[0:SEQ_LENGTH].values
else:
    input_seq = df_features[feature_cols].iloc[idx-SEQ_LENGTH:idx].values

# Scale input
input_scaled = scaler.transform(input_seq)
input_scaled = np.expand_dims(input_scaled, axis=0)  # shape (1, SEQ_LENGTH, n_features)

# Predict
predicted_close = model.predict(input_scaled)
predicted_close_value = float(predicted_close[0][0])

st.success(f"Predicted Sensex Close Price for next day: **{predicted_close_value:,.2f}**")
