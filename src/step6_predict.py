import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os
import mlflow
import mlflow.tensorflow

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

FEATURES_DATA_PATH = Path("data/processed/features.csv")
LSTM_MODEL_PATH = Path("data/models/lstm_sensex_model")
XGB_MODEL_PATH = Path("data/models/xgboost_sensex_model.joblib")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SensexPrediction")

SEQ_LENGTH = 20  # same as training

def predict_lstm(df: pd.DataFrame):
    logging.info("Predicting with LSTM model")
    # Load scaler and model
    scaler = joblib.load(LSTM_MODEL_PATH / "scaler.save")
    model = load_model(str(LSTM_MODEL_PATH / "lstm_model"))

    feature_cols = [col for col in df.columns if col not in ["date", "close"]]
    data_scaled = scaler.transform(df[feature_cols])

    # Prepare sequences
    X_test = []
    y_true = df["close"].values[SEQ_LENGTH:]
    for i in range(SEQ_LENGTH, len(df)):
        X_test.append(data_scaled[i-SEQ_LENGTH:i])
    X_test = np.array(X_test)

    # Predict
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()

    # Evaluate
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logging.info(f"LSTM Evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return y_pred, y_true

def predict_xgboost(df: pd.DataFrame):
    logging.info("Predicting with XGBoost model")
    from xgboost import XGBRegressor

    # Load model
    xgb_model = joblib.load(XGB_MODEL_PATH)
    feature_cols = [col for col in df.columns if col not in ["date", "close"]]
    X = df[feature_cols].iloc[:-1]
    y_true = df["close"].iloc[1:].values

    y_pred = xgb_model.predict(X)

    # Evaluate
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logging.info(f"XGBoost Evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return y_pred, y_true

def main():
    df = pd.read_csv(FEATURES_DATA_PATH)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # LSTM predictions
    lstm_pred, lstm_true = predict_lstm(df)

    # XGBoost predictions
    if XGB_MODEL_PATH.exists():
        xgb_pred, xgb_true = predict_xgboost(df)
    else:
        logging.warning(f"XGBoost model not found at {XGB_MODEL_PATH}")

if __name__ == "__main__":
    main()
