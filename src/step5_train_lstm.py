import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
import os

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------------------
# Paths (Windows-safe)
# ---------------------------
BASE_DIR = Path.cwd()  # assumes script is run from project root: D:\sensex-prediction
FEATURES_DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_SAVE_PATH = BASE_DIR / "models"

# MLflow tracking (Windows-safe)
MLFLOW_TRACKING_URI = f"file:///{(BASE_DIR / 'mlruns').as_posix()}"
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SensexPrediction")

print(f"Features path: {FEATURES_DATA_PATH}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# ---------------------------
# LSTM Hyperparameters
# ---------------------------
SEQ_LENGTH = 20
EPOCHS = 50
BATCH_SIZE = 16

# ---------------------------
# Helper functions
# ---------------------------

def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Prepare sequences for LSTM."""
    logging.info("Scaling features")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    for i in range(SEQ_LENGTH, len(df)):
        X.append(data_scaled[i-SEQ_LENGTH:i])
        y.append(df[target_col].iloc[i])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm(input_shape):
    """Build and compile LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------
# Main function
# ---------------------------

def main():
    # Check features file
    if not FEATURES_DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_DATA_PATH}")

    logging.info(f"Reading features from {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [col for col in df.columns if col not in ["date", "close"]]
    target_col = "close"

    # Prepare data
    X, y, scaler = prepare_data(df, feature_cols, target_col)
    logging.info(f"Prepared data shapes: X={X.shape}, y={y.shape}")

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LSTM_Sensex"):
        mlflow.log_param("seq_length", SEQ_LENGTH)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # Build LSTM
        model = build_lstm(input_shape=(X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        logging.info("Training LSTM model...")
        history = model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )

        # Log metrics
        final_loss = history.history["loss"][-1]
        mlflow.log_metric("final_loss", final_loss)
        logging.info(f"Training completed. Final loss: {final_loss:.6f}")

        # Log model to MLflow
        mlflow.tensorflow.log_model(model, artifact_path="lstm_model")

        # Save locally for Streamlit
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_PATH / "lstm_model.h5")
        joblib.dump(scaler, MODEL_SAVE_PATH / "scaler.save")
        logging.info(f"✅ Model and scaler saved to: {MODEL_SAVE_PATH}")

    logging.info("✅ LSTM training and saving complete.")

# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    main()
