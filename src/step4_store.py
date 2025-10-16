import pandas as pd
import logging
import os
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Paths
FEATURES_DATA_PATH = Path("data/processed/features.csv")

# PostgreSQL connection variables
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "newpassword")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sensex_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

TABLE_NAME = "sensex_features"

def store_features_to_postgres():
    """
    Store features CSV to PostgreSQL table.
    """
    try:
        # Read features CSV
        logging.info(f"Reading features from {FEATURES_DATA_PATH}")
        df = pd.read_csv(FEATURES_DATA_PATH)

        if df.empty:
            raise ValueError("Features CSV is empty.")

        # Create SQLAlchemy engine
        engine_str = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        engine = create_engine(engine_str)
        logging.info(f"Connecting to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

        # Store data
        df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
        logging.info(f"Data successfully stored in table '{TABLE_NAME}'")

    except Exception as e:
        logging.error(f"Error storing features to PostgreSQL: {e}")
        raise

if __name__ == "__main__":
    store_features_to_postgres()
