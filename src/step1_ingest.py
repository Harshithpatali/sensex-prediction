import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Environment variables
TICKER = os.getenv("YFINANCE_TICKER", "^BSESN")
START_DATE = os.getenv("START_DATE", "2000-01-01")
END_DATE = os.getenv("END_DATE", "2025-12-31")
RAW_DATA_PATH = Path("data/raw/sensex_raw.csv")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def ingest_data() -> pd.DataFrame:
    """
    Downloads Sensex historical data and saves it as CSV.

    Returns:
        pd.DataFrame: Downloaded data
    """
    try:
        logging.info(f"Downloading data for {TICKER} from {START_DATE} to {END_DATE}")
        df = yf.download(TICKER, start=START_DATE, end=END_DATE)
        
        if df.empty:
            raise ValueError("No data downloaded. Check ticker or date range.")

        df.reset_index(inplace=True)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logging.info(f"Data saved to {RAW_DATA_PATH}")
        return df

    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise

if __name__ == "__main__":
    df = ingest_data()
    print(df.head())
