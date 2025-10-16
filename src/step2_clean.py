import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

RAW_DATA_PATH = Path("data/raw/sensex_raw.csv")
CLEANED_DATA_PATH = Path("data/processed/sensex_cleaned.csv")

def clean_data() -> pd.DataFrame:
    """
    Cleans raw Sensex data:
    - Handles missing values
    - Drops duplicates
    - Renames columns consistently
    - Converts Date to datetime
    - Validates data quality
    """
    try:
        logging.info(f"Reading raw data from {RAW_DATA_PATH}")
        
        # Check if file exists
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")
        
        df = pd.read_csv(RAW_DATA_PATH)
        
        # Check for empty data
        if df.empty:
            raise ValueError("Raw data CSV is empty.")
        
        logging.info(f"Original data shape: {df.shape}")
        logging.info(f"Original columns: {list(df.columns)}")

        # Store original shape for logging
        original_shape = df.shape

        # Drop duplicates
        logging.info("Dropping duplicates")
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        logging.info(f"Removed {duplicates_removed} duplicate rows")

        # Clean column names
        logging.info("Renaming columns to consistent format")
        df.rename(columns=lambda x: x.strip().replace(" ", "_").lower(), inplace=True)
        logging.info(f"Cleaned columns: {list(df.columns)}")

        # Ensure 'date' column exists and is first column
        if 'date' not in df.columns:
            logging.warning("'date' column not found, using first column as date")
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)

        # Convert date to datetime with error handling
        logging.info("Converting date column to datetime")
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Check for failed date conversions
            invalid_dates = df['date'].isna().sum()
            if invalid_dates > 0:
                logging.warning(f"Found {invalid_dates} rows with invalid dates, these will be dropped")
                df = df.dropna(subset=['date'])
        except Exception as e:
            logging.error(f"Error converting dates: {e}")
            raise

        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)

        # Handle missing values - FIXED: Use ffill() and bfill() instead of fillna(method=)
        logging.info("Handling missing values")
        missing_before = df.isnull().sum().sum()
        
        # Forward fill then backward fill - UPDATED SYNTAX
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill (for any remaining NaNs at start)
        
        missing_after = df.isnull().sum().sum()
        logging.info(f"Missing values handled: {missing_before} before, {missing_after} after")

        # Data validation - FIXED: Ensure numeric columns are converted to float first
        logging.info("Performing data validation")
        
        # Convert numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill any NaN values created during conversion
                df[col] = df[col].ffill().bfill()

        # Check for negative prices (shouldn't exist for stock indices)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logging.warning(f"Found {negative_count} negative values in {col}, taking absolute values")
                    df[col] = df[col].abs()

        # Validate high >= low, high >= open, high >= close
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # Ensure all are numeric before comparison
            invalid_high_low = (df['high'] < df['low']).sum()
            invalid_high_open = (df['high'] < df['open']).sum()
            invalid_high_close = (df['high'] < df['close']).sum()
            
            if any([invalid_high_low, invalid_high_open, invalid_high_close]):
                logging.warning(f"Found inconsistencies in price data: "
                              f"high<low: {invalid_high_low}, "
                              f"high<open: {invalid_high_open}, "
                              f"high<close: {invalid_high_close}")

        # Ensure data quality metrics
        logging.info(f"Final data shape: {df.shape}")
        logging.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logging.info(f"Total trading days: {len(df)}")

        # Save cleaned data
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_DATA_PATH, index=False)
        logging.info(f"Cleaned data saved to {CLEANED_DATA_PATH}")

        # Log summary statistics
        if 'close' in df.columns:
            logging.info(f"Final Close price stats - Min: {df['close'].min():.2f}, "
                        f"Max: {df['close'].max():.2f}, "
                        f"Last: {df['close'].iloc[-1]:.2f}")

        return df

    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        logging.error("Data cleaning failed")
        raise

def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """
    Validate the cleaned data for basic quality checks
    """
    checks = []
    
    # Check if dataframe is not empty
    checks.append(not df.empty)
    
    # Check if date column exists and is datetime
    checks.append('date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']))
    
    # Check if date is sorted
    checks.append(df['date'].is_monotonic_increasing)
    
    # Check for no missing values
    checks.append(df.isnull().sum().sum() == 0)
    
    return all(checks)

if __name__ == "__main__":
    try:
        df_clean = clean_data()
        
        # Validate the cleaned data
        if validate_cleaned_data(df_clean):
            logging.info("✅ Data cleaning completed successfully and passed validation")
        else:
            logging.warning("⚠️ Data cleaning completed but validation checks failed")
        
        print("\nFirst 5 rows of cleaned data:")
        print(df_clean.head())
        print(f"\nData shape: {df_clean.shape}")
        print(f"Columns: {list(df_clean.columns)}")
        
    except Exception as e:
        logging.error(f"Script failed: {e}")