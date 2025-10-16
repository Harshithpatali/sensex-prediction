import pandas as pd
import numpy as np
import logging
from pathlib import Path
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator, WilliamsRIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

CLEANED_DATA_PATH = Path("data/processed/sensex_cleaned.csv")
FEATURES_DATA_PATH = Path("data/processed/features.csv")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 20+ quantitative finance features.
    """
    try:
        logging.info("Creating features...")

        # Ensure sorted by date
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Prices
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df['volume']

        # ----------------------
        # Trend indicators
        # ----------------------
        df['sma_5'] = close.rolling(5).mean()
        df['sma_10'] = close.rolling(10).mean()
        df['ema_5'] = close.ewm(span=5, adjust=False).mean()
        df['ema_10'] = close.ewm(span=10, adjust=False).mean()

        macd = MACD(close)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        adx = ADXIndicator(high, low, close)
        df['adx'] = adx.adx()

        cci = CCIIndicator(high, low, close)
        df['cci'] = cci.cci()

        # ----------------------
        # Momentum indicators
        # ----------------------
        rsi = RSIIndicator(close)
        df['rsi'] = rsi.rsi()

        willr = WilliamsRIndicator(high, low, close)
        df['willr'] = willr.williams_r()

        stoch = StochasticOscillator(high, low, close)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # ----------------------
        # Volatility indicators
        # ----------------------
        bb = BollingerBands(close)
        df['bb_h'] = bb.bollinger_hband()
        df['bb_l'] = bb.bollinger_lband()
        atr = AverageTrueRange(high, low, close)
        df['atr'] = atr.average_true_range()

        # ----------------------
        # Volume indicators
        # ----------------------
        obv = OnBalanceVolumeIndicator(close, volume)
        df['obv'] = obv.on_balance_volume()

        vwap = VolumeWeightedAveragePrice(high, low, close, volume)
        df['vwap'] = vwap.volume_weighted_average_price()

        # ----------------------
        # Returns & lag features
        # ----------------------
        df['daily_return'] = close.pct_change()
        for lag in range(1, 6):
            df[f'close_lag_{lag}'] = close.shift(lag)

        # Fill missing values (from rolling calculations)
        df.fillna(method='bfill', inplace=True)

        logging.info(f"Features created successfully. Total columns: {len(df.columns)}")
        return df

    except Exception as e:
        logging.error(f"Error creating features: {e}")
        raise

def main():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df_features = create_features(df)

    # Save features
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(FEATURES_DATA_PATH, index=False)
    logging.info(f"Features saved to {FEATURES_DATA_PATH}")
    print(df_features.head())

if __name__ == "__main__":
    main()
