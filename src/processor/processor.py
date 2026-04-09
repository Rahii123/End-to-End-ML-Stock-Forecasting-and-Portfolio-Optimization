import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data cleaning, normalization, and preparation for Prophet.
    """

    def align_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes NaNs and ensures all symbols have aligned dates.
        """
        logger.info("Aligning and cleaning data...")
        # Drop rows where any stock has missing data
        cleaned_df = df.dropna()
        logger.info(f"Data cleaned. Remaining data points: {len(cleaned_df)}")
        return cleaned_df

    def prepare_for_prophet(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Prepares a single symbol's data for Prophet.
        Prophet requires columns 'ds' (date) and 'y' (target).
        """
        if symbol not in df.columns:
            raise ValueError(f"Symbol {symbol} not found in DataFrame columns.")
        
        prophet_df = df[[symbol]].reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure 'ds' is timezone-naive as Prophet prefers it
        if prophet_df['ds'].dt.tz is not None:
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            
        return prophet_df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates daily percentage returns for portfolio optimization.
        """
        return df.pct_change().dropna()

if __name__ == "__main__":
    # Mock data for demonstration
    import numpy as np
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "AAPL": np.random.randn(10).cumsum() + 150,
        "GOOG": np.random.randn(10).cumsum() + 2800
    }, index=dates)
    
    processor = DataProcessor()
    prophet_aapl = processor.prepare_for_prophet(data, "AAPL")
    print("Prophet format for AAPL:")
    print(prophet_aapl.head())
    
    returns = processor.calculate_returns(data)
    print("\nDaily returns:")
    print(returns.head())
