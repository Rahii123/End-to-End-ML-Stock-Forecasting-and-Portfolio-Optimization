import yfinance as yf
import pandas as pd
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    """
    Handles data extraction from Yahoo Finance (yfinance).
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    def fetch_historical_data(self, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical OHLCV data for the given symbols.
        
        Args:
            period: Data period to download (e.g., '1y', '5y', 'max').
            interval: Data interval (e.g., '1d', '1wk', '1mo').
            
        Returns:
            pd.DataFrame: Multi-indexed DataFrame (Date, Ticker) with Adj Close prices.
        """
        logger.info(f"Fetching historical data for {self.symbols} over {period}...")
        try:
            # Download all symbols at once
            data = yf.download(
                tickers=self.symbols,
                period=period,
                interval=interval,
                group_by='column',
                auto_adjust=True
            )
            
            if data.empty:
                logger.warning("No data found for the given symbols.")
                return pd.DataFrame()

            # We primarily need Adjusted Close for stock analysis
            # In recent yfinance, if auto_adjust=True, 'Close' is the adjusted close.
            adj_close = data['Close']
            
            # If only one symbol, yf.download might return a Series. 
            # We want a DataFrame with columns as symbols.
            if isinstance(adj_close, pd.Series):
                adj_close = adj_close.to_frame(name=self.symbols[0])
                
            logger.info("Data extraction successful.")
            return adj_close

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "GOOG", "PLTR", "JPM"]
    extractor = DataExtractor(symbols)
    df = extractor.fetch_historical_data()
    print(df.tail())
