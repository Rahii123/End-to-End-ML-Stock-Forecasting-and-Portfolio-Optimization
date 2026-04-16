import pandas as pd
import numpy as np
import requests
import time
import os
from typing import List, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    """
    Handles data extraction from Alpha Vantage Official Financial API.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    def fetch_historical_data(self, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch real historical daily close data for the given symbols from Alpha Vantage.
        """
        if not self.api_key:
            logger.error("Alpha Vantage API Key missing from .env! Falling back to synthetic data.")
            return self._generate_synthetic_data(period)

        logger.info(f"Fetching real historical data from Alpha Vantage for {self.symbols}...")
        try:
            adj_close_list = []
            for symbol in self.symbols:
                logger.info(f"Downloading real data for {symbol}...")
                
                # outputsize='compact' gives us 100 days of data and is FREE.
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={self.api_key}"
                
                response = requests.get(url)
                data = response.json()
                
                # Check if API returned an error or rate limit warning
                if "Information" in data and "rate limit" in data["Information"].lower():
                    logger.error(f"Alpha Vantage Rate Limit Hit: {data['Information']}")
                    break
                
                if "Time Series (Daily)" in data:
                    # Convert complex JSON into a clean Pandas DataFrame
                    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index() # Sort oldest to newest
                    
                    # Extract just the closing price and name the column after the stock symbol
                    close_series = df['4. close'].astype(float).rename(symbol)
                    
                    # If we only want 5 years, filter the date range
                    if "5y" in period:
                        cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=5)
                        close_series = close_series[close_series.index >= cutoff_date]
                    
                    adj_close_list.append(close_series)
                else:
                    logger.warning(f"Failed to retrieve {symbol}. API said: {data}")
                
                # CRITICAL: Alpha Vantage free tier only allows a few requests per minute.
                # We must sleep for at least 12 seconds between downloads or they will ban the request.
                logger.info("Sleeping for 15 seconds to respect Alpha Vantage free tier limits...")
                time.sleep(15)

            if not adj_close_list:
                logger.error("No real data pulled. Falling back to synthetic.")
                return self._generate_synthetic_data(period)

            # Merge all stocks side-by-side
            combined_df = pd.concat(adj_close_list, axis=1)
            
            logger.info("Real Data extraction successful!")
            return combined_df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return self._generate_synthetic_data(period)

    def _generate_synthetic_data(self, period: str) -> pd.DataFrame:
        """Generates realistic stock data using Geometric Brownian Motion if API fails."""
        logger.warning(f"Generating realistic synthetic data for {self.symbols} to ensure pipeline execution.")
        
        days = 1260 if "5y" in period else 252
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')
        
        synth_data = {}
        for symbol in self.symbols:
            returns = np.random.normal(loc=0.0005, scale=0.02, size=days)
            price_path = 150 * np.exp(np.cumsum(returns))
            synth_data[symbol] = price_path
            
        df = pd.DataFrame(synth_data, index=dates)
        return df

if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "GOOG", "PLTR", "JPM"]
    extractor = DataExtractor(symbols)
    df = extractor.fetch_historical_data()
    print(df.tail())
