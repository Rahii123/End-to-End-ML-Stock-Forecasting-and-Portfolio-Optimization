from prophet import Prophet
import pandas as pd
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockForecaster:
    """
    Handles time-series forecasting using Facebook Prophet.
    """

    def __init__(self, changepoint_prior_scale: float = 0.05):
        self.changepoint_prior_scale = changepoint_prior_scale

    def forecast(self, df: pd.DataFrame, periods: int = 30) -> Tuple[pd.DataFrame, float]:
        """
        Trains a Prophet model and forecasts future prices.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns.
            periods: Number of days to forecast.
            
        Returns:
            lower_forecast: The full forecast DataFrame.
            next_day_prediction: The predicted price for the next business day.
        """
        logger.info(f"Training Prophet model for {len(df)} data points...")
        
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods, freq='B')
        forecast = model.predict(future)
        
        # Get the prediction for the next day (first row in the future part)
        next_day_prediction = forecast.iloc[-periods]['yhat']
        
        logger.info(f"Forecasting complete. Next day prediction: {next_day_prediction:.2f}")
        return forecast, next_day_prediction

if __name__ == "__main__":
    # Mock data
    import numpy as np
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    data = pd.DataFrame({
        "ds": dates,
        "y": np.random.randn(100).cumsum() + 150
    })
    
    forecaster = StockForecaster()
    forecast_df, pred = forecaster.forecast(data)
    print(f"Predicted next day price: {pred}")
    print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
