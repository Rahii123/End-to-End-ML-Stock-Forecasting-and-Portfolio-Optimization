import datetime
import logging
from typing import List

from src.extractor.extractor import DataExtractor
from src.processor.processor import DataProcessor
from src.model.model import StockForecaster
from src.optimizer.optimizer import PortfolioOptimizer
from src.database.database import SupabaseDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(symbols: List[str]):
    """
    Executes the full end-to-end ML pipeline.
    """
    logger.info("Starting ML Stock Forecasting Pipeline...")
    
    # 1. Extraction
    extractor = DataExtractor(symbols)
    raw_data = extractor.fetch_historical_data(period="5y")
    
    if raw_data.empty:
        logger.error("No data extracted. Exiting.")
        return

    # 2. Processing
    processor = DataProcessor()
    cleaned_data = processor.align_and_clean_data(raw_data)
    returns = processor.calculate_returns(cleaned_data)

    # 3. Forecasting
    forecaster = StockForecaster()
    predictions = []
    
    for symbol in symbols:
        try:
            prophet_df = processor.prepare_for_prophet(cleaned_data, symbol)
            # Forecast 5-days out using Autoregressive LSTM
            forecast_df, preds_list = forecaster.forecast(prophet_df, periods=5)
            
            for idx, pred_val in enumerate(preds_list):
                pred_date = forecast_df['ds'].iloc[idx].strftime('%Y-%m-%d')
                predictions.append({
                    "symbol": symbol,
                    "prediction_date": pred_date,
                    "predicted_price": float(pred_val)
                })
        except Exception as e:
            logger.error(f"Failed to forecast {symbol}: {e}")

    # 4. Optimization
    optimizer = PortfolioOptimizer()
    try:
        weights = optimizer.optimize_sharpe_ratio(returns)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        weights = {s: 1.0/len(symbols) for s in symbols}

    # 5. Database Storage
    db = SupabaseDatabase()
    try:
        if predictions:
            db.save_predictions(predictions)
        db.save_portfolio_weights(weights)
        logger.info("Pipeline executed and data saved successfully.")
    except Exception as e:
        logger.error(f"Database sync failed: {e}")

if __name__ == "__main__":
    initial_symbols = ["AAPL", "GOOG", "PLTR", "JPM"]
    run_pipeline(initial_symbols)
