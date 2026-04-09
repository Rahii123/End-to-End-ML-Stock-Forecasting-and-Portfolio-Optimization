import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseDatabase:
    """
    Handles connection and operations with Supabase.
    """

    def __init__(self):
        self.url: str = os.getenv("SUPABASE_URL", "")
        self.key: str = os.getenv("SUPABASE_KEY", "")
        
        if not self.url or not self.key:
            logger.warning("Supabase credentials missing. Database operations will fail.")
            self.client = None
        else:
            self.client: Optional[Client] = create_client(self.url, self.key)

    def save_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Saves a list of predictions to the 'predictions' table.
        Each prediction should have keys: symbol, prediction_date, predicted_price.
        """
        if not self.client:
            logger.error("Supabase client not initialized.")
            return

        try:
            logger.info(f"Saving {len(predictions)} predictions to Supabase...")
            # Supabase upsert requires a primary key or unique constraint (e.g., symbol + date)
            response = self.client.table("predictions").upsert(predictions).execute()
            logger.info("Predictions saved successfully.")
            return response
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise

    def get_latest_predictions(self) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent predictions from the database.
        """
        if not self.client:
            logger.error("Supabase client not initialized.")
            return []

        try:
            response = self.client.table("predictions").select("*").order("prediction_date", desc=True).limit(20).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching predictions: {e}")
            return []

    def save_portfolio_weights(self, weights: Dict[str, float]):
        """
        Saves optimized weights to the 'portfolio_weights' table.
        """
        if not self.client:
            return

        try:
            data = [{"symbol": s, "weight": w} for s, w in weights.items()]
            self.client.table("portfolio_weights").upsert(data).execute()
            logger.info("Portfolio weights saved successfully.")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")

if __name__ == "__main__":
    # Example usage (requires .env)
    db = SupabaseDatabase()
    # db.save_predictions([{"symbol": "AAPL", "prediction_date": "2024-04-10", "predicted_price": 175.5}])
