import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    A Deep Learning Long Short-Term Memory (LSTM) Neural Network.
    Designed to extract complex, non-linear sequences and chaotic noise from stock data.
    """
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # The LSTM cell: remembers "Short-Term" volatility up to X days in the past.
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        
        # The Linear layer: compresses the LSTM's complex thoughts into a single price prediction.
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class StockForecaster:
    """
    Enterprise-Grade Hybrid Forecaster.
    Combines Statistical Baseline Trenlines (EWMA) with Deep Learning Residual Chaos extraction (LSTM).
    """

    def __init__(self, changepoint_prior_scale: float = 0.05):
        self.changepoint_prior_scale = changepoint_prior_scale
        
    def create_sequences(self, data: np.ndarray, seq_length: int):
        """Pairs historical data into [Sequence] -> [Target Next Day] for PyTorch training."""
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            xs.append(data[i:(i+seq_length)])
            ys.append(data[i+seq_length])
        return np.array(xs), np.array(ys)

    def forecast(self, df: pd.DataFrame, periods: int = 5) -> Tuple[pd.DataFrame, List[float]]:
        """
        Trains the Hybrid Model and autoregressively forecasts multiple future prices.
        """
        logger.info(f"Training Hybrid LSTM Ensemble model on {len(df)} data points...")
        
        try:
            # ----------------------------------------------------------------
            # PHASE 1: The Statistical Macro-Trend Baseline
            # ----------------------------------------------------------------
            span = 20
            baseline_ewma = df['y'].ewm(span=span, adjust=False).mean()
            
            # Predict the baseline's trajectory into the future using momentum
            baseline_momentum = baseline_ewma.iloc[-1] - baseline_ewma.iloc[-2]
            
            # ----------------------------------------------------------------
            # PHASE 2: Residual Chaos Extraction
            # ----------------------------------------------------------------
            residuals = df['y'] - baseline_ewma
            
            # ----------------------------------------------------------------
            # PHASE 3: Deep Learning (LSTM)
            # ----------------------------------------------------------------
            scaler = MinMaxScaler(feature_range=(-1, 1))
            res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
            
            seq_length = 5
            if len(res_scaled) <= seq_length:
                raise ValueError("Dataset is too small to train an LSTM model.")
                
            X, y = self.create_sequences(res_scaled, seq_length)
            
            X_torch = torch.FloatTensor(X)
            y_torch = torch.FloatTensor(y)
            
            torch.manual_seed(42)
            lstm_net = LSTMModel()
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(lstm_net.parameters(), lr=0.01)
            
            logger.info("Initiating LSTM Neural Network Backpropagation (50 Epochs)...")
            epochs = 50
            for i in range(epochs):
                optimizer.zero_grad()
                y_pred = lstm_net(X_torch)
                loss = loss_function(y_pred, y_torch)
                loss.backward()
                optimizer.step()
                
            # ----------------------------------------------------------------
            # PHASE 4: Autoregressive Multi-Step Synthesis
            # ----------------------------------------------------------------
            current_seq_scaled = res_scaled[-seq_length:].copy()
            predicted_residuals_dollars = []
            
            # Loop recursively to project a full 'periods' distance into the future
            lstm_net.eval()
            with torch.no_grad():
                for _ in range(periods):
                    input_seq = torch.FloatTensor(current_seq_scaled).unsqueeze(0)
                    next_res_scaled = lstm_net(input_seq).item()
                    
                    pred_res_usd = scaler.inverse_transform([[next_res_scaled]])[0][0]
                    predicted_residuals_dollars.append(pred_res_usd)
                    
                    # Slide the sequence forward by dropping day 1 and appending the new fake prediction
                    current_seq_scaled = np.append(current_seq_scaled[1:], [[next_res_scaled]], axis=0)
            
            # Combine Future Baseline + Future Chaos
            final_target_prices = []
            for d in range(periods):
                future_baseline = baseline_ewma.iloc[-1] + (baseline_momentum * (d + 1))
                final_target_prices.append(future_baseline + predicted_residuals_dollars[d])
            
            # Compute future business dates
            future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=periods+1, freq='B')[1:]
            
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': final_target_prices,
                'yhat_lower': [p * 0.95 for p in final_target_prices],
                'yhat_upper': [p * 1.05 for p in final_target_prices],
            })
            
            logger.info(f"Hybrid Autoregressive Synthesis Complete for {periods} days!")
            return forecast, list(final_target_prices)

        except Exception as e:
            logger.error(f"Critical error during Hybrid Neural Network Training: {e}")
            raise

if __name__ == "__main__":
    # Example local test
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    data = pd.DataFrame({
        "ds": dates,
        "y": np.random.randn(100).cumsum() + 150
    })
    
    forecaster = StockForecaster()
    forecast_df, pred = forecaster.forecast(data)
