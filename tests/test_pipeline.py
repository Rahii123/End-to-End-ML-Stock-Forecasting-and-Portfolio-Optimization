import pytest
import pandas as pd
import numpy as np
from src.extractor.extractor import DataExtractor
from src.processor.processor import DataProcessor
from src.optimizer.optimizer import PortfolioOptimizer

def test_data_processor_prepare_for_prophet():
    processor = DataProcessor()
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"AAPL": np.random.randn(10)}, index=dates)
    
    prophet_df = processor.prepare_for_prophet(df, "AAPL")
    
    assert "ds" in prophet_df.columns
    assert "y" in prophet_df.columns
    assert len(prophet_df) == 10
    assert prophet_df["y"].iloc[0] == df["AAPL"].iloc[0]

def test_portfolio_optimizer():
    optimizer = PortfolioOptimizer()
    symbols = ["AAPL", "GOOG"]
    returns = pd.DataFrame(np.random.normal(0.001, 0.02, (100, 2)), columns=symbols)
    
    weights = optimizer.optimize_sharpe_ratio(returns)
    
    assert len(weights) == 2
    assert pytest.approx(sum(weights.values()), 0.001) == 1.0
    for weight in weights.values():
        assert 0.0 <= weight <= 1.0

def test_data_extractor_init():
    symbols = ["AAPL", "GOOG"]
    extractor = DataExtractor(symbols)
    assert extractor.symbols == symbols
