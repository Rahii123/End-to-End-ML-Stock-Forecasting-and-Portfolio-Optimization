import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Implements Modern Portfolio Theory (MPT) using SciPy's quadratic solver.
    Focuses on maximizing the Sharpe Ratio.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_statistics(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculates annualized mean returns and covariance matrix.
        """
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        return mean_returns, cov_matrix

    def _portfolio_performance(self, weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculates portfolio return and volatility.
        """
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std

    def _negative_sharpe_ratio(self, weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Objective function to minimize (negative Sharpe Ratio).
        """
        p_ret, p_std = self._portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - self.risk_free_rate) / p_std

    def optimize_sharpe_ratio(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Finds the weights that maximize the Sharpe Ratio.
        """
        logger.info("Starting portfolio optimization...")
        num_assets = len(returns.columns)
        mean_returns, cov_matrix = self.calculate_statistics(returns)

        # Constraints: components sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        # Initial guess: equal distribution
        initial_weights = num_assets * [1. / num_assets]

        result = minimize(
            self._negative_sharpe_ratio,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.error(f"Optimization failed: {result.message}")
            raise ValueError("Optimization failed")

        optimized_weights = dict(zip(returns.columns, result.x))
        logger.info("Optimization successful.")
        return optimized_weights

if __name__ == "__main__":
    # Mock data
    symbols = ["AAPL", "GOOG", "PLTR", "JPM"]
    returns = pd.DataFrame(np.random.normal(0.001, 0.02, (252, 4)), columns=symbols)
    
    optimizer = PortfolioOptimizer()
    weights = optimizer.optimize_sharpe_ratio(returns)
    print("Optimized Weights:")
    for symbol, weight in weights.items():
        print(f"{symbol}: {weight:.4f}")
