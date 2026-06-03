"""Volatility indicators."""
import numpy as np

def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands."""
    sma = np.convolve(prices, np.ones(period)/period, mode='valid')
    std = np.array([np.std(prices[max(0,i-period+1):i+1]) for i in range(period-1, len(prices))])
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    return np.convolve(tr, np.ones(period)/period, mode='valid')
