"""Trend indicators."""
import numpy as np

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2 / (period + 1)
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    return np.convolve(data, np.ones(period)/period, mode='valid')

def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume Weighted Average Price."""
    return np.sum(prices * volumes) / np.sum(volumes)
