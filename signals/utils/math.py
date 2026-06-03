"""Math utilities for signal processing."""
import numpy as np

def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range."""
    min_val, max_val = data.min(), data.max()
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def z_score(data: np.ndarray) -> np.ndarray:
    """Calculate z-scores."""
    return (data - data.mean()) / data.std()

def percentile_rank(data: np.ndarray, value: float) -> float:
    """Calculate percentile rank of a value."""
    return (data < value).sum() / len(data) * 100
