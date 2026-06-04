"""
Realized Volume Weighted Average Price (RVWAP)
===============================================
Extends classic VWAP by using a rolling realized volume profile to derive
a more accurate measure of fair value. Deviations signal market inefficiency.
"""

import numpy as np
import pandas as pd


def rvwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
          volume: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate Realized Volume Weighted Average Price (RVWAP).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume at each period
        window: Rolling lookback window (default: 20)

    Returns:
        Array of RVWAP values
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    cum_tp_vol = pd.Series(tp_vol).rolling(window=window, min_periods=1).sum().values
    cum_vol = pd.Series(volume).rolling(window=window, min_periods=1).sum().values

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)

    return result


def rvwap_deviation(close: np.ndarray, rvwap_values: np.ndarray) -> np.ndarray:
    """
    Calculate percentage deviation of price from RVWAP.

    Args:
        close: Close prices
        rvwap_values: RVWAP values from rvwap()

    Returns:
        Array of deviation percentages
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rvwap_values > 0, (close - rvwap_values) / rvwap_values * 100, 0)


def rvwap_zscore(deviation: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate Z-score of RVWAP deviation.

    Args:
        deviation: Deviation values from rvwap_deviation()
        window: Rolling window for mean/std calculation

    Returns:
        Array of Z-scores
    """
    dev_series = pd.Series(deviation)
    mean = dev_series.rolling(window=window, min_periods=1).mean().values
    std = dev_series.rolling(window=window, min_periods=1).std().values

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, (deviation - mean) / std, 0)


def efficiency_score(deviation: np.ndarray, z_score: np.ndarray,
                     volume: np.ndarray, lookback: int = 50) -> float:
    """
    Calculate market efficiency score (0-100) based on RVWAP metrics.

    Components:
        1. Deviation magnitude (0-30): smaller = more efficient
        2. Mean-reversion speed (0-30): more zero crossings = faster reversion
        3. Volume consistency (0-20): lower CV = more efficient
        4. Z-score stability (0-20): z-scores near 0 = efficient

    Args:
        deviation: Deviation array
        z_score: Z-score array
        volume: Volume array
        lookback: Number of recent periods to evaluate

    Returns:
        Efficiency score 0-100
    """
    n = min(lookback, len(deviation))
    if n < 5:
        return 50.0

    dev = deviation[-n:]
    z = z_score[-n:]
    vol = volume[-n:]

    # 1. Deviation score
    avg_abs_dev = np.nanmean(np.abs(dev))
    dev_score = max(0, min(30, 30 - avg_abs_dev * 10))

    # 2. Mean-reversion score (zero crossings)
    signs = np.sign(dev)
    zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
    reversion_score = min(30, zero_crossings * 2)

    # 3. Volume consistency
    vol_mean = np.nanmean(vol)
    vol_std = np.nanstd(vol)
    vol_cv = vol_std / vol_mean if vol_mean > 0 else 1
    vol_score = max(0, min(20, 20 - vol_cv * 5))

    # 4. Z-score stability
    avg_abs_z = np.nanmean(np.abs(z))
    z_stability = max(0, min(20, 20 - avg_abs_z * 5))

    return round(dev_score + reversion_score + vol_score + z_stability, 1)


def generate_signal(z_score: float) -> str:
    """
    Generate trading signal from current Z-score.

    Args:
        z_score: Current Z-score value

    Returns:
        Signal string: STRONG_LONG, WEAK_LONG, NEUTRAL, WEAK_SHORT, STRONG_SHORT
    """
    if z_score > 2.0:
        return "STRONG_SHORT"
    elif z_score > 1.0:
        return "WEAK_SHORT"
    elif z_score < -2.0:
        return "STRONG_LONG"
    elif z_score < -1.0:
        return "WEAK_LONG"
    return "NEUTRAL"
