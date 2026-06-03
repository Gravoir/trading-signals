"""Kalman filter for price smoothing."""
import numpy as np

def kalman_filter(measurements: np.ndarray, process_noise: float = 0.01, measurement_noise: float = 0.1):
    """Apply Kalman filter to smooth noisy price data."""
    n = len(measurements)
    estimates = np.zeros(n)
    
    # Initial estimates
    estimates[0] = measurements[0]
    P = 1.0  # Error covariance
    
    for i in range(1, n):
        # Predict
        prediction = estimates[i-1]
        P = P + process_noise
        
        # Update
        K = P / (P + measurement_noise)  # Kalman gain
        estimates[i] = prediction + K * (measurements[i] - prediction)
        P = (1 - K) * P
    
    return estimates
