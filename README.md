# Trading Signals 📈

Signal processing toolkit for algorithmic trading.

A collection of indicators, filters, and backtest utilities I use in my trading systems.

## Features

- **Indicators:** RSI, EMA, Bollinger Bands, VWAP, MACD, **RVWAP**
- **Filters:** Kalman filter, Butterworth low-pass, noise reduction
- **Backtest:** Event-driven backtesting engine with position management
- **Signals:** Multi-timeframe signal aggregation and scoring

## Quick Start

```python
from signals.indicators import rsi, ema, bollinger_bands
from signals.backtest import BacktestEngine

# Calculate indicators
rsi_values = rsi(closes, period=14)
upper, middle, lower = bollinger_bands(closes, period=20, std_dev=2)

# Run backtest
engine = BacktestEngine(initial_capital=10000)
results = engine.run(strategy=my_strategy, data=ohlcv_data)
print(f"Sharpe: {results.sharpe_ratio:.2f}")
```

## Structure

```
signals/
├── indicators/     # Technical indicators (RSI, EMA, VWAP, RVWAP, ...)
├── filters/        # Signal processing filters
├── backtest/       # Backtesting engine
├── models/         # Prediction models
└── utils/          # Helpers
tools/
└── rvwap_efficiency.py  # CLI: market efficiency analyzer
```

## RVWAP Market Efficiency

Analyze market efficiency using Realized Volume Weighted Average Price:

```bash
python tools/rvwap_efficiency.py --symbol BTCUSDT --interval 1h --chart --backtest
```

Scores market efficiency 0-100 based on deviation, mean-reversion speed, volume consistency, and z-score stability. Generates signals: `STRONG_LONG`, `WEAK_LONG`, `NEUTRAL`, `WEAK_SHORT`, `STRONG_SHORT`.

## Install

```bash
pip install -r requirements.txt
```

## License

MIT
