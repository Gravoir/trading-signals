"""Event-driven backtesting engine."""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    
class BacktestEngine:
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        
    def run(self, strategy: Callable, data: np.ndarray) -> BacktestResult:
        """Run backtest with given strategy on OHLCV data."""
        capital = self.initial_capital
        position = 0
        peak = capital
        max_dd = 0
        
        for i in range(len(data)):
            signal = strategy(data[:i+1])
            
            if signal > 0 and position == 0:
                shares = capital / data[i]['close']
                cost = capital * self.commission
                position = shares
                capital = cost
                self.trades.append(('BUY', data[i]['close'], i))
                
            elif signal < 0 and position > 0:
                revenue = position * data[i]['close']
                cost = revenue * self.commission
                capital = revenue - cost
                position = 0
                self.trades.append(('SELL', data[i]['close'], i))
            
            equity = capital + position * data[i]['close']
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        
        final = capital + position * data[-1]['close']
        total_return = (final - self.initial_capital) / self.initial_capital
        wins = sum(1 for i in range(0, len(self.trades)-1, 2) if i+1 < len(self.trades) and self.trades[i+1][1] > self.trades[i][1])
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=total_return / max(max_dd, 0.01),
            max_drawdown=max_dd,
            win_rate=wins / max(len(self.trades)//2, 1),
            total_trades=len(self.trades)
        )
