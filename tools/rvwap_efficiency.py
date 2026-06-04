#!/usr/bin/env python3
"""
RVWAP Market Efficiency CLI
============================
Analyze market efficiency using Realized Volume Weighted Average Price.

Usage:
    python tools/rvwap_efficiency.py --symbol BTCUSDT --interval 1h
    python tools/rvwap_efficiency.py --csv data.csv --window 20 --backtest
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent dir to path so we can import signals
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

from signals.indicators.rvwap import (
    rvwap, rvwap_deviation, rvwap_zscore, efficiency_score, generate_signal,
)


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["timestamp", "date", "time", "datetime"]:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col])
            df = df.set_index("timestamp")
            break
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


# ─── Multi-timeframe ──────────────────────────────────────────────────────────

def multi_window(high, low, close, volume, windows=(10, 20, 50, 100)):
    results = {}
    for w in windows:
        if len(close) < w:
            continue
        rv = rvwap(high, low, close, volume, w)
        dev = rvwap_deviation(close, rv)
        z = rvwap_zscore(dev, w)
        eff = efficiency_score(dev, z, volume, w)
        results[w] = {
            "rvwap": round(float(rv[-1]), 2),
            "deviation": round(float(dev[-1]), 4),
            "z_score": round(float(z[-1]), 3),
            "efficiency": eff,
            "signal": generate_signal(float(z[-1])),
        }
    return results


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(close, z, capital=10000, z_entry=1.5, z_exit=0.0, z_stop=3.0):
    position = 0
    entry_price = 0
    equity = capital
    peak = capital
    trades = []

    for i in range(1, len(close)):
        price = close[i]
        zi = z[i]

        if position == 0:
            if zi > z_entry:
                position = -1; entry_price = price
                trades.append({"type": "SHORT", "entry": price, "idx": i})
            elif zi < -z_entry:
                position = 1; entry_price = price
                trades.append({"type": "LONG", "entry": price, "idx": i})
        else:
            exit_now = False
            if position == 1 and zi >= z_exit: exit_now = True
            elif position == -1 and zi <= z_exit: exit_now = True
            elif position == 1 and zi < -z_stop: exit_now = True
            elif position == -1 and zi > z_stop: exit_now = True

            if exit_now:
                pnl_pct = (price - entry_price) / entry_price * position
                equity += equity * pnl_pct
                peak = max(peak, equity)
                trades[-1].update({"exit": price, "pnl_pct": round(pnl_pct * 100, 2)})
                position = 0

    closed = [t for t in trades if "exit" in t]
    wins = [t for t in closed if t["pnl_pct"] > 0]
    return {
        "final_equity": round(equity, 2),
        "return_pct": round((equity - capital) / capital * 100, 2),
        "trades": len(closed),
        "win_rate": round(len(wins) / max(1, len(closed)) * 100, 1),
        "recent": closed[-5:],
    }


# ─── Chart ────────────────────────────────────────────────────────────────────

def plot(df, rvwap_arr, dev_arr, z_arr, symbol, window, output="rvwap_chart.png"):
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle(f"RVWAP Market Efficiency — {symbol} ({window}-period)", fontsize=14)
    idx = df.index

    ax = axes[0]
    ax.plot(idx, df["close"], label="Price", color="#2196F3", linewidth=1)
    ax.plot(idx, rvwap_arr, label="RVWAP", color="#FF9800", linewidth=1.5, linestyle="--")
    ax.fill_between(idx, df["close"], rvwap_arr,
                    where=df["close"].values > rvwap_arr, alpha=0.15, color="#4CAF50")
    ax.fill_between(idx, df["close"], rvwap_arr,
                    where=df["close"].values < rvwap_arr, alpha=0.15, color="#F44336")
    ax.set_ylabel("Price"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ["#4CAF50" if d >= 0 else "#F44336" for d in dev_arr]
    ax.bar(idx, dev_arr, color=colors, alpha=0.6, width=0.02)
    ax.axhline(0, color="white", linewidth=0.5); ax.set_ylabel("Dev %"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(idx, z_arr, color="#9C27B0", linewidth=1)
    ax.axhline(1.5, color="#F44336", linestyle=":", alpha=0.5, label="Short entry")
    ax.axhline(-1.5, color="#4CAF50", linestyle=":", alpha=0.5, label="Long entry")
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_ylabel("Z-Score"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3]
    vc = ["#4CAF50" if df["close"].iloc[i] >= df["open"].iloc[i] else "#F44336"
          for i in range(len(df))]
    ax.bar(idx, df["volume"], color=vc, alpha=0.5, width=0.02)
    ax.set_ylabel("Volume"); ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Chart → {output}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RVWAP Market Efficiency Analyzer")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--csv", type=str)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--capital", type=float, default=10000)
    p.add_argument("--chart", action="store_true")
    p.add_argument("--output", default="rvwap_chart.png")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    df = load_csv(args.csv) if args.csv else fetch_binance(args.symbol, args.interval, args.limit)
    symbol = Path(args.csv).stem if args.csv else args.symbol

    h, l, c, v = df["high"].values, df["low"].values, df["close"].values, df["volume"].values

    rv = rvwap(h, l, c, v, args.window)
    dev = rvwap_deviation(c, rv)
    z = rvwap_zscore(dev, args.window)
    eff = efficiency_score(dev, z, v, args.window)
    sig = generate_signal(float(z[-1]))
    mtf = multi_window(h, l, c, v)

    if args.chart:
        plot(df, rv, dev, z, symbol, args.window, args.output)

    if args.json:
        out = {"symbol": symbol, "interval": args.interval, "window": args.window,
               "price": round(float(c[-1]), 2), "rvwap": round(float(rv[-1]), 2),
               "deviation": round(float(dev[-1]), 4), "z_score": round(float(z[-1]), 3),
               "efficiency": eff, "signal": sig, "multi_window": mtf}
        if args.backtest:
            out["backtest"] = backtest(c, z, args.capital)
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'='*55}")
        print(f"  RVWAP Market Efficiency Report")
        print(f"{'='*55}")
        print(f"  Symbol:       {symbol}")
        print(f"  Timeframe:    {args.interval}")
        print(f"  Price:        ${c[-1]:,.2f}")
        print(f"  RVWAP:        ${rv[-1]:,.2f}")
        print(f"  Deviation:    {dev[-1]:+.4f}%")
        print(f"  Z-Score:      {z[-1]:+.3f}")
        print(f"  Efficiency:   {eff}/100")
        print(f"  Signal:       {sig}")
        print(f"\n  Multi-Window:")
        print(f"  {'Win':<6} {'RVWAP':>12} {'Dev':>9} {'Z':>7} {'Eff':>5} {'Signal':<14}")
        print(f"  {'-'*55}")
        for w, d in mtf.items():
            print(f"  {w:<6} ${d['rvwap']:>10,.2f} {d['deviation']:>+8.4f}% {d['z_score']:>+6.3f} {d['efficiency']:>4} {d['signal']}")
        if args.backtest:
            bt = backtest(c, z, args.capital)
            print(f"\n  Backtest: {bt['return_pct']:+.2f}% | {bt['trades']} trades | {bt['win_rate']:.0f}% win")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
