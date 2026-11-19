"""Trend following (per ticker) and momentum rotation (portfolio) on the 21 TradingView tickers.

Both evaluated with the same walk-forward as the mean-reversion scan: optimise Sharpe on 3y, trade next 1y OOS.
Costs: 0.2% per side (slippage + commission). Signals use data up to close t; trades fill one bar later.
"""
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dino_mean_reversion"))
from backtest import CAPITAL, COST, metrics  # noqa: E402

DATA = os.path.join(HERE, "..", "mean_reversion_scan", "data")
TODAY = pd.Timestamp("2026-11-19")


def load_all():
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        d = pd.read_csv(f, parse_dates=["datetime"]).set_index("datetime")
        out[os.path.basename(f)[:-4]] = d[d.index < TODAY]  # drop today's forming bar
    return out


# ---------- 1. Trend following: long while SMA(fast) > SMA(slow) and close > SMA(slow) ----------
TREND_GRID = {"fast": [20, 50], "slow": [100, 150, 200]}


def trend_returns(d, fast, slow):
    c = d.close
    want = ((c.rolling(fast).mean() > c.rolling(slow).mean()) & (c > c.rolling(slow).mean())).astype(float)
    pos = want.shift(1).fillna(0)          # decided at close t-1
    # held through day t: entered at open t if pos changed 0->1, exited at open t if 1->0
    prev = pos.shift(1).fillna(0)
    r = pd.Series(0.0, index=d.index)
    hold = (prev == 1) & (pos == 1)
    enter = (prev == 0) & (pos == 1)
    exit_ = (prev == 1) & (pos == 0)
    r[hold] = (c / c.shift(1) - 1)[hold]
    r[enter] = (c / d.open * (1 - COST) - 1)[enter]
    r[exit_] = (d.open / c.shift(1) * (1 - COST) - 1)[exit_]
    return r, want


# ---------- 2. Momentum rotation: monthly, top-K by lookback return, only if above SMA200 ----------
ROT_GRID = {"lookback": [63, 126, 252], "k": [3, 5]}


def rotation_returns(closes, lookback, k):
    rets = closes.pct_change().fillna(0)
    mom = closes / closes.shift(lookback) - 1
    ok = closes > closes.rolling(200).mean()
    month_end = closes.index.to_series().groupby(closes.index.to_period("M")).transform("max") == closes.index
    w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for t in closes.index[month_end]:
        m = mom.loc[t].where(ok.loc[t]).dropna()
        row = pd.Series(0.0, index=closes.columns)
        top = m.nlargest(k)
        top = top[top > 0]
        if len(top):
            row[top.index] = 1.0 / k   # unfilled slots stay in cash
        w.loc[t] = row
    w = w.ffill().fillna(0).shift(2).fillna(0)  # decided at month-end close, executed next close
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    return (w * rets).sum(axis=1) - turnover * COST, w


def walk_forward(fn, grid, start_data, end_data, train_yrs=3):
    out, chosen = [], []
    start = start_data + pd.DateOffset(years=train_yrs + 1)  # +1y warm-up for 252d lookbacks
    while start < end_data:
        end = start + pd.DateOffset(years=1)
        best, best_s = None, -np.inf
        for v in itertools.product(*grid.values()):
            p = dict(zip(grid, v))
            r = fn(p)
            tr = r[(r.index >= start - pd.DateOffset(years=train_yrs)) & (r.index < start)]
            s = tr.mean() / tr.std() if tr.std() else -np.inf
            if s > best_s:
                best, best_s = p, s
        r = fn(best)
        out.append(r[(r.index >= start) & (r.index < end)])
        chosen.append((start.date(), best))
        start = end
    return pd.concat(out), chosen


def eq(r):
    return CAPITAL * (1 + r).cumprod()


def mc_daily_blocks(r, n=1000, block=21, seed=0):
    """Block bootstrap of daily returns (keeps autocorrelation); returns p5/p95 final equity and P(DD<=-50%)."""
    rng = np.random.default_rng(seed)
    a, L = r.values, len(r)
    finals, ruin = [], 0
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, L - block, L // block + 1)])[:L]
        path = CAPITAL * np.cumprod(1 + a[idx])
        finals.append(path[-1])
        ruin += (path / np.maximum.accumulate(path) - 1).min() <= -0.5
    return np.percentile(finals, 5), np.percentile(finals, 95), np.mean(np.array(finals) < CAPITAL), ruin / n


if __name__ == "__main__":
    data = load_all()
    first, last = max(d.index[0] for d in data.values()), min(d.index[-1] for d in data.values())
    rows = []
    for sym, d in data.items():
        cache = {}
        fn = lambda p, d=d: cache.setdefault(tuple(p.values()), trend_returns(d, **p)[0])
        oos, chosen = walk_forward(fn, TREND_GRID, d.index[0], d.index[-1])
        m = metrics(eq(oos), np.array([0.0]))
        bh = d.close[oos.index]
        mb = metrics(bh / bh.iloc[0] * CAPITAL, np.array([0.0]))
        p5, p95, ploss, ruin = mc_daily_blocks(oos)
        _, want = trend_returns(d, **chosen[-1][1])
        rows.append({"sym": sym, "oos_cagr": m["cagr"], "oos_sharpe": m["sharpe"], "oos_dd": m["max_dd"],
                     "bh_cagr": mb["cagr"], "bh_sharpe": mb["sharpe"], "bh_dd": mb["max_dd"], "expo": m["exposure"],
                     "mc_p5": p5, "mc_p95": p95, "mc_loss": ploss, "mc_ruin": ruin,
                     "params_last": chosen[-1][1], "signal": "LONG" if want.iloc[-1] else "FLAT",
                     "close": d.close.iloc[-1], "date": d.index[-1].date()})
    df = pd.DataFrame(rows).sort_values("oos_cagr", ascending=False)
    df.to_csv(os.path.join(HERE, "trend_results.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("=== TREND FOLLOWING (per ticker, walk-forward OOS) ===")
    print(df.round(3).to_string(index=False))

    closes = pd.DataFrame({s: d.close for s, d in data.items()}).loc[first:last].ffill()
    cache = {}
    fn = lambda p: cache.setdefault(tuple(p.values()), rotation_returns(closes, **p)[0])
    oos, chosen = walk_forward(fn, ROT_GRID, closes.index[0], closes.index[-1])
    m = metrics(eq(oos), np.array([0.0]))
    spy = closes.SPY[oos.index]
    ew = closes.pct_change().fillna(0).mean(axis=1)[oos.index]
    print("\n=== MOMENTUM ROTATION (21 tickers, walk-forward OOS)", oos.index[0].date(), "->", oos.index[-1].date())
    for name, e in [("rotation", eq(oos)), ("SPY b&h", spy / spy.iloc[0] * CAPITAL), ("equal-weight 21", eq(ew))]:
        mm = metrics(e, np.array([0.0]))
        print(f"  {name:16s} cagr {mm['cagr']:.1%}  sharpe {mm['sharpe']:.2f}  sortino {mm['sortino']:.2f}  maxdd {mm['max_dd']:.1%}")
    print("  MC (block bootstrap 1000):", [round(x, 3) for x in mc_daily_blocks(oos)])
    for s, p in chosen:
        print("   ", s, p)
    _, w = rotation_returns(closes, **chosen[-1][1])
    # current target = weights decided at the latest month-end (before execution lag)
    lw = w.iloc[-1]
    print("  current holdings:", {k: round(v, 3) for k, v in lw[lw > 0].items()})
