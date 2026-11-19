"""DINO (HF Sinclair) daily mean-reversion: long-only RSI(2) pullbacks inside an uptrend.

Rules (signals on close of day t, fills at open of t+1 -> no lookahead):
  Entry : close > SMA(trend) and RSI(2) < rsi_entry
  Exit  : close > SMA(5)  or  held >= max_hold days  or  intraday stop (entry - atr_mult * ATR14)
Costs : 0.10% slippage + 0.10% commission per side.
Data  : dino_1d.csv (Twelve Data, unadjusted for dividends).
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

COST = 0.002  # per side: slippage + commission
CAPITAL = 100_000


def load(path="dino_1d.csv"):
    d = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    if d["volume"].iloc[-1] < 0.2 * d["volume"].iloc[-21:-1].median():
        d = d.iloc[:-1]  # drop today's partial bar
    return d


def indicators(d, trend):
    c = d["close"]
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
    return pd.DataFrame({
        "rsi2": 100 - 100 / (1 + up / dn),
        "sma_t": c.rolling(trend).mean(),
        "sma5": c.rolling(5).mean(),
        "atr": tr.rolling(14).mean(),
    }, index=d.index)


def run(d, rsi_entry=10, trend=200, max_hold=10, atr_mult=3.0):
    ind = indicators(d, trend)
    o, h, l, c = d.open.values, d.high.values, d.low.values, d.close.values
    rsi, smat, sma5, atr = (ind[k].values for k in ["rsi2", "sma_t", "sma5", "atr"])
    eq, cash, shares = [], CAPITAL, 0.0
    trades, pending, entry_i, stop, entry_px = [], None, 0, 0.0, 0.0
    for i in range(len(d)):
        if pending == "buy":
            entry_px = o[i] * (1 + COST)
            shares, cash = cash / entry_px, 0.0
            entry_i, stop = i, o[i] - atr_mult * atr[i - 1]
        elif pending == "sell":
            px = o[i] * (1 - COST)
            trades.append(px / entry_px - 1); cash, shares = shares * px, 0.0
        pending = None
        if shares and l[i] <= stop:  # stop hit intraday (gap-down fills at open)
            px = min(o[i], stop) * (1 - COST)
            trades.append(px / entry_px - 1); cash, shares = shares * px, 0.0
        elif shares and (c[i] > sma5[i] or i - entry_i >= max_hold):
            pending = "sell"
        elif not shares and c[i] > smat[i] and rsi[i] < rsi_entry:
            pending = "buy"
        eq.append(cash + shares * c[i])
    return pd.Series(eq, index=d.index), np.array(trades), pending, shares > 0


def metrics(eq, trades):
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252
    dd = eq / eq.cummax() - 1
    neg = r[r < 0]
    wins, losses = trades[trades > 0], trades[trades <= 0]
    return {
        "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() else 0,
        "sortino": r.mean() / neg.std() * np.sqrt(252) if len(neg) else 0,
        "max_dd": dd.min(),
        "win_rate": len(wins) / len(trades) if len(trades) else 0,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() else np.inf,
        "trades": len(trades),
        "exposure": (eq.diff() != 0).mean(),
    }


GRID = {"rsi_entry": [5, 10, 15], "trend": [100, 200], "max_hold": [5, 10], "atr_mult": [2.0, 3.0]}


def walk_forward(d, train_yrs=3, test_yrs=1):
    """Re-optimise Sharpe on rolling 3y window, trade the next 1y out of sample."""
    oos_eq, oos_trades, chosen = [], [], []
    start = d.index[0] + pd.DateOffset(years=train_yrs)
    while start < d.index[-1]:
        end = start + pd.DateOffset(years=test_yrs)
        train = d[(d.index >= start - pd.DateOffset(years=train_yrs)) & (d.index < start)]
        best = max((dict(zip(GRID, v)) for v in itertools.product(*GRID.values())),
                   key=lambda p: metrics(*run(train, **p)[:2])["sharpe"])
        # warm-up: include 250 bars of history for indicators, then slice test period
        seg = d[(d.index >= start - pd.DateOffset(days=400)) & (d.index < end)]
        eq, _, _, _ = run(seg, **best)
        eq = eq[eq.index >= start]
        rets = eq.pct_change().fillna(0)
        oos_eq.append(rets)
        chosen.append((start.date(), best))
        start = end
    rets = pd.concat(oos_eq)
    return CAPITAL * (1 + rets).cumprod(), chosen


def monte_carlo(trades, n=1000, seed=0):
    """Bootstrap trade returns (resample with replacement), full compounding."""
    rng = np.random.default_rng(seed)
    finals, dds = [], []
    for _ in range(n):
        path = CAPITAL * np.cumprod(1 + rng.choice(trades, len(trades)))
        finals.append(path[-1])
        dds.append((path / np.maximum.accumulate(path) - 1).min())
    finals, dds = np.array(finals), np.array(dds)
    return {"p5": np.percentile(finals, 5), "p50": np.median(finals), "p95": np.percentile(finals, 95),
            "prob_loss": (finals < CAPITAL).mean(), "prob_ruin_dd50": (dds <= -0.5).mean(),
            "dd_p5": np.percentile(dds, 5)}


if __name__ == "__main__":
    d = load()
    fmt = lambda m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    base = dict(rsi_entry=10, trend=200, max_hold=10, atr_mult=3.0)
    eq, tr, pending, in_pos = run(d, **base)
    print("FULL SAMPLE (default params)", d.index[0].date(), "->", d.index[-1].date())
    print(json.dumps(fmt(metrics(eq, tr)), indent=1))
    bh = d.close / d.close.iloc[0] * CAPITAL
    print("Buy&hold:", json.dumps(fmt(metrics(bh, np.array([bh.iloc[-1] / CAPITAL - 1]))), indent=1))

    print("\nPARAM SENSITIVITY (full sample, sharpe / cagr / maxdd / trades)")
    for v in itertools.product(*GRID.values()):
        p = dict(zip(GRID, v)); m = metrics(*run(d, **p)[:2])
        print(f"  {p}  {m['sharpe']:.2f}  {m['cagr']:.1%}  {m['max_dd']:.1%}  {m['trades']}")

    wf_eq, chosen = walk_forward(d)
    print("\nWALK-FORWARD OOS (3y train / 1y test)", wf_eq.index[0].date(), "->", wf_eq.index[-1].date())
    m = metrics(wf_eq, tr); [m.pop(k) for k in ("win_rate", "profit_factor", "trades")]
    print(json.dumps(fmt(m), indent=1))
    for s, p in chosen:
        print("  ", s, p)

    print("\nMONTE CARLO (1000 bootstrap, default params trades)")
    print(json.dumps(fmt(monte_carlo(tr)), indent=1))

    ind = indicators(d, base["trend"]).iloc[-1]
    print("\nLAST BAR", d.index[-1].date(), "close", d.close.iloc[-1],
          "| RSI2 %.1f | SMA200 %.2f | SMA5 %.2f | ATR %.2f" % (ind.rsi2, ind.sma_t, ind.sma5, ind.atr),
          "| in_position", in_pos, "| pending", pending)
