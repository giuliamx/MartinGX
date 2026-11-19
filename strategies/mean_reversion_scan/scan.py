"""Run the DINO RSI(2) mean-reversion backtest over every CSV in data/ and rank by walk-forward OOS CAGR."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dino_mean_reversion"))
from backtest import CAPITAL, indicators, load, metrics, monte_carlo, run, walk_forward  # noqa: E402

BASE = dict(rsi_entry=10, trend=200, max_hold=10, atr_mult=3.0)
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv"))):
    sym = os.path.basename(f)[:-4]
    d = load(f)
    eq, tr, pending, in_pos = run(d, **BASE)
    m = metrics(eq, tr)
    wf_eq, _ = walk_forward(d)
    wf = metrics(wf_eq, tr)
    bh = metrics(d.close / d.close.iloc[0] * CAPITAL, np.array([0.0]))
    bh_oos = metrics(d.close[wf_eq.index] / d.close[wf_eq.index[0]] * CAPITAL, np.array([0.0]))
    mc = monte_carlo(tr)
    ind = indicators(d, BASE["trend"]).iloc[-1]
    rows.append({
        "sym": sym, "oos_cagr": wf["cagr"], "oos_sharpe": wf["sharpe"], "oos_dd": wf["max_dd"],
        "bh_oos_cagr": bh_oos["cagr"], "bh_oos_dd": bh_oos["max_dd"],
        "full_cagr": m["cagr"], "full_sharpe": m["sharpe"], "full_dd": m["max_dd"],
        "win": m["win_rate"], "pf": m["profit_factor"], "trades": m["trades"], "expo": m["exposure"],
        "mc_p5": mc["p5"], "mc_p95": mc["p95"], "mc_loss": mc["prob_loss"], "mc_ruin": mc["prob_ruin_dd50"],
        "last": d.index[-1].date(), "close": d.close.iloc[-1], "rsi2": ind.rsi2, "sma200": ind.sma_t,
        "in_pos": in_pos, "pending": pending,
    })

df = pd.DataFrame(rows).sort_values("oos_cagr", ascending=False)
df.to_csv(os.path.join(os.path.dirname(__file__), "results.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(df.round(3).to_string(index=False))
