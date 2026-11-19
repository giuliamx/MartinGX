"""Paper trading for the monthly momentum rotation (21 tickers).

Usage: python3 paper_trade.py --date YYYY-MM-DD --quotes '{"AAPL": 349.0, ...}'
  - Appends/overwrites the day's close in ../mean_reversion_scan/data/*.csv
  - Marks the paper portfolio to market
  - Rebalances (fills at the given quotes, 0.2% cost on traded notional) when a new month-end
    decision is available: decision = month's last trading-day close, execution = next trading day.
    On the first run the latest completed month-end decision is executed immediately.
State: paper/ledger.json, log: paper/log.md
"""
import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "mean_reversion_scan", "data")
STATE = os.path.join(HERE, "paper", "ledger.json")
LOG = os.path.join(HERE, "paper", "log.md")
COST = 0.002
PARAMS = {"lookback": 252, "k": 3}  # walk-forward choice for 2026; re-optimise each January


def update_closes(date, quotes):
    closes = {}
    for sym, px in quotes.items():
        f = os.path.join(DATA, f"{sym}.csv")
        d = pd.read_csv(f, parse_dates=["datetime"]).set_index("datetime")
        d = d[d.index != date]
        d.loc[date] = [px, px, px, px, 0]
        d.sort_index().to_csv(f, index_label="datetime")
        closes[sym] = d.sort_index().close
    return pd.DataFrame(closes).ffill()


def target(closes, decision_day):
    c = closes.loc[:decision_day]
    mom = c.iloc[-1] / c.iloc[-1 - PARAMS["lookback"]] - 1
    ok = c.iloc[-1] > c.iloc[-200:].mean()
    top = mom[ok].nlargest(PARAMS["k"])
    top = top[top > 0]
    return {s: 1.0 / PARAMS["k"] for s in top.index}, mom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--quotes", required=True)
    a = ap.parse_args()
    date = pd.Timestamp(a.date)
    quotes = json.loads(a.quotes)
    closes = update_closes(date, quotes)

    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {
        "start": a.date, "capital": 100_000.0, "cash": 100_000.0, "positions": {},
        "last_decision": None, "spy_start": quotes["SPY"], "history": []}

    px = closes.loc[date]
    equity = st["cash"] + sum(sh * px[s] for s, sh in st["positions"].items())

    # latest completed month-end strictly before today
    days = closes.index[closes.index < date]
    # a day is a month-end if the next trading day (or today) falls in a different month
    nxt = list(days[1:]) + [date]
    decision_day = [d for d, n in zip(days, nxt) if d.month != n.month][-1]

    lines = []
    if st["last_decision"] != str(decision_day.date()):
        tgt, mom = target(closes, decision_day)
        traded = 0.0
        new_pos = {}
        for s in set(tgt) | set(st["positions"]):
            want = equity * tgt.get(s, 0.0) / px[s]
            traded += abs(want - st["positions"].get(s, 0.0)) * px[s]
            if want > 0:
                new_pos[s] = want
        fee = traded * COST
        # scale down so fees come out of the invested amount
        scale = (equity - fee) / equity
        new_pos = {s: sh * scale for s, sh in new_pos.items()}
        st["positions"] = new_pos
        st["cash"] = equity - fee - sum(sh * px[s] for s, sh in new_pos.items())
        st["last_decision"] = str(decision_day.date())
        equity -= fee
        top = mom.sort_values(ascending=False).head(6).round(3).to_dict()
        lines.append(f"- **REBALANCE** (decision close {decision_day.date()}): "
                     f"target {list(tgt) or 'CASH'} | traded ${traded:,.0f}, fees ${fee:,.2f} | top momentum {top}")

    spy_ret = quotes["SPY"] / st["spy_start"] - 1
    ret = equity / st["capital"] - 1
    st["history"] = [h for h in st["history"] if h["date"] != a.date] + [
        {"date": a.date, "equity": round(equity, 2), "spy_ret": round(spy_ret, 5)}]
    json.dump(st, open(STATE, "w"), indent=1)

    hold = ", ".join(f"{s} {sh:.2f}sh @ {px[s]:.2f} (${sh * px[s]:,.0f})" for s, sh in st["positions"].items())
    lines.insert(0, f"### {a.date}\n- Equity **${equity:,.2f}** ({ret:+.2%}) vs SPY {spy_ret:+.2%} | cash ${st['cash']:,.2f}"
                    f"\n- Holdings: {hold or 'none'}")
    with open(LOG, "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
