import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG (use your local CSV)
# =============================
# ============================
CSV_PATH = r"E:\网页下载\FT5009\ASS3\Assignment_3_Group_15\materials\MAANG_2015_2019_close_stooq.csv"

START = "2015-01-01"
END   = "2019-12-31"

# --- Signals ---
MOM_LB = 60            # momentum lookback (days)
TOP_K_BULL = 2              # concentrate winners in bull
TOP_WEIGHT_BULL = 0.60      # % allocated to top K (rest spread across remaining longs)

EXIT_HOLD_DAYS = 10         # hysteresis: need N consecutive negative mom to exit long (bull)

# --- Regime (bull/bear) ---
MA_BULL = 200               # bull regime if index > MA200

# --- Vol targeting (dynamic) ---
VOL_LB = 30
TARGET_VOL_BULL = 0.18      # higher risk in bull
TARGET_VOL_BEAR = 0.10      # lower risk in bear
LEV_CAP_BULL = 2.0
LEV_CAP_BEAR = 1.5

# --- Optional simple costs ---
TCOST_BPS = 0.0             # e.g. 5 or 10

# =========================================================
# Helpers
# =========================================================
def load_prices(path: str) -> pd.DataFrame:
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    px.columns = [c.strip().upper() for c in px.columns]
    px = px.sort_index().ffill()
    px = px.loc[START:END]
    px = px.dropna(how="all")
    if px.empty:
        raise ValueError("Price data empty. Check CSV/date range.")
    return px

def compute_regime(prices: pd.DataFrame, ma_bull: int = 200) -> pd.Series:
    idx = prices.mean(axis=1)  # MAANG equal-weight index
    ma = idx.rolling(ma_bull).mean()
    bull = idx > ma
    return bull.fillna(False)

def momentum(prices: pd.DataFrame, lb: int) -> pd.DataFrame:
    return prices.pct_change(lb)

def make_bull_positions(prices: pd.DataFrame, mom: pd.DataFrame,
                        top_k: int = 2, top_weight: float = 0.60,
                        exit_hold_days: int = 10) -> pd.DataFrame:
    """
    Bull regime: long-only, concentrated in top_k momentum names.
    - Allocate top_weight equally across top_k.
    - Allocate remaining (1-top_weight) equally across the other (N-top_k) names.
    - Hysteresis exit: if a name was in top set, keep it until it has exit_hold_days
      consecutive days of negative momentum.
    """
    n = prices.shape[1]
    cols = prices.columns
    top_k = min(top_k, max(1, n - 1))

    # daily ranks by momentum
    ranks = mom.rank(axis=1, method="first", ascending=False)  # 1=highest
    top_now = ranks <= top_k  # boolean DataFrame

    # hysteresis: "sticky" top membership
    sticky = pd.DataFrame(False, index=prices.index, columns=cols)
    neg = (mom < 0).fillna(False)

    # Track consecutive negative days since last in top set
    neg_streak = pd.DataFrame(0, index=prices.index, columns=cols, dtype=int)

    for t in range(len(prices.index)):
        if t == 0:
            sticky.iloc[t] = top_now.iloc[t]
            neg_streak.iloc[t] = neg.iloc[t].astype(int)
            continue

        prev_sticky = sticky.iloc[t-1].copy()

        # update negative streak
        prev_streak = neg_streak.iloc[t-1].copy()
        curr_neg = neg.iloc[t]
        curr_streak = prev_streak.where(~curr_neg, prev_streak + 1)  # if neg, +1 else keep
        curr_streak = curr_streak.where(curr_neg, 0)                 # if not neg, reset to 0
        neg_streak.iloc[t] = curr_streak

        # determine who is newly top today
        curr_top = top_now.iloc[t]

        # keep previous sticky members unless they've been negative for exit_hold_days
        drop = (prev_sticky) & (curr_streak >= exit_hold_days)
        curr_sticky = (prev_sticky & ~drop) | curr_top
        sticky.iloc[t] = curr_sticky

    # Build weights: long-only, ensure sum = 1
    w = pd.DataFrame(0.0, index=prices.index, columns=cols)

    # At each day:
    # - if sticky has >= top_k names, use sticky top set; else use top_now
    for dt in prices.index:
        stick_set = sticky.loc[dt]
        if stick_set.sum() >= top_k:
            top_set = stick_set
        else:
            top_set = top_now.loc[dt]

        top_names = cols[top_set.values]
        other_names = cols[~top_set.values]

        if len(top_names) == 0:
            # fallback: equal weight
            w.loc[dt] = 1.0 / n
            continue

        # assign top block
        w.loc[dt, top_names] = top_weight / len(top_names)

        # assign remainder to others (if any)
        if len(other_names) > 0:
            w.loc[dt, other_names] = (1.0 - top_weight) / len(other_names)

        # safety normalize
        s = w.loc[dt].sum()
        if s > 0:
            w.loc[dt] = w.loc[dt] / s

    return w

def make_bear_positions(prices: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    """
    Bear regime: time-series momentum long/short, equal-weight across names:
    - mom>0 -> +1, mom<0 -> -1, 0 otherwise
    - normalize by gross exposure so sum(abs(weights)) = 1 (if any signal)
    """
    sig = np.sign(mom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = sig.abs().sum(axis=1).replace(0, np.nan)
    w = sig.div(gross, axis=0).fillna(0.0)
    return w

def vol_target_portfolio(pos: pd.DataFrame, returns: pd.DataFrame,
                         vol_lb: int, target_vol: float, cap: float):
    """
    Portfolio-level vol targeting; shift(1) to avoid look-ahead.
    """
    port_ret = (pos.shift(1) * returns).sum(axis=1)
    daily_target = target_vol / np.sqrt(252)
    roll_vol = port_ret.rolling(vol_lb).std()
    lev = (daily_target / roll_vol).clip(lower=0.0, upper=cap).shift(1).fillna(0.0)
    scaled_ret = lev * port_ret
    return scaled_ret, lev

def transaction_costs(pos: pd.DataFrame, bps: float) -> pd.Series:
    if bps <= 0:
        return pd.Series(0.0, index=pos.index)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    return turnover * (bps / 10000.0)

def perf(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        raise ValueError("Empty returns.")
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else np.nan
    equity = (1 + ret).cumprod()
    dd = equity / equity.cummax() - 1
    return {
        "AnnReturn": float(ann_ret),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd.min()),
        "Equity": equity,
        "Drawdown": dd
    }

# =========================================================
# Run
# =========================================================
prices = load_prices(CSV_PATH)
rets = prices.pct_change().fillna(0.0)

bull = compute_regime(prices, ma_bull=MA_BULL)
mom = momentum(prices, lb=MOM_LB)

# Positions by regime
pos_bull = make_bull_positions(
    prices, mom,
    top_k=TOP_K_BULL,
    top_weight=TOP_WEIGHT_BULL,
    exit_hold_days=EXIT_HOLD_DAYS
)
pos_bear = make_bear_positions(prices, mom)

pos = pos_bear.copy()
pos.loc[bull] = pos_bull.loc[bull]

# Dynamic vol targeting by regime
ret_bull, lev_bull = vol_target_portfolio(pos, rets, VOL_LB, TARGET_VOL_BULL, LEV_CAP_BULL)
ret_bear, lev_bear = vol_target_portfolio(pos, rets, VOL_LB, TARGET_VOL_BEAR, LEV_CAP_BEAR)

# Blend returns by regime (use matching leverage series)
port_ret_raw = (pos.shift(1) * rets).sum(axis=1)
lev = lev_bear.copy()
lev.loc[bull] = lev_bull.loc[bull]
strat_ret = lev * port_ret_raw

# costs
cost = transaction_costs(pos, TCOST_BPS)
net_ret = strat_ret - cost

m = perf(net_ret)

# =========================================================
# Exposure Analysis (Long / Short Visualization)
# =========================================================

# Net exposure (long - short)
net_exposure = pos.sum(axis=1)

# Gross exposure (total leverage before vol targeting)
gross_exposure = pos.abs().sum(axis=1)

df_expo = pd.DataFrame({
    "NetExposure": net_exposure,
    "GrossExposure": gross_exposure,
    "BullRegime": bull.astype(int)
})

# ---- Plot Net Exposure ----
plt.figure(figsize=(12,5))
df_expo["NetExposure"].plot(label="Net Exposure", color="blue")
plt.axhline(0, color="black", linestyle="--", linewidth=1)

# Shade bear periods
bear = df_expo["BullRegime"] == 0
plt.fill_between(
    df_expo.index,
    df_expo["NetExposure"].min(),
    df_expo["NetExposure"].max(),
    where=bear,
    color="red",
    alpha=0.1,
    label="Bear Regime"
)

plt.title("Portfolio Net Exposure (Long vs Short)")
plt.ylabel("Net Weight")
plt.legend()
plt.show()
print("Data columns:", list(prices.columns))
print("\nParams:")
print(f"MOM_LB={MOM_LB}, MA_BULL={MA_BULL}, TOP_K_BULL={TOP_K_BULL}, TOP_WEIGHT_BULL={TOP_WEIGHT_BULL}, EXIT_HOLD_DAYS={EXIT_HOLD_DAYS}")
print(f"VOL_LB={VOL_LB}, TARGET_VOL_BULL={TARGET_VOL_BULL}, TARGET_VOL_BEAR={TARGET_VOL_BEAR}, LEV_CAP_BULL={LEV_CAP_BULL}, LEV_CAP_BEAR={LEV_CAP_BEAR}, TCOST_BPS={TCOST_BPS}")

print("\nPerformance Metrics:")
print(f"AnnReturn: {m['AnnReturn']:.4f}")
print(f"AnnVol:    {m['AnnVol']:.4f}")
print(f"Sharpe:    {m['Sharpe']:.4f}")
print(f"MaxDD:     {m['MaxDD']:.4f}")

plt.figure()
m["Equity"].plot(title="Equity Curve - Bull-Enhanced Momentum (Offline)")
plt.ylabel("Equity")
plt.show()

plt.figure()
m["Drawdown"].plot(title="Drawdown - Bull-Enhanced Momentum (Offline)")
plt.ylabel("Drawdown")
plt.show()