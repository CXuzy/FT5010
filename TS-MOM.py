import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Local data path (your file)
# =========================
CSV_PATH = r"E:\网页下载\FT5009\ASS3\Assignment_3_Group_15\materials\MAANG_2015_2019_close_stooq.csv"

# =========================
# Strategy parameters
# =========================
LOOKBACK = 60          # momentum lookback (days)
K = 2                  # long top K, short bottom K
VOL_LB = 20            # vol lookback (days)
TARGET_VOL = 0.15      # annualized target vol
VOL_CAP = 2.0          # leverage cap

USE_REGIME = True
MA_FAST = 20
MA_SLOW = 80
REGIME_MODE = "short_in_bear"  # "short_in_bear" / "flat_in_bear" / "half_in_bear"

TCOST_BPS = 0.0        # optional: e.g., 5 or 10
# =========================


def load_prices(csv_path: str) -> pd.DataFrame:
    px = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Clean column names (stooq file might have META/AAPL/...)
    px.columns = [c.strip().upper() for c in px.columns]

    # Sort by date ascending and forward-fill
    px = px.sort_index().ffill()

    # Keep only 2015-2019 (safety)
    px = px.loc["2015-01-01":"2019-12-31"]

    # Drop rows that are all NaN
    px = px.dropna(how="all")

    if px.empty:
        raise ValueError("Loaded price data is empty. Check CSV content/date range.")

    return px


def make_cs_mom_positions(prices: pd.DataFrame, lookback: int = 60, k: int = 2) -> pd.DataFrame:
    """
    Cross-sectional momentum:
    - compute trailing returns over lookback
    - rank across assets each day
    - long top-k, short bottom-k, equal weight, market-neutral
    """
    mom = prices.pct_change(lookback)
    ranks = mom.rank(axis=1, method="first", ascending=True)

    n = prices.shape[1]
    k = min(k, max(1, n // 2))  # safety: cannot exceed half universe

    long_mask = ranks >= (n - k + 1)
    short_mask = ranks <= k

    long_w = long_mask.div(long_mask.sum(axis=1), axis=0)
    short_w = short_mask.div(short_mask.sum(axis=1), axis=0)

    pos = (long_w - short_w).fillna(0.0)
    return pos


def apply_regime_filter(pos: pd.DataFrame, prices: pd.DataFrame,
                        ma_fast: int = 20, ma_slow: int = 80,
                        mode: str = "short_in_bear") -> pd.DataFrame:
    """
    Market regime from MAANG equal-weight index:
    bull: fast > slow
    bear: depending on mode
    """
    idx = prices.mean(axis=1)
    fast = idx.rolling(ma_fast).mean()
    slow = idx.rolling(ma_slow).mean()
    bull = fast > slow

    pos_f = pos.copy()
    if mode == "short_in_bear":
        pos_f.loc[~bull] = pos_f.loc[~bull].clip(upper=0.0)  # keep shorts only
    elif mode == "flat_in_bear":
        pos_f.loc[~bull] = 0.0
    elif mode == "half_in_bear":
        pos_f.loc[~bull] = 0.5 * pos_f.loc[~bull]
    else:
        raise ValueError("Unknown REGIME_MODE")

    return pos_f


def vol_target_returns(pos: pd.DataFrame, returns: pd.DataFrame,
                       vol_lb: int = 20, target_vol: float = 0.15, cap: float = 2.0):
    """
    Vol targeting at portfolio level:
    - compute portfolio returns using pos.shift(1)
    - estimate rolling vol
    - compute leverage factor and shift(1)
    """
    port_ret = (pos.shift(1) * returns).sum(axis=1)

    daily_target = target_vol / np.sqrt(252)
    roll_vol = port_ret.rolling(vol_lb).std()

    lev = (daily_target / roll_vol).clip(lower=0.0, upper=cap)
    lev = lev.shift(1).fillna(0.0)

    scaled_ret = lev * port_ret
    return scaled_ret, lev


def apply_transaction_costs(pos: pd.DataFrame, tcost_bps: float = 0.0) -> pd.Series:
    """
    Simple linear transaction cost:
    cost_t = turnover_t * (tcost_bps / 10000)
    turnover approx = sum(abs(delta_position))
    """
    if tcost_bps <= 0:
        return pd.Series(0.0, index=pos.index)

    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (tcost_bps / 10000.0)
    return cost


def perf_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        raise ValueError("Return series is empty. Check strategy logic/data.")

    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else np.nan

    equity = (1 + ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    maxdd = dd.min()

    return {
        "AnnReturn": float(ann_ret),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(maxdd),
        "Equity": equity,
        "Drawdown": dd
    }


# =========================
# Main
# =========================
prices = load_prices(CSV_PATH)
rets = prices.pct_change().fillna(0.0)

pos = make_cs_mom_positions(prices, lookback=LOOKBACK, k=K)

if USE_REGIME:
    pos = apply_regime_filter(pos, prices, ma_fast=MA_FAST, ma_slow=MA_SLOW, mode=REGIME_MODE)

strat_ret, lev = vol_target_returns(pos, rets, vol_lb=VOL_LB, target_vol=TARGET_VOL, cap=VOL_CAP)

cost = apply_transaction_costs(pos, tcost_bps=TCOST_BPS)
net_ret = strat_ret - cost

m = perf_metrics(net_ret)

print("Data columns:", list(prices.columns))
print("\nSelected Params:")
print(f"LOOKBACK={LOOKBACK}, K={K}, VOL_LB={VOL_LB}, TARGET_VOL={TARGET_VOL}, CAP={VOL_CAP}, "
      f"USE_REGIME={USE_REGIME}, MA_FAST={MA_FAST}, MA_SLOW={MA_SLOW}, REGIME_MODE={REGIME_MODE}, TCOST_BPS={TCOST_BPS}")

print("\nPerformance Metrics:")
print(f"AnnReturn: {m['AnnReturn']:.4f}")
print(f"AnnVol:    {m['AnnVol']:.4f}")
print(f"Sharpe:    {m['Sharpe']:.4f}")
print(f"MaxDD:     {m['MaxDD']:.4f}")

plt.figure()
m["Equity"].plot(title="Equity Curve - TS-Mom Long/Short (MAANG) [Offline]")
plt.ylabel("Equity")
plt.show()

plt.figure()
m["Drawdown"].plot(title="Drawdown - TS-Mom Long/Short (MAANG) [Offline]")
plt.ylabel("Drawdown")
plt.show()