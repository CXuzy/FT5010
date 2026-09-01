# =========================================================
# strategy.py
# =========================================================

import numpy as np
import pandas as pd

from live_trading.config import (
    TARGET_VOL_BULL,
    TARGET_VOL_BEAR,
    LEV_CAP_BULL,
    LEV_CAP_BEAR,
)


def compute_raw_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def usd_aligned_returns(returns: pd.DataFrame) -> pd.DataFrame:
    aligned = returns.copy()

    for col in aligned.columns:
        if col.endswith("USD"):
            aligned[col] = -aligned[col]
        elif col.startswith("USD"):
            aligned[col] = aligned[col]
        else:
            raise ValueError(f"Unexpected pair format: {col}")

    return aligned


def compute_regime_from_usd_basket(prices: pd.DataFrame, ma_bull: int = 200):
    raw_rets = compute_raw_returns(prices)
    aligned_rets = usd_aligned_returns(raw_rets)

    basket_ret = aligned_rets.mean(axis=1)
    basket_idx = (1 + basket_ret).cumprod()
    basket_ma = basket_idx.rolling(ma_bull).mean()

    bull = (basket_idx > basket_ma).fillna(False)
    return bull, basket_idx, basket_ma


def ma_signal(prices: pd.DataFrame, short_lb: int = 20, long_lb: int = 60) -> pd.DataFrame:
    ma_short = prices.rolling(short_lb).mean()
    ma_long = prices.rolling(long_lb).mean()

    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sig[ma_short > ma_long] = 1.0
    sig[ma_short < ma_long] = -1.0
    return sig


def make_fx_positions_from_signal(signal: pd.DataFrame) -> pd.DataFrame:
    gross = signal.abs().sum(axis=1).replace(0, np.nan)
    w = signal.div(gross, axis=0).fillna(0.0)
    return w


def compute_realized_portfolio_vol(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    vol_lookback: int = 60,
    annualization_factor: int = 252 * 24 * 60,
) -> pd.Series:
    """
    Estimate realized annualized vol of the strategy using lagged positions.
    Assumes M1 bars by default, so annualization uses 252 * 24 * 60.
    """
    raw_rets = compute_raw_returns(prices)
    aligned_rets = usd_aligned_returns(raw_rets)

    shifted_pos = positions.shift(1).fillna(0.0)
    portfolio_ret = (shifted_pos * aligned_rets).sum(axis=1)

    realized_vol = portfolio_ret.rolling(vol_lookback).std() * np.sqrt(annualization_factor)
    return realized_vol


def compute_dynamic_leverage(
    realized_vol: pd.Series,
    bull_regime: pd.Series,
) -> pd.Series:
    """
    leverage = target_vol / realized_vol
    then cap by regime-specific leverage cap
    """
    lev = pd.Series(index=realized_vol.index, dtype=float)

    for idx in realized_vol.index:
        rv = realized_vol.loc[idx]
        is_bull = bool(bull_regime.loc[idx]) if idx in bull_regime.index else False

        target_vol = TARGET_VOL_BULL if is_bull else TARGET_VOL_BEAR
        lev_cap = LEV_CAP_BULL if is_bull else LEV_CAP_BEAR

        if pd.isna(rv) or rv <= 0:
            lev.loc[idx] = 1.0
        else:
            lev.loc[idx] = min(target_vol / rv, lev_cap)

    return lev.fillna(1.0)


def get_latest_strategy_state(
    prices: pd.DataFrame,
    short_ma: int = 20,
    long_ma: int = 60,
    regime_ma: int = 200,
    vol_lookback: int = 60,
):
    """
    Given a historical price DataFrame, return the latest strategy state.
    """

    bull, basket_idx, basket_ma = compute_regime_from_usd_basket(prices, ma_bull=regime_ma)
    signal = ma_signal(prices, short_lb=short_ma, long_lb=long_ma)
    pos = make_fx_positions_from_signal(signal)

    realized_vol = compute_realized_portfolio_vol(
        prices=prices,
        positions=pos,
        vol_lookback=vol_lookback,
    )
    leverage = compute_dynamic_leverage(
        realized_vol=realized_vol,
        bull_regime=bull,
    )

    latest_idx = prices.index[-1]

    state = {
        "timestamp": latest_idx,
        "is_bull": bool(bull.loc[latest_idx]),
        "latest_signal": signal.loc[latest_idx].to_dict(),
        "latest_position": pos.loc[latest_idx].to_dict(),
        "basket_index": float(basket_idx.loc[latest_idx]),
        "basket_ma": float(basket_ma.loc[latest_idx]) if pd.notna(basket_ma.loc[latest_idx]) else np.nan,
        "latest_realized_vol": float(realized_vol.loc[latest_idx]) if pd.notna(realized_vol.loc[latest_idx]) else np.nan,
        "latest_leverage": float(leverage.loc[latest_idx]) if pd.notna(leverage.loc[latest_idx]) else 1.0,
    }

    return state