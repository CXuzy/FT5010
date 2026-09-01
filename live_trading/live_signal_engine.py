# =========================================================
# live_signal_engine.py
# =========================================================

import pandas as pd

from live_trading.config import (
    INSTRUMENTS,
    SHORT_MA,
    LONG_MA,
    REGIME_MA,
    GRANULARITY,
    MAX_CANDLES
)
from live_trading.oanda_client import OandaClient
from live_trading.strategy import get_latest_strategy_state


def fetch_multi_instrument_close(client: OandaClient,
                                 instruments,
                                 count: int = 300,
                                 granularity: str = "H1") -> pd.DataFrame:
    """
    Fetch close prices for multiple instruments from Oanda
    and combine them into one DataFrame.
    """
    frames = []

    for inst in instruments:
        df = client.get_candles(inst, count=count, granularity=granularity)
        s = df["close"].copy()
        s.name = inst.replace("_", "")
        frames.append(s)

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.ffill().dropna()

    return prices


def generate_live_signal(client: OandaClient):
    """
    Pull latest prices from Oanda and generate latest strategy state.
    """
    prices = fetch_multi_instrument_close(
        client=client,
        instruments=INSTRUMENTS,
        count=MAX_CANDLES,
        granularity=GRANULARITY
    )

    state = get_latest_strategy_state(
        prices=prices,
        short_ma=SHORT_MA,
        long_ma=LONG_MA,
        regime_ma=REGIME_MA
    )

    return prices, state


def print_live_signal(state: dict):
    print("===== LIVE STRATEGY STATE =====")
    print("Timestamp:", state["timestamp"])
    print("Bull regime:", state["is_bull"])
    print("Basket index:", state["basket_index"])
    print("Basket MA:", state["basket_ma"])
    print("Latest realized vol:", state.get("latest_realized_vol"))
    print("Latest leverage:", state.get("latest_leverage"))
    print()

    print("Latest signal:")
    for k, v in state["latest_signal"].items():
        print(f"  {k}: {v}")

    print()
    print("Latest target position:")
    for k, v in state["latest_position"].items():
        print(f"  {k}: {v:.4f}")