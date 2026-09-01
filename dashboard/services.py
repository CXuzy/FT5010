import json
import os

import pandas as pd

from live_trading.config import (
    INSTRUMENTS,
    STATUS_FILE,
    TRADE_LOG_FILE,
)
from live_trading.live_signal_engine import generate_live_signal


def safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def parse_json_cell(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}


def fmt_num(x, digits=2):
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "-"


def get_account_snapshot(client):
    summary = client.get_account_summary()
    return {
        "nav": float(summary.get("NAV", 0.0)),
        "balance": float(summary.get("balance", 0.0)),
        "unrealized": float(summary.get("unrealizedPL", 0.0)),
        "realized": float(summary.get("pl", 0.0)),
        "margin_used": float(summary.get("marginUsed", 0.0)),
        "margin_available": float(summary.get("marginAvailable", 0.0)),
    }


def get_live_positions_df(client):
    positions = client.get_positions_map()
    rows = []

    for inst in INSTRUMENTS:
        rows.append({
            "instrument": inst,
            "current_units_live": int(positions.get(inst, 0))
        })

    return pd.DataFrame(rows)


def get_latest_status():
    df = safe_read_csv(STATUS_FILE)
    if df.empty:
        return {
            "timestamp_utc": "-",
            "nav": None,
            "regime_bull": None,
            "signal": {},
            "target_position": {},
            "current_positions": {},
            "target_units": {},
            "order_deltas": {}
        }

    last = df.iloc[-1].to_dict()

    return {
        "timestamp_utc": last.get("timestamp_utc", "-"),
        "nav": last.get("nav", None),
        "regime_bull": last.get("regime_bull", None),
        "signal": parse_json_cell(last.get("signal_json")),
        "target_position": parse_json_cell(last.get("target_position_json")),
        "current_positions": parse_json_cell(last.get("current_positions_json")),
        "target_units": parse_json_cell(last.get("target_units_json")),
        "order_deltas": parse_json_cell(last.get("order_deltas_json")),
    }


def get_status_history():
    df = safe_read_csv(STATUS_FILE)
    if df.empty:
        return df

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        df = df.sort_values("timestamp_utc")

    return df


def get_trade_history():
    df = safe_read_csv(TRADE_LOG_FILE)
    if df.empty:
        return df

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        df = df.sort_values("timestamp_utc", ascending=False)

    return df


def get_live_strategy_state(client):
    try:
        _, state = generate_live_signal(client)
        return state
    except Exception:
        return None