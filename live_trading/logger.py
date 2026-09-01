import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict
from live_trading.config import TRADE_LOG_FILE, KILL_LOG_FILE, STATUS_FILE


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_row(file_path: str, row: Dict[str, Any]) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    df_new = pd.DataFrame([row])
    file_exists = os.path.exists(file_path)

    df_new.to_csv(
        file_path,
        mode="a",
        header=not file_exists,
        index=False
    )


def write_trade_log(event_type: str,
                    instrument: str,
                    units: int,
                    nav: float,
                    regime: bool,
                    signal: Dict[str, Any],
                    target_position: Dict[str, Any],
                    current_positions: Dict[str, Any],
                    target_units: Dict[str, Any],
                    order_deltas: Dict[str, Any],
                    response: Dict[str, Any]) -> None:
    row = {
        "timestamp_utc": utc_now_str(),
        "event_type": event_type,
        "instrument": instrument,
        "units": units,
        "nav": nav,
        "regime_bull": regime,
        "signal_json": json.dumps(signal, default=str),
        "target_position_json": json.dumps(target_position, default=str),
        "current_positions_json": json.dumps(current_positions, default=str),
        "target_units_json": json.dumps(target_units, default=str),
        "order_deltas_json": json.dumps(order_deltas, default=str),
        "response_json": json.dumps(response, default=str)
    }
    _append_row(TRADE_LOG_FILE, row)


def write_kill_log(instrument: str,
                   response: Dict[str, Any]) -> None:
    row = {
        "timestamp_utc": utc_now_str(),
        "instrument": instrument,
        "response_json": json.dumps(response, default=str)
    }
    _append_row(KILL_LOG_FILE, row)


def write_status(nav: float,
                 regime: bool,
                 signal: Dict[str, Any],
                 target_position: Dict[str, Any],
                 current_positions: Dict[str, Any],
                 target_units: Dict[str, Any],
                 order_deltas: Dict[str, Any]) -> None:
    row = {
        "timestamp_utc": utc_now_str(),
        "nav": nav,
        "regime_bull": regime,
        "signal_json": json.dumps(signal, default=str),
        "target_position_json": json.dumps(target_position, default=str),
        "current_positions_json": json.dumps(current_positions, default=str),
        "target_units_json": json.dumps(target_units, default=str),
        "order_deltas_json": json.dumps(order_deltas, default=str)
    }
    _append_row(STATUS_FILE, row)

def reset_log_files():
    """
    Remove old log files at program startup so each run starts fresh.
    """
    for file_path in [STATUS_FILE, TRADE_LOG_FILE, KILL_LOG_FILE]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"[WARN] Failed to remove log file {file_path}: {e}")