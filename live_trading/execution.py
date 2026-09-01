from live_trading.config import NOTIONAL_FRACTION, MIN_ORDER_UNITS


def get_mid_price_from_last_close(prices, instrument: str) -> float:
    col = instrument.replace("_", "")
    return float(prices[col].iloc[-1])


def convert_usd_notional_to_units(instrument: str, usd_notional: float, price: float) -> int:
    """
    Convert USD notional into broker units depending on pair convention.
    """
    if instrument.endswith("_USD"):
        units = usd_notional / price
    elif instrument.startswith("USD_"):
        units = usd_notional
    else:
        raise ValueError(f"Unsupported instrument format for sizing: {instrument}")

    return int(units)


def compute_target_units(state: dict, prices, nav: float, instruments):
    """
    Convert target portfolio weights into target OANDA units.

    target_notional_usd = weight * NAV * NOTIONAL_FRACTION * latest_leverage
    """
    target_units = {}

    latest_leverage = float(state.get("latest_leverage", 1.0))
    if latest_leverage <= 0:
        latest_leverage = 1.0

    for inst in instruments:
        col = inst.replace("_", "")
        weight = float(state["latest_position"].get(col, 0.0))
        px = get_mid_price_from_last_close(prices, inst)

        usd_notional = weight * nav * NOTIONAL_FRACTION * latest_leverage
        units = convert_usd_notional_to_units(inst, usd_notional, px)

        target_units[inst] = units

    return target_units


def compute_order_deltas(current_units: dict, target_units: dict, instruments):
    deltas = {}

    for inst in instruments:
        curr = int(current_units.get(inst, 0))
        tgt = int(target_units.get(inst, 0))
        deltas[inst] = tgt - curr

    return deltas


def filter_small_orders(deltas: dict, min_order_units: int = MIN_ORDER_UNITS):
    filtered = {}

    for inst, units in deltas.items():
        if abs(units) >= min_order_units:
            filtered[inst] = units

    return filtered