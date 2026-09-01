# =========================================================
# kill_switch.py
# =========================================================

from live_trading.oanda_client import OandaClient

def get_nonzero_positions(client: OandaClient):
    """
    Return only positions with non-zero long or short units.
    """
    positions = client.get_open_positions()
    active = []

    for p in positions:
        long_units = int(float(p.get("long", {}).get("units", "0")))
        short_units = int(float(p.get("short", {}).get("units", "0")))

        if long_units != 0 or short_units != 0:
            active.append({
                "instrument": p["instrument"],
                "long_units": long_units,
                "short_units": short_units
            })

    return active


def close_all_positions(client: OandaClient):
    """
    Close all active positions in the account.
    Only close the side that actually exists.
    """
    active_positions = get_nonzero_positions(client)

    results = []

    for p in active_positions:
        inst = p["instrument"]

        if p["long_units"] != 0 and p["short_units"] != 0:
            resp = client.close_position(inst, side="ALL")
        elif p["long_units"] != 0:
            resp = client.close_position(inst, side="long")
        elif p["short_units"] != 0:
            resp = client.close_position(inst, side="short")
        else:
            continue

        results.append({
            "instrument": inst,
            "response": resp
        })

    return results