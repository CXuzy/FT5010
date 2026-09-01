# =========================================================
# run_live_trader.py
# =========================================================

import time
import traceback

from live_trading.config import (
    OANDA_API_KEY,
    OANDA_ACCOUNT_ID,
    OANDA_ENV,
    INSTRUMENTS,
)
from live_trading.oanda_client import OandaClient
from live_trading.live_signal_engine import generate_live_signal, print_live_signal
from live_trading.execution import (
    compute_target_units,
    compute_order_deltas,
    filter_small_orders,
)
from live_trading.logger import write_trade_log, write_status, reset_log_files

# how often to check for a new completed bar
POLL_SECONDS = 60


def execute_rebalance_once(client, last_processed_bar=None):
    """
    Pull latest complete candles, generate signal, and rebalance
    only if a new completed bar has appeared.
    """
    prices, state = generate_live_signal(client)
    latest_bar = state["timestamp"]

    # skip if this bar has already been processed
    if last_processed_bar is not None and latest_bar <= last_processed_bar:
        print(f"[SKIP] No new completed bar. latest_bar={latest_bar}")
        return last_processed_bar

    print("\n===== NEW BAR DETECTED =====")
    print_live_signal(state)
    print()

    # account snapshot
    summary = client.get_account_summary()
    nav = float(summary["NAV"])

    # live positions
    current_positions = client.get_positions_map()

    # target positions
    target_units = compute_target_units(
        state=state,
        prices=prices,
        nav=nav,
        instruments=INSTRUMENTS
    )

    # required order changes
    deltas = compute_order_deltas(
        current_units=current_positions,
        target_units=target_units,
        instruments=INSTRUMENTS
    )

    filtered_deltas = filter_small_orders(deltas)

    print("===== ACCOUNT =====")
    print("NAV:", nav)
    print()

    print("===== CURRENT POSITIONS =====")
    print(current_positions)
    print()

    print("===== TARGET UNITS =====")
    print(target_units)
    print()

    print("===== FILTERED ORDER DELTAS =====")
    print(filtered_deltas)
    print()

    # always write latest status for dashboard/logging
    write_status(
        nav=nav,
        regime=state["is_bull"],
        signal=state["latest_signal"],
        target_position=state["latest_position"],
        current_positions=current_positions,
        target_units=target_units,
        order_deltas=filtered_deltas
    )

    # if no actual rebalance needed, just move on
    if not filtered_deltas:
        print("[INFO] No orders to send.")
        return latest_bar

    # send orders automatically
    print("===== AUTO SENDING ORDERS =====")
    for inst, units in filtered_deltas.items():
        print(f"Sending market order: {inst}, units={units}")
        resp = client.place_market_order(inst, units)
        print(resp)
        print()

        write_trade_log(
            event_type="EXECUTE_REBALANCE_AUTO",
            instrument=inst,
            units=units,
            nav=nav,
            regime=state["is_bull"],
            signal=state["latest_signal"],
            target_position=state["latest_position"],
            current_positions=current_positions,
            target_units=target_units,
            order_deltas=filtered_deltas,
            response=resp
        )

    return latest_bar


def main():
    reset_log_files()
    client = OandaClient(
        api_key=OANDA_API_KEY,
        account_id=OANDA_ACCOUNT_ID,
        env=OANDA_ENV
    )

    last_processed_bar = None

    print("===== LIVE AUTO TRADER STARTED =====")
    print("OANDA ENV:", OANDA_ENV)
    print("Polling every", POLL_SECONDS, "seconds")
    print()

    while True:
        try:
            last_processed_bar = execute_rebalance_once(
                client=client,
                last_processed_bar=last_processed_bar
            )
        except Exception as e:
            print("[ERROR]", str(e))
            traceback.print_exc()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()