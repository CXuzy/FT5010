# =========================================================
# run_kill_switch.py
# =========================================================

from live_trading.config import OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV
from live_trading.oanda_client import OandaClient
from live_trading.kill_switch import get_nonzero_positions, close_all_positions
from live_trading.logger import write_kill_log


def main():
    client = OandaClient(
        api_key=OANDA_API_KEY,
        account_id=OANDA_ACCOUNT_ID,
        env=OANDA_ENV
    )

    try:
        active_positions = get_nonzero_positions(client)

        print("===== ACTIVE POSITIONS =====")
        if not active_positions:
            print("No active positions found.")
            return

        for p in active_positions:
            print(
                f"{p['instrument']} | "
                f"long_units={p['long_units']} | "
                f"short_units={p['short_units']}"
            )

        print()
        confirm = input("Type CLOSE to close ALL positions: ").strip()

        if confirm != "CLOSE":
            print("Cancelled.")
            return

        print("\n===== CLOSING POSITIONS =====")
        results = close_all_positions(client)

        if not results:
            print("No positions were closed.")
            return

        for item in results:
            inst = item.get("instrument", "UNKNOWN")
            resp = item.get("response", {})

            print(f"Closed: {inst}")
            print(resp)
            print()

            write_kill_log(
                instrument=inst,
                response=resp
            )

        print("===== KILL SWITCH COMPLETED =====")

    except Exception as e:
        print("ERROR:", str(e))


if __name__ == "__main__":
    main()