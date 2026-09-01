# =========================================================
# oanda_client.py
# =========================================================

import requests
import pandas as pd


class OandaClient:
    def __init__(self, api_key: str, account_id: str, env: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id

        if env == "practice":
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, params=None):
        resp = self.session.get(self._url(path), params=params, timeout=20)
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, payload: dict):
        resp = self.session.post(self._url(path), json=payload, timeout=20)
        self._raise_for_status(resp)
        return resp.json()

    def _put(self, path: str, payload: dict):
        resp = self.session.put(self._url(path), json=payload, timeout=20)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp):
        if not resp.ok:
            try:
                msg = resp.json()
            except Exception:
                msg = resp.text
            raise RuntimeError(f"OANDA API error {resp.status_code}: {msg}")

    def get_account_summary(self) -> dict:
        path = f"/v3/accounts/{self.account_id}/summary"
        data = self._get(path)
        return data.get("account", data)

    def get_candles(self, instrument: str, count: int = 300, granularity: str = "H1", price: str = "M") -> pd.DataFrame:
        """
        instrument: e.g. EUR_USD
        granularity: e.g. H1
        price: M(mid), B(bid), A(ask)
        """
        path = f"/v3/instruments/{instrument}/candles"
        params = {
            "count": count,
            "granularity": granularity,
            "price": price
        }
        data = self._get(path, params=params)

        candles = data.get("candles", [])
        rows = []

        for c in candles:
            # skip incomplete latest candle
            if not c.get("complete", False):
                continue

            price_obj = c.get("mid") or c.get("bid") or c.get("ask")
            if not price_obj:
                continue

            rows.append({
                "time": c["time"],
                "open": float(price_obj["o"]),
                "high": float(price_obj["h"]),
                "low": float(price_obj["l"]),
                "close": float(price_obj["c"]),
                "volume": int(c.get("volume", 0))
            })

        if not rows:
            raise ValueError(f"No complete candles returned for {instrument}")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        return df

    def get_open_positions(self):
        path = f"/v3/accounts/{self.account_id}/openPositions"
        data = self._get(path)
        return data.get("positions", [])

    def place_market_order(self, instrument: str, units: int):
        """
        units > 0 => buy
        units < 0 => sell
        """
        path = f"/v3/accounts/{self.account_id}/orders"
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        return self._post(path, payload)

    def close_position(self, instrument: str, side: str = "ALL"):
        """
        Close long / short / all of one instrument.
        side:
          - "long"
          - "short"
          - "ALL"
        """
        path = f"/v3/accounts/{self.account_id}/positions/{instrument}/close"

        if side == "long":
            payload = {"longUnits": "ALL"}
        elif side == "short":
            payload = {"shortUnits": "ALL"}
        else:
            payload = {"longUnits": "ALL", "shortUnits": "ALL"}

        return self._put(path, payload)

    def get_positions_map(self):
        """
        Return current net units by instrument.
        Example:
        {
            "EUR_USD": -1000,
            "GBP_USD": 0,
            "USD_JPY": 2000
        }
        """
        positions = self.get_open_positions()
        pos_map = {}

        for p in positions:
            inst = p["instrument"]
            long_units = int(float(p.get("long", {}).get("units", "0")))
            short_units = int(float(p.get("short", {}).get("units", "0")))
            net_units = long_units + short_units   # short side is negative in Oanda payload
            pos_map[inst] = net_units

        return pos_map