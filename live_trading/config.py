# =========================================================
# config.py
# =========================================================

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Oanda account info
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")   # practice for demo account

# Execution sizing
NOTIONAL_FRACTION = 0.20
MIN_ORDER_UNITS = 1

# Trading universe
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]

# Strategy parameters
SHORT_MA = 3
LONG_MA = 8
REGIME_MA = 60

# Risk settings
TARGET_VOL_BULL = 0.30
TARGET_VOL_BEAR = 0.20
LEV_CAP_BULL = 3.0
LEV_CAP_BEAR = 2.0

# Execution settings
GRANULARITY = "M1"
MAX_CANDLES = 300

# Portfolio settings
BENCHMARK = "EUR_USD"

# Logging files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

STATUS_FILE = os.path.join(DATA_DIR, "status.csv")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
KILL_LOG_FILE = os.path.join(DATA_DIR, "kill_log.csv")
