# OANDA API-Based Foreign Exchange Momentum Trading System

This is a fintech engineering project that builds a small foreign exchange trading system. It can read market data, generate trading signals, simulate or place orders through the OANDA practice API, save trading logs, and show the results on a dashboard.

In simple terms, the system answers one question:

> If several currency pairs are moving in different directions, can a program detect the stronger trends, decide what positions to hold, control risk, and record every trading decision?

The project uses a practice trading environment, not real-money trading.

---

## What The Project Does

The system follows a complete trading workflow:

1. Collect recent price data for major currency pairs.
2. Calculate simple trend signals using moving averages.
3. Decide whether each currency pair should be long, short, or neutral.
4. Convert those decisions into target portfolio weights.
5. Adjust exposure based on market regime and volatility.
6. Compare target positions with current positions.
7. Send only the required rebalance orders.
8. Save status and trade logs as CSV files.
9. Display NAV, benchmark comparison, positions, and trades in a dashboard.

The main trading universe is:

- `EUR_USD`
- `GBP_USD`
- `USD_JPY`

---

## Why This Project Matters

Many trading ideas look good in a notebook but fail when they need to run as a system. This project focuses on the engineering side of a trading strategy:

- turning raw market data into structured signals
- separating strategy, execution, logging, and dashboard modules
- making position sizing configurable
- tracking every order decision through logs
- adding a kill switch for emergency position closeout
- using a dashboard to monitor system state

So the project is not only about foreign exchange. It is also a demonstration of data pipeline design, automation, monitoring, and risk-aware decision logic.

---

## Strategy Logic

### Momentum Signal

For each currency pair, the system compares a short moving average with a long moving average.

- If the short moving average is above the long moving average, the signal is positive.
- If the short moving average is below the long moving average, the signal is negative.
- If there is no clear direction, the signal is neutral.

This is a simple way to estimate whether the recent price trend is upward or downward.

### Portfolio Construction

The system converts the signals into target portfolio weights. If more than one currency pair has an active signal, the portfolio is normalized so that exposure is distributed across the active opportunities.

### Regime Filter

The strategy also builds a broad market indicator from the currency basket. This helps classify the environment as a bull or bear regime, so the system can use different risk settings under different market conditions.

### Volatility Targeting

The system estimates recent volatility and adjusts leverage. When the market is more volatile, the system can reduce exposure. When the environment is more stable, it can allow higher exposure within a configured cap.

### Transaction And Execution Awareness

Instead of assuming every signal should become a new trade immediately, the system compares current positions with target positions and only sends the difference as an order. This reduces unnecessary trading and keeps the logs easier to audit.

---

## System Modules

```text
.
├── backtesting/
│   └── backtest.ipynb
├── dashboard/
│   ├── app.py
│   ├── callbacks.py
│   ├── components.py
│   ├── layout.py
│   ├── services.py
│   └── utils.py
├── data/
│   ├── status.csv
│   ├── trade_log.csv
│   └── kill_log.csv
├── live_trading/
│   ├── config.py
│   ├── execution.py
│   ├── kill_switch.py
│   ├── live_signal_engine.py
│   ├── logger.py
│   ├── oanda_client.py
│   ├── run_kill_switch.py
│   ├── run_live_trader.py
│   └── strategy.py
├── .env.example
├── requirements.txt
└── README.md
```

### `live_trading/`

Contains the trading engine. It connects to OANDA, retrieves candle data, generates signals, calculates target units, sends rebalance orders, and writes logs.

### `dashboard/`

Contains the monitoring dashboard built with Dash and Plotly. It reads CSV logs and shows portfolio NAV, benchmark movement, current positions, target units, order deltas, and recent trades.

### `data/`

Contains sample output logs used by the dashboard. These files make it possible to view the dashboard without running a live trading session.

### `backtesting/`

Contains the notebook used for strategy testing and analysis.

---

## Security Note

Real OANDA credentials are not stored in this repository.

The project reads credentials from environment variables:

- `OANDA_API_KEY`
- `OANDA_ACCOUNT_ID`
- `OANDA_ENV`

Use `.env.example` as a template and create your own local `.env` file. The `.env` file is ignored by Git.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_oanda_account_id
OANDA_ENV=practice
```

---

## Run The Trading Engine

Start the live trading loop:

```bash
python -m live_trading.run_live_trader
```

Run the emergency kill switch:

```bash
python -m live_trading.run_kill_switch
```

---

## Run The Dashboard

Start the dashboard:

```bash
python dashboard/app.py
```

Then open:

```text
http://127.0.0.1:8050
```

The dashboard can display the sample CSV logs in `data/`, so it can be used as a project demo even without connecting to OANDA.

---

## Key Takeaways

This project demonstrates:

- Python data processing with pandas and NumPy
- REST API integration with OANDA
- modular strategy and execution design
- automated position sizing and rebalancing
- risk control through volatility targeting and leverage caps
- structured CSV logging
- dashboard-based monitoring and visualization
