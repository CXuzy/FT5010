# FT5010 Final Project  
## Cross-Sectional FX Momentum Live Trading System with Real-Time Dashboard

This project implements a modular live FX trading system built on the OANDA practice environment. It combines live market data retrieval, cross-sectional momentum signal generation, dynamic portfolio sizing, automated trade execution, risk control, persistent logging, and a real-time monitoring dashboard.

The system is designed as an end-to-end validation framework for live strategy deployment rather than a pure backtesting-only prototype.

---

# 1. Project Objective

The goal of this project is to design and validate a live foreign exchange trading workflow that can:

- generate systematic trading signals from live FX data
- allocate target positions across multiple currency pairs
- convert target exposure into executable broker units
- rebalance positions automatically through OANDA
- maintain risk control through leverage and kill switch logic
- record status and trade logs for analysis
- visualize live portfolio state through a professional dashboard

This project focuses on **practical strategy validation in a live trading environment** using a modular and extensible system architecture.

---

# 2. Strategy Overview

The trading strategy is a **cross-sectional FX momentum model** applied to a small basket of currency pairs:

- `EUR_USD`
- `GBP_USD`
- `USD_JPY`

## Core logic

### 2.1 Trend signal
For each instrument, the system computes:

- short moving average
- long moving average

Signal rule:

- `+1` if short MA > long MA
- `-1` if short MA < long MA
- `0` otherwise

### 2.2 Cross-sectional portfolio construction
Signals are converted into normalized target portfolio weights based on active long/short opportunities.

### 2.3 Regime filter
A USD-aligned basket index is constructed from the FX universe.  
A moving average is applied to classify the market regime:

- **Bull regime**
- **Bear regime**

### 2.4 Volatility targeting and leverage adjustment
The system estimates realized portfolio volatility and adjusts leverage dynamically according to:

- regime-specific target volatility
- regime-specific leverage cap

### 2.5 Execution
Target portfolio weights are transformed into broker units using:

- portfolio NAV
- notional fraction
- latest leverage
- latest market price
- instrument quotation convention

The system then compares:

- current live positions
- target positions

and submits only the required rebalance orders.

---

# 3. System Features

## Live trading engine
- fetches live candle data from OANDA
- generates latest strategy state
- computes target units
- compares against current holdings
- sends rebalance orders automatically

## Real-time dashboard
- displays account NAV and balance
- shows unrealized and realized PnL
- monitors margin usage
- visualizes strategy NAV history
- compares strategy vs benchmark
- displays current live positions vs target units
- shows recent trade logs
- provides emergency kill switch control

## Logging
The system maintains structured CSV logs for:

- strategy status
- executed trades
- kill switch actions

## Risk control
- leverage cap
- minimum order filtering
- manual kill switch for emergency position closeout

---

# 4. Project Structure

```text
.
│
├── dashboard/
│   ├── app.py
│   ├── callbacks.py
│   ├── components.py
│   ├── layout.py
│   ├── services.py
│   └── utils.py
│
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
│
├── data/
│   ├── status.csv
│   ├── trade_log.csv
│   └── kill_log.csv
│
└── README.md
```

---

# 5. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file or set environment variables before connecting to OANDA:

```bash
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_oanda_account_id
OANDA_ENV=practice
```

The repository intentionally does not store real OANDA credentials. Use `.env.example` as a template.

---

# 6. Run

Start the live trading loop:

```bash
python -m live_trading.run_live_trader
```

Run the kill switch:

```bash
python -m live_trading.run_kill_switch
```

Start the dashboard:

```bash
python dashboard/app.py
```

The dashboard reads the CSV files in `data/` and visualizes NAV, benchmark comparison, current positions, target units, order deltas, and trade logs.
