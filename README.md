# Volatility Regime Classifier — Nifty 50

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Classifies Nifty 50 market conditions into three volatility regimes — **Low**, **Mid**, and **High** — using GARCH(1,1) conditional volatility and a Hidden Markov Model (HMM). Regime labels are explained via SHAP feature importance and validated through a vectorbt regime-based backtest.

---

## Architecture

```
Yahoo Finance (yfinance)
        ↓
  Feature Engineering          15 features: realized vol, OHLC estimators,
  (features.py)                momentum, RSI, volume ratio, drawdown
        ↓
  GARCH(1,1)                   Conditional volatility (annualized)
  (garch_model.py)
        ↓
  Gaussian HMM (3 states)      Regime labels: Low / Mid / High Vol
  (hmm_model.py)
        ↓
  SHAP (GBM surrogate)         Feature importance explaining regime assignments
        ↓
  vectorbt Backtest            Regime-based strategy vs Buy & Hold
  (backtest.py)
        ↓
  FastAPI  +  Streamlit        REST API and interactive dashboard
  (app/)
```

---

## Strategy Logic

| Regime   | Signal | Rationale                             |
|----------|--------|---------------------------------------|
| Low Vol  | ENTRY  | Trending market, low uncertainty      |
| Mid Vol  | HOLD   | Stay in existing position             |
| High Vol | EXIT   | Protect capital, reduce drawdown risk |

Fees: 0.1% per trade · Slippage: 0.1% · Initial capital: ₹1,00,000

---

## Quickstart

## Live Demo

| Service | URL |
|---|---|
| Dashboard (Streamlit) | https://volatility-cassifier-simplified.streamlit.app/ |
| REST API (FastAPI) | https://volatility-cassifier-simplified.up.railway.app/ |
| API Docs | https://volatility-cassifier-simplified.up.railway.app/docs |

### 1. Clone and install

```bash
git clone https://github.com/RaajitSingh1306/volatility-classifier-simplified.git
cd volatility-classifier-simplified
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run.py
```

This will:
- Download Nifty 50 OHLCV data (2014–2026) and cache it in `data/`
- Engineer 15 features
- Fit GARCH(1,1) and save conditional volatility
- Fit a 3-state Gaussian HMM and assign regime labels
- Generate SHAP feature importance
- Run the regime-based backtest
- Save all outputs to `data/` and `outputs/`

### 3. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

### 4. Launch the API

```bash
uvicorn app.main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint              | Description                            |
|--------|-----------------------|----------------------------------------|
| GET    | `/`                   | Health check                           |
| GET    | `/current`            | Latest regime and GARCH vol            |
| GET    | `/regimes?last_n=252` | Historical regime time series          |
| GET    | `/stats`              | Per-regime aggregate statistics        |
| GET    | `/backtest-summary`   | Strategy vs Buy & Hold metrics         |

---

## Outputs

| File                         | Description                              |
|------------------------------|------------------------------------------|
| `outputs/garch_vol.png`      | GARCH conditional volatility chart       |
| `outputs/regimes.png`        | Price chart with regime shading          |
| `outputs/shap_summary.png`   | SHAP feature importance bar chart        |
| `outputs/backtest.png`       | Equity curve and drawdown comparison     |
| `data/labeled_data.csv`      | Full dataset with regime labels          |
| `data/backtest_summary.csv`  | Strategy vs benchmark performance table  |

---

## Project Structure

```
volatility-regime-classifier/
├── data.py             ← yfinance download + log-returns
├── features.py         ← 15 engineered features
├── garch_model.py      ← GARCH(1,1) conditional volatility
├── hmm_model.py        ← HMM regime labeling + SHAP
├── backtest.py         ← vectorbt regime strategy
├── run.py              ← master pipeline
├── requirements.txt
├── LICENSE
├── app/
│   ├── __init__.py
│   ├── main.py         ← FastAPI REST backend
│   └── dashboard.py    ← Streamlit frontend
└── tests/
    ├── __init__.py
    └── test_features.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dependencies

- **Data**: [yfinance](https://github.com/ranaroussi/yfinance)
- **GARCH**: [arch](https://github.com/bashtage/arch)
- **HMM**: [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- **Explainability**: [SHAP](https://github.com/shap/shap)
- **Backtesting**: [vectorbt](https://github.com/polakowo/vectorbt)
- **Dashboard**: [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/)
- **API**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past performance of any modelled strategy is not indicative of future results.

---
