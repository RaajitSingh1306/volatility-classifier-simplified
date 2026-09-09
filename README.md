# Volatility Regime Classifier - Nifty 50

An end-to-end Python project that classifies Nifty 50 market conditions into three volatility regimes: **Low Vol**, **Mid Vol**, and **High Vol**.
The project combines:

- Nifty 50 OHLCV data downloaded from Yahoo Finance
- 15 volatility, momentum, volume, and drawdown features
- A GARCH(1,1) model for annualized conditional volatility
- A three-state Gaussian Hidden Markov Model (HMM)
- SHAP explanations through a Gradient Boosting surrogate model
- A vectorbt regime-based strategy backtest against Buy & Hold
- A Streamlit dashboard
- A FastAPI REST service

This project is intended for educational and research use. It is not financial advice, and the strategy results are not a guarantee of future performance.
## What The Project Does

The complete pipeline is run by `run.py`:
1. Downloads Nifty 50 data for the period beginning on `2014-01-01`.
2. Loads a local OHLCV cache when it is less than seven days old.
3. Computes daily percentage returns and log returns.
4. Engineers the feature set described below and removes incomplete rolling rows.
5. Fits a GARCH(1,1) model to daily log returns and annualizes its conditional volatility.
6. Standardizes the 15 features and fits a three-state Gaussian HMM.
7. Relabels HMM states by their average GARCH volatility so labels are stable:
      `0 = Low Vol`, `1 = Mid Vol`, and `2 = High Vol`.
8. Trains a Gradient Boosting classifier to reproduce the HMM labels and uses permutation SHAP values to explain feature importance.
9. Runs a regime-based vectorbt backtest.
10. Saves the labeled data, backtest summary, and diagnostic plots.

The current data end date is the date on which the pipeline is run. Yahoo Finance data is automatically refreshed when the cache is older than seven days.
## Architecture

```text
Yahoo Finance (^NSEI)
      |
      v
data.py: OHLCV download, cache, returns
      |
      v
features.py: 15 engineered features
      |
      +----------------------+
      |                      |
      v                      v
garch_model.py           hmm_model.py
GARCH(1,1)               StandardScaler + Gaussian HMM
annualized volatility    stable Low/Mid/High labels
      |                      |
      +----------+-----------+
               v
        SHAP explainability
        Gradient Boosting surrogate
               |
               v
        backtest.py: vectorbt strategy
               |
        +--------+---------+
        v                  v
   Streamlit dashboard   FastAPI REST API
```

## Regime And Strategy Logic

| Regime | Numeric label | Strategy action | Interpretation |
|---|---:|---|---|
| Low Vol | `0` | Entry | Lower uncertainty and a potentially more stable trend |
| Mid Vol | `1` | Hold | Remain in the existing position |
| High Vol | `2` | Exit | Reduce exposure during elevated uncertainty |

The backtest enters when the regime is Low Vol and exits when it is High Vol.
Mid Vol produces no new signal. The strategy uses:

- Initial capital: `100,000`
- Fees: `0.1%` per trade
- Slippage: `0.1%`
- Frequency: daily (`D`)
- Benchmark: Buy & Hold with the same initial capital and no fees

The backtest reports total return, CAGR, Sharpe ratio, maximum drawdown, Calmar ratio, win rate, and number of trades.
## Engineered Features

All features are annualized or normalized where appropriate and are defined in `features.py`.

| Feature | Description |
|---|---|
| `rv_5d`, `rv_10d`, `rv_21d`, `rv_63d` | Rolling close-to-close realized volatility from log returns |
| `vol_ratio_5_21` | 5-day realized volatility divided by 21-day realized volatility |
| `vol_ratio_21_63` | 21-day realized volatility divided by 63-day realized volatility |
| `gk_vol_21` | 21-day annualized Garman-Klass OHLC volatility |
| `park_vol_21` | 21-day annualized Parkinson high-low volatility |
| `mom_5d`, `mom_10d`, `mom_21d`, `mom_63d` | Close-price percentage momentum over each lookback |
| `rsi_14` | 14-period Wilder RSI |
| `vol_ma_ratio` | Volume divided by its 21-day moving average |
| `drawdown` | Current close relative to the running all-time close peak |

Rows containing rolling-window NaN values are dropped before modelling.
## GARCH Model

`garch_model.py` fits a normal-innovation GARCH(1,1) model using the `arch` package. Returns are temporarily multiplied by 100 for numerical stability;
conditional volatility is then scaled back and annualized using 252 trading days.

The model exposes its AIC, BIC, and annualized `garch_vol` series. The generated GARCH diagnostic plot contains:

- Nifty 50 closing price
- Daily log returns
- Annualized conditional volatility
- Reference lines at 15% and 30% annualized volatility
## HMM And Regime Labels

The HMM uses:

- `GaussianHMM` from `hmmlearn`
- Three hidden states
- Full covariance matrices
- Up to 200 fitting iterations
- Random seed `42`
- Standardized engineered features

HMM state numbers are not inherently meaningful, so the raw states are sorted by their mean `garch_vol`. This makes the final labels consistent across model initializations. The labeled data contains both `raw_regime` and the remapped `regime`, plus the human-readable `regime_name`.
## SHAP Explainability

The HMM itself does not provide direct feature importance. The project therefore trains a post-hoc `GradientBoostingClassifier` on the engineered features and the HMM regime labels. SHAP permutation explanations are calculated on up to:

- 100 background samples
- 250 explanation samples

The resulting mean absolute SHAP importance is saved as a horizontal bar chart and returned as a feature-indexed pandas Series. This explains which features help reproduce the HMM assignments; it is not a causal explanation of market movements.
## Installation

Python 3.11 or newer is recommended.

```bash
git clone <repository-url>
cd volatility-classifier-simplified
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include pandas, NumPy, yfinance, arch, hmmlearn, scikit-learn, SHAP, vectorbt, Matplotlib, Plotly, Streamlit, FastAPI, Uvicorn, Pydantic, and pytest.
## Run The Pipeline

From the project root:

```bash
python run.py
```

The pipeline needs network access when `data/nifty_ohlcv.csv` is absent or more than seven days old. It creates `data/` and `outputs/` automatically.

Individual modules also expose standalone entry points:

```bash
python data.py
python features.py
python garch_model.py
python hmm_model.py
python backtest.py
```

The standalone model scripts expect the preceding pipeline data to exist where appropriate. Running `python run.py` is the supported end-to-end workflow.
## Streamlit Dashboard

Generate `data/labeled_data.csv` first, then start the dashboard:

```bash
streamlit run app/dashboard.py
```

The dashboard provides:

- Current regime, annualized GARCH volatility, latest Nifty 50 price, and days in the current regime
- A configurable lookback slider from 252 trading days to the full dataset
- A checkbox to show or hide GARCH volatility
- A multi-select filter for Low, Mid, and High Vol regimes
- Price history with regime shading and an optional GARCH volatility overlay
- Regime distribution for the selected period
- Per-regime days, mean volatility, mean return, and return standard deviation
- A violin plot of volatility distribution by regime
- The last 100 rows of selected raw data

If `data/labeled_data.csv` is missing, the dashboard attempts to run the full pipeline automatically. For repeatable deployments, run the pipeline before starting Streamlit and provide the generated data file.
## FastAPI Service

Generate the pipeline outputs, then start the API from the project root:

```bash
uvicorn app.main:app --reload
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Response |
|---|---|---|
| `GET` | `/` | Health status and API version |
| `GET` | `/current` | Most recent date, regime, regime name, GARCH volatility percentage, and close |
| `GET` | `/regimes?last_n=252` | Up to 10,000 recent trading-day regime points |
| `GET` | `/stats` | Aggregate days, mean volatility, mean return, and return standard deviation per regime |
| `GET` | `/backtest-summary` | Strategy and Buy & Hold metrics from `data/backtest_summary.csv` |

`/regimes` accepts `last_n` values from 1 through 10,000. The API reads the labeled CSV once and caches it in memory. Restart the service after regenerating the CSV so it loads the new data.

If `data/labeled_data.csv` has not been generated, regime endpoints return HTTP 503 with an instruction to run the pipeline. If the backtest summary is absent, `/backtest-summary` returns HTTP 404.
## Generated Files

| Path | Generated by | Description |
|---|---|---|
| `data/nifty_ohlcv.csv` | `data.py` | Cached Nifty 50 OHLCV data plus percentage and log returns |
| `data/labeled_data.csv` | `run.py` / `hmm_model.py` | Features, GARCH volatility, raw and remapped HMM regimes, and regime names |
| `data/backtest_summary.csv` | `run.py` / `backtest.py` | Strategy versus Buy & Hold metrics |
| `outputs/garch_vol.png` | `garch_model.py` | Price, returns, and GARCH volatility chart |
| `outputs/regimes.png` | `hmm_model.py` | Price regime shading and regime-colored volatility chart |
| `outputs/shap_summary.png` | `hmm_model.py` | Mean absolute SHAP feature importance chart |
| `outputs/backtest.png` | `backtest.py` | Equity curves and drawdown comparison |

The CSV files under `data/` are useful for local reuse. Generated plots under `outputs/` are ignored by Git by default.
## Project Structure

```text
volatility-classifier-simplified/
|-- data.py                 # Yahoo Finance download, cache, and returns
|-- features.py             # Feature engineering and volatility estimators
|-- garch_model.py          # GARCH fitting and diagnostic plot
|-- hmm_model.py            # HMM labels, regime plot, and SHAP explanation
|-- backtest.py             # vectorbt strategy and benchmark comparison
|-- run.py                  # End-to-end pipeline
|-- requirements.txt        # Runtime and test dependencies
|-- pytest.ini              # Pytest configuration
|-- app/
|   |-- main.py             # FastAPI application
|   |-- dashboard.py        # Streamlit dashboard
|-- data/
|   |-- nifty_ohlcv.csv     # Cached input data
|   |-- labeled_data.csv    # Pipeline output
|   |-- backtest_summary.csv
|-- outputs/                # Generated PNG diagnostics
|-- tests/
|   |-- test_features.py
|   |-- test_garch.py
|   |-- test_hmm.py
|   |-- test_backtest.py
|-- .github/workflows/ci.yml
```

## Run Tests

```bash
pytest tests/ -v
```

The test suite covers:

- Garman-Klass and Parkinson volatility estimators
- Feature columns, null removal, and rolling-window behavior
- GARCH output presence and positive volatility
- HMM state count, valid labels, and null-free assignments
- Backtest summary output and positive portfolio value

Continuous integration runs the same test command on Python 3.11 after installing `requirements.txt`.

## Important Limitations

- Yahoo Finance is an external data source and may be unavailable or change its response format.
- The cache is local and expires after seven days; it is not a production data store.
- The HMM and scaler are fitted during the pipeline run but are not serialized for later online inference.
- The API serves classifications already saved in `data/labeled_data.csv`; it does not classify new prices on request.
- The SHAP model is a surrogate for explaining HMM labels, not the HMM itself and not a causal market model.
- The backtest is historical and does not model taxes, market impact, liquidity constraints, or live order execution.
- Strategy signals are based on classifications available in the historical labeled dataset and should not be treated as a live trading system.

## License

No license file is currently included in this repository. Add a license before redistributing the project.
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
