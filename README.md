# Volatility Regime Classifier — Nifty 50 (Simplified v2 Refactor)

> [!WARNING]
> **SUPERSEDED ARCHITECTURE (v2 Single-Asset Refactor)**
> This version is **SUPERSEDED** by the production-grade **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)** (v3 flagship).
> While this repository represents the clean, 15-feature GARCH + HMM modular pipeline with vectorbt backtesting, the **Volatility Intelligence Platform** expands upon it by introducing:
> - **Forward-looking XGBoost predictive layer** (predicting Day $T+1$ regime from today's signals, OOF AUC: 0.7453)
> - **MLflow experiment tracking** (`mlflow.db`)
> - **Next.js 14 Web Dashboard** (replacing Streamlit with an institutional UI)
> - **Docker containerization & multi-service architecture**
> 
> For the active production deployment, visit the [VIP repository](https://github.com/RaajitSingh1306/volatility-intelligence-platform).

[![Status: Superseded](https://img.shields.io/badge/Status-SUPERSEDED%20(v2%20Refactor)-critical)](#evolution-lineage)
[![CI Tests](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen)](#8-results--evaluation)
[![HMM States](https://img.shields.io/badge/HMM%20States-3%20(Gaussian)-blue)](#6-step-by-step-pipeline)
[![GARCH](https://img.shields.io/badge/GARCH(1%2C1)-Annualized%20Vol-orange)](#6-step-by-step-pipeline)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#12-deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

| Resource | URL |
|---|---|
| **Live REST API** | [https://volatility-classifier-simplified.onrender.com](https://volatility-classifier-simplified.onrender.com) |
| **Interactive API Docs** | [https://volatility-classifier-simplified.onrender.com/docs](https://volatility-classifier-simplified.onrender.com/docs) |
| **Local Dashboard** | `streamlit run app/dashboard.py` |

An end-to-end quantitative pipeline classifying **Nifty 50 (`^NSEI`)** market conditions into three volatility regimes (Low, Mid, High). The project downloads daily OHLCV data from Yahoo Finance, engineers **15 volatility and momentum features**, fits conditional heteroskedasticity with GARCH(1,1), identifies latent market regimes with a 3-state Gaussian HMM, explains decisions via a Gradient Boosting surrogate model with TreeSHAP, and runs a realistic fee-adjusted backtest using **vectorbt**, served via **FastAPI** and **Streamlit**.

---

## Evolution Lineage

1. **v1 — Volatility Regime Classifier**: [GitHub: Volatility-Regime-Classifier](https://github.com/RaajitSingh1306/Volatility-Regime-Classifier) — Initial dual-asset prototype with 3 features and Kernel SHAP. *(Superseded)*
2. **v2 — Volatility Classifier Simplified (This Repository)**: [GitHub: volatility-classifier-simplified](https://github.com/RaajitSingh1306/volatility-classifier-simplified) — Modular single-asset pipeline refactor with 15 features, vectorbt backtesting, and surrogate TreeSHAP. *(Superseded)*
3. **v3 — Volatility Intelligence Platform (Final Production Flagship)**: [GitHub: volatility-intelligence-platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) — Production platform adding forward-looking XGBoost predictive layer, walk-forward CV (AUC 0.7453), MLflow experiment tracking, Docker containerization, and Next.js 14 web interface. *(Active)*

---

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Why It Was Built](#2-why-it-was-built)
- [3. System Architecture](#3-system-architecture)
- [4. Tech Stack & Libraries](#4-tech-stack--libraries)
- [5. Data](#5-data)
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
- [7. Problems Faced & How We Solved Them](#7-problems-faced--how-we-solved-them)
- [8. Results & Evaluation](#8-results--evaluation)
- [9. Project Structure](#9-project-structure)
- [10. Getting Started](#10-getting-started)
- [11. API Reference](#11-api-reference)
- [12. Deployment](#12-deployment)
- [13. Connected Portfolio Projects](#13-connected-portfolio-projects)
- [14. Limitations & Known Issues](#14-limitations--known-issues)
- [15. Roadmap / Evolution Path](#15-roadmap--evolution-path)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. What This Project Does

Given daily historical price data for the Nifty 50 index (`^NSEI`), the system:

- **Downloads Market Data**: Fetches OHLCV records via Yahoo Finance with an automated 7-day TTL disk cache.
- **Engineers 15 Advanced Quant Features**: Computes realized rolling volatilities, volatility-of-volatility ratios, Parkinson and Garman-Klass intraday range estimators, momentum windows, RSI, and drawdown metrics.
- **Models Conditional Variance**: Estimates GARCH(1,1) time-varying conditional volatility updated daily.
- **Discovers Latent Regimes**: Applies an unsupervised 3-state Gaussian Hidden Markov Model (HMM) to group market states into Low, Mid, and High volatility without arbitrary manual cutoffs.
- **Explains Decisions via Surrogate SHAP**: Trains a Gradient Boosting surrogate model on HMM states to unlock exact, sub-25ms TreeSHAP feature attributions.
- **Executes Realistic Backtest (vectorbt)**: Simulates a regime-switching strategy with 0.1% fees, 0.1% slippage, and a 1-day execution lag.
- **Serves Dual Frontends**: Delivers a production FastAPI backend on Render and a local Streamlit visualization dashboard.

### Regime Definitions & Systematic Actions

| Regime | Label | Market Environment | Systematic Strategy Action |
|---|:---:|---|---|
| **Low Vol** | `0` | Calm, sustained bull trend — low uncertainty | **Enter / Long Exposure (1.0x)** |
| **Mid Vol** | `1` | Rangebound consolidation — intermediate volatility | **Hold Current Exposure** |
| **High Vol** | `2` | Acute turbulence, panic liquidation — crisis risk | **Exit to Cash (0.0x)** |

---

## 2. Why It Was Built

- **Replacing Market Intuition with Statistics**: Retail investors and portfolio managers often rely on qualitative sentiment to gauge market risk. This leads to late exits during rising volatility and delayed entries during market recovery rallies.
- **Objective, Unsupervised Regime Discovery**: Instead of guessing whether a 20% VIX reading represents high risk, the Gaussian HMM clusters multivariate return distributions into statistically distinct regimes.
- **Modular Single-Asset Refactoring**: Solved the limitations of the v1 prototype (which required precomputed CSVs and lacked reproducible backtests) by packaging an automated, self-contained Python pipeline.

---

## 3. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                            │
│                              (data.py)                                 │
│                                                                        │
│   - Yahoo Finance API (^NSEI, 2014 to present)                         │
│   - 7-Day TTL Local Disk Cache to eliminate redundant network calls    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                         │
│                            (features.py)                               │
│                                                                        │
│   15 Engineered Features:                                              │
│   - Realized Vol (rv_5d, rv_10d, rv_21d, rv_63d)                       │
│   - Vol-of-Vol Ratios (vol_ratio_5_21, vol_ratio_21_63)                │
│   - OHLC Range Estimators (Parkinson, Garman-Klass)                    │
│   - Momentum Windows (mom_5d, mom_10d, mom_21d, mom_63d)               │
│   - Microstructure & Drawdown (rsi_14, vol_ma_ratio, drawdown)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│           GARCH(1,1) ENGINE           │ │      GAUSSIAN HMM ENGINE     │
│           (garch_model.py)            │ │        (hmm_model.py)        │
│                                       │ │                              │
│  - Conditional heteroskedasticity     │ │  - 3-State Gaussian HMM      │
│  - Annualized conditional volatility  │ │  - Monotonic state remapping │
└───────────────────────────────────────┘ └──────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       TREE SHAP EXPLAINABILITY                         │
│                                                                        │
│   Gradient Boosting Surrogate Model ──► shap.TreeExplainer             │
│   Fast, exact polynomial-time attribution of HMM state decisions       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     BACKTESTING & SERVING LAYER                        │
│                                                                        │
│   vectorbt Engine (backtest.py):                                       │
│   - 0.1% fees, 0.1% slippage, 1-day lag vs Nifty 50 Buy & Hold         │
│         │                                                              │
│         ├───────────────────────────────┐                              │
│         ▼                               ▼                              │
│   FastAPI Backend (app/main.py)    Streamlit Dashboard                 │
│   (Render Cloud Web Service)       (app/dashboard.py)                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `>=3.11` | Core programming runtime | Modern high-performance scientific execution |
| **yfinance** | `^0.2.36` | Market data ingestion | Access to Nifty 50 adjusted daily OHLCV bars with local caching |
| **arch** | `^6.2.0` | Econometric volatility modeling | Industry-standard conditional heteroskedasticity GARCH(1,1) modeling |
| **hmmlearn** | `^0.3.0` | Unsupervised regime discovery | Efficient Baum-Welch EM implementation of Gaussian Hidden Markov Models |
| **scikit-learn** | `^1.3.0` | Preprocessing & surrogate modeling | `StandardScaler` and `GradientBoostingClassifier` for surrogate SHAP fitting |
| **SHAP** | `^0.44.0` | Model explainability | `TreeExplainer` providing exact polynomial-time Shapley attributions |
| **vectorbt** | `^0.25.0` | Vectorized backtesting | High-speed backtesting engine with realistic fees, slippage, and position tracking |
| **FastAPI** | `^0.109.0` | REST API backend | Asynchronous endpoints with Pydantic validation and automatic OpenAPI docs |
| **Streamlit** | `^1.31.0` | Interactive web dashboard | Rapid UI development for regime band visualization and backtest charts |
| **pytest** | `^8.0.0` | Automated test suite | 12 automated unit tests across feature math, GARCH bounds, and HMM states |

---

## 5. Data

- **Asset**: Nifty 50 Index (`^NSEI`).
- **Source**: Yahoo Finance API via `yfinance`.
- **Time Horizon**: 2014-01-01 to present (~2,700+ daily trading sessions).
- **Caching Mechanism**: Local CSV cache with a **7-day TTL check** to prevent redundant API queries.
- **15 Engineered Features**:

| Category | Features | Description |
|---|---|---|
| **Realized Volatility** | `rv_5d`, `rv_10d`, `rv_21d`, `rv_63d` | Rolling standard deviation of continuous log returns scaled by $\sqrt{252}$ |
| **Vol-of-Vol Ratios** | `vol_ratio_5_21`, `vol_ratio_21_63` | Short-term to medium/long-term realized volatility ratios |
| **OHLC Estimators** | `park_vol_21`, `gk_vol_21` | Parkinson (High-Low) and Garman-Klass (OHLC) intraday range volatility |
| **Momentum** | `mom_5d`, `mom_10d`, `mom_21d`, `mom_63d` | Multi-horizon cumulative price percentage changes |
| **Microstructure** | `rsi_14`, `vol_ma_ratio`, `drawdown` | Relative Strength Index, 20d volume moving average ratio, peak-to-trough drawdown |

---

## 6. Step-by-Step Pipeline

1. **Market Data Ingestion (`data.py`)**: Download Nifty 50 OHLCV data from Yahoo Finance; store to local cache if older than 7 days; calculate percentage and continuous log returns.
2. **Feature Engineering (`features.py`)**: Calculate the 15 quantitative features across realized volatility, Parkinson/Garman-Klass estimators, momentum, and RSI.
3. **GARCH(1,1) Volatility Modeling (`garch_model.py`)**: Scale returns by $\times 100$ and fit constant-mean GARCH(1,1):
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
4. **Gaussian HMM Regime Discovery (`hmm_model.py`)**: Standardize the 15-feature matrix; fit a 3-state Gaussian HMM with full covariance; sort states ascending by mean GARCH volatility so that $0 = \text{Low}$, $1 = \text{Mid}$, $2 = \text{High}$.
5. **Surrogate Model & SHAP Attribution (`hmm_model.py`)**: Train a `GradientBoostingClassifier` surrogate model on the HMM regime labels; initialize `shap.TreeExplainer` for instant feature importance extraction.
6. **Vectorized Backtesting (`backtest.py`)**: Simulate the regime-switching strategy using `vectorbt` with 0.1% fees, 0.1% slippage, and 1-day execution lag.
7. **Pipeline Orchestration (`run.py`)**: Execute all steps sequentially, persist `data/labeled_data.csv` and `data/backtest_summary.csv`, and render diagnostic plots to `outputs/`.
8. **Serving (`app/main.py`, `app/dashboard.py`)**: Expose REST endpoints via FastAPI and interactive charts via Streamlit.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **HMM has no native feature importance** | HMM's Baum-Welch EM algorithm does not produce feature weights or coefficients, rendering regime classifications black-box | Trained a **Gradient Boosting surrogate model** on the same features with HMM labels as targets, then applied `shap.TreeExplainer` on the surrogate for fast, exact Shapley attribution |
| **HMM state index permutation** (inherited from v1) | Random state numbering after convergence caused regime numbers to switch semantics arbitrarily across runs | Applied **monotonic ascending sort by mean GARCH volatility**, ensuring consistent $0 = \text{Low}$, $1 = \text{Mid}$, $2 = \text{High}$ semantics across every execution |
| **yfinance data freshness on repeated runs** | Re-downloading 10+ years of daily OHLCV data on every pipeline execution caused unnecessary network delays | Implemented a **7-day local disk cache** with TTL checking — data is downloaded fresh from Yahoo Finance only if the cached file is older than 7 days |
| **vectorbt dependency complexity** | vectorbt has heavy compiled dependencies (`numba`, LLVM) that triggered version conflicts on some environments | Pinned compatible package versions in `requirements.txt` and provided a Docker containerization configuration to bypass local compilation issues |
| **Surrogate SHAP approximation error is unquantified** | The surrogate gradient boosting model might not perfectly replicate every HMM decision boundary, introducing explanation variance | Documented this architectural limitation transparently. In v3 (VIP), this was resolved by replacing the surrogate with XGBoost as the primary classifier, allowing native `TreeExplainer` SHAP directly |

---

## 8. Results & Evaluation

### Backtest Performance Comparison (2014 – Present)

| Metric | Regime-Switching Strategy | Nifty 50 Buy & Hold Benchmark | Outperformance / Protection |
|---|:---:|:---:|:---:|
| **Sharpe Ratio ($R_f = 6.5\%$)** | **0.968** | 0.873 | **+0.095** |
| **Maximum Drawdown** | **-25.92%** | -38.44% | **+12.52% Capital Preservation** |
| **Total Cumulative Return** | **+262.01%** | +249.48% | **+12.53%** |
| **CAGR** | **16.61%** | 16.13% | **+0.48% / year** |
| **Calmar Ratio** | **0.641** | 0.419 | **+0.222** |

*Takeaway*: The regime strategy captures equivalent compounding returns while cutting maximum drawdown by more than 12.5 percentage points by shifting to cash during crisis drawdowns.

### Automated Test Suite (12/12 Tests Passing)

```bash
pytest tests/ -v
```

| Test Module | Tests | Verifications |
|---|:---:|---|
| `test_features.py` | 3 | Validates 15-feature count, NaN cleaning, and column headers |
| `test_garch.py` | 3 | Checks stationarity ($\alpha + \beta < 1$), volatility bounds, output arrays |
| `test_hmm.py` | 3 | Verifies 3-state count, label monotonicity ($0 < 1 < 2$), regime mapping |
| `test_backtest.py` | 3 | Confirms strategy signal execution, Sharpe calculation, drawdown sanity |

---

## 9. Project Structure

```text
volatility-classifier-simplified/
├── data.py                  # Yahoo Finance OHLCV download with 7-day TTL cache
├── features.py              # 15 technical and volatility feature calculations
├── garch_model.py           # GARCH(1,1) conditional volatility estimation
├── hmm_model.py             # 3-state Gaussian HMM + surrogate SHAP explainability
├── backtest.py              # vectorbt regime-switching backtesting engine
├── run.py                   # Master pipeline CLI orchestrator
│
├── app/
│   ├── main.py              # FastAPI REST service (/current, /regimes, /stats)
│   └── dashboard.py         # Streamlit interactive visualization frontend
│
├── data/                    # Generated datasets
│   ├── labeled_data.csv     # Full time series with regime labels and features
│   └── backtest_summary.csv # Strategy vs Buy & Hold performance metrics
│
├── outputs/                 # Generated diagnostic plots
│   ├── garch_vol.png        # GARCH conditional volatility timeline
│   ├── regimes.png          # Regime classification historical timeline
│   ├── shap_summary.png     # SHAP feature importance summary
│   └── backtest.png         # Strategy equity curve vs benchmark
│
├── tests/                   # Pytest test suite (12 tests)
│   ├── test_features.py
│   ├── test_garch.py
│   ├── test_hmm.py
│   └── test_backtest.py
│
├── Dockerfile               # Container definition with dynamic $PORT binding
├── docker-compose.yml       # Local container orchestration
├── render.yaml              # Render Blueprint deployment manifest
├── requirements.txt         # Production dependencies
└── README.md                # Project documentation
```

---

## 10. Getting Started

### 1. Environment Setup

```bash
cd "Volatility Classifier_simplified"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Master Pipeline

```bash
python run.py
```

Generates `data/labeled_data.csv`, `data/backtest_summary.csv`, and all diagnostic PNGs in `outputs/`.

### 3. Run Automated Tests

```bash
pytest tests/ -v
```

### 4. Launch Services

**FastAPI Backend (Terminal 1):**
```bash
uvicorn app.main:app --reload --port 8000
```
Swagger UI: `http://localhost:8000/docs`

**Streamlit Dashboard (Terminal 2):**
```bash
streamlit run app/dashboard.py
```
Web UI: `http://localhost:8501`

---

## 11. API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check and version status |
| `GET` | `/current` | Latest regime classification, date, close price, GARCH volatility |
| `GET` | `/regimes?last_n=252` | Historical regime series (default: 252 trading sessions) |
| `GET` | `/stats` | Per-regime aggregate metrics (days, mean volatility, mean returns) |
| `GET` | `/backtest-summary` | Strategy vs Buy & Hold performance metrics |

---

## 12. Deployment

### Render Deployment

The repository includes `render.yaml` for 1-click Render Blueprint deployment.
- **Service Type**: Web Service
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt && python run.py`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Docker Containerization

```bash
docker compose build
docker compose up
```

---

## 13. Connected Portfolio Projects

- **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**: Flagship v3 evolution adding forward XGBoost predictive modeling, MLflow experiment tracking, Docker, and Next.js 14.
- **[SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)**: Multi-agent compliance assistant that queries this volatility API to inject real-time risk intelligence into regulatory responses.
- **[Nifty-Time-Series](https://github.com/RaajitSingh1306/Nifty-Time-Series)**: Econometric baseline demonstrating why return variance must be modeled instead of directional price drift.

---

## 14. Limitations & Known Issues

- **Superseded by v3 VIP**: This v2 refactor has been superseded by the production Volatility Intelligence Platform.
- **Retrospective State Filtering**: Uses HMM filtering to identify current latent states; does not train a forward-looking predictive probability layer for Day $T+1$ (added in VIP).
- **Surrogate-Dependent Explainability**: Explains regime states via a Gradient Boosting surrogate model rather than native HMM parameter attribution.
- **No Experiment Tracking**: Hyperparameters and model iterations are stored locally without an experiment registry (MLflow added in VIP).
- **Streamlit Prototyping UI**: Dashboard is built in Streamlit rather than an enterprise responsive web framework (Next.js 14 added in VIP).

---

## 15. Roadmap / Evolution Path

This repository served as the bridge between the v1 prototype and the v3 flagship:
- **v1**: Multi-asset prototype with 3 features and manual CSV dependencies.
- **v2 (This Repo)**: 15 features, automated data caching, vectorbt backtesting, surrogate SHAP, 12 pytest tests.
- **v3 (VIP)**: Forward-looking XGBoost predictive layer ($P(z_{T+1} \mid x_T)$), walk-forward CV (AUC 0.7453), MLflow, Next.js 14 dark-mode dashboard.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This software is intended strictly for quantitative finance research and algorithmic backtesting education. It does not constitute investment advice.
