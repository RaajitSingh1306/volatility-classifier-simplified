# Volatility Regime Classifier — Nifty 50 (Simplified)

> [!WARNING]
> **SUPERSEDED PROJECT (v2 Single-Asset Refactor)**
> This version is **SUPERSEDED** by the production-grade **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)** (the final shipped version).
> While this repository represents the clean, 15-feature GARCH + HMM single-asset pipeline with vectorbt backtesting, the **Volatility Intelligence Platform** expands upon it by introducing:
> - **Forward-looking XGBoost predictive layer** (predicts tomorrow's regime from today's signals, OOF AUC: 0.7453)
> - **MLflow experiment tracking** (`mlflow.db`)
> - **Next.js 14 Web Dashboard** (replacing Streamlit with an institutional UI)
> - **Docker & multi-service architecture**
> 
> For the active production deployment, visit the [Volatility Intelligence Platform repository](https://github.com/RaajitSingh1306/volatility-intelligence-platform).

[![CI Tests](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen)](#testing)
[![HMM States](https://img.shields.io/badge/HMM%20States-3%20(Gaussian)-blue)](#how)
[![GARCH](https://img.shields.io/badge/GARCH(1%2C1)-Annualized%20Vol-orange)](#how)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| **REST API** | https://volatility-classifier-simplified.onrender.com |
| **API Docs** | https://volatility-classifier-simplified.onrender.com/docs |
| **Dashboard** | `streamlit run app/dashboard.py` (local) |

---

## What

This project classifies Nifty 50 market conditions into three **volatility regimes** — Low, Mid, and High — using an end-to-end quantitative pipeline.

Given today's Nifty 50 price data, the system:

1. Downloads OHLCV data from Yahoo Finance (`^NSEI`, 2014 – present)
2. Engineers **15 volatility, momentum, and risk features**
3. Estimates conditional volatility with **GARCH(1,1)**
4. Identifies latent market states with a **3-state Gaussian Hidden Markov Model (HMM)**
5. Explains which features drive each regime classification using **SHAP**
6. Backtests a regime-switching strategy against Buy & Hold using **vectorbt**
7. Serves results through a **FastAPI REST API** and **Streamlit dashboard**

### What Each Regime Means

| Regime | Label | Market Conditions | Strategy Action |
|---|---:|---|---|
| **Low Vol** | `0` | Calm, trending markets — low uncertainty | **Enter** (go long) |
| **Mid Vol** | `1` | Moderate fluctuation — consolidation phase | **Hold** current position |
| **High Vol** | `2` | Turbulent, elevated risk — crisis periods | **Exit** to cash |

---

## Why

### The Problem

Most retail investors and portfolio managers rely on intuition to assess "how risky the market feels right now." This leads to:

- **Late exits** — staying invested during rising volatility until a drawdown forces capitulation
- **Missed entries** — waiting too long after turbulence ends, missing the recovery rally
- **No systematic framework** — no objective, repeatable way to define what "low volatility" or "high volatility" actually means on any given day

### The Solution

This project replaces intuition with a statistical framework:

| Problem | Solution |
|---|---|
| "How volatile is the market right now?" | GARCH(1,1) provides a **conditional volatility estimate** updated daily |
| "What regime are we in?" | HMM discovers **three latent states** from 15 features — no arbitrary thresholds |
| "Why is it classified this way?" | SHAP explainability shows **which features** pushed the model toward each state |
| "Does this regime signal actually work for trading?" | vectorbt backtest with realistic fees and slippage proves **strategy vs Buy & Hold** |

### Why "Simplified"?

This is a **lightweight, self-contained version** of the full [Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform). The simplified version:

- Uses **HMM-only** classification (no XGBoost prediction layer)
- Has **no MLflow** experiment tracking
- Has **no Next.js frontend** (uses Streamlit instead)
- Deploys with a **Python runtime** on Render (no Docker build required)
- Requires less RAM and builds faster on free-tier hosting

Use this project if you want to understand the core GARCH → HMM → Backtest pipeline without the additional ML prediction and experiment tracking layers.

---

## How

### Pipeline Architecture

```
Yahoo Finance (^NSEI)
      │
      ▼
data.py ─── Download OHLCV, 7-day TTL cache, compute returns
      │
      ▼
features.py ─── Engineer 15 features
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
garch_model.py           hmm_model.py
GARCH(1,1)               StandardScaler + 3-State Gaussian HMM
Annualized volatility    Monotonic state sorting by mean vol
      │                      │
      └──────────┬───────────┘
                 ▼
         SHAP Explainability
         (Gradient Boosting surrogate)
                 │
                 ▼
         backtest.py ─── vectorbt regime-switching strategy
                 │
         ┌───────┴────────┐
         ▼                ▼
   Streamlit Dashboard   FastAPI REST API
```

### Step 1: Data Ingestion (`data.py`)

Downloads Nifty 50 OHLCV data from Yahoo Finance for the period 2014-01-01 to present. Implements a **7-day local cache** to avoid redundant API calls. Computes daily percentage returns and continuous log returns.

### Step 2: Feature Engineering (`features.py`)

Computes 15 quantitative features from the raw OHLCV data:

| Category | Features | Count |
|---|---|---|
| **Realized Volatility** | `rv_5d`, `rv_10d`, `rv_21d`, `rv_63d` — rolling std of log returns | 4 |
| **Volatility-of-Volatility** | `vol_ratio_5_21`, `vol_ratio_21_63` — short/long vol ratios | 2 |
| **OHLC Estimators** | `gk_vol_21` (Garman-Klass), `park_vol_21` (Parkinson) — intraday range-based | 2 |
| **Momentum** | `mom_5d`, `mom_10d`, `mom_21d`, `mom_63d` — cumulative returns | 4 |
| **Microstructure** | `rsi_14`, `vol_ma_ratio`, `drawdown` — RSI, volume trend, peak-to-trough | 3 |

### Step 3: GARCH(1,1) Volatility (`garch_model.py`)

Fits a GARCH(1,1) model to daily log returns to estimate **time-varying conditional volatility**:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

The model captures **volatility clustering** — the empirical observation that large price moves tend to follow large price moves. Returns are scaled by 100 before estimation for numerical stability.

### Step 4: HMM Regime Classification (`hmm_model.py`)

Fits a **3-state Gaussian HMM** with full covariance matrices on the standardized 15-feature set. The HMM discovers three latent market states through unsupervised learning, without requiring manually defined thresholds.

**State sorting:** HMM hidden states have arbitrary indices. The pipeline sorts states by ascending mean GARCH volatility so that `0` always means Low Vol, `1` always means Mid Vol, and `2` always means High Vol — guaranteeing consistent semantics across training runs.

### Step 5: SHAP Explainability

Since HMM doesn't natively support feature importance, a **Gradient Boosting surrogate model** is trained on the same features with HMM labels as targets. SHAP TreeExplainer then computes exact Shapley values showing which features contribute most to each regime classification.

### Step 6: Backtest (`backtest.py`)

Runs a **regime-switching strategy** using vectorbt:

| Parameter | Value |
|---|---|
| **Initial Capital** | ₹1,00,000 |
| **Entry Signal** | Regime = Low Vol (`0`) |
| **Exit Signal** | Regime = High Vol (`2`) |
| **Fees** | 0.1% per trade |
| **Slippage** | 0.1% |
| **Benchmark** | Nifty 50 Buy & Hold |

---

## Evaluation & Verification

### Backtest Results (2014 – Present)

| Metric | Regime Strategy | Buy & Hold |
|---|---|---|
| **Sharpe Ratio** (Rf = 6.5%) | 0.968 | 0.873 |
| **Max Drawdown** | -25.92% | -38.44% |
| **Total Return** | +262.01% | +249.48% |
| **CAGR** | 16.61% | 16.13% |
| **Calmar Ratio** | 0.641 | 0.419 |

Key takeaway: The regime strategy achieves a **comparable return** with **12.5% less max drawdown** — the primary value is **capital protection** during turbulent periods.

### Testing

12/12 tests passing:

| Module | Tests | Coverage |
|---|---|---|
| `test_features.py` | 3 | Feature count, NaN handling, column names |
| `test_garch.py` | 3 | Stationarity (α+β < 1), volatility bounds, output shape |
| `test_hmm.py` | 3 | State count, label monotonicity, regime name mapping |
| `test_backtest.py` | 3 | Strategy execution, Sharpe calculation, drawdown bounds |

```bash
pytest tests/ -v
```

---

## Where — Project Structure

```
volatility-classifier-simplified/
│
├── data.py                  # Yahoo Finance OHLCV download with 7-day cache
├── features.py              # 15 technical and volatility features
├── garch_model.py           # GARCH(1,1) conditional volatility estimation
├── hmm_model.py             # 3-state Gaussian HMM + SHAP explainability
├── backtest.py              # vectorbt regime-switching strategy backtest
├── run.py                   # Master pipeline orchestrator (runs all steps)
│
├── app/
│   ├── main.py              # FastAPI REST API (/current, /regimes, /stats, /backtest-summary)
│   └── dashboard.py         # Streamlit interactive visualization dashboard
│
├── data/                    # Generated artifacts (after running pipeline)
│   ├── labeled_data.csv     # Full time series with regime labels and features
│   └── backtest_summary.csv # Strategy vs B&H performance metrics
│
├── outputs/                 # Generated diagnostic plots
│   ├── garch_vol.png        # GARCH conditional volatility over time
│   ├── regimes.png          # Regime classification timeline
│   ├── shap_summary.png     # SHAP feature importance
│   └── backtest.png         # Strategy equity curve vs benchmark
│
├── tests/                   # Pytest suite (12 tests)
│   ├── test_features.py
│   ├── test_garch.py
│   ├── test_hmm.py
│   └── test_backtest.py
│
├── Dockerfile               # Container with dynamic $PORT binding
├── docker-compose.yml       # Local multi-container orchestration
├── render.yaml              # Render Blueprint deployment manifest
├── requirements.txt         # Python dependencies
└── pytest.ini               # Pytest configuration
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### 1. Install & Run Pipeline

```bash
git clone https://github.com/RaajitSingh1306/volatility-classifier-simplified.git
cd volatility-classifier-simplified

pip install -r requirements.txt
python run.py
```

This generates:
- `data/labeled_data.csv` — processed time series with regime labels
- `data/backtest_summary.csv` — strategy performance metrics
- `outputs/` — diagnostic plots (GARCH, regimes, SHAP, backtest)

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Launch Services

**FastAPI:**
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for interactive Swagger documentation.

**Streamlit Dashboard:**
```bash
streamlit run app/dashboard.py
```
Open http://localhost:8501 for the interactive visualization.

### 4. Docker (Alternative)

```bash
docker compose build
docker compose up
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — `{"status": "ok", "message": "Volatility Regime Classifier API v1.0"}` |
| `GET` | `/current` | Latest regime classification, date, close price, GARCH volatility |
| `GET` | `/regimes?last_n=252` | Historical regime time series (default: 252 trading days) |
| `GET` | `/stats` | Per-regime aggregate statistics (days, mean vol, mean return) |
| `GET` | `/backtest-summary` | Strategy vs Buy & Hold performance comparison |

---

## Deployment

### FastAPI Backend → Render

**Option A: Blueprint (1-click)**
1. Push to GitHub
2. Render Dashboard → **New +** → **Blueprint** → select repo
3. Render reads `render.yaml` and configures automatically
4. Click **Apply**

**Option B: Manual**
1. Render → **New +** → **Web Service** → select repo
2. Configure:
   - **Runtime**: Python 3
   - **Build**: `pip install -r requirements.txt && python run.py`
   - **Start**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check**: `/`
   - **Plan**: Free
3. Click **Create Web Service**

### Streamlit Dashboard → Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Select repository → set Main file to `app/dashboard.py`
3. Deploy

---

## Related Projects

| Project | Difference |
|---|---|
| [Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) | Full version: adds XGBoost prediction layer, MLflow tracking, Next.js frontend, `/predict` and `/market-summary` endpoints |
| [SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot) | Compliance Q&A assistant that consumes the volatility API for real-time market intelligence |

---

## License & Disclaimer

MIT License. This project is for **educational and research purposes only** and does not constitute investment advice. Past performance of the backtested strategy does not guarantee future results.
