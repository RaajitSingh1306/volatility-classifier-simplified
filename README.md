# Volatility Regime Classifier — Nifty 50

An end-to-end quantitative pipeline that classifies Nifty 50 market conditions into three volatility regimes: **Low Vol**, **Mid Vol**, and **High Vol**.

The project combines:
- Nifty 50 OHLCV data downloaded from Yahoo Finance (`^NSEI`)
- 15 engineered volatility, momentum, volume, and drawdown features
- GARCH(1,1) model for annualized conditional volatility
- Three-state Gaussian Hidden Markov Model (HMM)
- SHAP explanations via Gradient Boosting surrogate model
- vectorbt regime-based strategy backtest against Buy & Hold
- Interactive Streamlit dashboard
- High-performance FastAPI REST service deployed on Render

[![CI Tests](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen)](#)
[![HMM States](https://img.shields.io/badge/HMM%20States-3%20(Gaussian)-blue)](#)
[![GARCH](https://img.shields.io/badge/GARCH(1%2C1)-Annualized%20Vol-orange)](#)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20App-Live-FF4B4B)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Live Deployments

| Service | Platform | URL |
|---|---|---|
| **Dashboard** | Streamlit Cloud | `https://volatility-classifier-simplified.streamlit.app/` |
| **REST API** | Render | `https://volatility-classifier-simplified.onrender.com` |
| **API Docs (Swagger)** | Render | `https://volatility-classifier-simplified.onrender.com/docs` |
| **Health Check** | Render | `https://volatility-classifier-simplified.onrender.com/` |

---

## 1. What The Project Does

The master pipeline is orchestrated by `run.py`:
1. **Downloads** Nifty 50 data (`2014-01-01` to present) with local 7-day TTL caching.
2. **Computes** daily percentage returns and continuous log returns.
3. **Engineers** 15 quantitative features (realized vol, Garman-Klass, Parkinson, momentum, RSI, volume ratio, drawdown).
4. **Fits GARCH(1,1)** to daily log returns and estimates annualized conditional volatility.
5. **Fits 3-State Gaussian HMM** on standardized features and sorts states monotonically by mean volatility:
   - `0 = Low Vol` (trending, calm)
   - `1 = Mid Vol` (neutral, consolidation)
   - `2 = High Vol` (turbulent, elevated risk)
6. **Explains Predictions** with SHAP feature importance via a Gradient Boosting surrogate model.
7. **Executes Backtest** using `vectorbt` entering on Low Vol and exiting on High Vol.
8. **Saves Artifacts** to `data/labeled_data.csv` and `data/backtest_summary.csv`.

---

## 2. Architecture

```text
Yahoo Finance (^NSEI)
      │
      ▼
data.py: OHLCV download, 7-day cache, returns
      │
      ▼
features.py: 15 engineered features
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
garch_model.py           hmm_model.py
GARCH(1,1)               StandardScaler + Gaussian HMM
annualized volatility    stable Low/Mid/High labels
      │                      │
      └──────────┬───────────┘
                 ▼
         SHAP explainability
         Gradient Boosting surrogate
                 │
                 ▼
         backtest.py: vectorbt strategy
                 │
         ┌───────┴────────┐
         ▼                ▼
   Streamlit Dashboard   FastAPI on Render
```

---

## 3. Regime And Strategy Logic

| Regime | Numeric Label | Strategy Action | Economic Interpretation |
|---|---:|---|---|
| **Low Vol** | `0` | **ENTRY** | Low market uncertainty, favorable trending environment |
| **Mid Vol** | `1` | **HOLD** | Moderate fluctuation; retain current position |
| **High Vol** | `2` | **EXIT** | Elevated uncertainty; exit to cash to protect capital |

**Backtest Parameters:**
- **Initial Capital**: ₹1,00,000
- **Fees**: 0.1% per trade · **Slippage**: 0.1%
- **Benchmark**: Nifty 50 Buy & Hold

---

## 4. Quickstart (Local Development)

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Install
```bash
git clone https://github.com/RaajitSingh1306/volatility-classifier-simplified.git
cd volatility-classifier-simplified
pip install -r requirements.txt
```

### 2. Run Pipeline & Train
```bash
python run.py
```
This generates:
- `data/labeled_data.csv`
- `data/backtest_summary.csv`
- Plots in `outputs/` (`garch_vol.png`, `regimes.png`, `shap_summary.png`, `backtest.png`)

### 3. Run Test Suite
```bash
pytest tests/ -v
```

### 4. Launch Services Locally
- **FastAPI Backend**:
  ```bash
  uvicorn app.main:app --reload --port 8000
  ```
  Open `http://localhost:8000/docs` for the interactive OpenAPI documentation.
- **Streamlit Dashboard**:
  ```bash
  streamlit run app/dashboard.py
  ```
  Open `http://localhost:8501`.

### 5. Docker Compose (One-Click)
```bash
docker compose build
docker compose up
```

---

## 5. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check (`{"status":"ok", "message":"Volatility Regime Classifier API v1.0"}`) |
| `GET` | `/current` | Latest regime classification, date, close, and GARCH volatility |
| `GET` | `/regimes?last_n=252` | Historical regime time series |
| `GET` | `/stats` | Per-regime summary statistics |
| `GET` | `/backtest-summary` | Strategy vs Buy & Hold performance metrics |

---

## 6. Render Deployment Guide (FastAPI Backend)

This repository includes [`render.yaml`](render.yaml) for automated Blueprint deployment.

### Option A: Render Blueprint (Recommended — 1-Click)
1. Push repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect this repository (`volatility-classifier-simplified`).
5. Render will detect `render.yaml` and configure the Python web service.
6. Click **Apply**.

### Option B: Render Web Service (Manual Setup)
1. In Render, click **New +** → **Web Service**.
2. Select your repository.
3. Configure:
   - **Name**: `volatility-classifier-simplified`
   - **Environment**: `Python 3` (or `Docker`)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt && python run.py`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/`
   - **Plan**: `Free`
4. Click **Create Web Service**. Your service will be live at `https://volatility-classifier-simplified.onrender.com`.

---

## 7. Streamlit Dashboard Cloud Deployment

For the interactive visualization frontend:
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Select your repository (`volatility-classifier-simplified`).
3. Set **Main file path** to `app/dashboard.py`.
4. Click **Deploy**.

---

## 8. Repository Structure

```
volatility-classifier-simplified/
├── data.py                      # Yahoo Finance data ingestion & caching
├── features.py                  # 15 technical and volatility features
├── garch_model.py               # GARCH(1,1) conditional volatility modeling
├── hmm_model.py                 # 3-state Gaussian HMM + SHAP explainability
├── backtest.py                  # vectorbt regime-based strategy backtest
├── run.py                       # Pipeline runner
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── Dockerfile                   # Container definition with dynamic $PORT binding
├── docker-compose.yml           # Local multi-container orchestrator
├── render.yaml                  # Render Blueprint deployment manifest
│
├── app/
│   ├── main.py                  # FastAPI REST backend
│   └── dashboard.py             # Streamlit visualization dashboard
│
├── data/
│   ├── labeled_data.csv         # Processed time series with regime labels
│   └── backtest_summary.csv     # Performance statistics
│
├── outputs/                     # Generated diagnostic plots
│   ├── garch_vol.png
│   ├── regimes.png
│   ├── shap_summary.png
│   └── backtest.png
│
└── tests/                       # Pytest test suite
    ├── test_features.py
    ├── test_garch.py
    ├── test_hmm.py
    └── test_backtest.py
```

---

## 9. License & Disclaimer

MIT License. This project is for educational and research purposes only and does not constitute investment advice.
