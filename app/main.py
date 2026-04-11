"""
app/main.py
-----------
FastAPI backend exposing regime classification results via REST API.

Run
---
    uvicorn app.main:app --reload

Endpoints
---------
GET /                   Health check
GET /current            Latest regime and GARCH vol
GET /regimes?last_n=N   Historical time series (default: last 252 days)
GET /stats              Per-regime aggregate statistics
GET /backtest-summary   Strategy vs Buy & Hold performance table
"""

import os
import sys
from pathlib import Path
from functools import lru_cache

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Volatility Regime Classifier API",
    description="HMM-based volatility regime classification for Nifty 50.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DATA_PATH = Path("data/labeled_data.csv")
BACKTEST_PATH = Path("data/backtest_summary.csv")


# ── Data loading ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "labeled_data.csv not found. Run `python run.py` first."
        )
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


def _get_df() -> pd.DataFrame:
    try:
        return _load_df()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RegimePoint(BaseModel):
    date: str
    close: float
    regime: int
    regime_name: str
    garch_vol_pct: float
    returns_pct: float


class CurrentRegime(BaseModel):
    date: str
    regime: int
    regime_name: str
    garch_vol_pct: float
    close: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root() -> dict:
    """Health check."""
    return {"status": "ok", "message": "Volatility Regime Classifier API v1.0"}


@app.get("/current", response_model=CurrentRegime, tags=["Regime"])
def current_regime() -> CurrentRegime:
    """Return the most recent regime classification and GARCH volatility."""
    df = _get_df()
    row = df.iloc[-1]
    return CurrentRegime(
        date=str(df.index[-1].date()),
        regime=int(row["regime"]),
        regime_name=str(row["regime_name"]),
        garch_vol_pct=round(float(row["garch_vol"]) * 100, 2),
        close=round(float(row["Close"]), 2),
    )


@app.get("/regimes", response_model=list[RegimePoint], tags=["Regime"])
def get_regimes(
    last_n: int = Query(default=252, ge=1, le=10_000, description="Number of trading days to return"),
) -> list[RegimePoint]:
    """Return historical regime time series."""
    df = _get_df().tail(last_n)
    return [
        RegimePoint(
            date=str(idx.date()),
            close=round(float(row["Close"]), 2),
            regime=int(row["regime"]),
            regime_name=str(row["regime_name"]),
            garch_vol_pct=round(float(row["garch_vol"]) * 100, 2),
            returns_pct=round(float(row["returns"]) * 100, 4),
        )
        for idx, row in df.iterrows()
    ]


@app.get("/stats", tags=["Analytics"])
def regime_stats() -> list[dict]:
    """Return aggregate statistics per volatility regime."""
    df = _get_df()
    stats = (
        df.groupby("regime_name")
        .agg(
            days=("regime", "count"),
            mean_vol_pct=("garch_vol", lambda x: round(x.mean() * 100, 2)),
            mean_ret_pct=("returns", lambda x: round(x.mean() * 100, 4)),
            ret_std_pct=("returns", lambda x: round(x.std() * 100, 4)),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return stats


@app.get("/backtest-summary", tags=["Analytics"])
def backtest_summary() -> dict:
    """Return strategy vs Buy & Hold performance metrics."""
    if not BACKTEST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Backtest results not found. Run `python run.py` first.",
        )
    df = pd.read_csv(BACKTEST_PATH, index_col=0)
    return df.to_dict()