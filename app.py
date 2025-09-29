from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

# -----------------------------
# Configuration
# -----------------------------
BASE_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models")).resolve()
# Simple caches so we don't reload on every request
_MODEL_CACHE: dict[str, keras.Model] = {}
_SCALER_CACHE: dict[str, tuple[float, float, int, int, str]] = {}


# -----------------------------
# Pydantic models
# -----------------------------
class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    steps: Optional[int] = Field(None, description="Forecast horizon (<= trained horizon)")


class PredictPoint(BaseModel):
    date: str
    price: float


class PredictResponse(BaseModel):
    ticker: str
    window: int
    horizon: int
    used_steps: int
    forecast: List[PredictPoint]


# -----------------------------
# Utilities
# -----------------------------
def business_days(start: date, n: int) -> List[date]:
    out: List[date] = []
    d = start
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Monday..Friday
            out.append(d)
    return out


def fetch_last_window_prices(ticker: str, window: int) -> tuple[np.ndarray, date]:
    # Download recent data (use a reasonably long period to ensure we have enough business days)
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise HTTPException(status_code=400, detail=f"Failed to download prices for {ticker}")
    closes = df["Close"].dropna().to_numpy(dtype=np.float32)
    if closes.size < window:
        raise HTTPException(status_code=400,
                            detail=f"Not enough data for {ticker}; need at least {window} closes, got {closes.size}")
    last_hist_date = df.index.max().date()
    return closes[-window:], last_hist_date


def load_scaler(path: str):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    scale = float(s["scale"])  # MinMaxScaler scale_
    bmin = float(s["min"])  # MinMaxScaler min_
    window = int(s["window"])  # training window
    horizon = int(s["horizon"])  # training horizon
    ticker = s.get("ticker", "TICKER")
    if scale == 0:
        raise ValueError("Invalid scaler: scale == 0")
    return scale, bmin, window, horizon, ticker


def transform(x: np.ndarray, scale: float, bmin: float) -> np.ndarray:
    # MinMaxScaler: x_scaled = x * scale + min
    return x * scale + bmin


def inverse(xs: np.ndarray, scale: float, bmin: float) -> np.ndarray:
    # inverse: x = (x_scaled - min) / scale
    return (xs - bmin) / scale


# -----------------------------
# Per-ticker artifact loading
# -----------------------------

def _resolve_ticker_dir(ticker: str) -> Path:
    d = BASE_MODELS_DIR / ticker.upper()
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail=f"No artifacts found for ticker {ticker} under {BASE_MODELS_DIR}")
    return d


def _find_latest_keras(model_dir: Path) -> Path:
    candidates = list(model_dir.glob("*.keras"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No .keras model found in {model_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_scaler_for(ticker: str) -> tuple[float, float, int, int, str]:
    if ticker in _SCALER_CACHE:
        return _SCALER_CACHE[ticker]
    d = _resolve_ticker_dir(ticker)
    scaler_path = d / "scaler_params.json"
    if not scaler_path.exists():
        raise HTTPException(status_code=404, detail=f"scaler_params.json not found for {ticker} in {d}")
    params = load_scaler(str(scaler_path))
    _SCALER_CACHE[ticker] = params
    return params


def _load_model_for(ticker: str) -> keras.Model:
    if ticker in _MODEL_CACHE:
        return _MODEL_CACHE[ticker]
    d = _resolve_ticker_dir(ticker)
    model_path = _find_latest_keras(d)
    model = keras.models.load_model(str(model_path))
    _MODEL_CACHE[ticker] = model
    return model


# -----------------------------
# App + model loading
# -----------------------------
app = FastAPI(title="LSTM Stock Forecaster (.keras)")


@app.get("/healthz")
def healthz():
    available = sorted([p.name for p in BASE_MODELS_DIR.iterdir() if
                        p.is_dir() and (p / "scaler_params.json").exists()]) if BASE_MODELS_DIR.exists() else []
    return {
        "status": "ok",
        "models_dir": str(BASE_MODELS_DIR),
        "available_tickers": available,
        "note": "Artifacts are loaded lazily per ticker (model.keras + scaler_params.json)"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Load artifacts for requested ticker
    try:
        SCALE, BMIN, WINDOW, HORIZON, TRAINED_TICKER = _load_scaler_for(req.ticker)
        MODEL = _load_model_for(req.ticker)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load artifacts for {req.ticker}: {e}")

    steps = req.steps if req.steps is not None else HORIZON
    if steps <= 0 or steps > HORIZON:
        raise HTTPException(status_code=400, detail=f"steps must be in [1, {HORIZON}]")

    # Fetch latest window closes for the requested ticker
    last, last_hist_date = fetch_last_window_prices(req.ticker, WINDOW)

    x_scaled = transform(last, SCALE, BMIN).reshape(1, WINDOW, 1)

    # Predict with Keras model (.keras)
    y_scaled = MODEL.predict(x_scaled, verbose=0)  # shape: (1, HORIZON)
    y_scaled = y_scaled[:, :steps]

    # Inverse transform back to price space
    y = inverse(y_scaled, SCALE, BMIN).astype(float)[0]  # (steps,)

    # Produce business dates starting from the last historical trading day
    dates = business_days(last_hist_date, steps)

    forecast = [PredictPoint(date=d.strftime("%Y-%m-%d"), price=round(float(p), 2)) for d, p in zip(dates, y)]

    return PredictResponse(
        ticker=req.ticker,
        window=WINDOW,
        horizon=HORIZON,
        used_steps=steps,
        forecast=forecast,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
