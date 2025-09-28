from __future__ import annotations

import json
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "lstm_AAPL_2015-01-01_2025-09-28.keras"  # path to your .keras file
SCALER_JSON = "scaler_params.json"               # saved after training

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

def fetch_last_window_prices(ticker: str, window: int) -> np.ndarray:
    # Download recent data (use a reasonably long period to ensure we have enough business days)
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise HTTPException(status_code=400, detail=f"Failed to download prices for {ticker}")
    closes = df["Close"].dropna().to_numpy(dtype=np.float32)
    if closes.size < window:
        raise HTTPException(status_code=400, detail=f"Not enough data for {ticker}; need at least {window} closes, got {closes.size}")
    return closes[-window:]


def load_scaler(path: str):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    scale = float(s["scale"])          # MinMaxScaler scale_
    bmin = float(s["min"])             # MinMaxScaler min_
    window = int(s["window"])          # training window
    horizon = int(s["horizon"])        # training horizon
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
# App + model loading
# -----------------------------
app = FastAPI(title="LSTM Stock Forecaster (.keras)")

SCALE, BMIN, WINDOW, HORIZON, TICKER = load_scaler(SCALER_JSON)
MODEL = keras.models.load_model(MODEL_PATH)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "ticker": TICKER,
        "window": WINDOW,
        "horizon": HORIZON,
        "model": MODEL_PATH,
        "trained_ticker": TICKER,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    steps = req.steps if req.steps is not None else HORIZON
    if steps <= 0 or steps > HORIZON:
        raise HTTPException(status_code=400, detail=f"steps must be in [1, {HORIZON}]")

    # Fetch latest window closes for the requested ticker
    last = fetch_last_window_prices(req.ticker, WINDOW)

    x_scaled = transform(last, SCALE, BMIN).reshape(1, WINDOW, 1)

    # Predict with Keras model (.keras)
    y_scaled = MODEL.predict(x_scaled, verbose=0)  # shape: (1, HORIZON)
    y_scaled = y_scaled[:, :steps]

    # Inverse transform back to price space
    y = inverse(y_scaled, SCALE, BMIN).astype(float)[0]  # (steps,)

    # Produce business dates starting from today
    dates = business_days(date.today(), steps)

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