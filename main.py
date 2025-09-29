import logging
import os
import sys
from datetime import date

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPricePredictor:
    def __init__(self, ticker="AAPL", start="2015-01-01", end=None, window=60, batch_size=64, epochs=100, horizon=1):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.window = window
        self.batch_size = batch_size
        self.epochs = epochs
        self.horizon = horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.df = None
        self.scaled = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_data(self):
        logger.info(f"Downloading data for {self.ticker} from {self.start} to {self.end or 'today'}")
        self.df = yf.download(self.ticker, start=self.start, end=self.end)
        self.df = self.df[['Close']].dropna().astype('float32')
        self.scaled = self.scaler.fit_transform(self.df[['Close']].values)
        self.X, self.y = self._make_sequences(self.scaled, self.window, self.horizon)
        split = int(len(self.X) * 0.8)
        self.X_train, self.y_train = self.X[:split], self.y[:split]
        self.X_val, self.y_val = self.X[split:], self.y[split:]
        logger.info(f"Number of training samples: {len(self.X_train)}, validation samples: {len(self.X_val)}")

    def _make_sequences(self, arr, window, horizon):
        X, y = [], []
        # ensure we have enough room for `horizon` future steps
        for i in range(window, len(arr) - horizon + 1):
            X.append(arr[i - window:i, 0])
            # next `horizon` points as target
            y.append(arr[i:i + horizon, 0])
        X = np.array(X)
        y = np.array(y)
        return X[..., None], y  # X: (N, window, 1), y: (N, horizon)

    def build_model(self):
        keras.utils.set_random_seed(42)
        self.model = keras.Sequential([
            keras.layers.Input(shape=(self.window, 1)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(self.horizon)
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        logger.info("Model built.")

    def train(self):
        cb = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        logger.info("Starting model training...")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=0
        )
        logger.info("Training completed.")

    def evaluate(self):
        pred_val = self.model.predict(self.X_val, verbose=0)
        # scaled RMSE per step
        rmse_scaled_per_h = []
        rmse_unscaled_per_h = []
        for h in range(self.horizon):
            rmse_h = sqrt(mean_squared_error(self.y_val[:, h], pred_val[:, h]))
            rmse_scaled_per_h.append(rmse_h)
            y_true_h = self.scaler.inverse_transform(self.y_val[:, h].reshape(-1, 1))
            y_pred_h = self.scaler.inverse_transform(pred_val[:, h].reshape(-1, 1))
            rmse_unscaled_per_h.append(sqrt(mean_squared_error(y_true_h, y_pred_h)))
        # aggregate (mean) RMSE
        rmse_scaled = float(np.mean(rmse_scaled_per_h))
        rmse_unscaled = float(np.mean(rmse_unscaled_per_h))
        logger.info(f"Val RMSE (scaled, mean over {self.horizon} steps): {rmse_scaled:.4f}")
        logger.info(f"Val RMSE (unscaled $, mean over {self.horizon} steps): {rmse_unscaled:.4f}")
        for h, (rs, ru) in enumerate(zip(rmse_scaled_per_h, rmse_unscaled_per_h), start=1):
            logger.info(f"  step+{h}: RMSE_scaled={rs:.4f}, RMSE_unscaled$={ru:.4f}")
        return rmse_scaled, rmse_unscaled

    def forecast(self, steps=None):
        if steps is None:
            steps = self.horizon
        last_window = self.scaled[-self.window:].reshape(1, self.window, 1)
        next_scaled_vec = self.model.predict(last_window, verbose=0)[0]  # shape: (horizon,)
        if steps <= self.horizon:
            next_scaled_vec = next_scaled_vec[:steps]
        # inverse-transform each predicted point
        next_prices = [float(self.scaler.inverse_transform([[v]])[0, 0]) for v in next_scaled_vec]
        # Generate forecast dates starting from the last date in self.df.index
        last_date = self.df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')
        # Log each predicted date with its price
        for date, price in zip(forecast_dates, next_prices):
            logger.info(f"Forecast for {self.ticker} on {date.date()}: {price:.2f} USD")
        return list(zip(forecast_dates, next_prices))

    def save_scaler_params(self, path="scaler_params.json"):
        """Save MinMaxScaler parameters and training config for inference services."""
        import json
        if not hasattr(self.scaler, "scale_") or not hasattr(self.scaler, "min_"):
            raise RuntimeError("Scaler has not been fit yet. Call load_data() before saving scaler params.")
        params = {
            "scale": float(self.scaler.scale_[0]),
            "min": float(self.scaler.min_[0]),
            "feature_range": [0.0, 1.0],
            "window": int(self.window),
            "horizon": int(self.horizon),
            "ticker": str(self.ticker),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Scaler params saved to file: {path}")

    def save_model(self, path="lstm_stock_forecast.keras"):
        self.model.save(path)
        logger.info(f"Model saved to file: {path}")

    def save_savedmodel(self, path="saved_model"):
        """Export model as TensorFlow SavedModel. Prefer Keras export() if available, otherwise fall back to tf.saved_model.save.
        Creates the target directory if it doesn't exist and registers a default serving signature.
        """
        os.makedirs(path, exist_ok=True)

        # Prefer Keras 3 export if present
        if hasattr(self.model, "export"):
            try:
                self.model.export(path)
                logger.info(f"SavedModel exported via Keras export() to: {path}")
                return
            except Exception as e:
                logger.warning(f"keras.Model.export() failed: {e}; falling back to tf.saved_model.save")

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, self.window, 1], dtype=tf.float32, name="inputs")
        ])
        def serving_fn(x):
            # Return a single Tensor so we can pass a ConcreteFunction as signature
            return self.model(x, training=False)

        try:
            concrete = serving_fn.get_concrete_function()
            tf.saved_model.save(self.model, path, signatures=concrete)
            logger.info(f"SavedModel exported to directory: {path}")
        except TypeError as e:
            logger.error(
                "Failed to export SavedModel. This can be due to TensorFlow/Python version incompatibility (e.g., Python 3.13).\n"
                f"Python: {sys.version}\nTensorFlow: {tf.__version__}\nError: {e}"
            )
            raise


if __name__ == "__main__":
    # Train and export models for multiple tickers
    tickers = [
        "AAPL",  # Apple
        "AMD",  # Advanced Micro Devices
        "INTC",  # Intel Corp
        "MSFT",  # Microsoft
        "NDAQ",  # NASDAQ
        "NVDA",  # NVIDIA Corp
        "QCOM",  # Qualcomm Inc
    ]

    common_cfg = dict(
        start="2015-01-01",
        end=None,
        window=60,
        batch_size=64,
        epochs=100,
        horizon=5,
    )

    all_forecasts = {}

    for tk in tickers:
        logger.info("\n==============================")
        logger.info(f"Starting pipeline for ticker: {tk}")
        try:
            predictor = StockPricePredictor(ticker=tk, **common_cfg)
            predictor.load_data()
            predictor.build_model()
            predictor.train()
            predictor.evaluate()
            fc = predictor.forecast(steps=common_cfg["horizon"])  # list[(pd.Timestamp, float)]
            all_forecasts[tk] = fc

            end_str = predictor.end if predictor.end is not None else date.today().strftime("%Y-%m-%d")
            out_dir = os.path.join("models", predictor.ticker)
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f"lstm_{predictor.ticker}_{predictor.start}_{end_str}.keras")
            predictor.save_model(filename)

            scaler_path = os.path.join(out_dir, "scaler_params.json")
            predictor.save_scaler_params(scaler_path)

            # If you also want SavedModel per ticker, uncomment and give a per-ticker dir, e.g.:
            # predictor.save_savedmodel(f"saved_model_{predictor.ticker}")

            logger.info(f"Finished pipeline for {tk}")
        except Exception as e:
            logger.exception(f"Pipeline failed for {tk}: {e}")
            continue

    # === Summary artifacts ===
    summary_dir = os.path.join("models", "_summary")
    os.makedirs(summary_dir, exist_ok=True)
    txt_path = os.path.join(summary_dir, "forecast_summary.txt")
    png_path = os.path.join(summary_dir, "forecast_summary.png")

    # Write text summary
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LSTM multi-ticker forecast summary\n")
        f.write(f"Window={common_cfg['window']}, Horizon={common_cfg['horizon']}\n\n")
        for tk in tickers:
            if tk not in all_forecasts:
                continue
            f.write(f"Ticker: {tk}\n")
            for d, p in all_forecasts[tk]:
                # d may be pandas.Timestamp; ensure date string
                try:
                    ds = d.strftime("%Y-%m-%d")
                except Exception:
                    ds = str(d)
                f.write(f"  {ds}: {p:.2f}\n")
            f.write("\n")

    # Draw combined plot with subplots (one subplot per ticker)
    n = len(tickers)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True)
    # Normalize axes list
    if rows * cols == 1:
        ax_list = [axes]
    else:
        ax_list = axes.ravel().tolist()

    idx = 0
    for tk in tickers:
        if tk not in all_forecasts:
            continue
        ax = ax_list[idx]
        dates = []
        prices = []
        for d, p in all_forecasts[tk]:
            try:
                dates.append(pd.to_datetime(d))
            except Exception:
                dates.append(pd.Timestamp(d))
            prices.append(p)
        ax.plot(dates, prices, marker="o", linestyle="-", linewidth=2, markersize=4)
        ax.grid(True, linestyle="--", alpha=0.6)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        if idx % cols == 0:
            ax.set_ylabel("Predicted Close (USD)")
        ax.set_title(tk)
        idx += 1

    # Hide any unused subplots
    for j in range(idx, len(ax_list)):
        ax_list[j].set_visible(False)

    fig.autofmt_xdate()
    fig.suptitle("LSTM Forecasts by Ticker", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    logger.info(f"Summary written: {txt_path}")
    logger.info(f"Plot saved: {png_path}")
