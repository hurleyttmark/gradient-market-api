import pandas as pd
import yfinance as yf
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# -----------------------------
# APP
# -----------------------------
app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
ticker_file = r"C:\Users\hurle\OneDrive\Documents\tickers.txt"

LOOKBACK_DAYS = 365 * 3
SCAN_INTERVAL = 15
GRADIENT_WINDOW = 20

# -----------------------------
# SAFE CACHE (NEVER BREAK FRONTEND)
# -----------------------------
scan_cache = {
    "data": [],
    "timestamp": 0
}

# -----------------------------
# TODAY-BASED DATA WINDOW
# -----------------------------
def get_date_window(days_back):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    return start, end

# -----------------------------
# GRADIENT ENGINE
# -----------------------------
def compute_gradient(close_series, window=20):
    close_series = close_series.dropna()

    if len(close_series) < window + 5:
        return np.array([])

    streak = 0
    codes = []

    diffs = close_series.diff().fillna(0).values

    for d in diffs:
        if d > 0:
            streak = streak + 1 if streak >= 0 else 1
        elif d < 0:
            streak = streak - 1 if streak <= 0 else -1
        else:
            streak = 0

        codes.append(streak)

    codes = np.array(codes)

    gradient = []
    for i in range(len(codes)):
        start = max(0, i - window + 1)
        window_vals = codes[start:i+1]

        score = (window_vals > 0).sum() - (window_vals < 0).sum()
        gradient.append(score)

    return np.array(gradient)

# -----------------------------
# SCANNER
# -----------------------------
def run_scan():
    results = []

    start_date, end_date = get_date_window(LOOKBACK_DAYS)

    with open(ticker_file, "r") as f:
        tickers = [t.strip().upper() for t in f if t.strip()]

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df:
                continue

            close = df["Close"].dropna()

            if len(close) < GRADIENT_WINDOW + 10:
                continue

            grad = compute_gradient(close, GRADIENT_WINDOW)

            if grad is None or len(grad) == 0:
                continue

            score = float(grad[-1])

            results.append({
                "ticker": ticker,
                "score": round(score, 2),
                "signal": (
                    "BULLISH" if score > 0 else
                    "BEARISH" if score < 0 else
                    "NEUTRAL"
                )
            })

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
            continue

    return results

# -----------------------------
# BACKGROUND SCANNER LOOP
# -----------------------------
def scanner_loop():
    global scan_cache

    while True:
        try:
            data = run_scan()

            if data and len(data) > 0:
                scan_cache["data"] = data
                scan_cache["timestamp"] = time.time()
                print(f"Scan updated → {len(data)} tickers")
            else:
                print("Scan skipped (no valid data)")

        except Exception as e:
            print(f"Scanner error: {e}")

        time.sleep(SCAN_INTERVAL)

# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"status": "running", "mode": "today_based"}

@app.get("/scan")
def scan():
    return JSONResponse({
        "results": scan_cache["data"],
        "updated": scan_cache["timestamp"]
    })

# -----------------------------
# START BACKGROUND THREAD
# -----------------------------
threading.Thread(target=scanner_loop, daemon=True).start()