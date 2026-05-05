from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
import traceback
import time
from threading import Thread
from fastapi.responses import HTMLResponse

app = FastAPI()

# =============================
# CONFIG
# =============================
GRADIENT_WINDOW = 20
CACHE_TTL = 60

TICKERS = [
    "AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL",
    "SPY","QQQ","IWM","NFLX","AMD"
]

cache = {}
scan_cache = {"data": None, "timestamp": 0}

# =============================
# CACHE
# =============================
def get_cached(ticker):
    if ticker in cache:
        entry = cache[ticker]
        if time.time() - entry["time"] < CACHE_TTL:
            return entry["data"]
    return None

def set_cache(ticker, data):
    cache[ticker] = {"data": data, "time": time.time()}

# =============================
# SAFE DATA LOADER (IMPORTANT FIX)
# =============================
def load_data(ticker):
    df = yf.download(
        ticker,
        period="5y",          # longer history = more stable gradient
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return None

    df = df[['Open','High','Low','Close','Volume']].dropna()

    # force chronological order (IMPORTANT FIX)
    df = df.sort_index()

    return df

# =============================
# CORE ENGINE (YOUR LOGIC, STABILIZED)
# =============================
def compute_gradient(df):
    df = df.copy()

    # -----------------------------
    # CANDLE METRICS
    # -----------------------------
    df['range'] = df['High'] - df['Low']
    df['body'] = (df['Close'] - df['Open']).abs()

    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']

    # -----------------------------
    # SIGNAL
    # -----------------------------
    signal = []

    for i in range(len(df)):
        r = df['range'].iloc[i]
        if r == 0:
            signal.append(0)
            continue

        body = df['Close'].iloc[i] - df['Open'].iloc[i]
        body_abs = abs(body)
        upper = df['upper_wick'].iloc[i]
        lower = df['lower_wick'].iloc[i]

        s = 0

        if body > 0:
            if body_abs / r >= 0.65:
                s = 1
            elif lower >= 1.3 * body_abs:
                s = 1
            elif body_abs / r >= 0.25:
                s = 1

        elif body < 0:
            if body_abs / r >= 0.65:
                s = -1
            elif upper >= 1.3 * body_abs:
                s = -1
            elif body_abs / r >= 0.25:
                s = -1

        signal.append(s)

    df['signal'] = signal

    # -----------------------------
    # STREAK
    # -----------------------------
    streak = []
    c = 0

    for s in signal:
        if s == 1:
            c = c + 1 if c > 0 else 1
        elif s == -1:
            c = c - 1 if c < 0 else -1
        else:
            c = 0
        streak.append(c)

    df['streak'] = streak

    # -----------------------------
    # 3-DAY ENGULF (SAFE)
    # -----------------------------
    N = 3
    scenario = np.zeros(len(df))

    for i in range(2*N, len(df)):
        first = df.iloc[i-2*N:i-N]
        second = df.iloc[i-N:i]

        f_open, f_close = first['Open'].iloc[0], first['Close'].iloc[-1]
        s_open, s_close = second['Open'].iloc[0], second['Close'].iloc[-1]

        s_body = abs(s_close - s_open)
        s_range = second['High'].max() - second['Low'].min()

        vol_ok = second['Volume'].sum() >= 1.25 * first['Volume'].mean() * N

        if f_close < f_open and s_close > s_open and s_body >= 0.7 * s_range and vol_ok:
            scenario[i] = 3

        elif f_close > f_open and s_close < s_open and s_body >= 0.7 * s_range and vol_ok:
            scenario[i] = -3

    df['scenario'] = scenario

    # -----------------------------
    # GRADIENT SCORE
    # -----------------------------
    grad = []

    for i in range(len(df)):
        start = max(0, i - GRADIENT_WINDOW + 1)
        window = df['streak'].iloc[start:i+1]

        score = (window > 0).sum() - (window < 0).sum()

        if df['scenario'].iloc[i] != 0:
            score += int(np.sign(df['scenario'].iloc[i]))

        score = max(-5, min(5, score))
        grad.append(score)

    return np.array(grad)

# =============================
# ANALYZE
# =============================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):
    try:
        ticker = ticker.upper()

        cached = get_cached(ticker)
        if cached:
            return cached

        df = load_data(ticker)

        if df is None:
            return {"error": "No data returned from yfinance"}

        grad = compute_gradient(df)

        result = {
            "ticker": ticker,
            "gradient_score": float(round(grad[-1], 3)),
            "signal": "bullish" if grad[-1] > 1 else "bearish" if grad[-1] < -1 else "neutral",
            "data_points": int(len(df))
        }

        set_cache(ticker, result)
        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# =============================
# SCANNER
# =============================
def scan_loop():
    while True:
        results = []

        for t in TICKERS:
            try:
                df = load_data(t)
                if df is None:
                    continue

                grad = compute_gradient(df)

                results.append({
                    "ticker": t,
                    "score": float(round(grad[-1], 3)),
                    "signal": "bullish" if grad[-1] > 1 else "bearish" if grad[-1] < -1 else "neutral"
                })

            except:
                continue

        scan_cache["data"] = sorted(results, key=lambda x: x["score"], reverse=True)
        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

Thread(target=scan_loop, daemon=True).start()

# =============================
# SCAN ENDPOINT
# =============================
@app.get("/scan")
def scan():
    return scan_cache

# =============================
# DASHBOARD
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return "<html><body><h1>Gradient Live</h1></body></html>"

# =============================
# ROOT
# =============================
@app.get("/")
def root():
    return {
        "status": "running",
        "scan": "/scan",
        "dashboard": "/dashboard"
    }