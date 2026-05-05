from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
import traceback
import time
import os
from threading import Thread
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# =============================
# APP LIFECYCLE
# =============================

GRADIENT_WINDOW = 20
CACHE_TTL = 60

TICKERS = [
    "AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL",
    "SPY","QQQ","IWM","NFLX","AMD"
]

cache = {}
scan_cache = {"data": [], "timestamp": 0}

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
# CORE ENGINE
# =============================

def compute_gradient(df):
    df = df.copy()
    df = df[['Open','High','Low','Close','Volume']].dropna()

    df['range'] = df['High'] - df['Low']
    df['body'] = (df['Close'] - df['Open']).abs()
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']

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
            if body_abs / r >= 0.65 or lower >= 1.3 * body_abs or body_abs / r >= 0.25:
                s = 1
        elif body < 0:
            if body_abs / r >= 0.65 or upper >= 1.3 * body_abs or body_abs / r >= 0.25:
                s = -1

        signal.append(s)

    df['signal'] = signal

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

    N = 3
    scenario = np.zeros(len(df))

    for i in range(2 * N, len(df)):
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

    grad = []

    for i in range(len(df)):
        start = max(0, i - GRADIENT_WINDOW + 1)
        window = df['streak'].iloc[start:i+1]

        score = (window > 0).sum() - (window < 0).sum()

        if df['scenario'].iloc[i] != 0:
            score += int(np.sign(df['scenario'].iloc[i]))

        grad.append(max(-5, min(5, score)))

    return np.array(grad)

# =============================
# SCANNER
# =============================

def scan_loop():
    while True:
        results = []

        for t in TICKERS:
            try:
                df = yf.download(t, period="1y", auto_adjust=True, progress=False)
                if df is None or df.empty:
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

# =============================
# FASTAPI LIFESPAN (FIXED FOR RENDER)
# =============================

@asynccontextmanager
def lifespan(app: FastAPI):
    thread = Thread(target=scan_loop, daemon=True)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

# =============================
# ROUTES
# =============================

@app.get("/scan")
def scan():
    return scan_cache

@app.get("/")
def root():
    return {"status": "running", "scan": "/scan", "dashboard": "/dashboard"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
    <body>
        <h1>Gradient Live Scanner</h1>
        <pre id='data'>Loading...</pre>
        <script>
        async function load(){
            const res = await fetch('/scan');
            const data = await res.json();
            document.getElementById('data').innerText = JSON.stringify(data, null, 2);
        }
        load();
        setInterval(load, 5000);
        </script>
    </body>
    </html>
    """

# =============================
# LOCAL RUN
# =============================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
