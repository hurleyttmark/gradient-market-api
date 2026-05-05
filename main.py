from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
import time
import os
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

scan_cache = {"data": [], "timestamp": 0}

# =============================
# CORE GRADIENT ENGINE
# =============================
def compute_gradient(df):
    df = df.copy()
    df = df[['Open','High','Low','Close','Volume']].dropna()

    df["range"] = df["High"] - df["Low"]
    df["body"] = (df["Close"] - df["Open"]).abs()

    signals = []

    for i in range(len(df)):
        r = df["range"].iloc[i]
        if r == 0:
            signals.append(0)
            continue

        body = df["Close"].iloc[i] - df["Open"].iloc[i]
        body_abs = abs(body)

        s = 0

        if body > 0:
            if body_abs / r > 0.5:
                s = 1
        elif body < 0:
            if body_abs / r > 0.5:
                s = -1

        signals.append(s)

    df["signal"] = signals

    # streak
    streak = []
    c = 0

    for s in signals:
        if s == 1:
            c = c + 1 if c > 0 else 1
        elif s == -1:
            c = c - 1 if c < 0 else -1
        else:
            c = 0
        streak.append(c)

    df["streak"] = streak

    # gradient score
    grad = []

    for i in range(len(df)):
        start = max(0, i - GRADIENT_WINDOW)
        window = df["streak"].iloc[start:i+1]

        score = (window > 0).sum() - (window < 0).sum()
        score = max(-5, min(5, score))

        grad.append(score)

    return np.array(grad)

# =============================
# SCANNER LOOP
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
                    "score": float(grad[-1]),
                    "signal": "bullish" if grad[-1] > 1 else "bearish" if grad[-1] < -1 else "neutral"
                })

            except:
                continue

        scan_cache["data"] = sorted(results, key=lambda x: x["score"], reverse=True)
        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

# =============================
# STARTUP (RENDER SAFE)
# =============================
@app.on_event("startup")
def startup():
    thread = Thread(target=scan_loop, daemon=True)
    thread.start()

# =============================
# API ROUTES
# =============================
@app.get("/scan")
def scan():
    return scan_cache

@app.get("/")
def home():
    return {"status": "running", "scan": "/scan", "dashboard": "/dashboard"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
    <head>
        <title>Gradient Scanner</title>
    </head>
    <body>
        <h2>Live Gradient Scanner</h2>
        <pre id="data">Loading...</pre>

        <script>
        async function load(){
            const res = await fetch('/scan');
            const data = await res.json();
            document.getElementById('data').innerText =
                JSON.stringify(data, null, 2);
        }

        load();
        setInterval(load, 5000);
        </script>
    </body>
    </html>
    """

# =============================
# LOCAL RUN (optional)
# =============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)