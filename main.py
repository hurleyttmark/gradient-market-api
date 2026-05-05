from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
import traceback
import time
from threading import Thread
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta

app = FastAPI()

# =============================
# CORE SETTINGS
# =============================
GRADIENT_WINDOW = 10
CACHE_TTL = 60

LOOKBACK_YEARS = 3

# =============================
# CACHE
# =============================
cache = {}
scan_cache = {
    "data": [],
    "timestamp": 0
}

# =============================
# TODAY-BASED DATE WINDOW
# =============================
def get_dates():
    end = datetime.now()
    start = end - timedelta(days=365 * LOOKBACK_YEARS)
    return start, end

# =============================
# GRADIENT ENGINE (UNCHANGED CORE LOGIC)
# =============================
def compute_gradient(df):
    df = df.copy()

    volatility = df['Close'].rolling(10).std()
    volatility = volatility.replace(0, np.nan)

    momentum = df['Close'].diff(3)
    momentum_norm = (momentum / volatility).replace([np.inf, -np.inf], np.nan).fillna(0)

    price_change = df['Close'].diff()

    streak = []
    s = 0

    for d in price_change:
        if d > 0:
            s = s + 1 if s > 0 else 1
        elif d < 0:
            s = s - 1 if s < 0 else -1
        else:
            s = 0
        streak.append(s)

    streak = np.array(streak)

    raw = np.tanh(momentum_norm) * 5 + np.tanh(streak / 5) * 2
    raw = np.clip(raw, -5, 5)

    return raw

# =============================
# CACHE HELPERS
# =============================
def get_cached(ticker):
    if ticker in cache:
        entry = cache[ticker]
        if time.time() - entry["time"] < CACHE_TTL:
            return entry["data"]
    return None

def set_cache(ticker, data):
    cache[ticker] = {
        "data": data,
        "time": time.time()
    }

# =============================
# ANALYZE ENDPOINT
# =============================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):
    try:
        ticker = ticker.upper()

        cached = get_cached(ticker)
        if cached:
            return cached

        start, end = get_dates()

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return {"error": "No data found"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Close']].dropna()

        grad = compute_gradient(df)
        score = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(score, 3),
            "signal": "bullish" if score > 1 else "bearish" if score < -1 else "neutral",
            "data_points": len(df),
            "cached": False
        }

        set_cache(ticker, result)
        return result

    except Exception as e:
        return {
            "error": "Server error",
            "details": str(e),
            "trace": traceback.format_exc()
        }

# =============================
# SCANNER (RESTORED FULL TICKER LIST)
# =============================
def update_scan_loop():
    tickers = [
        "AAPL", "MSFT", "NVDA", "TSLA",
        "AMZN", "META", "GOOGL", "SPY",
        "QQQ", "IWM", "AMD", "NFLX"
    ]

    while True:
        results = []

        start, end = get_dates()

        for t in tickers:
            try:
                df = yf.download(
                    t,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False
                )

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[['Close']].dropna()

                grad = compute_gradient(df)

                if len(grad) == 0:
                    continue

                results.append({
                    "ticker": t,
                    "score": round(float(grad[-1]), 3),
                    "signal": "bullish" if grad[-1] > 1 else "bearish" if grad[-1] < -1 else "neutral"
                })

            except Exception as e:
                print(f"Ticker error {t}: {e}")
                continue

        if results:
            scan_cache["data"] = sorted(results, key=lambda x: x["score"], reverse=True)

        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

Thread(target=update_scan_loop, daemon=True).start()

# =============================
# SCAN ENDPOINT
# =============================
@app.get("/scan")
def scan():
    return {
        "live": True,
        "last_updated": scan_cache["timestamp"],
        "results": scan_cache["data"]
    }

# =============================
# DASHBOARD (UNCHANGED UI)
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Gradient Heat Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background:#0b1220;color:white;font-family:Arial;text-align:center;">

<h1>Gradient Heat Dashboard</h1>

<input id="ticker" placeholder="AAPL">
<button onclick="analyze()">Analyze</button>

<div id="result"></div>

<table id="table"></table>

<script>
const API = window.location.origin;

async function analyze(){
    const t = document.getElementById('ticker').value;
    const res = await fetch(`${API}/analyze?ticker=${t}`);
    const d = await res.json();
    document.getElementById('result').innerHTML =
        `${d.ticker} | ${d.gradient_score} | ${d.signal}`;
}

async function loadScan(){
    const res = await fetch(`${API}/scan`);
    const d = await res.json();

    let html = "<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>";

    (d.results || []).forEach(r => {
        html += `<tr><td>${r.ticker}</td><td>${r.score}</td><td>${r.signal}</td></tr>`;
    });

    document.getElementById("table").innerHTML = html;
}

loadScan();
setInterval(loadScan, 15000);
</script>

</body>
</html>
"""
    return HTMLResponse(content=html)

# =============================
# ROOT
# =============================
@app.get("/")
def root():
    return {
        "status": "running",
        "endpoints": ["/analyze", "/scan", "/dashboard"]
    }