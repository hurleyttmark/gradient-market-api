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
# CORE SETTINGS
# =============================
GRADIENT_WINDOW = 10
CACHE_TTL = 60  # seconds (LIVE UPDATE INTERVAL)

# =============================
# SIMPLE IN-MEMORY CACHE (MAKES IT "LIVE")
# =============================
cache = {}
scan_cache = {
    "data": None,
    "timestamp": 0
}

# =============================
# IMPROVED GRADIENT ENGINE (STABLE + HYBRID MOMENTUM)
# =============================

def compute_gradient(df):
    df = df.copy()

    # -----------------------------
    # MOMENTUM (VOL NORMALIZED)
    # -----------------------------
    volatility = df['Close'].rolling(10).std()
    volatility = volatility.replace(0, np.nan)

    momentum = df['Close'].diff(3)
    momentum_norm = (momentum / volatility).replace([np.inf, -np.inf], np.nan).fillna(0)

    # -----------------------------
    # STREAK (DIRECTIONAL PRESSURE)
    # -----------------------------
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

    # -----------------------------
    # FINAL GRADIENT (STABLE HYBRID)
    # -----------------------------
    raw = np.tanh(momentum_norm) * 5 + np.tanh(streak / 5) * 2

    raw = np.clip(raw, -5, 5)

    return raw

# =============================
# LIVE CACHE HELPERS
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
# TICKER ANALYZER (LIVE)
# =============================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):
    try:
        ticker = ticker.upper()

        cached = get_cached(ticker)
        if cached:
            return cached

        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {"error": "No data found"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Close']].dropna()

        grad = compute_gradient(df)
        latest_score = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(latest_score, 3),
            "signal": "bullish" if latest_score > 1 else "bearish" if latest_score < -1 else "neutral",
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
# BACKGROUND LIVE SCANNER
# =============================

def update_scan_loop():
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]

    while True:
        results = []

        for t in tickers:
            try:
                df = yf.download(t, period="1y", auto_adjust=True, progress=False)

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[['Close']].dropna()
                grad = compute_gradient(df)

                results.append({
                    "ticker": t,
                    "score": round(float(grad[-1]), 3),
                    "signal": "bullish" if grad[-1] > 1 else "bearish" if grad[-1] < -1 else "neutral"
                })

            except:
                continue

        scan_cache["data"] = sorted(results, key=lambda x: x["score"], reverse=True)
        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

Thread(target=update_scan_loop, daemon=True).start()

# =============================
# LIVE SCANNER ENDPOINT
# =============================
@app.get("/scan")
def scan():
    return {
        "live": True,
        "last_updated": scan_cache["timestamp"],
        "results": scan_cache["data"]
    }

# =============================
# DASHBOARD
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Gradient Heat Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>

    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            background: radial-gradient(circle at top, #0f1a2b, #0b1220);
            color: #ffffff;
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial;
            overflow-x: hidden;
            text-align: center;
        }

        .container {
            width: 100%;
            max-width: 1100px;
            margin: 0 auto;
            padding: 80px 20px;
        }

        h1 {
            font-size: 42px;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .subtitle {
            color: #9aa4b2;
            margin: 0 auto 30px auto;
            font-size: 16px;
            line-height: 1.6;
            max-width: 750px;
        }

        .input-row {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-bottom: 35px;
            flex-wrap: wrap;
        }

        input {
            padding: 14px 16px;
            font-size: 16px;
            border-radius: 12px;
            border: 1px solid #263244;
            background: #0f1a2b;
            color: white;
            width: 260px;
            text-align: center;
            outline: none;
        }

        button {
            padding: 14px 20px;
            font-size: 16px;
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: black;
            font-weight: 700;
            cursor: pointer;
        }

        .card {
            margin: 0 auto 40px auto;
            padding: 34px;
            background: #111c2e;
            border-radius: 16px;
            max-width: 320px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.55);
        }

        #symbol { font-size: 20px; color: #cbd5e1; }
        #score { font-size: 56px; font-weight: 700; }
        #signal { font-size: 16px; color: #9aa4b2; }

        table {
            margin: 30px auto;
            width: 100%;
            max-width: 900px;
            border-collapse: collapse;
            background: #0f1a2b;
        }

        th, td {
            padding: 14px;
            border-bottom: 1px solid #1f2a3a;
        }

        th { color: #cbd5e1; }
        td { color: #e2e8f0; }
    </style>
</head>
<body>

<div class="container">

    <h1>🔥 Gradient Heat Dashboard</h1>

    <div class="subtitle">
        Real-time momentum + structure scoring system measuring trend strength and reversal pressure.
    </div>

    <div class="input-row">
        <input id="ticker" placeholder="Enter ticker (AAPL)" />
        <button onclick="analyze()">Analyze</button>
    </div>

    <div class="card">
        <div id="symbol">---</div>
        <div id="score">0</div>
        <div id="signal">---</div>
    </div>

    <button onclick="loadScan()">Refresh Scan</button>

    <table id="table"></table>

</div>

<script>
const API = window.location.origin;

async function analyze() {
    const t = document.getElementById('ticker').value;
    const res = await fetch(`${API}/analyze?ticker=${t}`);
    const data = await res.json();

    document.getElementById('symbol').innerText = data.ticker;
    document.getElementById('score').innerText = data.gradient_score;
    document.getElementById('signal').innerText = data.signal;
}

async function loadScan() {
    const res = await fetch(`${API}/scan`);
    const data = await res.json();

    let html = '<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>';

    (data.results || []).forEach(r => {
        html += `<tr><td>${r.ticker}</td><td>${r.score}</td><td>${r.signal}</td></tr>`;
    });

    document.getElementById('table').innerHTML = html;
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
        "message": "LIVE Gradient Heat API running",
        "dashboard": "/dashboard",
        "endpoints": {
            "/analyze?ticker=AAPL": "live cached gradient score",
            "/scan": "live market heatmap"
        }
    }
