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

CACHE_TTL = 60

# =============================
# CACHE
# =============================

cache = {}

scan_cache = {
    "data": [],
    "timestamp": 0
}

# =============================
# MODERN GRADIENT ENGINE
# =============================

def compute_gradient(df):

    df = df.copy()

    # ---------------------------------
    # RETURNS
    # ---------------------------------

    df["returns"] = df["Close"].pct_change()

    # 3-day smoothed move
    df["short_return"] = (
        df["Close"] / df["Close"].shift(3) - 1
    )

    # Longer-term structure
    df["long_return"] = (
        df["Close"] / df["Close"].shift(20) - 1
    )

    # ---------------------------------
    # VOLATILITY NORMALIZATION
    # ---------------------------------

    df["volatility"] = (
        df["returns"]
        .rolling(20)
        .std()
    )

    df["volatility"] = (
        df["volatility"]
        .replace(0, np.nan)
    )

    df["momentum"] = (
        df["short_return"] / df["volatility"]
    )

    df["momentum"] = (
        df["momentum"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------
    # TREND COMPONENTS
    # ---------------------------------

    # Medium trend
    df["trend"] = (
        df["momentum"]
        .rolling(5)
        .mean()
        .fillna(0)
    )

    # Long-term trend
    df["macro_trend"] = (
        df["long_return"]
        .rolling(10)
        .mean()
        .fillna(0)
    )

    # Momentum acceleration
    df["acceleration"] = (
        df["trend"]
        .diff()
        .fillna(0)
    )

    # ---------------------------------
    # EMA STRUCTURE
    # ---------------------------------

    df["ema_fast"] = (
        df["Close"]
        .ewm(span=10)
        .mean()
    )

    df["ema_slow"] = (
        df["Close"]
        .ewm(span=30)
        .mean()
    )

    df["ema_signal"] = np.where(
        df["ema_fast"] > df["ema_slow"],
        1,
        -1
    )

    # ---------------------------------
    # VOLUME CONFIRMATION
    # ---------------------------------

    if "Volume" in df.columns:

        df["vol_ma"] = (
            df["Volume"]
            .rolling(20)
            .mean()
        )

        df["volume_signal"] = np.where(
            df["Volume"] > df["vol_ma"],
            1,
            0
        )

    else:
        df["volume_signal"] = 0

    # ---------------------------------
    # FINAL REGIME MODEL
    # ---------------------------------

    regime_raw = (
        0.35 * df["trend"] +
        0.30 * df["macro_trend"] +
        0.20 * df["acceleration"] +
        0.10 * df["ema_signal"] +
        0.05 * df["volume_signal"]
    )

    # Smooth final score
    regime_smoothed = (
        regime_raw
        .rolling(3)
        .mean()
        .fillna(0)
    )

    # Final bounded score
    df["gradient"] = np.tanh(regime_smoothed) * 5

    return df["gradient"].values

# =============================
# SIGNAL ENGINE
# =============================

def get_signal(score):

    if score >= 2:
        return "strong bullish"

    elif score >= 0.75:
        return "bullish"

    elif score <= -2:
        return "strong bearish"

    elif score <= -0.75:
        return "bearish"

    else:
        return "neutral"

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

        df = yf.download(
            ticker,
            period="3y",
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return {"error": "No data found"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        grad = compute_gradient(df)

        latest_score = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(latest_score, 3),
            "signal": get_signal(latest_score),
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
# LIVE SCANNER LOOP
# =============================

def update_scan_loop():

    tickers = [
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "AMZN",
        "META",
        "GOOGL"
    ]

    while True:

        results = []

        for t in tickers:

            try:

                df = yf.download(
                    t,
                    period="1y",
                    auto_adjust=True,
                    progress=False
                )

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.dropna()

                grad = compute_gradient(df)

                latest = float(grad[-1])

                results.append({
                    "ticker": t,
                    "score": round(latest, 3),
                    "signal": get_signal(latest)
                })

            except:
                continue

        scan_cache["data"] = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

Thread(
    target=update_scan_loop,
    daemon=True
).start()

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
# DASHBOARD
# =============================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    html = """

    <!DOCTYPE html>

    <html>

    <head>

        <title>Gradient Heat Dashboard</title>

        <style>

            body {
                font-family: Arial;
                background: #0f172a;
                color: white;
                text-align: center;
                margin: 0;
                padding: 20px;
            }

            .container {
                max-width: 1000px;
                margin: auto;
            }

            input, button {
                padding: 10px;
                margin: 5px;
                font-size: 16px;
                border-radius: 6px;
                border: none;
            }

            button {
                background: #2563eb;
                color: white;
                cursor: pointer;
            }

            .card-row {
                display: flex;
                gap: 20px;
                justify-content: center;
                flex-wrap: wrap;
                margin-top: 20px;
            }

            .card {
                background: #1e293b;
                padding: 20px;
                border-radius: 10px;
                min-width: 220px;
            }

            table {
                width: 100%;
                margin-top: 30px;
                border-collapse: collapse;
            }

            td, th {
                padding: 12px;
                border-bottom: 1px solid #334155;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>🔥 Gradient Heat Dashboard</h1>

            <input id="ticker" placeholder="Enter ticker (AAPL)" />

            <button onclick="analyze()">
                Analyze
            </button>

            <div class="card-row">

                <div class="card">
                    <h2 id="symbol">---</h2>
                </div>

                <div class="card">
                    <h2>Score</h2>
                    <h1 id="score">0</h1>
                </div>

                <div class="card">
                    <h2>Signal</h2>
                    <h2 id="signal">---</h2>
                </div>

            </div>

            <h2 style="margin-top:40px;">
                📊 Live Heatmap
            </h2>

            <button onclick="loadScan()">
                Refresh Scan
            </button>

            <table id="table"></table>

        </div>

        <script>

        async function analyze() {

            const t = document.getElementById('ticker').value;

            const res = await fetch(`/analyze?ticker=${t}`);

            const data = await res.json();

            document.getElementById('symbol').innerText =
                data.ticker || "---";

            document.getElementById('score').innerText =
                data.gradient_score || 0;

            document.getElementById('signal').innerText =
                data.signal || "---";
        }

        async function loadScan() {

            const res = await fetch('/scan');

            const data = await res.json();

            let html =
                '<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>';

            data.results.forEach(r => {

                html += `
                    <tr>
                        <td>${r.ticker}</td>
                        <td>${r.score}</td>
                        <td>${r.signal}</td>
                    </tr>
                `;
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
            "/analyze?ticker=AAPL":
                "live gradient score",

            "/scan":
                "live market heatmap"
        }
    }