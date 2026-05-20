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
# SIMPLE IN-MEMORY CACHE
# =============================
cache = {}

scan_cache = {
    "data": None,
    "timestamp": 0
}

# =============================
# 3-DAY GRADIENT ENGINE
# =============================
def compute_gradient(df):

    df = df.copy()

    # ---------------------------------
    # 3-DAY PRICE STRUCTURE
    # ---------------------------------
    df['close_3d'] = df['Close'].pct_change(3)

    # ---------------------------------
    # VOLATILITY NORMALIZATION
    # ---------------------------------
    df['volatility'] = df['close_3d'].rolling(20).std()
    df['volatility'] = df['volatility'].replace(0, np.nan)

    df['momentum'] = (
        df['close_3d'] / df['volatility']
    )

    df['momentum'] = (
        df['momentum']
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------
    # 3-DAY STREAK SYSTEM
    # ---------------------------------
    streak = []
    s = 0

    for val in df['close_3d'].fillna(0):

        if val > 0:
            s = s + 1 if s > 0 else 1

        elif val < 0:
            s = s - 1 if s < 0 else -1

        else:
            s = 0

        streak.append(s)

    df['streak'] = streak

    # ---------------------------------
    # TREND REGIME
    # ---------------------------------
    df['trend'] = (
        df['momentum']
        .rolling(5)
        .mean()
        .fillna(0)
    )

    # ---------------------------------
    # ACCELERATION
    # ---------------------------------
    df['accel'] = (
        df['trend']
        .diff(3)
        .fillna(0)
    )

    # ---------------------------------
    # VOLUME CONFIRMATION
    # ---------------------------------
    df['vol_ma'] = (
        df['Volume']
        .rolling(20)
        .mean()
    )

    df['vol_boost'] = np.where(
        df['Volume'] > df['vol_ma'],
        1,
        0
    )

    # ---------------------------------
    # FINAL REGIME SCORE
    # ---------------------------------
    regime_raw = (
        0.55 * df['trend'] +
        0.30 * np.tanh(df['streak'] / 4) +
        0.10 * df['accel'] +
        0.05 * df['vol_boost']
    )

    # ---------------------------------
    # FINAL SMOOTHED GRADIENT
    # ---------------------------------
    df['gradient'] = (
        np.tanh(
            regime_raw
            .rolling(3)
            .mean()
            .fillna(0)
        ) * 5
    )

    return df['gradient'].fillna(0).values


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
# TICKER ANALYZER
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

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        grad = compute_gradient(df)

        latest_score = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(latest_score, 3),
            "signal": (
                "bullish"
                if latest_score > 1
                else "bearish"
                if latest_score < -1
                else "neutral"
            ),
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

    tickers = [
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

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

                grad = compute_gradient(df)

                score = float(grad[-1])

                results.append({
                    "ticker": t,
                    "score": round(score, 3),
                    "signal": (
                        "bullish"
                        if score > 1
                        else "bearish"
                        if score < -1
                        else "neutral"
                    )
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

        <style>

            body {
                font-family: Arial;
                background:#0f172a;
                color:white;
                text-align:center;
            }

            input, button {
                padding:10px;
                margin:5px;
                font-size:16px;
            }

            .card {
                margin-top:20px;
                padding:20px;
                background:#1e293b;
                display:inline-block;
                border-radius:10px;
            }

            table {
                margin:auto;
                margin-top:20px;
                border-collapse: collapse;
            }

            td, th {
                padding:10px 20px;
                border-bottom:1px solid #334155;
            }

        </style>

    </head>

    <body>

        <h1>🔥 Gradient Heat Dashboard</h1>

        <input id="ticker" placeholder="Enter ticker (AAPL)" />

        <button onclick="analyze()">
            Analyze
        </button>

        <div class="card">

            <h2 id="symbol">---</h2>

            <h1 id="score">0</h1>

            <div id="signal">---</div>

        </div>

        <h2>📊 Live Heatmap</h2>

        <button onclick="loadScan()">
            Refresh Scan
        </button>

        <table id="table"></table>

        <script>

        async function analyze() {

            const t =
                document.getElementById('ticker').value;

            const res =
                await fetch(`/analyze?ticker=${t}`);

            const data =
                await res.json();

            document.getElementById('symbol').innerText =
                data.ticker;

            document.getElementById('score').innerText =
                data.gradient_score;

            document.getElementById('signal').innerText =
                data.signal;
        }

        async function loadScan() {

            const res =
                await fetch('/scan');

            const data =
                await res.json();

            let html =
                '<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>';

            if (data.results) {

                data.results.forEach(r => {

                    html += `
                        <tr>
                            <td>${r.ticker}</td>
                            <td>${r.score}</td>
                            <td>${r.signal}</td>
                        </tr>
                    `;
                });
            }

            document.getElementById('table').innerHTML =
                html;
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
                "live cached gradient score",

            "/scan":
                "live market heatmap"
            @app.get("/chart")
def chart(ticker: str = Query(...)):
    ticker = ticker.upper()

    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return {"error": "No data"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close']].dropna()

    # --- reuse candle logic (simplified but consistent) ---
    df['range'] = df['High'] - df['Low']
    df['body'] = abs(df['Close'] - df['Open'])
    df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)

    def candle_signal(r):
        if r['range'] == 0:
            return 0

        if r['body_pct'] <= 0.2:
            return 0

        body = r['Close'] - r['Open']
        body_abs = abs(body)

        if body > 0:
            if body_abs / r['range'] >= 0.25:
                return 1

        if body < 0:
            if body_abs / r['range'] >= 0.25:
                return -1

        return 0

    df['signal'] = df.apply(candle_signal, axis=1)

    return {
        "ticker": ticker,
        "index": df.index.astype(str).tolist(),
        "open": df['Open'].tolist(),
        "high": df['High'].tolist(),
        "low": df['Low'].tolist(),
        "close": df['Close'].tolist(),
        "signal": df['signal'].tolist()
    }
            
        }
    }
