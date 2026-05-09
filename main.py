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

def compute_gradient(df):
    df = df.copy()

    # =============================
    # BASE FEATURES (UNCHANGED CORE)
    # =============================
    df['returns'] = df['Close'].pct_change()
    df['vol'] = df['returns'].rolling(10).std()

    df['vol'] = df['vol'].replace(0, np.nan)
    df['momentum'] = df['returns'] / df['vol']
    df['momentum'] = df['momentum'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['trend'] = df['momentum'].rolling(5).mean().fillna(0)
    df['accel'] = df['momentum'].diff().fillna(0)

    # Volume confirmation
    if 'Volume' in df.columns:
        df['vol_ma'] = df['Volume'].rolling(10).mean()
        df['vol_boost'] = np.where(df['Volume'] > df['vol_ma'], 1, 0)
    else:
        df['vol_boost'] = 0

    # =============================
    # 3-DAY CANDLE STRUCTURE LOGIC
    # =============================
    N = 3

    engulf_signal = np.zeros(len(df))
    pattern_signal = np.zeros(len(df))

    for i in range(2 * N, len(df)):

        first = df.iloc[i-2*N:i-N]
        second = df.iloc[i-N:i]

        # -------- 3-day aggregation --------
        f_open = first['Open'].iloc[0]
        f_close = first['Close'].iloc[-1]
        f_high = first['High'].max()
        f_low = first['Low'].min()
        f_body = abs(f_close - f_open)
        f_range = f_high - f_low

        s_open = second['Open'].iloc[0]
        s_close = second['Close'].iloc[-1]
        s_high = second['High'].max()
        s_low = second['Low'].min()
        s_body = abs(s_close - s_open)
        s_range = s_high - s_low

        # Volume filter
        if 'Volume' in df.columns:
            vol_ok = second['Volume'].sum() >= 1.2 * first['Volume'].mean() * N
        else:
            vol_ok = True

        # =============================
        # 3-DAY ENGULFING
        # =============================

        # Bullish engulf
        if (
            f_close < f_open and f_body >= 0.5 * f_range and
            s_close > s_open and s_body >= 0.6 * s_range and
            s_close > f_open and vol_ok
        ):
            engulf_signal[i-N:i] = 1

        # Bearish engulf
        if (
            f_close > f_open and f_body >= 0.5 * f_range and
            s_close < s_open and s_body >= 0.6 * s_range and
            s_close < f_open and vol_ok
        ):
            engulf_signal[i-N:i] = -1

        # =============================
        # 3x3 STRUCTURE SHIFT (TREND BREAK)
        # =============================

        if (
            f_close < f_open and s_body < f_body * 0.3 and s_close > s_open
        ):
            pattern_signal[i] = 1

        if (
            f_close > f_open and s_body < f_body * 0.3 and s_close < s_open
        ):
            pattern_signal[i] = -1

    # =============================
    # COMBINE STRUCTURE SIGNALS
    # =============================
    structure = engulf_signal + pattern_signal

    # =============================
    # FINAL GRADIENT (CONSISTENT SCALE)
    # =============================
    raw = (
        0.5 * df['trend'] +
        0.3 * df['accel'] +
        0.1 * df['vol_boost'] +
        0.6 * structure
    )

    # normalize + clamp to same range (-5 to 5)
    gradient = np.tanh(raw) * 5

    return gradient.values

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
# DASHBOARD (NEW FRONTEND)
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gradient Heat Dashboard</title>
        <style>
            body { font-family: Arial; background:#0f172a; color:white; text-align:center; }
            input, button { padding:10px; margin:5px; font-size:16px; }
            .card { margin-top:20px; padding:20px; background:#1e293b; display:inline-block; border-radius:10px; }
            table { margin:auto; margin-top:20px; border-collapse: collapse; }
            td, th { padding:10px 20px; border-bottom:1px solid #334155; }
        </style>
    </head>
    <body>
        <h1>🔥 Gradient Heat Dashboard</h1>

        <input id="ticker" placeholder="Enter ticker (AAPL)" />
        <button onclick="analyze()">Analyze</button>

        <div class="card">
            <h2 id="symbol">---</h2>
            <h1 id="score">0</h1>
            <div id="signal">---</div>
        </div>

        <h2>📊 Live Heatmap</h2>
        <button onclick="loadScan()">Refresh Scan</button>
        <table id="table"></table>

        <script>
        async function analyze() {
            const t = document.getElementById('ticker').value;
            const res = await fetch(`/analyze?ticker=${t}`);
            const data = await res.json();

            document.getElementById('symbol').innerText = data.ticker;
            document.getElementById('score').innerText = data.gradient_score;
            document.getElementById('signal').innerText = data.signal;
        }

        async function loadScan() {
            const res = await fetch('/scan');
            const data = await res.json();

            let html = '<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>';

            data.results.forEach(r => {
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
