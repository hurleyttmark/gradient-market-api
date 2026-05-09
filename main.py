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
# SETTINGS
# =============================
CACHE_TTL = 60

cache = {}
scan_cache = {
    "data": None,
    "timestamp": 0
}

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
# DATA LOADER (SAFE)
# =============================
def load_data(ticker):
    df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    return df


# =============================
# GRADIENT ENGINE (YOUR ORIGINAL + SAFE 3-DAY ADDITION)
# =============================
def compute_gradient(df):
    df = df.copy()

    # =============================
    # BASE MOMENTUM SYSTEM (UNCHANGED)
    # =============================
    df['returns'] = df['Close'].pct_change()
    df['vol'] = df['returns'].rolling(10).std()

    df['vol'] = df['vol'].replace(0, np.nan)
    df['momentum'] = df['returns'] / df['vol']
    df['momentum'] = df['momentum'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['trend'] = df['momentum'].rolling(5).mean().fillna(0)
    df['accel'] = df['momentum'].diff().fillna(0)

    if 'Volume' in df.columns:
        df['vol_ma'] = df['Volume'].rolling(10).mean()
        df['vol_boost'] = np.where(df['Volume'] > df['vol_ma'], 1, 0)
    else:
        df['vol_boost'] = 0

    base = (
        0.6 * df['trend'] +
        0.3 * df['accel'] +
        0.1 * df['vol_boost']
    )

    # =============================
    # SAFE 3-DAY STRUCTURE LAYER
    # =============================
    N = 3
    structure = np.zeros(len(df))

    for i in range(2 * N, len(df)):

        first = df.iloc[i-2*N:i-N]
        second = df.iloc[i-N:i]

        # safety check (prevents crash)
        if len(first) < 1 or len(second) < 1:
            continue

        # FIRST BLOCK (3-day group)
        f_open = first['Open'].iloc[0]
        f_close = first['Close'].iloc[-1]

        # SECOND BLOCK (next 3-day group)
        s_open = second['Open'].iloc[0]
        s_close = second['Close'].iloc[-1]

        f_body = abs(f_close - f_open)
        s_body = abs(s_close - s_open)

        # =============================
        # 3-DAY TREND SHIFT
        # =============================
        if f_close < f_open and s_close > s_open:
            structure[i] += 1

        if f_close > f_open and s_close < s_open:
            structure[i] -= 1

        # =============================
        # BREAKOUT EXPANSION
        # =============================
        if s_body > (f_body * 1.2):
            structure[i] += np.sign(s_close - s_open)

    # =============================
    # FINAL SCORE (SAFE COMBINATION)
    # =============================
    raw = base + 0.7 * structure

    raw = np.nan_to_num(raw)

    return np.tanh(raw) * 5


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

        df = load_data(ticker)
        if df is None:
            return {"error": "No data found"}

        grad = compute_gradient(df)
        latest = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(latest, 3),
            "signal": "bullish" if latest > 1 else "bearish" if latest < -1 else "neutral",
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
# SCANNER
# =============================
def update_scan_loop():
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]

    while True:
        results = []

        for t in tickers:
            try:
                df = load_data(t)
                if df is None:
                    continue

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


@app.get("/scan")
def scan():
    return scan_cache


# =============================
# DASHBOARD
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Gradient Dashboard</title>
    <style>
        body { background:#0f172a; color:white; font-family:Arial; text-align:center; }
        input, button { padding:10px; margin:5px; }
        .card { background:#1e293b; padding:20px; border-radius:10px; display:inline-block; margin:20px; }
        table { margin:auto; border-collapse:collapse; }
        td,th { padding:10px 20px; border-bottom:1px solid #334155; }
    </style>
</head>
<body>

<h1>🔥 Gradient Heat Dashboard</h1>

<input id="ticker" placeholder="Enter ticker (AAPL)" />
<button onclick="run()">Analyze</button>

<div class="card">
    <h2 id="sym">---</h2>
    <h1 id="score">0</h1>
    <div id="signal">---</div>
</div>

<h3>Live Scanner</h3>
<button onclick="scan()">Refresh</button>
<table id="table"></table>

<script>
async function run(){
    const t = document.getElementById("ticker").value;
    const r = await fetch("/analyze?ticker="+t);
    const d = await r.json();

    document.getElementById("sym").innerText = d.ticker;
    document.getElementById("score").innerText = d.gradient_score;
    document.getElementById("signal").innerText = d.signal;
}

async function scan(){
    const r = await fetch("/scan");
    const d = await r.json();

    let h = "<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>";

    d.results.forEach(x=>{
        h += `<tr><td>${x.ticker}</td><td>${x.score}</td><td>${x.signal}</td></tr>`;
    });

    document.getElementById("table").innerHTML = h;
}

scan();
setInterval(scan, 15000);
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
        "dashboard": "/dashboard",
        "analyze": "/analyze?ticker=AAPL",
        "scan": "/scan"
    }
