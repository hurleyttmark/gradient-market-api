from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
import traceback
import time
from threading import Thread
from fastapi.responses import HTMLResponse
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# =============================
# CORE SETTINGS
# =============================
GRADIENT_WINDOW = 10
CACHE_TTL = 60

cache = {}
scan_cache = {"data": None, "timestamp": 0}

# =============================
# GRADIENT ENGINE (UNCHANGED)
# =============================
def compute_gradient(df):

    df = df.copy()
    df['close_3d'] = df['Close'].pct_change(3)
    df['volatility'] = df['close_3d'].rolling(20).std().replace(0, np.nan)

    df['momentum'] = (
        df['close_3d'] / df['volatility']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

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

    df['trend'] = df['momentum'].rolling(5).mean().fillna(0)
    df['accel'] = df['trend'].diff(3).fillna(0)

    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['vol_boost'] = np.where(df['Volume'] > df['vol_ma'], 1, 0)

    regime_raw = (
        0.55 * df['trend'] +
        0.30 * np.tanh(df['streak'] / 4) +
        0.10 * df['accel'] +
        0.05 * df['vol_boost']
    )

    df['gradient'] = (
        np.tanh(regime_raw.rolling(3).mean().fillna(0)) * 5
    )

    return df['gradient'].fillna(0).values


# =============================
# CACHE
# =============================
def get_cached(ticker):
    if ticker in cache:
        if time.time() - cache[ticker]["time"] < CACHE_TTL:
            return cache[ticker]["data"]
    return None


def set_cache(ticker, data):
    cache[ticker] = {"data": data, "time": time.time()}


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

        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']].dropna()

        grad = compute_gradient(df)

        result = {
            "ticker": ticker,
            "gradient_score": float(round(grad[-1], 3)),
            "signal": (
                "bullish" if grad[-1] > 1
                else "bearish" if grad[-1] < -1
                else "neutral"
            ),
            "cached": False
        }

        set_cache(ticker, result)
        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# =============================
# LIVE SCANNER (UNCHANGED)
# =============================
def update_scan_loop():

    tickers = ["AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL"]

    while True:

        results = []

        for t in tickers:

            try:
                df = yf.download(t, period="1y", auto_adjust=True, progress=False)
                if df is None or df.empty:
                    continue

                df = df[['Open','High','Low','Close','Volume']].dropna()
                grad = compute_gradient(df)
                score = float(grad[-1])

                results.append({
                    "ticker": t,
                    "score": round(score, 3),
                    "signal": (
                        "bullish" if score > 1
                        else "bearish" if score < -1
                        else "neutral"
                    )
                })

            except:
                continue

        scan_cache["data"] = sorted(results, key=lambda x: x["score"], reverse=True)
        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)


Thread(target=update_scan_loop, daemon=True).start()


# =============================
# SCAN ENDPOINT
# =============================
@app.get("/scan")
def scan():
    return scan_cache


# =============================
# CHART ENDPOINT (NEW FIX)
# =============================
@app.get("/chart")
def chart():

    ticker = "SPY"

    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']].dropna()

    # candle logic (simplified arrows only)
    df['signal'] = np.where(df['Close'] > df['Open'], 1, -1)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(df.index, df['Close'], color='black', linewidth=2)

    price_range = df['High'].max() - df['Low'].min()
    arrow_size = price_range * 0.02

    for i in range(len(df)):

        x = df.index[i]

        # bullish arrow
        if df['signal'].iloc[i] == 1:
            ax.annotate(
                '',
                xy=(x, df['Low'].iloc[i]),
                xytext=(x, df['Low'].iloc[i] - arrow_size),
                arrowprops=dict(color='green', arrowstyle='simple')
            )

        # bearish arrow
        else:
            ax.annotate(
                '',
                xy=(x, df['High'].iloc[i]),
                xytext=(x, df['High'].iloc[i] + arrow_size),
                arrowprops=dict(color='red', arrowstyle='simple')
            )

    ax.set_title(f"{ticker} Price Action")
    ax.grid(alpha=0.2)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return HTMLResponse(f"<img style='width:100%;height:100%' src='data:image/png;base64,{img}'/>")


# =============================
# DASHBOARD (FIXED SPLIT LAYOUT)
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    html = """
<!DOCTYPE html>
<html>
<head>
<title>Gradient Dashboard</title>

<style>

body {
    margin: 0;
    background: #0b0f14;
    color: white;
    font-family: Arial;
    height: 100vh;
    overflow: hidden;
}

.header {
    padding: 10px;
    font-weight: bold;
    border-bottom: 1px solid #1f2a37;
}

.container {
    display: flex;
    height: calc(100vh - 40px);
}

.left {
    width: 25%;
    background: #0f141b;
    padding: 10px;
    overflow: hidden;
    border-right: 1px solid #1f2a37;
}

.right {
    width: 75%;
}

iframe {
    width: 100%;
    height: 100%;
    border: none;
}

.card {
    background: #111827;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

table {
    width: 100%;
    font-size: 12px;
}

td, th {
    border-bottom: 1px solid #1f2a37;
    padding: 5px;
}

button {
    background: #1f2937;
    color: white;
    border: none;
    padding: 6px;
    margin-top: 5px;
    border-radius: 6px;
    cursor: pointer;
}

</style>
</head>

<body>

<div class="header">🔥 Gradient Flow Dashboard</div>

<div class="container">

<div class="left">

    <div class="card">
        <input id="ticker" placeholder="AAPL" style="width:100%">
        <button onclick="analyze()">Analyze</button>

        <div id="symbol">---</div>
        <div id="score">0</div>
        <div id="signal">---</div>
    </div>

    <button onclick="loadScan()">Refresh Scan</button>

    <table id="table"></table>

</div>

<div class="right">
    <iframe src="/chart"></iframe>
</div>

</div>

<script>

async function analyze(){
    const t = document.getElementById("ticker").value;
    const r = await fetch(`/analyze?ticker=${t}`);
    const d = await r.json();

    document.getElementById("symbol").innerText = d.ticker;
    document.getElementById("score").innerText = d.gradient_score;
    document.getElementById("signal").innerText = d.signal;
}

async function loadScan(){
    const r = await fetch("/scan");
    const d = await r.json();

    let html = "<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>";

    if(d.data){
        d.data.forEach(x=>{
            html += `<tr><td>${x.ticker}</td><td>${x.score}</td><td>${x.signal}</td></tr>`;
        });
    }

    document.getElementById("table").innerHTML = html;
}

loadScan();
setInterval(loadScan,15000);

</script>

</body>
</html>
"""

    return HTMLResponse(html)


# =============================
# ROOT
# =============================
@app.get("/")
def root():
    return {
        "dashboard": "/dashboard",
        "analyze": "/analyze?ticker=AAPL",
        "scan": "/scan",
        "chart": "/chart"
    }
