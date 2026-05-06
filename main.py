import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import yfinance as yf
import numpy as np
import pandas as pd
import time
from threading import Thread

app = FastAPI()

CACHE_TTL = 60
cache = {}
scan_cache = {"data": None, "timestamp": 0}

# =============================
# GRADIENT ENGINE
# =============================
def compute_gradient(df):
    df = df.copy()

    df["returns"] = df["Close"].pct_change()
    df["vol"] = df["returns"].rolling(10).std()

    df["vol"] = df["vol"].replace(0, np.nan)
    df["momentum"] = df["returns"] / df["vol"]
    df["momentum"] = df["momentum"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["trend"] = df["momentum"].rolling(5).mean().fillna(0)
    df["accel"] = df["momentum"].diff().fillna(0)

    if "Volume" in df.columns:
        df["vol_ma"] = df["Volume"].rolling(10).mean()
        df["vol_boost"] = np.where(df["Volume"] > df["vol_ma"], 1, 0)
    else:
        df["vol_boost"] = 0

    regime = 0.6*df["trend"] + 0.3*df["accel"] + 0.1*df["vol_boost"]
    df["gradient"] = np.tanh(regime) * 5

    return df["gradient"].values


# =============================
# ANALYZE + PLOT (FIXED)
# =============================
@app.get("/analyze_full")
def analyze_full(ticker: str = Query(...)):
    ticker = ticker.upper()

    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return {"error": "No data"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()

    grad = compute_gradient(df)
    score = float(grad[-1])

    # ---------------- PLOT ----------------
    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(df.index, df["Close"], color="white", linewidth=2)

    for i in range(1, len(df)):
        g = grad[i]
        color = (0, min(1, g/5), 0, 0.25) if g > 0 else (min(1, -g/5), 0, 0, 0.25)
        ax.axvspan(df.index[i-1], df.index[i], color=color)

    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.set_title(f"{ticker} Gradient Chart", color="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode()

    return {
        "ticker": ticker,
        "score": round(score, 3),
        "signal": "bullish" if score > 1 else "bearish" if score < -1 else "neutral",
        "image": img
    }


# =============================
# SCAN LOOP
# =============================
def scan_loop():
    tickers = ["AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","SPY"]

    while True:
        results = []

        for t in tickers:
            try:
                df = yf.download(t, period="1y", auto_adjust=True, progress=False)
                if df is None or df.empty:
                    continue

                df = df[["Close"]].dropna()
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

Thread(target=scan_loop, daemon=True).start()


@app.get("/scan")
def scan():
    return scan_cache


# =============================
# DASHBOARD (FIXED PRO UI)
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<html>
<head>
<style>
body {
    background:#0b1220;
    color:white;
    font-family:Arial;
    margin:0;
}

.container {
    max-width:1200px;
    margin:auto;
    padding:20px;
}

.row {
    display:flex;
    gap:20px;
    flex-wrap:wrap;
}

.card {
    background:#111c33;
    padding:20px;
    border-radius:12px;
    flex:1;
    min-width:250px;
}

img {
    width:100%;
    margin-top:20px;
    border-radius:10px;
}

table {
    width:100%;
    margin-top:20px;
    border-collapse:collapse;
}

td, th {
    padding:10px;
    border-bottom:1px solid #223;
}
button, input {
    padding:10px;
    border-radius:6px;
}
</style>
</head>

<body>

<div class="container">

<h1>🔥 Gradient Dashboard</h1>

<input id="t" placeholder="AAPL"/>
<button onclick="run()">Analyze</button>

<div class="row">
    <div class="card">
        <h2>Score</h2>
        <h1 id="score">--</h1>
    </div>

    <div class="card">
        <h2>Signal</h2>
        <h1 id="signal">--</h1>
    </div>
</div>

<img id="chart"/>

<h2>📊 Scan</h2>
<button onclick="scan()">Refresh</button>

<table id="table"></table>

</div>

<script>

async function run(){
    let t=document.getElementById("t").value;

    let r=await fetch("/analyze_full?ticker="+t);
    let d=await r.json();

    document.getElementById("score").innerText=d.score;
    document.getElementById("signal").innerText=d.signal;

    document.getElementById("chart").src =
        "data:image/png;base64," + d.image;
}

async function scan(){
    let r=await fetch("/scan");
    let d=await r.json();

    let html="<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>";

    d.data.forEach(x=>{
        html+=`<tr><td>${x.ticker}</td><td>${x.score}</td><td>${x.signal}</td></tr>`;
    });

    document.getElementById("table").innerHTML=html;
}

scan();

</script>

</body>
</html>
"""


@app.get("/")
def root():
    return {"status":"running","dashboard":"/dashboard"}