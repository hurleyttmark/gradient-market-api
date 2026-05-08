from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import yfinance as yf
import numpy as np
import pandas as pd
import time
from threading import Thread
import traceback

app = FastAPI()

CACHE_TTL = 60
cache = {}
scan_cache = {"data": None, "timestamp": 0}

# =============================
# CACHE
# =============================
def get_cached(ticker):
    if ticker in cache:
        if time.time() - cache[ticker]["time"] < CACHE_TTL:
            return cache[ticker]["data"]
    return None

def set_cached(ticker, data):
    cache[ticker] = {"data": data, "time": time.time()}

# =============================
# GRADIENT ENGINE (FIXED SIGN LOGIC)
# =============================
def compute_gradient(df):
    df = df.copy()

    df["returns"] = df["Close"].pct_change()

    # volatility
    df["vol"] = df["returns"].rolling(10).std()
    df["vol"] = df["vol"].replace(0, np.nan)

    # raw momentum (IMPORTANT FIX: sign preserved)
    df["momentum"] = df["returns"] / df["vol"]
    df["momentum"] = df["momentum"].replace([np.inf, -np.inf], 0).fillna(0)

    # trend + accel
    df["trend"] = df["momentum"].rolling(3).mean().fillna(0)
    df["accel"] = df["momentum"].diff().fillna(0)

    # volume filter
    if "Volume" in df.columns:
        df["vol_ma"] = df["Volume"].rolling(10).mean()
        df["vol_boost"] = np.where(df["Volume"] > df["vol_ma"], 1, 0)
    else:
        df["vol_boost"] = 0

    # FINAL REGIME SCORE
    regime = (
        0.65 * df["trend"] +
        0.25 * df["accel"] +
        0.10 * df["vol_boost"]
    )

    # clamp to -5 to +5
    df["gradient"] = np.tanh(regime) * 5

    return df["gradient"].values

# =============================
# ANALYZE
# =============================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):
    ticker = ticker.upper()

    cached = get_cached(ticker)
    if cached:
        return cached

    df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return {"error": "No data"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()

    grad = compute_gradient(df)
    score = float(grad[-1])

    result = {
        "ticker": ticker,
        "gradient_score": round(score, 3),
        "signal": "bullish" if score > 1 else "bearish" if score < -1 else "neutral"
    }

    set_cached(ticker, result)
    return result

# =============================
# PLOT (FIXED CANDLESTYLE HEAT)
# =============================
@app.get("/plot")
def plot(ticker: str = Query(...)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64

    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return {"error": "No data"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()

    grad = compute_gradient(df)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(df.index, df["Close"], color="black", linewidth=1.8)

    # FIXED COLOR LOGIC (NO MORE WRONG SIGNALS)
    for i in range(1, len(df)):
        g = grad[i]
        price_change = df["Close"].iloc[i] - df["Close"].iloc[i-1]

        # TRUE alignment rule:
        # bullish only if BOTH gradient AND price agree
        if g > 0 and price_change > 0:
            color = (0, 1, 0, 0.25)
        elif g < 0 and price_change < 0:
            color = (1, 0, 0, 0.25)
        else:
            color = (0.5, 0.5, 0.5, 0.12)

        ax.axvspan(df.index[i-1], df.index[i], color=color)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode()

    return {"image": img}

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

# =============================
# SCAN
# =============================
@app.get("/scan")
def scan():
    return scan_cache

# =============================
# DASHBOARD (FIXED + WORKING)
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
    max-width:1000px;
    margin:auto;
    padding:20px;
}

.card {
    background:#111c33;
    padding:15px;
    border-radius:10px;
    margin:10px 0;
}

input, button {
    padding:10px;
    border-radius:6px;
    border:none;
}

img {
    width:100%;
    margin-top:10px;
    border-radius:10px;
}
</style>
</head>

<body>

<div class="container">

<h2>🔥 Gradient Engine</h2>

<input id="t" placeholder="AAPL"/>
<button onclick="run()">Analyze</button>

<div class="card">
    <h3>Score: <span id="score">--</span></h3>
    <h3>Signal: <span id="signal">--</span></h3>
</div>

<img id="chart"/>

<h3>Scan</h3>
<button onclick="scan()">Refresh</button>
<table id="table"></table>

</div>

<script>

async function run(){
    let t=document.getElementById("t").value;

    let r=await fetch("/analyze?ticker="+t);
    let d=await r.json();

    document.getElementById("score").innerText=d.gradient_score;
    document.getElementById("signal").innerText=d.signal;

    let p=await fetch("/plot?ticker="+t);
    let img=await p.json();

    document.getElementById("chart").src =
        "data:image/png;base64," + img.image;
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

# =============================
# ROOT
# =============================
@app.get("/")
def root():
    return {
        "status": "running",
        "dashboard": "/dashboard",
        "endpoints": ["/analyze", "/plot", "/scan"]
    }