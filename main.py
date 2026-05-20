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
# CACHE
# =============================
cache = {}
scan_cache = {"data": None, "timestamp": 0}

CACHE_TTL = 60


# =============================
# GRADIENT ENGINE
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

    for v in df['close_3d'].fillna(0):
        if v > 0:
            s = s + 1 if s > 0 else 1
        elif v < 0:
            s = s - 1 if s < 0 else -1
        else:
            s = 0
        streak.append(s)

    df['streak'] = streak

    df['trend'] = df['momentum'].rolling(5).mean().fillna(0)
    df['accel'] = df['trend'].diff(3).fillna(0)

    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['vol_boost'] = np.where(df['Volume'] > df['vol_ma'], 1, 0)

    regime = (
        0.55 * df['trend'] +
        0.30 * np.tanh(df['streak'] / 4) +
        0.10 * df['accel'] +
        0.05 * df['vol_boost']
    )

    df['gradient'] = np.tanh(regime.rolling(3).mean().fillna(0)) * 5

    return df['gradient'].values


# =============================
# ANALYZE ENDPOINT (FIXED SAFE OUTPUT)
# =============================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):

    try:
        ticker = ticker.upper()

        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {
                "ticker": ticker,
                "gradient_score": 0,
                "signal": "neutral"
            }

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']].dropna()

        grad = compute_gradient(df)

        if len(grad) == 0:
            return {
                "ticker": ticker,
                "gradient_score": 0,
                "signal": "neutral"
            }

        score = float(round(grad[-1], 3))

        return {
            "ticker": ticker,
            "gradient_score": score,
            "signal": (
                "bullish" if score > 1
                else "bearish" if score < -1
                else "neutral"
            )
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "gradient_score": 0,
            "signal": "neutral",
            "error": str(e)
        }


# =============================
# SCAN LOOP
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
# SCAN
# =============================
@app.get("/scan")
def scan():
    return scan_cache


# =============================
# CHART (FULLY DYNAMIC + SAFE)
# =============================
@app.get("/chart")
def chart(ticker: str = "SPY"):

    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return HTMLResponse("<h3>No data</h3>")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']].dropna()

    df['bullish_3x3'] = df['Close'] > df['Open']
    df['bearish_3x3'] = df['Close'] < df['Open']

    fig, ax = plt.subplots(figsize=(10,5), dpi=120)

    ax.plot(df.index, df['Close'], color='black', linewidth=1.5)

    price_range = df['High'].max() - df['Low'].min()
    arrow = price_range * 0.02

    for i in range(len(df)):

        x = df.index[i]

        if df['bullish_3x3'].iloc[i]:
            ax.annotate(
                '',
                xy=(x, df['Low'].iloc[i]),
                xytext=(x, df['Low'].iloc[i] - arrow),
                arrowprops=dict(color='green', arrowstyle='simple')
            )

        if df['bearish_3x3'].iloc[i]:
            ax.annotate(
                '',
                xy=(x, df['High'].iloc[i]),
                xytext=(x, df['High'].iloc[i] + arrow),
                arrowprops=dict(color='red', arrowstyle='simple')
            )

    ax.set_title(f"{ticker} Price Chart")
    ax.grid(alpha=0.2)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode()
    plt.close()

    return HTMLResponse(f"""
    <div style="width:100%;height:100%;display:flex;">
        <img src="data:image/png;base64,{img}"
             style="width:100%;height:100%;object-fit:contain;">
    </div>
    """)


# =============================
# DASHBOARD (FIXED + SYNCED)
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    html = """
<!DOCTYPE html>
<html>
<head>
<style>

body {
    margin:0;
    height:100vh;
    display:flex;
    flex-direction:column;
    overflow:hidden;
    background:#0b0f14;
    color:white;
    font-family:Arial;
}

.header {
    height:40px;
    line-height:40px;
    padding-left:10px;
    border-bottom:1px solid #1f2a37;
}

.container {
    flex:1;
    display:flex;
    min-height:0;
}

.left {
    width:25%;
    padding:10px;
    overflow:auto;
    border-right:1px solid #1f2a37;
}

.right {
    flex:1;
    min-width:0;
}

iframe {
    width:100%;
    height:100%;
    border:none;
}

table {
    width:100%;
    font-size:12px;
}

td,th {
    border-bottom:1px solid #1f2a37;
    padding:5px;
}

button {
    margin-top:5px;
    background:#1f2937;
    color:white;
    border:none;
    padding:6px;
    cursor:pointer;
}

input {
    width:100%;
    padding:6px;
}

</style>
</head>

<body>

<div class="header">🔥 Gradient Dashboard</div>

<div class="container">

<div class="left">

<input id="ticker" placeholder="AAPL">

<button onclick="analyze()">Analyze</button>

<div id="symbol">---</div>
<div id="score">0</div>
<div id="signal">---</div>

<button onclick="loadScan()">Refresh Scan</button>

<table id="table"></table>

</div>

<div class="right">
<iframe id="chartFrame" src="/chart?ticker=SPY"></iframe>
</div>

</div>

<script>

async function analyze(){

    const t = document.getElementById("ticker").value;

    if(!t){
        alert("Enter ticker");
        return;
    }

    const r = await fetch(`/analyze?ticker=${t}`);
    const d = await r.json();

    document.getElementById("symbol").innerText = d.ticker || "---";
    document.getElementById("score").innerText = d.gradient_score ?? 0;
    document.getElementById("signal").innerText = d.signal || "neutral";

    // FIX: update chart
    document.getElementById("chartFrame").src = `/chart?ticker=${t}`;
}

async function loadScan(){

    const r = await fetch("/scan");
    const d = await r.json();

    let html = "<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>";

    if(d.data){
        d.data.forEach(x=>{
            html += `<tr>
                        <td>${x.ticker}</td>
                        <td>${x.score}</td>
                        <td>${x.signal}</td>
                     </tr>`;
        });
    }

    document.getElementById("table").innerHTML = html;
}

loadScan();

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
        "chart": "/chart?ticker=TSLA",
        "scan": "/scan"
    }
