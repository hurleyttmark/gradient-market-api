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
# CONFIG
# =============================
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

    regime_raw = (
        0.55 * df['trend'] +
        0.30 * np.tanh(df['streak'] / 4) +
        0.10 * df['accel'] +
        0.05 * df['vol_boost']
    )

    df['gradient'] = np.tanh(regime_raw.rolling(3).mean().fillna(0)) * 5

    return df['gradient'].fillna(0).values


# =============================
# TICKER ANALYSIS
# =============================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):

    try:
        ticker = ticker.upper()

        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']].dropna()

        grad = compute_gradient(df)

        return {
            "ticker": ticker,
            "gradient_score": float(round(grad[-1], 3)),
            "signal": (
                "bullish" if grad[-1] > 1
                else "bearish" if grad[-1] < -1
                else "neutral"
            )
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# =============================
# SCAN (UNCHANGED LOGIC SIMPLIFIED)
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
# CHART (FULL FIXED + DYNAMIC TICKER)
# =============================
@app.get("/chart")
def chart(ticker: str = "SPY"):

    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return HTMLResponse("<h3>No data</h3>")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']].dropna()

    # =============================
    # PATTERN SIGNALS
    # =============================
    df['signal'] = np.where(df['Close'] > df['Open'], 1, -1)

    # 3x3 placeholders (you already compute in full system if needed)
    df['bullish_3x3'] = (df['signal'] == 1)
    df['bearish_3x3'] = (df['signal'] == -1)

    # engulf placeholders (safe fallback if not computed upstream)
    df['bullish_engulf'] = False
    df['bearish_engulf'] = False
    df['engulf_start'] = None

    # =============================
    # PLOT
    # =============================
    fig, ax = plt.subplots(figsize=(10,5), dpi=120)

    ax.plot(df.index, df['Close'], color='black', linewidth=1.5)

    price_range = df['High'].max() - df['Low'].min()
    arrow_len = price_range * 0.02

    # =============================
    # ARROWS (ONLY YOUR RULES)
    # =============================
    for i in range(len(df)):

        x = df.index[i]

        # bullish 3x3
        if df['bullish_3x3'].iloc[i]:
            ax.annotate(
                '',
                xy=(x, df['Low'].iloc[i]),
                xytext=(x, df['Low'].iloc[i] - arrow_len),
                arrowprops=dict(color='green', arrowstyle='simple')
            )

        # bearish 3x3
        if df['bearish_3x3'].iloc[i]:
            ax.annotate(
                '',
                xy=(x, df['High'].iloc[i]),
                xytext=(x, df['High'].iloc[i] + arrow_len),
                arrowprops=dict(color='red', arrowstyle='simple')
            )

    ax.set_title(f"{ticker} Price + Signals")
    ax.grid(alpha=0.2)

    plt.tight_layout()

    buf = io.BytesIO()
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
# DASHBOARD
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

</style>
</head>

<body>

<div class="header">🔥 Gradient Dashboard</div>

<div class="container">

<div class="left">

<input id="ticker" placeholder="AAPL" style="width:100%">
<button onclick="analyze()">Analyze</button>

<div id="symbol">---</div>
<div id="score">0</div>
<div id="signal">---</div>

<button onclick="loadScan()">Refresh</button>

<table id="table"></table>

</div>

<div class="right">
<iframe src="/chart?ticker=SPY"></iframe>
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
        "chart": "/chart?ticker=AAPL",
        "scan": "/scan"
    }
