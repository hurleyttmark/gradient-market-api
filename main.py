from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import traceback

app = FastAPI()

# =============================
# CORE ENGINE
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

    regime = (
        0.6 * df["trend"] +
        0.3 * df["accel"] +
        0.1 * df["vol_boost"]
    )

    df["gradient"] = np.tanh(regime) * 5

    return df["gradient"].values


# =============================
# ANALYZE + PLOT
# =============================

@app.get("/analyze")
def analyze(ticker: str = Query(...)):
    try:
        ticker = ticker.upper()

        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {"error": "No data"}

        df = df.dropna()

        grad = compute_gradient(df)
        score = float(grad[-1])

        # =============================
        # TREND LINE (EMA)
        # =============================
        df["ema"] = df["Close"].ewm(span=20).mean()

        # =============================
        # PLOT
        # =============================
        fig, ax = plt.subplots(figsize=(12, 5))

        # price
        ax.plot(df.index, df["Close"], label="Price", color="white", linewidth=1.5)

        # trend
        ax.plot(df.index, df["ema"], label="Trend (EMA 20)", color="cyan", linewidth=1)

        # gradient heat overlay
        for i in range(1, len(df)):
            g = grad[i]

            if g > 0:
                color = (0, min(1, g/5), 0, 0.15)
            else:
                color = (min(1, -g/5), 0, 0, 0.15)

            ax.axvspan(df.index[i-1], df.index[i], color=color)

        ax.set_title(f"{ticker} | Gradient Score: {round(score, 2)}")
        ax.set_facecolor("#0b1220")
        fig.patch.set_facecolor("#0b1220")
        ax.tick_params(colors="white")
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        img = base64.b64encode(buf.read()).decode()

        # =============================
        # SIGNAL
        # =============================
        if score > 1:
            signal = "BULLISH"
        elif score < -1:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "ticker": ticker,
            "score": round(score, 3),
            "signal": signal,
            "image": img
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }


# =============================
# DASHBOARD (CLEAN UI)
# =============================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<html>
<head>
<style>

body {
    margin:0;
    font-family: Arial;
    background:#0b1220;
    color:white;
}

.container {
    max-width:1000px;
    margin:auto;
    padding:20px;
    text-align:center;
}

input, button {
    padding:10px;
    border-radius:6px;
    border:none;
    margin:5px;
}

button {
    background:#2563eb;
    color:white;
    cursor:pointer;
}

.card {
    display:inline-block;
    background:#1e293b;
    padding:15px;
    margin:10px;
    border-radius:10px;
    min-width:150px;
}

img {
    width:100%;
    margin-top:20px;
    border-radius:10px;
}

</style>
</head>

<body>

<div class="container">

    <h1>🔥 Gradient Signal Dashboard</h1>

    <input id="t" placeholder="Enter ticker (AAPL)" />
    <button onclick="run()">Analyze</button>

    <div>
        <div class="card">
            <div>Score</div>
            <h2 id="score">--</h2>
        </div>

        <div class="card">
            <div>Signal</div>
            <h2 id="signal">--</h2>
        </div>
    </div>

    <img id="chart"/>

</div>

<script>

async function run(){

    let t = document.getElementById("t").value;

    let r = await fetch("/analyze?ticker=" + t);
    let d = await r.json();

    document.getElementById("score").innerText = d.score;
    document.getElementById("signal").innerText = d.signal;

    document.getElementById("chart").src =
        "data:image/png;base64," + d.image;
}

</script>

</body>
</html>
"""


@app.get("/")
def root():
    return {
        "status": "running",
        "dashboard": "/dashboard",
        "endpoint": "/analyze?ticker=AAPL"
    }