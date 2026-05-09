from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import time
import traceback
from threading import Thread
from fastapi.responses import HTMLResponse

app = FastAPI()

# =========================
# CORS FIX (IMPORTANT)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CACHE
# =========================
cache = {}
CACHE_TTL = 60


def compute_gradient(df):
    df = df.copy()

    df["returns"] = df["Close"].pct_change()
    df["vol"] = df["returns"].rolling(10).std()

    df["vol"] = df["vol"].replace(0, np.nan)
    df["momentum"] = df["returns"] / df["vol"]
    df["momentum"] = df["momentum"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["trend"] = df["momentum"].rolling(5).mean().fillna(0)
    df["accel"] = df["momentum"].diff().fillna(0)

    df["regime"] = 0.6 * df["trend"] + 0.3 * df["accel"]

    df["gradient"] = np.tanh(df["regime"]) * 5

    return df["gradient"].values


def get_cached(ticker):
    if ticker in cache and time.time() - cache[ticker]["time"] < CACHE_TTL:
        return cache[ticker]["data"]
    return None


def set_cached(ticker, data):
    cache[ticker] = {"data": data, "time": time.time()}


# =========================
# API
# =========================
@app.get("/analyze")
def analyze(ticker: str = Query(...)):
    try:
        ticker = ticker.upper()

        cached = get_cached(ticker)
        if cached:
            return cached

        df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {"error": "No data"}

        df = df[["Close"]].dropna()

        grad = compute_gradient(df)
        score = float(grad[-1])

        result = {
            "ticker": ticker,
            "score": round(score, 3),
            "signal": "bullish" if score > 1 else "bearish" if score < -1 else "neutral"
        }

        set_cached(ticker, result)
        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# =========================
# DASHBOARD (EMBED SAFE)
# =========================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
<style>
body {
    margin:0;
    font-family: Arial;
    background:#0b0f14;
    color:white;
}

.box {
    max-width:700px;
    margin:60px auto;
    padding:30px;
    background:#111827;
    border-radius:12px;
    text-align:center;
}

input {
    padding:10px;
    width:60%;
    border-radius:6px;
    border:none;
}

button {
    padding:10px 15px;
    border:none;
    background:#2563eb;
    color:white;
    border-radius:6px;
    cursor:pointer;
}

.card {
    margin-top:20px;
    padding:20px;
    background:#1f2937;
    border-radius:10px;
}
</style>
</head>

<body>

<div class="box">

    <h2>Gradient Score Scanner</h2>

    <input id="ticker" placeholder="Enter ticker (AAPL)" />
    <button onclick="run()">Analyze</button>

    <div class="card">
        <h3 id="t">---</h3>
        <h1 id="s">0</h1>
        <div id="sig">---</div>
    </div>

</div>

<script>

async function run() {
    let t = document.getElementById("ticker").value;

    try {
        let res = await fetch("/analyze?ticker=" + encodeURIComponent(t));
        let data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById("t").innerText = data.ticker;
        document.getElementById("s").innerText = data.score;
        document.getElementById("sig").innerText = data.signal;

    } catch (err) {
        alert("API fetch failed. Check deployment.");
    }
}

</script>

</body>
</html>
"""


@app.get("/")
def root():
    return {
        "status": "ok",
        "dashboard": "/dashboard",
        "analyze": "/analyze?ticker=AAPL"
    }
