from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
import time
import traceback
from threading import Thread
from fastapi.responses import HTMLResponse

app = FastAPI()

# =============================
# SETTINGS
# =============================
CACHE_TTL = 60

cache = {}
scan_cache = {
    "data": [],
    "timestamp": 0
}

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
# CORE GRADIENT ENGINE (YOUR ORIGINAL BASE + SAFE UPGRADE)
# =============================
def compute_gradient(df):
    df = df.copy()

    # -------------------------
    # BASE MOMENTUM SYSTEM (UNCHANGED CORE)
    # -------------------------
    df['returns'] = df['Close'].pct_change()
    df['vol'] = df['returns'].rolling(10).std()
    df['vol'] = df['vol'].replace(0, np.nan)

    df['momentum'] = (df['returns'] / df['vol']).replace([np.inf, -np.inf], 0).fillna(0)

    df['trend'] = df['momentum'].rolling(5).mean().fillna(0)
    df['accel'] = df['momentum'].diff().fillna(0)

    # -------------------------
    # VOLUME CONFIRMATION
    # -------------------------
    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(10).mean()
        df["vol_boost"] = (df["Volume"] > vol_ma).astype(int)
    else:
        df["vol_boost"] = 0

    # -------------------------
    # SAFE 3-DAY CANDLE PRESSURE (NO LOOPS, NO BREAKS)
    # -------------------------
    if "Open" in df.columns:
        body = (df["Close"] - df["Open"]).abs()
        rng = (df["High"] - df["Low"]).replace(0, np.nan)

        body_ratio = (body / rng).fillna(0)

        # bullish/bearish pressure
        direction = np.sign(df["Close"] - df["Open"])

        df["candle_pressure"] = (
            direction.rolling(3).sum().fillna(0) / 3
        )
    else:
        df["candle_pressure"] = 0

    # -------------------------
    # FINAL SCORE (STABLE HYBRID)
    # -------------------------
    raw = (
        0.55 * df['trend'] +
        0.25 * df['accel'] +
        0.10 * df['vol_boost'] +
        0.10 * df['candle_pressure']
    )

    return np.tanh(raw) * 5


# =============================
# ANALYZE
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
            return {"error": "no data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        grad = compute_gradient(df)
        score = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(score, 3),
            "signal": "bullish" if score > 1 else "bearish" if score < -1 else "neutral"
        }

        set_cache(ticker, result)
        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# =============================
# SCANNER LOOP (STABLE)
# =============================
def scanner_loop():
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

    while True:
        results = []

        for t in tickers:
            try:
                df = yf.download(t, period="1y", auto_adjust=True, progress=False)

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.dropna()

                score = float(compute_gradient(df)[-1])

                results.append({
                    "ticker": t,
                    "score": round(score, 3),
                    "signal": "bullish" if score > 1 else "bearish" if score < -1 else "neutral"
                })

            except:
                continue

        scan_cache["data"] = results
        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)


Thread(target=scanner_loop, daemon=True).start()


# =============================
# SCAN ENDPOINT
# =============================
@app.get("/scan")
def scan():
    return {
        "live": True,
        "last_updated": scan_cache["timestamp"],
        "data": scan_cache.get("data", [])
    }


# =============================
# DASHBOARD (CLEAN + SAFE)
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<title>Gradient Dashboard</title>

<style>
body {
    background:#0b1220;
    color:white;
    font-family:Arial;
    text-align:center;
}

.card {
    background:#111c2e;
    padding:20px;
    width:260px;
    margin:20px auto;
    border-radius:12px;
}

input,button { padding:10px; margin:5px; }

table {
    margin:auto;
    width:80%;
    border-collapse:collapse;
}

td,th {
    padding:10px;
    border-bottom:1px solid #2a3b55;
}
</style>

</head>

<body>

<h2>🔥 Gradient Heat Dashboard</h2>

<input id="ticker" placeholder="AAPL">
<button onclick="run()">Analyze</button>

<div class="card">
<h3 id="t">---</h3>
<h1 id="s">0</h1>
<div id="sig">---</div>
</div>

<button onclick="load()">Refresh Scan</button>

<table id="tbl"></table>

<script>

async function run(){
    const t=document.getElementById("ticker").value;

    const r=await fetch("/analyze?ticker="+t);
    const d=await r.json();

    document.getElementById("t").innerText=d.ticker||"---";
    document.getElementById("s").innerText=d.gradient_score??0;
    document.getElementById("sig").innerText=d.signal||"---";
}

async function load(){
    const r=await fetch("/scan");
    const d=await r.json();

    const rows = d?.data ?? [];

    let h="<tr><th>Ticker</th><th>Score</th><th>Signal</th></tr>";

    rows.forEach(x=>{
        h += `<tr>
                <td>${x.ticker}</td>
                <td>${x.score}</td>
                <td>${x.signal}</td>
              </tr>`;
    });

    document.getElementById("tbl").innerHTML=h;
}

load();
setInterval(load,10000);

</script>

</body>
</html>
""")


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
