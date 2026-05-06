import matplotlib
matplotlib.use("Agg")  # IMPORTANT (server-safe)

import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
import traceback
import time
from threading import Thread

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

# =============================
# GRADIENT ENGINE (REGIME MODEL)
# =============================

def compute_gradient(df):
    df = df.copy()

    df['returns'] = df['Close'].pct_change()
    df['vol'] = df['returns'].rolling(10).std()

    df['vol'] = df['vol'].replace(0, np.nan)
    df['momentum'] = df['returns'] / df['vol']
    df['momentum'] = df['momentum'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # REGIME COMPONENTS
    df['trend'] = df['momentum'].rolling(5).mean().fillna(0)
    df['accel'] = df['momentum'].diff().fillna(0)

    if 'Volume' in df.columns:
        df['vol_ma'] = df['Volume'].rolling(10).mean()
        df['vol_boost'] = np.where(df['Volume'] > df['vol_ma'], 1, 0)
    else:
        df['vol_boost'] = 0

    regime_raw = (
        0.6 * df['trend'] +
        0.3 * df['accel'] +
        0.1 * df['vol_boost']
    )

    df['gradient'] = np.tanh(regime_raw) * 5

    return df['gradient'].values

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
            return {"error": "No data found"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Close']].dropna()

        grad = compute_gradient(df)
        latest = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(latest, 3),
            "signal": "bullish" if latest > 1 else "bearish" if latest < -1 else "neutral"
        }

        set_cache(ticker, result)
        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# =============================
# 📊 PLOT ENDPOINT (HEAT + PRICE)
# =============================
@app.get("/plot")
def plot(ticker: str = Query(...)):
    try:
        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

        if df is None or df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Close']].dropna()

        grad = compute_gradient(df)

        fig, ax = plt.subplots(figsize=(12,5))

        ax.plot(df.index, df['Close'], color='black', linewidth=2)

        # gradient heat overlay
        for i in range(1, len(df)):
            g = grad[i]
            color = (0, min(1, g/5), 0, 0.2) if g > 0 else (min(1, -g/5), 0, 0, 0.2)
            ax.axvspan(df.index[i-1], df.index[i], color=color)

        ax.set_title(f"{ticker.upper()} Price + Gradient Heat")
        ax.grid(alpha=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        img = base64.b64encode(buf.read()).decode()

        return JSONResponse({
            "ticker": ticker,
            "image": img
        })

    except Exception as e:
        return {"error": str(e)}

# =============================
# SCANNER LOOP
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
# SCAN
# =============================
@app.get("/scan")
def scan():
    return scan_cache

# =============================
# DASHBOARD
# =============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
    <body style='background:#0f172a;color:white;text-align:center;font-family:Arial;'>
        <h1>Gradient Dashboard</h1>
        <input id='t'/>
        <button onclick='go()'>Run</button>
        <div id='out'></div>

        <script>
        async function go(){
            let t=document.getElementById('t').value;
            let r=await fetch('/plot?ticker='+t);
            let d=await r.json();
            document.getElementById('out').innerHTML=
                '<img style="width:90%" src="data:image/png;base64,'+d.image+'"/>';
        }
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
        "endpoints": ["/analyze", "/scan", "/plot", "/dashboard"]
    }
