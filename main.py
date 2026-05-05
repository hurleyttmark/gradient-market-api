from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
import time
import os
from threading import Thread
from fastapi.responses import HTMLResponse

app = FastAPI()

# =============================
# CONFIG
# =============================
GRADIENT_WINDOW = 20
CACHE_TTL = 60

TICKERS = [
    "AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL",
    "SPY","QQQ","IWM","NFLX","AMD"
]

scan_cache = {"data": [], "timestamp": 0}

# =============================
# REGIME-AWARE GRADIENT ENGINE (NEW)
# =============================
def compute_gradient(df):
    df = df.copy()
    df = df[['Open','High','Low','Close','Volume']].dropna()

    # -------------------------
    # CORE FEATURES
    # -------------------------
    df["range"] = df["High"] - df["Low"]
    df["body"] = (df["Close"] - df["Open"]).abs()

    df["return"] = df["Close"].pct_change().fillna(0)
    df["vol_change"] = df["Volume"].pct_change().fillna(0)

    # -------------------------
    # SIGNAL GENERATION
    # -------------------------
    signals = []

    for i in range(len(df)):
        r = df["range"].iloc[i]
        if r == 0:
            signals.append(0)
            continue

        body = df["Close"].iloc[i] - df["Open"].iloc[i]
        body_abs = abs(body)

        vol = df["Volume"].iloc[i]
        vol_ma = df["Volume"].rolling(10).mean().iloc[i]

        s = 0

        # REGIME SHIFT DETECTION (NEW)
        trend_strength = df["return"].rolling(5).mean().iloc[i]

        # bullish regime
        if body > 0:
            if body_abs / r > 0.5:
                s = 1
            if trend_strength > 0:
                s += 1
            if vol_ma > 0 and vol > vol_ma:
                s += 1

        # bearish regime
        elif body < 0:
            if body_abs / r > 0.5:
                s = -1
            if trend_strength < 0:
                s -= 1
            if vol_ma > 0 and vol > vol_ma:
                s -= 1

        signals.append(np.clip(s, -3, 3))

    df["signal"] = signals

    # -------------------------
    # STREAK ENGINE
    # -------------------------
    streak = []
    c = 0

    for s in signals:
        if s > 0:
            c = c + 1 if c > 0 else 1
        elif s < 0:
            c = c - 1 if c < 0 else -1
        else:
            c = 0
        streak.append(c)

    df["streak"] = streak

    # -------------------------
    # SCENARIO BOOST (3-CANDLE REGIME)
    # -------------------------
    N = 3
    scenario = np.zeros(len(df))

    for i in range(2 * N, len(df)):
        first = df.iloc[i-2*N:i-N]
        second = df.iloc[i-N:i]

        f_open, f_close = first["Open"].iloc[0], first["Close"].iloc[-1]
        s_open, s_close = second["Open"].iloc[0], second["Close"].iloc[-1]

        s_body = abs(s_close - s_open)
        s_range = second["High"].max() - second["Low"].min()

        vol_ok = second["Volume"].sum() >= 1.2 * first["Volume"].mean() * N

        if f_close < f_open and s_close > s_open and s_body >= 0.6 * s_range and vol_ok:
            scenario[i] = 3

        elif f_close > f_open and s_close < s_open and s_body >= 0.6 * s_range and vol_ok:
            scenario[i] = -3

    df["scenario"] = scenario

    # -------------------------
    # FINAL GRADIENT SCORE (IMPROVED)
    # -------------------------
    grad = []

    for i in range(len(df)):
        start = max(0, i - GRADIENT_WINDOW + 1)
        window = df["streak"].iloc[start:i+1]

        base = (window > 0).sum() - (window < 0).sum()

        # scenario boost
        if df["scenario"].iloc[i] != 0:
            base += int(np.sign(df["scenario"].iloc[i])) * 2

        # volatility expansion bias (NEW)
        vol_boost = df["range"].iloc[i] / (df["range"].rolling(10).mean().iloc[i] + 1e-9)

        if vol_boost > 1.5:
            base += np.sign(base)

        grad.append(max(-5, min(5, base)))

    return np.array(grad)

# =============================
# SCANNER LOOP
# =============================
def scan_loop():
    while True:
        results = []

        for t in TICKERS:
            try:
                # ALWAYS FRESH DATA (fixes "old date" issue)
                df = yf.download(
                    t,
                    period="6mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )

                if df is None or df.empty:
                    continue

                grad = compute_gradient(df)

                results.append({
                    "ticker": t,
                    "score": float(round(grad[-1], 3)),
                    "signal": (
                        "bullish" if grad[-1] > 1
                        else "bearish" if grad[-1] < -1
                        else "neutral"
                    )
                })

            except:
                continue

        scan_cache["data"] = sorted(results, key=lambda x: x["score"], reverse=True)
        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

# =============================
# STARTUP
# =============================
@app.on_event("startup")
def startup_event():
    Thread(target=scan_loop, daemon=True).start()

# =============================
# ROUTES
# =============================
@app.get("/scan")
def scan():
    return scan_cache

@app.get("/")
def root():
    return {
        "status": "running",
        "scan": "/scan",
        "note": "regime upgraded scanner active"
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
    <body>
        <h1>Gradient Regime Scanner</h1>
        <pre id="data">Loading...</pre>

        <script>
        async function load(){
            const res = await fetch('/scan');
            const data = await res.json();
            document.getElementById('data').innerText =
                JSON.stringify(data, null, 2);
        }

        load();
        setInterval(load, 5000);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)