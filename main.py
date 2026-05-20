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
CACHE_TTL = 60

cache = {}
scan_cache = {"data": None, "timestamp": 0}

# =============================
# GRADIENT ENGINE (UNCHANGED LOGIC STYLE)
# =============================
def compute_gradient(df):

    df = df.copy()

    df['close_3d'] = df['Close'].pct_change(3)
    df['volatility'] = df['close_3d'].rolling(20).std().replace(0, np.nan)

    df['momentum'] = (
        df['close_3d'] / df['volatility']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # streak
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

    df['gradient'] = np.tanh(regime_raw.rolling(3).mean().fillna(0)) * 5

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
# LIVE SCAN LOOP
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
# CHART (FIXED - ANY TICKER + CLEAN SIGNALS)
# =============================
@app.get("/chart")
def chart(t: str = "SPY"):

    ticker = t.upper()

    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return HTMLResponse(f"<h3>No data for {ticker}</h3>")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']].dropna()

    # =============================
    # CLEAN CANDLE SIGNAL LOGIC
    # =============================

    df['range'] = df['High'] - df['Low']
    df['body'] = abs(df['Close'] - df['Open'])
    df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)

    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']

    def classify(row):

        if row['range'] == 0:
            return 0

        if row['body_pct'] <= 0.20:
            return 0

        body = row['Close'] - row['Open']
        body_abs = abs(body)

        if body > 0:
            if body_abs / row['range'] >= 0.65:
                return 1
            if row['lower_wick'] >= 1.3 * body_abs:
                return 1

        if body < 0:
            if body_abs / row['range'] >= 0.65:
                return -1
            if row['upper_wick'] >= 1.3 * body_abs:
                return -1

        return 0

    df['signal'] = df.apply(classify, axis=1)

    # =============================
    # PLOT
    # =============================

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df['Close'], color='black', linewidth=2)

    price_range = df['High'].max() - df['Low'].min()
    arrow_size = price_range * 0.015

    for i in range(len(df)):

        x = df.index[i]

        if df['signal'].iloc[i] == 1:
            ax.annotate(
                '',
                xy=(x, df['Low'].iloc[i]),
                xytext=(x, df['Low'].iloc[i] - arrow_size),
                arrowprops=dict(color='green', arrowstyle='simple'),
                zorder=5
            )

        elif df['signal'].iloc[i] == -1:
            ax.annotate(
                '',
                xy=(x, df['High'].iloc[i]),
                xytext=(x, df['High'].iloc[i] + arrow_size),
                arrowprops=dict(color='red', arrowstyle='simple'),
                zorder=5
            )

    ax.set_title(f"{ticker} Price Action (Filtered Signals)")
    ax.grid(alpha=0.25)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return HTMLResponse(
        f"<img style='width:100%;height:100%' src='data:image/png;base64,{img}'/>"
    )


# =============================
# ROOT
# =============================
@app.get("/")
def root():
    return {
        "dashboard": "/dashboard",
        "analyze": "/analyze?ticker=AAPL",
        "scan": "/scan",
        "chart": "/chart?t=TSLA"
    }
