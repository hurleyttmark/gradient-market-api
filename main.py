import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64
import time
import traceback
from threading import Thread

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI()

# =========================================================
# SETTINGS
# =========================================================

CACHE_TTL = 60

cache = {}

scan_cache = {
    "data": [],
    "timestamp": 0
}

# =========================================================
# GRADIENT ENGINE
# =========================================================

def compute_gradient(df):

    df = df.copy()

    # =====================================================
    # CANDLE METRICS
    # =====================================================

    df['range'] = df['High'] - df['Low']

    df['body'] = abs(
        df['Close'] - df['Open']
    )

    df['upper_wick'] = (
        df['High'] -
        df[['Open', 'Close']].max(axis=1)
    )

    df['lower_wick'] = (
        df[['Open', 'Close']].min(axis=1) -
        df['Low']
    )

    df['body_pct'] = (
        df['body'] /
        df['range'].replace(0, np.nan)
    ).fillna(0)

    # =====================================================
    # CANDLE CLASSIFIER
    # =====================================================

    def classify_candle(row):

        if row['range'] == 0:
            return 0

        neutral = (
            row['body_pct'] <= 0.20 and
            abs(row['upper_wick'] - row['lower_wick'])
            <= 0.20 * row['range']
        )

        if neutral:
            return 0

        body = row['Close'] - row['Open']
        body_abs = abs(body)

        upper = row['upper_wick']
        lower = row['lower_wick']

        # bullish
        if body > 0:

            if body_abs / row['range'] >= 0.65:
                return 1

            if lower >= 1.3 * body_abs:
                return 1

            if body_abs / row['range'] >= 0.25:
                return 1

            if upper > 1.5 * body_abs:
                return 0

        # bearish
        if body < 0:

            if body_abs / row['range'] >= 0.65:
                return -1

            if upper >= 1.3 * body_abs:
                return -1

            if body_abs / row['range'] >= 0.25:
                return -1

            if lower > 1.5 * body_abs:
                return 0

        return 0

    df['signal'] = df.apply(classify_candle, axis=1)

    # =====================================================
    # STREAK ENGINE
    # =====================================================

    streak = []
    count = 0

    for s in df['signal']:

        if s == 1:
            count = count + 1 if count > 0 else 1

        elif s == -1:
            count = count - 1 if count < 0 else -1

        else:
            count = 0

        streak.append(count)

    df['streak'] = streak

    # =====================================================
    # 3-DAY ENGULFING
    # =====================================================

    N = 3

    df['bullish_engulf'] = False
    df['bearish_engulf'] = False

    for i in range(2 * N, len(df)):

        first = df.iloc[i - 2*N:i - N]
        second = df.iloc[i - N:i]

        first_open = first['Open'].iloc[0]
        first_close = first['Close'].iloc[-1]

        second_open = second['Open'].iloc[0]
        second_close = second['Close'].iloc[-1]

        first_range = (
            first['High'].max() -
            first['Low'].min()
        )

        second_range = (
            second['High'].max() -
            second['Low'].min()
        )

        first_body = abs(first_close - first_open)
        second_body = abs(second_close - second_open)

        avg_first_vol = first['Volume'].mean()
        second_vol = second['Volume'].sum()

        volume_ok = (
            second_vol >=
            1.25 * avg_first_vol * N
        )

        # bullish engulf
        if (
            first_close < first_open and
            first_body >= 0.5 * first_range and
            second_close > second_open and
            second_body >= 0.7 * second_range and
            second_open <= first_open and
            second_close >= first_close and
            volume_ok
        ):

            df.loc[
                df.index[i-N:i],
                'bullish_engulf'
            ] = True

        # bearish engulf
        if (
            first_close > first_open and
            first_body >= 0.5 * first_range and
            second_close < second_open and
            second_body >= 0.7 * second_range and
            second_open >= first_open and
            second_close <= first_close and
            volume_ok
        ):

            df.loc[
                df.index[i-N:i],
                'bearish_engulf'
            ] = True

    # =====================================================
    # 3x3 PATTERNS
    # =====================================================

    df['bullish_3x3'] = False
    df['bearish_3x3'] = False

    for i in range(6, len(df)):

        first = df.iloc[i-6:i-3]
        second = df.iloc[i-3:i]
        third = df.iloc[i-2:i+1]

        f_open = first['Open'].iloc[0]
        f_close = first['Close'].iloc[-1]

        f_body = abs(f_close - f_open)

        f_range = (
            first['High'].max() -
            first['Low'].min()
        )

        s_body = abs(
            second['Close'].iloc[-1] -
            second['Open'].iloc[0]
        )

        t_open = third['Open'].iloc[0]
        t_close = third['Close'].iloc[-1]

        t_body = abs(t_close - t_open)

        t_range = (
            third['High'].max() -
            third['Low'].min()
        )

        # bullish
        if (
            f_close < f_open and
            f_body >= 0.7 * f_range and
            s_body <= 0.25 * f_body and
            t_close > t_open and
            t_body >= 0.7 * t_range
        ):

            df.at[
                df.index[i-1],
                'bullish_3x3'
            ] = True

        # bearish
        if (
            f_close > f_open and
            f_body >= 0.7 * f_range and
            s_body <= 0.25 * f_body and
            t_close < t_open and
            t_body >= 0.7 * t_range
        ):

            df.at[
                df.index[i-1],
                'bearish_3x3'
            ] = True

    # =====================================================
    # SCENARIO STRENGTH
    # =====================================================

    df['scenario_strength'] = 0

    df.loc[
        df['bullish_engulf'],
        'scenario_strength'
    ] += 2

    df.loc[
        df['bearish_engulf'],
        'scenario_strength'
    ] -= 2

    df.loc[
        df['bullish_3x3'],
        'scenario_strength'
    ] += 3

    df.loc[
        df['bearish_3x3'],
        'scenario_strength'
    ] -= 3

    # =====================================================
    # TREND ENGINE
    # =====================================================

    df['price_delta'] = (
        df['Close'].diff().abs()
    )

    df['trend_raw'] = (
        df['streak'] *
        df['price_delta'] *
        df['Volume']
    )

    df['trend_raw'] = (
        df['trend_raw'].fillna(0)
    )

    scale = (
        df['trend_raw']
        .abs()
        .rolling(200)
        .max()
    )

    df['trend_norm'] = (
        df['trend_raw'] /
        scale.replace(0, np.nan)
    )

    df['trend_norm'] = (
        df['trend_norm'].fillna(0)
    )

    # =====================================================
    # SLOPE FILTER
    # =====================================================

    slope = df['Close'].diff(10)

    gated = np.where(
        slope > 0,
        df['streak'].clip(lower=0),
        df['streak'].clip(upper=0)
    )

    def trend_code(v):

        if v >= 2:
            return 2

        if v >= 1:
            return 1

        if v <= -2:
            return -2

        if v <= -1:
            return -1

        return 0

    df['trend_code'] = (
        pd.Series(gated, index=df.index)
        .apply(trend_code)
    )

    # =====================================================
    # FINAL GRADIENT SCORE
    # =====================================================

    scores = []

    for i in range(len(df)):

        start = max(0, i - 20)

        window = (
            df['trend_code']
            .iloc[start:i+1]
        )

        score = (
            (window > 0).sum() -
            (window < 0).sum()
        )

        score = np.clip(score, -5, 5)

        scenario = (
            df['scenario_strength']
            .iloc[i]
        )

        if scenario != 0:
            score += np.sign(scenario)

        score = np.clip(score, -5, 5)

        scores.append(score)

    return np.array(scores)

# =========================================================
# CACHE
# =========================================================

def get_cached(ticker):

    if ticker in cache:

        entry = cache[ticker]

        if time.time() - entry["time"] < CACHE_TTL:
            return entry["data"]

    return None


def set_cache(ticker, data):

    cache[ticker] = {
        "data": data,
        "time": time.time()
    }

# =========================================================
# ANALYZE ENDPOINT
# =========================================================

@app.get("/analyze")
def analyze(ticker: str = Query(...)):

    try:

        ticker = ticker.upper()

        cached = get_cached(ticker)

        if cached:
            return cached

        df = yf.download(
            ticker,
            period="3y",
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[
            ['Open','High','Low','Close','Volume']
        ].dropna()

        grad = compute_gradient(df)

        latest = float(grad[-1])

        result = {
            "ticker": ticker,
            "gradient_score": round(latest, 2),
            "signal":
                "bullish" if latest > 1 else
                "bearish" if latest < -1 else
                "neutral"
        }

        set_cache(ticker, result)

        return result

    except Exception as e:

        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

# =========================================================
# PLOT ENDPOINT
# =========================================================

@app.get("/plot")
def plot(ticker: str = Query(...)):

    try:

        ticker = ticker.upper()

        df = yf.download(
            ticker,
            period="1y",
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[
            ['Open','High','Low','Close','Volume']
        ].dropna()

        grad = compute_gradient(df)

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            df.index,
            df['Close'],
            color='black',
            linewidth=2
        )

        # gradient heat
        for i in range(1, len(df)):

            g = grad[i]

            if g > 0:
                color = (
                    0,
                    min(1, g / 5),
                    0,
                    0.20
                )
            else:
                color = (
                    min(1, abs(g) / 5),
                    0,
                    0,
                    0.20
                )

            ax.axvspan(
                df.index[i-1],
                df.index[i],
                color=color
            )

        ax.set_title(
            f"{ticker} Gradient Heat"
        )

        ax.grid(alpha=0.2)

        buf = io.BytesIO()

        plt.savefig(
            buf,
            format='png',
            bbox_inches='tight'
        )

        plt.close(fig)

        buf.seek(0)

        img = base64.b64encode(
            buf.read()
        ).decode()

        return JSONResponse({
            "ticker": ticker,
            "image": img
        })

    except Exception as e:

        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

# =========================================================
# LIVE SCANNER
# =========================================================

def update_scan_loop():

    tickers = [
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "META"
    ]

    while True:

        results = []

        for t in tickers:

            try:

                df = yf.download(
                    t,
                    period="1y",
                    auto_adjust=True,
                    progress=False
                )

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = (
                        df.columns
                        .get_level_values(0)
                    )

                df = df[
                    ['Open','High','Low','Close','Volume']
                ].dropna()

                grad = compute_gradient(df)

                latest = float(grad[-1])

                results.append({
                    "ticker": t,
                    "score": round(latest, 2),
                    "signal":
                        "bullish" if latest > 1 else
                        "bearish" if latest < -1 else
                        "neutral"
                })

            except:
                continue

        scan_cache["data"] = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        scan_cache["timestamp"] = time.time()

        time.sleep(CACHE_TTL)

Thread(
    target=update_scan_loop,
    daemon=True
).start()

# =========================================================
# SCAN ENDPOINT
# =========================================================

@app.get("/scan")
def scan():

    return scan_cache

# =========================================================
# DASHBOARD
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    return """
<!DOCTYPE html>
<html>
<head>

<title>Gradient Dashboard</title>

<style>

body{
    margin:0;
    background:#0f172a;
    color:white;
    font-family:Arial;
}

.wrapper{
    max-width:1100px;
    margin:auto;
    padding:40px 20px;
}

h1{
    font-size:42px;
    margin-bottom:10px;
}

.sub{
    color:#94a3b8;
    margin-bottom:30px;
}

input{
    padding:12px;
    width:220px;
    border:none;
    border-radius:8px;
    margin-right:10px;
}

button{
    padding:12px 18px;
    border:none;
    border-radius:8px;
    background:#2563eb;
    color:white;
    cursor:pointer;
}

.cards{
    display:flex;
    gap:20px;
    margin-top:25px;
    flex-wrap:wrap;
}

.card{
    background:#1e293b;
    padding:20px;
    border-radius:12px;
    min-width:180px;
}

.label{
    color:#94a3b8;
    margin-bottom:10px;
}

.value{
    font-size:34px;
    font-weight:bold;
}

#chart{
    width:100%;
    margin-top:30px;
    border-radius:12px;
}

table{
    width:100%;
    margin-top:30px;
    border-collapse:collapse;
}

th,td{
    padding:14px;
    border-bottom:1px solid #334155;
    text-align:left;
}

th{
    color:#94a3b8;
}

</style>
</head>

<body>

<div class="wrapper">

<h1>Gradient Flow Dashboard</h1>

<div class="sub">
Longer-term multi-candle gradient scoring engine
using 3-day engulfing and 3x3 structures.
</div>

<input
    id="ticker"
    placeholder="Enter ticker"
/>

<button onclick="analyze()">
Analyze
</button>

<div class="cards">

    <div class="card">
        <div class="label">Ticker</div>
        <div class="value" id="symbol">---</div>
    </div>

    <div class="card">
        <div class="label">Gradient Score</div>
        <div class="value" id="score">0</div>
    </div>

    <div class="card">
        <div class="label">Signal</div>
        <div class="value" id="signal">---</div>
    </div>

</div>

<img id="chart"/>

<h2 style="margin-top:40px;">
Live Scanner
</h2>

<table id="scanner"></table>

</div>

<script>

async function analyze(){

    const ticker =
        document
        .getElementById("ticker")
        .value;

    const res =
        await fetch(
            "/analyze?ticker=" + ticker
        );

    const data = await res.json();

    document.getElementById("symbol").innerText =
        data.ticker;

    document.getElementById("score").innerText =
        data.gradient_score;

    document.getElementById("signal").innerText =
        data.signal;

    const p =
        await fetch(
            "/plot?ticker=" + ticker
        );

    const plot =
        await p.json();

    document.getElementById("chart").src =
        "data:image/png;base64," +
        plot.image;
}

async function loadScanner(){

    const res =
        await fetch("/scan");

    const data =
        await res.json();

    let html = `
    <tr>
        <th>Ticker</th>
        <th>Score</th>
        <th>Signal</th>
    </tr>
    `;

    data.data.forEach(r => {

        html += `
        <tr>
            <td>${r.ticker}</td>
            <td>${r.score}</td>
            <td>${r.signal}</td>
        </tr>
        `;
    });

    document.getElementById("scanner").innerHTML =
        html;
}

loadScanner();

setInterval(
    loadScanner,
    15000
);

</script>

</body>
</html>
"""

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "routes": [
            "/dashboard",
            "/analyze",
            "/plot",
            "/scan"
        ]
    }
