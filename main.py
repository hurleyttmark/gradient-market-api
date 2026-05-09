from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
EMA_SPAN_LOWER_PLOT = 10
EMA_SPAN_UPPER      = 100
EMA_SPAN_LOWER      = 100
GRADIENT_WINDOW     = 20
PRICE_SLOPE_WINDOW  = 10

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def classify_candle(row):
    if row['range'] == 0:
        return 0
    neutral = (row['body_pct'] <= 0.20 and
               abs(row['upper_wick'] - row['lower_wick']) <= 0.20 * row['range'])
    if neutral:
        return 0
    body = row['Close'] - row['Open']
    body_abs = abs(body)
    upper = row['upper_wick']
    lower = row['lower_wick']
    if body > 0:
        if body_abs / row['range'] >= 0.65: return 1
        if lower >= 1.3 * body_abs:         return 1
        if body_abs / row['range'] >= 0.25: return 1
        if upper > 1.5 * body_abs:          return 0
    if body < 0:
        if body_abs / row['range'] >= 0.65: return -1
        if upper >= 1.3 * body_abs:         return -1
        if body_abs / row['range'] >= 0.25: return -1
        if lower > 1.5 * body_abs:          return 0
    return 0


def iterative_ema(series, length):
    alpha = 2 / (length + 1)
    ema = np.zeros(len(series))
    ema[length - 1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        ema[i] = ema[i - 1] + alpha * (series.iloc[i] - ema[i - 1])
    ema[:length - 1] = ema[length - 1]
    return ema


def trend_code(val):
    if val >= 2:  return 2
    if val >= 1:  return 1
    if val <= -2: return -2
    if val <= -1: return -1
    return 0


def compute(symbol: str):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)

    df = yf.download(symbol, start=start_date, end=end_date,
                     auto_adjust=True, progress=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.columns = df.columns.get_level_values(0)

    # ── Candle metrics ──────────────────────────────────────────────────────
    df['range']      = df['High'] - df['Low']
    df['body']       = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body_pct']   = df['body'] / df['range'].replace(0, np.nan)
    df['signal']     = df.apply(classify_candle, axis=1)

    # ── Streak ──────────────────────────────────────────────────────────────
    streak, count = [], 0
    for s in df['signal']:
        if s == 1:   count = count + 1 if count > 0 else 1
        elif s == -1: count = count - 1 if count < 0 else -1
        else:         count = 0
        streak.append(count)
    df['streak'] = streak

    # ── Engulfing (3-day) ────────────────────────────────────────────────────
    N = 3
    df['bullish_engulf'] = False
    df['bearish_engulf'] = False
    df['scenario_strength'] = 0

    for i in range(2 * N, len(df)):
        first  = df.iloc[i - 2 * N: i - N]
        second = df.iloc[i - N: i]
        f_open, f_close = first['Open'].iloc[0], first['Close'].iloc[-1]
        s_open, s_close = second['Open'].iloc[0], second['Close'].iloc[-1]
        f_range = first['High'].max() - first['Low'].min()
        s_range = second['High'].max() - second['Low'].min()
        f_body  = abs(f_close - f_open)
        s_body  = abs(s_close - s_open)
        vol_ok  = second['Volume'].sum() >= 1.25 * first['Volume'].mean() * N

        if (f_close < f_open and f_body >= 0.5 * f_range and
                s_close > s_open and s_body >= 0.7 * s_range and
                s_open <= f_open and s_close >= f_close and vol_ok):
            df.loc[df.index[i - N: i], 'bullish_engulf'] = True
            df.loc[df.index[i - N: i], 'scenario_strength'] = 3

        if (f_close > f_open and f_body >= 0.5 * f_range and
                s_close < s_open and s_body >= 0.7 * s_range and
                s_open >= f_open and s_close <= f_close and vol_ok):
            df.loc[df.index[i - N: i], 'bearish_engulf'] = True
            df.loc[df.index[i - N: i], 'scenario_strength'] = -3

    # ── 3×3 patterns ────────────────────────────────────────────────────────
    df['bullish_3x3'] = False
    df['bearish_3x3'] = False
    for i in range(6, len(df)):
        first  = df.iloc[i - 6: i - 3]
        second = df.iloc[i - 3: i]
        third  = df.iloc[i - 2: i + 1]
        f_open, f_close = first['Open'].iloc[0], first['Close'].iloc[-1]
        t_open, t_close = third['Open'].iloc[0], third['Close'].iloc[-1]
        f_body   = abs(f_close - f_open)
        f_range  = first['High'].max() - first['Low'].min()
        s_body   = abs(second['Close'].iloc[-1] - second['Open'].iloc[0])
        t_body   = abs(t_close - t_open)
        t_range  = third['High'].max() - third['Low'].min()
        if (f_close < f_open and f_body >= 0.7 * f_range and
                s_body <= 0.25 * f_body and t_close > t_open and t_body >= 0.7 * t_range):
            df.at[df.index[i - 1], 'bullish_3x3'] = True
            df.at[df.index[i - 1], 'scenario_strength'] = 3
        if (f_close > f_open and f_body >= 0.7 * f_range and
                s_body <= 0.25 * f_body and t_close < t_open and t_body >= 0.7 * t_range):
            df.at[df.index[i - 1], 'bearish_3x3'] = True
            df.at[df.index[i - 1], 'scenario_strength'] = -3

    # ── Volume-weighted trend ────────────────────────────────────────────────
    df['price_delta'] = df['Close'].diff().abs()
    df['trend_raw']   = df['streak'] * df['price_delta'] * df['Volume']
    df['trend_raw']   = df['trend_raw'].fillna(0)
    scale             = df['trend_raw'].abs().rolling(200).max()
    df['trend_norm']  = (df['trend_raw'] / scale.replace(0, np.nan)).fillna(0)

    df['trend_ema20']        = iterative_ema(df['trend_norm'], EMA_SPAN_LOWER_PLOT)
    df['trend_ema20_smooth'] = iterative_ema(df['trend_ema20'], EMA_SPAN_LOWER_PLOT)
    df['upper_ema']          = iterative_ema(df['trend_norm'].clip(lower=0), EMA_SPAN_UPPER)
    df['lower_ema']          = iterative_ema(df['trend_norm'].clip(upper=0), EMA_SPAN_LOWER)

    price_slope = df['Close'].diff(PRICE_SLOPE_WINDOW)
    gated       = np.where(price_slope > 0,
                           df['streak'].clip(lower=0),
                           df['streak'].clip(upper=0))
    df['trend_code'] = pd.Series(gated, index=df.index).apply(trend_code)

    # ── Gradient score ───────────────────────────────────────────────────────
    gradient_score = np.zeros(len(df))
    for i in range(len(df)):
        start  = max(0, i - GRADIENT_WINDOW + 1)
        window = df['trend_code'].iloc[start: i + 1]
        score  = np.clip((window > 0).sum() - (window < 0).sum(), -5, 5)
        ss     = df['scenario_strength'].iloc[i]
        if ss != 0:
            score = np.clip(score + np.sign(ss), -5, 5)
        gradient_score[i] = score
    df['gradient_score'] = gradient_score

    # ── Return last N rows for chart (180 trading days ≈ 9 months) ──────────
    tail = df.tail(180)
    dates      = [d.strftime('%Y-%m-%d') for d in tail.index]
    ema_smooth = tail['trend_ema20_smooth'].tolist()
    upper_ema  = tail['upper_ema'].tolist()
    lower_ema  = tail['lower_ema'].tolist()
    grad_arr   = tail['gradient_score'].tolist()
    close_arr  = tail['Close'].tolist()

    latest_score = float(round(df['gradient_score'].iloc[-1], 2))

    if latest_score >= 3:   signal = "Very Bullish"
    elif latest_score >= 1: signal = "Bullish"
    elif latest_score <= -3: signal = "Very Bearish"
    elif latest_score <= -1: signal = "Bearish"
    else:                    signal = "Neutral"

    return {
        "ticker":         symbol.upper(),
        "gradient_score": latest_score,
        "signal":         signal,
        "chart": {
            "dates":      dates,
            "close":      [round(v, 2) for v in close_arr],
            "ema_smooth": [round(v, 4) for v in ema_smooth],
            "upper_ema":  [round(v, 4) for v in upper_ema],
            "lower_ema":  [round(v, 4) for v in lower_ema],
            "gradient":   [round(v, 2) for v in grad_arr],
        }
    }


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get("/analyze")
def analyze(ticker: str):
    try:
        return compute(ticker.upper())
    except Exception as e:
        return {"error": str(e)}


@app.get("/watchlist")
def watchlist():
    symbols = ["NVDA", "AAPL", "GOOG", "SPY", "AMZN"]
    results = []
    for sym in symbols:
        try:
            d = compute(sym)
            results.append({
                "ticker":         d["ticker"],
                "gradient_score": d["gradient_score"],
                "signal":         d["signal"],
            })
        except Exception as e:
            results.append({"ticker": sym, "gradient_score": 0, "signal": "Error", "error": str(e)})
    return results
