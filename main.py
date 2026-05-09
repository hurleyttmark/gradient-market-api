import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -----------------------------
# CONFIG
# -----------------------------
symbol = "gddy"
start_date = datetime(2021,1,13)
end_date   = datetime(2023,1,17)

gradient_window = 20

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
df = yf.download(symbol, start=start_date, end=end_date,
                 auto_adjust=True, progress=False)

df = df[['Open','High','Low','Close','Volume']].dropna()

# -----------------------------
# CANDLE METRICS
# -----------------------------
df['range'] = df['High'] - df['Low']
df['body'] = abs(df['Close'] - df['Open'])
df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
df['body_pct'] = df['body'] / df['range']

# -----------------------------
# CANDLE CLASSIFICATION
# -----------------------------
def classify_candle(row):
    if row['range'] == 0:
        return 0

    neutral = (
        row['body_pct'] <= 0.20 and
        abs(row['upper_wick'] - row['lower_wick']) <= 0.20 * row['range']
    )
    if neutral:
        return 0

    body = row['Close'] - row['Open']
    body_abs = abs(body)
    upper = row['upper_wick']
    lower = row['lower_wick']

    if body > 0:
        if body_abs / row['range'] >= 0.65: return 1
        if lower >= 1.3 * body_abs: return 1
        if body_abs / row['range'] >= 0.25: return 1
        if upper > 1.5 * body_abs: return 0

    if body < 0:
        if body_abs / row['range'] >= 0.65: return -1
        if upper >= 1.3 * body_abs: return -1
        if body_abs / row['range'] >= 0.25: return -1
        if lower > 1.5 * body_abs: return 0

    return 0

df['signal'] = df.apply(classify_candle, axis=1)

# -----------------------------
# STREAK (FIXED LOGIC)
# -----------------------------
price_change = df['Close'].diff()

streak = []
s = 0

for d in price_change:
    if d > 0:
        s = s + 1 if s > 0 else 1
    elif d < 0:
        s = s - 1 if s < 0 else -1
    else:
        s = 0
    streak.append(s)

df['streak'] = streak

# -----------------------------
# 3x3 PATTERNS
# -----------------------------
df['bullish_3x3'] = False
df['bearish_3x3'] = False

for i in range(6, len(df)):
    first = df.iloc[i-6:i-3]
    second = df.iloc[i-3:i]
    third = df.iloc[i-2:i+1]

    f_open, f_close = first['Open'].iloc[0], first['Close'].iloc[-1]
    f_body = abs(f_close - f_open)
    f_range = first['High'].max() - first['Low'].min()

    s_body = abs(second['Close'].iloc[-1] - second['Open'].iloc[0])

    t_open, t_close = third['Open'].iloc[0], third['Close'].iloc[-1]
    t_body = abs(t_close - t_open)
    t_range = third['High'].max() - third['Low'].min()

    if (
        f_close < f_open and
        f_body >= 0.7 * f_range and
        s_body <= 0.25 * f_body and
        t_close > t_open and
        t_body >= 0.7 * t_range
    ):
        df.at[df.index[i-1], 'bullish_3x3'] = True

    if (
        f_close > f_open and
        f_body >= 0.7 * f_range and
        s_body <= 0.25 * f_body and
        t_close < t_open and
        t_body >= 0.7 * t_range
    ):
        df.at[df.index[i-1], 'bearish_3x3'] = True

# -----------------------------
# ENGULFING (SIMPLIFIED BUT CORRECT)
# -----------------------------
df['bullish_engulf'] = False
df['bearish_engulf'] = False

N = 3

for i in range(2*N, len(df)):
    first = df.iloc[i-2*N:i-N]
    second = df.iloc[i-N:i]

    f_open, f_close = first['Open'].iloc[0], first['Close'].iloc[-1]
    s_open, s_close = second['Open'].iloc[0], second['Close'].iloc[-1]

    f_range = first['High'].max() - first['Low'].min()
    s_range = second['High'].max() - second['Low'].min()

    f_body = abs(f_close - f_open)
    s_body = abs(s_close - s_open)

    # Bullish
    if (
        f_close < f_open and
        s_close > s_open and
        s_body > 0.5 * s_range and
        s_open <= f_open and
        s_close >= f_close
    ):
        df.loc[df.index[i-N:i], 'bullish_engulf'] = True

    # Bearish
    if (
        f_close > f_open and
        s_close < s_open and
        s_body > 0.5 * s_range and
        s_open >= f_open and
        s_close <= f_close
    ):
        df.loc[df.index[i-N:i], 'bearish_engulf'] = True

# -----------------------------
# STRUCTURE SIGNAL (LIGHT WEIGHT)
# -----------------------------
df['structure_signal'] = 0
df.loc[df['bullish_3x3'], 'structure_signal'] = 1
df.loc[df['bearish_3x3'], 'structure_signal'] = -1
df.loc[df['bullish_engulf'], 'structure_signal'] = 2
df.loc[df['bearish_engulf'], 'structure_signal'] = -2

# -----------------------------
# NORMALIZED MOMENTUM
# -----------------------------
volatility = df['Close'].rolling(10).std()
volatility = volatility.replace(0, np.nan)

momentum = df['Close'].diff(3)

df['momentum_norm'] = (momentum / volatility).fillna(0)

# -----------------------------
# GRADIENT SCORE (FINAL CLEAN VERSION)
# -----------------------------
gradient_score = []

for i in range(len(df)):
    start = max(0, i - gradient_window + 1)

    mom_window = df['momentum_norm'].iloc[start:i+1]
    struct_window = df['structure_signal'].iloc[start:i+1]

    momentum_score = np.tanh(mom_window.mean()) * 5
    structure_score = struct_window.sum() * 0.5

    score = momentum_score + structure_score
    score = np.clip(score, -5, 5)

    gradient_score.append(score)

df['gradient'] = gradient_score

# -----------------------------
# VISUALIZATION (UNCHANGED STYLE)
# -----------------------------
dates = df.index

fig, ax = plt.subplots(figsize=(16,8))

cmap = LinearSegmentedColormap.from_list(
    "trend",
    ['#5b0000','#b30000','#ff6600','#ffd700','#bfff00','#66cc66','#006400']
)
norm = Normalize(vmin=-5, vmax=5)

for i in range(len(df)-1):
    color = cmap(norm(df['gradient'].iloc[i]))
    ax.axvspan(dates[i], dates[i+1], color=color, alpha=0.6)

ax.plot(dates, df['Close'], color='black', linewidth=2)

ax.set_title("Gradient + Structure Momentum Indicator (Revised)")
ax.grid(alpha=0.3)

plt.show()
