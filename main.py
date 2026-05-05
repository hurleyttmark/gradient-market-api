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

ema_span_lower_plot = 10
ema_span_upper = 100
ema_span_lower = 100
gradient_window = 20
price_slope_window = 10

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
df = yf.download(symbol, start=start_date, end=end_date,
                 auto_adjust=True, progress=False)
df = df[['Open','High','Low','Close','Volume']].dropna()
df.columns = df.columns.get_level_values(0)

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
    if row['range'] == 0: return 0
    neutral = (row['body_pct'] <= 0.20 and abs(row['upper_wick'] - row['lower_wick']) <= 0.20 * row['range'])
    if neutral: return 0

    body = row['Close'] - row['Open']
    body_abs = abs(body)
    upper = row['upper_wick']
    lower = row['lower_wick']

    # BULLISH
    if body > 0:
        if body_abs / row['range'] >= 0.65: return 1
        if lower >= 1.3*body_abs: return 1
        if body_abs / row['range'] >= 0.25: return 1
        if upper > 1.5*body_abs: return 0

    # BEARISH
    if body < 0:
        if body_abs / row['range'] >= 0.65: return -1
        if upper >= 1.3*body_abs: return -1
        if body_abs / row['range'] >= 0.25: return -1
        if lower > 1.5*body_abs: return 0

    return 0

df['signal'] = df.apply(classify_candle, axis=1)

# -----------------------------
# CONSECUTIVE STREAK
# -----------------------------
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

# -----------------------------
# 3-DAY ENGULFING DETECTION (NOW EACH SIGNAL COVERS 3 DAYS)
# -----------------------------
N = 3
df['bullish_engulf'] = False
df['bearish_engulf'] = False
df['engulf_start'] = None
df['engulf_end'] = None

for i in range(2*N, len(df)):
    first = df.iloc[i-2*N:i-N]
    second = df.iloc[i-N:i]

    first_open, first_close = first['Open'].iloc[0], first['Close'].iloc[-1]
    second_open, second_close = second['Open'].iloc[0], second['Close'].iloc[-1]
    first_range = first['High'].max() - first['Low'].min()
    second_range = second['High'].max() - second['Low'].min()
    first_body = abs(first_close - first_open)
    second_body = abs(second_close - second_open)
    avg_first_vol = first['Volume'].mean()
    second_vol = second['Volume'].sum()
    volume_ok = second_vol >= 1.25 * avg_first_vol * N

    # Bullish engulf covering 3-day period
    if first_close < first_open and first_body >= 0.5*first_range and \
       second_close > second_open and second_body >= 0.7*second_range and \
       second_open <= first_open and second_close >= first_close and volume_ok:
        df.loc[df.index[i-N:i], 'bullish_engulf'] = True
        df.loc[df.index[i-N], 'engulf_start'] = df.index[i-N]
        df.loc[df.index[i-1], 'engulf_end'] = df.index[i-1]

    # Bearish engulf covering 3-day period
    if first_close > first_open and first_body >= 0.5*first_range and \
       second_close < second_open and second_body >= 0.7*second_range and \
       second_open >= first_open and second_close <= first_close and volume_ok:
        df.loc[df.index[i-N:i], 'bearish_engulf'] = True
        df.loc[df.index[i-N], 'engulf_start'] = df.index[i-N]
        df.loc[df.index[i-1], 'engulf_end'] = df.index[i-1]

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

    if f_close < f_open and f_body >= 0.7*f_range and s_body <= 0.25*f_body and \
       t_close > t_open and t_body >= 0.7*t_range:
        df.at[df.index[i-1], 'bullish_3x3'] = True

    if f_close > f_open and f_body >= 0.7*f_range and s_body <= 0.25*f_body and \
       t_close < t_open and t_body >= 0.7*t_range:
        df.at[df.index[i-1], 'bearish_3x3'] = True

# -----------------------------
# SCENARIO STRENGTH
# -----------------------------
df['scenario_strength'] = 0
df.loc[df['bullish_engulf'], 'scenario_strength'] = 3
df.loc[df['bearish_engulf'], 'scenario_strength'] = -3
df.loc[df['bullish_3x3'], 'scenario_strength'] = 3
df.loc[df['bearish_3x3'], 'scenario_strength'] = -3

# -----------------------------
# VOLUME-WEIGHTED TREND
# -----------------------------
df['price_delta'] = df['Close'].diff().abs()
df['trend_raw'] = df['streak'] * df['price_delta'] * df['Volume']
df['trend_raw'] = df['trend_raw'].fillna(0)
scale = df['trend_raw'].abs().rolling(200).max()
df['trend_norm'] = df['trend_raw'] / scale.replace(0,np.nan)
df['trend_norm'] = df['trend_norm'].fillna(0)

# -----------------------------
# ITERATIVE EMA
# -----------------------------
def iterative_ema(series,length):
    alpha = 2/(length+1)
    ema = np.zeros(len(series))
    ema[length-1] = series.iloc[:length].mean()
    for i in range(length,len(series)):
        ema[i] = ema[i-1] + alpha*(series.iloc[i]-ema[i-1])
    ema[:length-1] = ema[length-1]
    return ema

df['trend_ema20'] = iterative_ema(df['trend_norm'], ema_span_lower_plot)
df['trend_ema20_smooth'] = iterative_ema(df['trend_ema20'], ema_span_lower_plot)
ema_vals = df['trend_ema20_smooth'].values
df['upper_ema'] = iterative_ema(df['trend_norm'].clip(lower=0), ema_span_upper)
df['lower_ema'] = iterative_ema(df['trend_norm'].clip(upper=0), ema_span_lower)

# -----------------------------
# DISCRETE TREND CODE
# -----------------------------
def trend_code(val):
    if val>=2: return 2
    if val>=1: return 1
    if val<=-2: return -2
    if val<=-1: return -1
    return 0

price_slope = df['Close'].diff(price_slope_window)
gated_streak = np.where(price_slope>0, df['streak'].clip(lower=0), df['streak'].clip(upper=0))
df['trend_code'] = pd.Series(gated_streak, index=df.index).apply(trend_code)

# -----------------------------
# GRADIENT
# -----------------------------
gradient_colors = ['#5b0000','#b30000','#ff6600','#ffd700','#bfff00','#66cc66','#006400']
cmap = LinearSegmentedColormap.from_list("trend_gradient", gradient_colors)
norm = Normalize(vmin=-5,vmax=5)

gradient_score = np.zeros(len(df))
gradient_rgb = np.zeros((len(df),3))
for i in range(len(df)):
    start = max(0,i-gradient_window+1)
    window = df['trend_code'].iloc[start:i+1]
    score = np.clip((window>0).sum() - (window<0).sum(), -5, 5)
    if df['scenario_strength'].iloc[i]!=0:
        score += np.sign(df['scenario_strength'].iloc[i])
        score = np.clip(score,-5,5)
    gradient_score[i] = score
    gradient_rgb[i] = cmap(norm(score))[:3]

# -----------------------------
# PLOT
# -----------------------------
dates = df.index
fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(16,14),
                                  gridspec_kw={'height_ratios':[2,0.25,1]},
                                  sharex=True)

trend_color_map = {3:'#0b3d91',2:'#2f6fd6',1:'#9bbcf2',
                   0:'#d9d9d9',-1:'#f4a6a6',-2:'#e06666',-3:'#8b0000'}

for i in range(len(dates)-1):
    tc = df['trend_code'].iloc[i]
    ss = df['scenario_strength'].iloc[i]
    close = df['Close'].iloc[i]
    open_ = df['Open'].iloc[i]

    # Default neutral
    color = '#d9d9d9'

    # Only allow positive shades if Close > Open
    if tc > 0 and close > open_:
        color = trend_color_map[tc]
    # Only allow negative shades if Close < Open
    elif tc < 0 and close < open_:
        color = trend_color_map[tc]

    # Scenario strength overrides, but still obey Close/Open
    if ss != 0:
        if ss > 0 and close > open_:
            color = trend_color_map[ss]
        elif ss < 0 and close < open_:
            color = trend_color_map[ss]

    ax1.axvspan(dates[i], dates[i+1], facecolor=color, alpha=0.65, zorder=0)

ax1.plot(dates, df['Close'], color='black', linewidth=3, zorder=5)
ax1.grid(alpha=0.3)

# -----------------------------
# ARROWS FOR 3x3 AND ENGULFING (3-day period)
# -----------------------------
price_range = df['High'].max() - df['Low'].min()
arrow_length = price_range*0.10*2

for i in range(len(df)):
    x = dates[i]

    # Bullish 3x3 (dark green)
    if df['bullish_3x3'].iloc[i]:
        ax1.annotate('', xy=(x, df['Low'].iloc[i]*0.995),
                     xytext=(x, df['Low'].iloc[i]*0.995 - arrow_length),
                     arrowprops=dict(facecolor='darkgreen', edgecolor='darkgreen', arrowstyle='simple', lw=2.2), zorder=6)
    # Bullish engulf (light green, 3-day span)
    if df['bullish_engulf'].iloc[i] and df['engulf_start'].iloc[i] is not None:
        start_idx = df.index.get_loc(df['engulf_start'].iloc[i])
        end_idx = df.index.get_loc(df.index[i])
        ax1.annotate('', xy=(dates[end_idx], df['Low'].iloc[end_idx]*0.995),
                     xytext=(dates[start_idx], df['Low'].iloc[start_idx]*0.995 - arrow_length),
                     arrowprops=dict(facecolor='limegreen', edgecolor='limegreen', arrowstyle='simple', lw=2.2), zorder=6)
    # Bearish 3x3 (dark red)
    if df['bearish_3x3'].iloc[i]:
        ax1.annotate('', xy=(x, df['High'].iloc[i]*1.005),
                     xytext=(x, df['High'].iloc[i]*1.005 + arrow_length),
                     arrowprops=dict(facecolor='darkred', edgecolor='darkred', arrowstyle='simple', lw=2.2), zorder=6)
    # Bearish engulf (light red, 3-day span)
    if df['bearish_engulf'].iloc[i] and df['engulf_start'].iloc[i] is not None:
        start_idx = df.index.get_loc(df['engulf_start'].iloc[i])
        end_idx = df.index.get_loc(df.index[i])
        ax1.annotate('', xy=(dates[end_idx], df['High'].iloc[end_idx]*1.005),
                     xytext=(dates[start_idx], df['High'].iloc[start_idx]*1.005 + arrow_length),
                     arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='simple', lw=2.2), zorder=6)

# -----------------------------
# GRADIENT SUBPLOT
# -----------------------------
ax2.imshow([gradient_rgb], aspect='auto', extent=[dates[0],dates[-1],0,1], origin='lower')
ax2.set_yticks([]); ax2.set_xticks([])

# -----------------------------
# EMA SUBPLOT
# -----------------------------
for i in range(1,len(df)):
    c = 'green' if ema_vals[i]>=0 else 'red'
    ax3.plot(dates[i-1:i+1], ema_vals[i-1:i+1], color=c, linewidth=2)

ax3.plot(dates, df['upper_ema'], color='darkblue', linewidth=2.5)
ax3.plot(dates, df['lower_ema'], color='darkred', linewidth=2.5)
ax3.axhline(0, color='black', linewidth=1.2)

plt.tight_layout()
plt.show()
