# ProsperVista v3.0 — Technical Pattern Recognition Engine
# Automated candlestick & chart pattern detection with Fibonacci levels

import numpy as np
import pandas as pd
import ta

# ==========================================
# CANDLESTICK PATTERN DETECTION
# ==========================================

def detect_candlestick_patterns(df):
    """
    Detect major candlestick patterns on the last N candles.
    Returns list of detected patterns with their signal direction.
    """
    if df is None or len(df) < 5:
        return []
    
    df = df.copy().tail(50)
    patterns = []
    o, h, l, c = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    body = np.abs(c - o)
    avg_body = pd.Series(body).rolling(14).mean().values
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    
    for i in range(2, len(df)):
        date = df.index[i]
        
        # DOJI: Body < 10% of range
        total_range = h[i] - l[i]
        if total_range > 0 and body[i] / total_range < 0.1:
            patterns.append({"date": date, "pattern": "Doji", "signal": "NEUTRAL",
                "description": "Indecision candle — trend reversal possible", "strength": 2})
        
        # HAMMER: Small body at top, long lower wick (2x body)
        if body[i] > 0 and lower_wick[i] > 2 * body[i] and upper_wick[i] < body[i] * 0.5:
            if c[i-1] < o[i-1]:  # Prior downtrend
                patterns.append({"date": date, "pattern": "Hammer", "signal": "BULLISH",
                    "description": "Bullish reversal — buyers absorbed selling pressure", "strength": 3})
        
        # INVERTED HAMMER
        if body[i] > 0 and upper_wick[i] > 2 * body[i] and lower_wick[i] < body[i] * 0.5:
            if c[i-1] < o[i-1]:
                patterns.append({"date": date, "pattern": "Inverted Hammer", "signal": "BULLISH",
                    "description": "Potential bullish reversal after downtrend", "strength": 2})
        
        # BULLISH ENGULFING
        if i >= 1 and c[i-1] < o[i-1] and c[i] > o[i]:
            if o[i] <= c[i-1] and c[i] >= o[i-1]:
                patterns.append({"date": date, "pattern": "Bullish Engulfing", "signal": "BULLISH",
                    "description": "Strong reversal — buyers overwhelmed sellers", "strength": 4})
        
        # BEARISH ENGULFING
        if i >= 1 and c[i-1] > o[i-1] and c[i] < o[i]:
            if o[i] >= c[i-1] and c[i] <= o[i-1]:
                patterns.append({"date": date, "pattern": "Bearish Engulfing", "signal": "BEARISH",
                    "description": "Strong reversal — sellers overwhelmed buyers", "strength": 4})
        
        # MORNING STAR (3-candle bullish reversal)
        if i >= 2:
            if c[i-2] < o[i-2] and body[i-1] < avg_body[i-1] * 0.5 and c[i] > o[i]:
                if c[i] > (o[i-2] + c[i-2]) / 2:
                    patterns.append({"date": date, "pattern": "Morning Star", "signal": "BULLISH",
                        "description": "3-candle bullish reversal — high reliability", "strength": 5})
        
        # EVENING STAR (3-candle bearish reversal)
        if i >= 2:
            if c[i-2] > o[i-2] and body[i-1] < avg_body[i-1] * 0.5 and c[i] < o[i]:
                if c[i] < (o[i-2] + c[i-2]) / 2:
                    patterns.append({"date": date, "pattern": "Evening Star", "signal": "BEARISH",
                        "description": "3-candle bearish reversal — high reliability", "strength": 5})
        
        # THREE WHITE SOLDIERS
        if i >= 2:
            all_green = c[i] > o[i] and c[i-1] > o[i-1] and c[i-2] > o[i-2]
            ascending = c[i] > c[i-1] > c[i-2]
            small_wicks = upper_wick[i] < body[i] * 0.3
            if all_green and ascending and small_wicks:
                patterns.append({"date": date, "pattern": "Three White Soldiers", "signal": "BULLISH",
                    "description": "Strong bullish continuation — institutional buying", "strength": 5})
        
        # THREE BLACK CROWS
        if i >= 2:
            all_red = c[i] < o[i] and c[i-1] < o[i-1] and c[i-2] < o[i-2]
            descending = c[i] < c[i-1] < c[i-2]
            small_lower = lower_wick[i] < body[i] * 0.3
            if all_red and descending and small_lower:
                patterns.append({"date": date, "pattern": "Three Black Crows", "signal": "BEARISH",
                    "description": "Strong bearish continuation — institutional selling", "strength": 5})
    
    # Return most recent patterns (max 10)
    return sorted(patterns, key=lambda x: x['date'], reverse=True)[:10]

# ==========================================
# SUPPORT & RESISTANCE DETECTION
# ==========================================

def detect_support_resistance(df, window=20, num_levels=5):
    """
    Detect key support and resistance levels using pivot points.
    Returns dict with support and resistance arrays.
    """
    if df is None or len(df) < window * 2:
        return {"support": [], "resistance": []}
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(df) - window):
        # Resistance: Local maximum
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance_levels.append(round(float(highs[i]), 2))
        # Support: Local minimum
        if lows[i] == min(lows[i-window:i+window+1]):
            support_levels.append(round(float(lows[i]), 2))
    
    # Cluster nearby levels (within 1% of each other)
    resistance_levels = _cluster_levels(resistance_levels)[:num_levels]
    support_levels = _cluster_levels(support_levels)[:num_levels]
    
    current_price = float(closes[-1])
    # Filter: resistance above price, support below
    resistance_levels = sorted([r for r in resistance_levels if r > current_price])
    support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)
    
    return {"support": support_levels[:num_levels], "resistance": resistance_levels[:num_levels]}

def _cluster_levels(levels, tolerance=0.01):
    """Cluster price levels that are within tolerance of each other."""
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters = [[sorted_levels[0]]]
    for level in sorted_levels[1:]:
        if abs(level - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [round(np.mean(c), 2) for c in clusters]

# ==========================================
# FIBONACCI RETRACEMENT
# ==========================================

FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

def calculate_fibonacci(df, lookback=90):
    """
    Calculate Fibonacci retracement levels from recent swing high/low.
    """
    if df is None or len(df) < lookback:
        return []
    
    recent = df.tail(lookback)
    swing_high = float(recent['High'].max())
    swing_low = float(recent['Low'].min())
    diff = swing_high - swing_low
    
    levels = []
    for ratio in FIB_RATIOS:
        price = swing_high - (diff * ratio)
        levels.append({
            "ratio": f"{ratio*100:.1f}%",
            "price": round(price, 2),
            "type": "SUPPORT" if ratio > 0.5 else "RESISTANCE"
        })
    
    return levels

# ==========================================
# BOLLINGER BAND SQUEEZE DETECTION
# ==========================================

def detect_bb_squeeze(df, window=20, num_std=2):
    """
    Detect Bollinger Band squeeze (low volatility = breakout imminent).
    Returns squeeze status and band data.
    """
    if df is None or len(df) < window + 10:
        return {"in_squeeze": False, "squeeze_duration": 0, "bands": None}
    
    df = df.copy()
    df['BB_Mid'] = df['Close'].rolling(window).mean()
    df['BB_Upper'] = df['BB_Mid'] + num_std * df['Close'].rolling(window).std()
    df['BB_Lower'] = df['BB_Mid'] - num_std * df['Close'].rolling(window).std()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    
    # Squeeze = width below 20th percentile of its own history
    width_percentile = df['BB_Width'].rank(pct=True).iloc[-1]
    in_squeeze = width_percentile < 0.2
    
    # How long has squeeze lasted?
    squeeze_days = 0
    if in_squeeze:
        threshold = df['BB_Width'].quantile(0.2)
        for i in range(len(df) - 1, -1, -1):
            if df['BB_Width'].iloc[i] < threshold:
                squeeze_days += 1
            else:
                break
    
    return {
        "in_squeeze": in_squeeze,
        "squeeze_duration": squeeze_days,
        "band_width": round(float(df['BB_Width'].iloc[-1]) * 100, 2),
        "width_percentile": round(width_percentile * 100, 1),
        "upper": round(float(df['BB_Upper'].iloc[-1]), 2),
        "mid": round(float(df['BB_Mid'].iloc[-1]), 2),
        "lower": round(float(df['BB_Lower'].iloc[-1]), 2)
    }

# ==========================================
# VOLUME PROFILE ANALYSIS
# ==========================================

def volume_profile(df, num_bins=20):
    """
    Calculate Volume Profile (price-volume distribution).
    Returns price levels with highest trading activity.
    """
    if df is None or len(df) < 20:
        return []
    
    prices = df['Close'].values
    volumes = df['Volume'].values
    
    price_range = np.linspace(prices.min(), prices.max(), num_bins + 1)
    profile = []
    
    for i in range(len(price_range) - 1):
        mask = (prices >= price_range[i]) & (prices < price_range[i + 1])
        vol = volumes[mask].sum()
        profile.append({
            "price_low": round(float(price_range[i]), 2),
            "price_high": round(float(price_range[i + 1]), 2),
            "price_mid": round(float((price_range[i] + price_range[i + 1]) / 2), 2),
            "volume": int(vol)
        })
    
    # Point of Control (POC): Price level with most volume
    poc = max(profile, key=lambda x: x['volume'])
    
    return {"profile": profile, "poc": poc["price_mid"], "poc_volume": poc["volume"]}

# ==========================================
# COMPREHENSIVE PATTERN REPORT
# ==========================================

def generate_pattern_report(df):
    """Generate a full technical pattern analysis report."""
    candle_patterns = detect_candlestick_patterns(df)
    sr_levels = detect_support_resistance(df)
    fib_levels = calculate_fibonacci(df)
    bb_squeeze = detect_bb_squeeze(df)
    vol_prof = volume_profile(df)
    
    # Overall Technical Signal
    bullish_count = sum(1 for p in candle_patterns if p['signal'] == 'BULLISH')
    bearish_count = sum(1 for p in candle_patterns if p['signal'] == 'BEARISH')
    
    if bullish_count > bearish_count + 1:
        overall = "BULLISH"
    elif bearish_count > bullish_count + 1:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"
    
    return {
        "patterns": candle_patterns,
        "support_resistance": sr_levels,
        "fibonacci": fib_levels,
        "bb_squeeze": bb_squeeze,
        "volume_profile": vol_prof,
        "overall_signal": overall,
        "bullish_patterns": bullish_count,
        "bearish_patterns": bearish_count
    }
