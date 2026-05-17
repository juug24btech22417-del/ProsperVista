# ProsperVista v3.0 — Market Screener Engine
# Multi-condition stock scanner with pre-built screens

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STOCK UNIVERSES
# ==========================================

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "ONGC.NS", "JSWSTEEL.NS",
    "TECHM.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "DRREDDY.NS",
    "BAJAJFINSV.NS", "CIPLA.NS", "EICHERMOT.NS", "TATACONSUM.NS", "HEROMOTOCO.NS",
    "COALINDIA.NS", "BPCL.NS", "GRASIM.NS", "BRITANNIA.NS", "SBILIFE.NS",
    "INDUSINDBK.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "TATAPOWER.NS", "ZOMATO.NS"
]

US_LARGECAP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "KO", "PEP", "AVGO", "COST", "TMO", "MCD", "WMT",
    "CSCO", "ACN", "ABT", "DHR", "NEE", "NKE", "AMD", "ORCL"
]

UNIVERSES = {
    "Nifty 50 (India)": NIFTY_50,
    "US Large Cap": US_LARGECAP
}

# ==========================================
# STOCK DATA FETCHER (with caching in session)
# ==========================================

def fetch_stock_data(ticker, period="3mo"):
    """Fetch OHLCV data for a single stock."""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 20:
            return None
        return df
    except Exception:
        return None

def compute_indicators(df):
    """Compute technical indicators for screening."""
    df = df.copy()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean() * 100
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Price change percentages
    df['Change_1D'] = df['Close'].pct_change(1) * 100
    df['Change_5D'] = df['Close'].pct_change(5) * 100
    df['Change_20D'] = df['Close'].pct_change(20) * 100
    
    # 52-week high/low proximity
    if len(df) >= 252:
        df['High_52W'] = df['High'].rolling(252).max()
        df['Low_52W'] = df['Low'].rolling(252).min()
    else:
        df['High_52W'] = df['High'].max()
        df['Low_52W'] = df['Low'].min()
    
    df['Pct_From_52W_High'] = (df['Close'] - df['High_52W']) / df['High_52W'] * 100
    df['Pct_From_52W_Low'] = (df['Close'] - df['Low_52W']) / df['Low_52W'] * 100
    
    return df

# ==========================================
# FILTER CONDITIONS
# ==========================================

def apply_filters(df, filters):
    """
    Apply filter conditions to the latest data point.
    
    filters: list of dicts with:
        - indicator: column name
        - operator: '>', '<', '>=', '<=', '=='
        - value: threshold value
    
    Returns True if all conditions met.
    """
    if df is None or df.empty:
        return False
    
    latest = df.iloc[-1]
    
    for f in filters:
        indicator = f.get("indicator")
        operator = f.get("operator", ">")
        value = f.get("value", 0)
        
        if indicator not in latest.index:
            return False
        
        actual = latest[indicator]
        if pd.isna(actual):
            return False
        
        if operator == ">" and not (actual > value):
            return False
        elif operator == "<" and not (actual < value):
            return False
        elif operator == ">=" and not (actual >= value):
            return False
        elif operator == "<=" and not (actual <= value):
            return False
        elif operator == "==" and not (actual == value):
            return False
    
    return True

# ==========================================
# PRE-BUILT SCREENS
# ==========================================

PREBUILT_SCREENS = {
    "Oversold Bounce": {
        "description": "RSI < 35, Volume > 1.1x average — potential bounce candidates",
        "filters": [
            {"indicator": "RSI", "operator": "<", "value": 35},
            {"indicator": "Volume_Ratio", "operator": ">", "value": 1.1}
        ]
    },
    "Momentum Breakout": {
        "description": "Price > MA50, RSI 45-70, Volume > 1.2x — strong uptrend breakouts",
        "filters": [
            {"indicator": "RSI", "operator": ">", "value": 45},
            {"indicator": "RSI", "operator": "<", "value": 70},
            {"indicator": "Volume_Ratio", "operator": ">", "value": 1.2}
        ]
    },
    "Value Play": {
        "description": "Near 52-week low, RSI < 45 — potential value entry",
        "filters": [
            {"indicator": "Pct_From_52W_Low", "operator": "<", "value": 15},
            {"indicator": "RSI", "operator": "<", "value": 45}
        ]
    },
    "High Volatility": {
        "description": "Volatility > 2%, Volume > 1.2x — active intraday movers",
        "filters": [
            {"indicator": "Volatility", "operator": ">", "value": 2.0},
            {"indicator": "Volume_Ratio", "operator": ">", "value": 1.2}
        ]
    },
    "Golden Cross": {
        "description": "MA50 just crossed above MA200 — classic bullish signal",
        "filters": [
            {"indicator": "Change_20D", "operator": ">", "value": 0}
        ]
    },
    "Overbought Alert": {
        "description": "RSI > 65, Price near 52W high — potential short/exit candidates",
        "filters": [
            {"indicator": "RSI", "operator": ">", "value": 65},
            {"indicator": "Pct_From_52W_High", "operator": ">", "value": -10}
        ]
    }
}

# ==========================================
# MAIN SCREENER
# ==========================================

def run_screen(screen_name=None, custom_filters=None, universe="Nifty 50 (India)", progress_callback=None):
    """
    Run a market screen and return matching stocks.
    
    Args:
        screen_name: Name of pre-built screen (from PREBUILT_SCREENS)
        custom_filters: List of custom filter dicts (overrides screen_name)
        universe: "Nifty 50 (India)" or "US Large Cap"
        progress_callback: Optional callback(current, total) for progress tracking
    
    Returns:
        list of matching stock dicts with indicator values
    """
    # Determine filters
    if custom_filters:
        filters = custom_filters
    elif screen_name and screen_name in PREBUILT_SCREENS:
        filters = PREBUILT_SCREENS[screen_name]["filters"]
    else:
        filters = []
    
    # Get universe
    tickers = UNIVERSES.get(universe, NIFTY_50)
    results = []
    
    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, len(tickers))
        
        df = fetch_stock_data(ticker, period="1y")
        if df is None:
            continue
        
        df = compute_indicators(df)
        
        # Apply golden cross special check
        if screen_name == "Golden Cross":
            latest = df.iloc[-1]
            prev_5 = df.iloc[-5] if len(df) > 5 else df.iloc[0]
            if pd.notna(latest.get('MA50')) and pd.notna(latest.get('MA200')):
                if not (latest['MA50'] > latest['MA200'] and 
                        (pd.isna(prev_5.get('MA50')) or prev_5.get('MA50', 0) <= prev_5.get('MA200', float('inf')))):
                    continue
            else:
                continue
        
        if apply_filters(df, filters):
            latest = df.iloc[-1]
            results.append({
                "ticker": ticker,
                "price": round(float(latest['Close']), 2),
                "change_1d": round(float(latest.get('Change_1D', 0)), 2),
                "change_5d": round(float(latest.get('Change_5D', 0)), 2),
                "change_20d": round(float(latest.get('Change_20D', 0)), 2),
                "rsi": round(float(latest.get('RSI', 0)), 1),
                "volume_ratio": round(float(latest.get('Volume_Ratio', 0)), 2),
                "volatility": round(float(latest.get('Volatility', 0)), 2),
                "from_52w_high": round(float(latest.get('Pct_From_52W_High', 0)), 1),
                "from_52w_low": round(float(latest.get('Pct_From_52W_Low', 0)), 1),
                "ma50": round(float(latest.get('MA50', 0)), 2),
                "ma200": round(float(latest.get('MA200', 0)), 2)
            })
    
    # Sort by 1D change descending
    results.sort(key=lambda x: x.get('change_1d', 0), reverse=True)
    return results

def get_screen_descriptions():
    """Get descriptions of all pre-built screens."""
    return {name: info["description"] for name, info in PREBUILT_SCREENS.items()}
