import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = 'AAPL'
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
END_DATE = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
# ==========================================

def fetch_data(ticker, start, end):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data found.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def fetch_intraday_data(ticker, interval="5m", period="5d"):
    print(f"Fetching intraday data ({interval}) for {ticker}...")
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError("No intraday data found.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def calculate_intraday_pivots(df):
    """Calculates Standard Pivot Points using the previous day's data."""
    df_copy = df.copy()
    df_copy['Date'] = df_copy.index.date
    dates = df_copy['Date'].unique()
    if len(dates) > 1:
        prev_day = df_copy[df_copy['Date'] == dates[-2]]
        h = prev_day['High'].max()
        l = prev_day['Low'].min()
        c = prev_day['Close'].iloc[-1]
    else:
        h = df_copy['High'].max()
        l = df_copy['Low'].min()
        c = df_copy['Close'].iloc[-1]
    
    pivot = (h + l + c) / 3
    r1 = (2 * pivot) - l
    r2 = pivot + (h - l)
    s1 = (2 * pivot) - h
    s2 = pivot - (h - l)
    return {"P": pivot, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

def detect_volatility_squeeze(df):
    """Detects if Bollinger Bands are inside Keltner Channels (Squeeze)."""
    if len(df) < 20: return False
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    bb_upper = sma + (2 * std)
    bb_lower = sma - (2 * std)
    
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20).mean()
    
    kc_upper = sma + (1.5 * atr)
    kc_lower = sma - (1.5 * atr)
    
    squeeze_on = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
    return squeeze_on

def calculate_alpha_divergence(ticker, df_intraday):
    """Checks if the stock is diverging from the broader index (NIFTY/S&P)."""
    index_ticker = "^NSEI" if ".NS" in ticker or ".BO" in ticker else "^GSPC"
    try:
        idx_df = yf.download(index_ticker, interval="5m", period="5d", progress=False)
        if idx_df.empty or len(df_intraday) < 12 or len(idx_df) < 12: return False, 0, 0
        if isinstance(idx_df.columns, pd.MultiIndex):
            idx_df.columns = idx_df.columns.droplevel(1)
        
        stock_ret = float((df_intraday['Close'].iloc[-1] - df_intraday['Close'].iloc[-12]) / df_intraday['Close'].iloc[-12] * 100)
        idx_ret = float((idx_df['Close'].iloc[-1] - idx_df['Close'].iloc[-12]) / idx_df['Close'].iloc[-12] * 100)
        
        if idx_ret < -0.2 and stock_ret > 0.2: # Significant Alpha
            return True, stock_ret, idx_ret
        return False, stock_ret, idx_ret
    except:
        return False, 0, 0

def prepare_intraday_features(df):
    df = df.copy()
    
    # VWAP Calculation
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    # VWAP needs to reset daily ideally, but for a simple rolling over 5 days, cumsum is acceptable for intraday trend
    # A robust intraday VWAP groups by date.
    df['Date'] = df.index.date
    df['VWAP'] = df.groupby('Date').apply(lambda x: (x['Close'] * x['Volume']).cumsum() / x['Volume'].cumsum()).reset_index(level=0, drop=True)
    df.drop('Date', axis=1, inplace=True)
    
    df['MA3'] = df['Close'].rolling(window=3).mean()
    df['MA9'] = df['Close'].rolling(window=9).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=7)
    df['ROC3'] = df['Close'].pct_change(3) * 100
    
    df['Close_Lag1'] = df['Close'].shift(1)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Vol_Lag1'] = df['Volume'].shift(1)
    
    df.dropna(inplace=True)
    
    # Predict absolute price of the next candle
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    
    feature_names = ['VWAP', 'MA3', 'MA9', 'RSI', 'ROC3', 'Close_Lag1', 'RSI_Lag1', 'Vol_Lag1']
    X = df[feature_names]
    y = df['Target']
    return X, y, feature_names, df.index

def detect_micro_whales(df, window=10):
    avg_vol = df['Volume'].rolling(window=window).mean()
    vol_spike = df['Volume'] > (avg_vol * 2.5) # 2.5x volume burst
    price_change = df['Close'] - df['Open']
    
    is_whale = vol_spike.iloc[-1]
    if not is_whale: return False, "STABLE"
    if price_change.iloc[-1] > 0: return True, "ACCUMULATION"
    return True, "DISTRIBUTION"

def get_intraday_signals(df):
    """Generates institutional intraday technical signal events from the last 20 periods."""
    signals = []
    if len(df) < 25: return signals
    
    df_copy = df.copy()
    
    # Calculate VWAP
    df_copy['Date'] = df_copy.index.date
    df_copy['VWAP'] = df_copy.groupby('Date').apply(lambda x: (x['Close'] * x['Volume']).cumsum() / x['Volume'].cumsum()).reset_index(level=0, drop=True)
    
    # Calculate Squeeze
    sma = df_copy['Close'].rolling(window=20).mean()
    std = df_copy['Close'].rolling(window=20).std()
    bb_upper = sma + (2 * std)
    bb_lower = sma - (2 * std)
    tr1 = df_copy['High'] - df_copy['Low']
    tr2 = (df_copy['High'] - df_copy['Close'].shift(1)).abs()
    tr3 = (df_copy['Low'] - df_copy['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20).mean()
    kc_upper = sma + (1.5 * atr)
    kc_lower = sma - (1.5 * atr)
    df_copy['Squeeze'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Micro-whales over the history
    avg_vol = df_copy['Volume'].rolling(window=10).mean()
    df_copy['Whale_Spike'] = df_copy['Volume'] > (avg_vol * 2.5)
    df_copy['Whale_Type'] = np.where(df_copy['Close'] > df_copy['Open'], "ACCUMULATION", "DISTRIBUTION")
    
    # Loop over the last 15 bars to generate chronological events
    recent = df_copy.tail(15)
    for idx, row in recent.iterrows():
        t_str = idx.strftime('%H:%M')
        # Check Whale activity
        if row['Whale_Spike']:
            w_type = row['Whale_Type']
            clr = "#00FF9D" if w_type == "ACCUMULATION" else "#FF4B4B"
            signals.append({
                "time": t_str,
                "type": "WHALE",
                "color": clr,
                "msg": f" Whale {w_type} detected on volume at ₹{row['Close']:.2f}"
            })
            
        # Check VWAP crossovers
        prev_idx = df_copy.index.get_loc(idx) - 1
        if prev_idx >= 0:
            prev_row = df_copy.iloc[prev_idx]
            # Price cross VWAP upward
            if prev_row['Close'] < prev_row['VWAP'] and row['Close'] > row['VWAP']:
                signals.append({
                    "time": t_str,
                    "type": "VWAP_CROSS",
                    "color": "#00FF9D",
                    "msg": f" VWAP CROSSOVER: Price crossed ABOVE VWAP (Bullish)"
                })
            # Price cross VWAP downward
            elif prev_row['Close'] > prev_row['VWAP'] and row['Close'] < row['VWAP']:
                signals.append({
                    "time": t_str,
                    "type": "VWAP_CROSS",
                    "color": "#FF4B4B",
                    "msg": f" VWAP BREAKDOWN: Price dropped BELOW VWAP (Bearish)"
                })
                
        # Check Volatility Squeeze changes
        if prev_idx >= 0:
            prev_row = df_copy.iloc[prev_idx]
            if not prev_row['Squeeze'] and row['Squeeze']:
                signals.append({
                    "time": t_str,
                    "type": "SQUEEZE",
                    "color": "#FF9900",
                    "msg": f" VOLATILITY SQUEEZE: Bollinger Bands squeezed inside Keltner Channels"
                })
            elif prev_row['Squeeze'] and not row['Squeeze']:
                # Squeeze Release (breakout)
                direction = "UPWARD" if row['Close'] > row['Open'] else "DOWNWARD"
                clr = "#00FF9D" if direction == "UPWARD" else "#FF4B4B"
                signals.append({
                    "time": t_str,
                    "type": "SQUEEZE_RELEASE",
                    "color": clr,
                    "msg": f" SQUEEZE RELEASE: High-volatility {direction} breakout underway!"
                })
                
    return signals

def get_radar_metrics(df):
    if len(df) < 25:
        return {
            'price': 0.0,
            'vwap_status': 'N/A',
            'vwap_color': '#8b949e',
            'squeeze_status': 'N/A',
            'squeeze_color': '#8b949e',
            'whale_status': 'Normal',
            'whale_color': '#8b949e',
            'change_pct': 0.0
        }
    
    df_copy = df.copy()
    
    # Calculate VWAP
    df_copy['Date'] = df_copy.index.date
    df_copy['VWAP'] = df_copy.groupby('Date').apply(lambda x: (x['Close'] * x['Volume']).cumsum() / x['Volume'].cumsum()).reset_index(level=0, drop=True)
    
    # Calculate Squeeze
    sma = df_copy['Close'].rolling(window=20).mean()
    std = df_copy['Close'].rolling(window=20).std()
    bb_upper = sma + (2 * std)
    bb_lower = sma - (2 * std)
    tr1 = df_copy['High'] - df_copy['Low']
    tr2 = (df_copy['High'] - df_copy['Close'].shift(1)).abs()
    tr3 = (df_copy['Low'] - df_copy['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20).mean()
    kc_upper = sma + (1.5 * atr)
    kc_lower = sma - (1.5 * atr)
    df_copy['Squeeze'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Calculate Whale Spike
    avg_vol = df_copy['Volume'].rolling(window=10).mean()
    df_copy['Whale_Spike'] = df_copy['Volume'] > (avg_vol * 2.5)
    df_copy['Whale_Type'] = np.where(df_copy['Close'] > df_copy['Open'], "ACCUMULATION", "DISTRIBUTION")
    
    latest = df_copy.iloc[-1]
    prev = df_copy.iloc[-2]
    
    price = float(latest['Close'])
    prev_close = float(prev['Close'])
    change_pct = ((price - prev_close) / prev_close) * 100
    
    # VWAP status
    if latest['Close'] > latest['VWAP']:
        vwap_status = 'Bullish'
        vwap_color = '#00FF9D'
    else:
        vwap_status = 'Bearish'
        vwap_color = '#FF4B4B'
        
    # Squeeze status
    if latest['Squeeze']:
        squeeze_status = 'Squeezing'
        squeeze_color = '#FF9900'
    else:
        squeeze_status = 'Normal'
        squeeze_color = '#8b949e'
        
    # Whale status (check last 3 bars for recent activity)
    recent_whales = df_copy.tail(3)
    whale_status = 'Normal'
    whale_color = '#8b949e'
    for idx, row in recent_whales.iterrows():
        if row['Whale_Spike']:
            whale_status = row['Whale_Type']
            whale_color = '#00FF9D' if whale_status == 'ACCUMULATION' else '#FF4B4B'
            break
            
    return {
        'price': price,
        'vwap_status': vwap_status,
        'vwap_color': vwap_color,
        'squeeze_status': squeeze_status,
        'squeeze_color': squeeze_color,
        'whale_status': whale_status,
        'whale_color': whale_color,
        'change_pct': change_pct
    }

def prepare_features(df):
    print("Engineering features...")
    df = df.copy()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Advanced Technicals
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    
    # MACD & Momentum
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['ROC5'] = df['Close'].pct_change(5) * 100

    # TEMPORAL MEMORY (The Lags)
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Vol_Lag1'] = df['Volume'].shift(1)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    features = [
        'Open', 'High', 'Low', 'Volume',
        'MA7', 'MA21', 'MA50', 'RSI', 'MFI', 'ADX',
        'MACD', 'ROC5', 'Volatility',
        'Close_Lag1', 'Close_Lag2', 'RSI_Lag1', 'Vol_Lag1'
    ]

    df['Target'] = df['Close'].shift(-1)
    
    df = df.dropna()
    X = df[features]
    y = df['Target']
    dates = df.index
    return X, y, features, dates

def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1e-9))) * 100
    directional = np.mean(np.sign(np.diff(y_test.values)) == np.sign(np.diff(preds))) * 100
    print(f"  {model_name}: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f} MAPE={mape:.2f}% Dir={directional:.1f}%")
    return preds, {'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
                   'R2': round(r2, 4), 'MAPE': round(mape, 4),
                   'Directional': round(directional, 2)}

def tune_model(name, estimator, param_grid, X_train, y_train):
    print(f"  Tuning {name}...")
    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(estimator, param_grid, cv=tscv,
                      scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    print(f"    Best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_

def detect_whales(df, window=20):
    """
    Detects 'Smart Money' footprints by scanning for high-volume 
    absorption patterns (High Volume + Tight Price Range).
    """
    df = df.copy()
    avg_vol = df['Volume'].rolling(window=window).mean()
    vol_std = df['Volume'].rolling(window=window).std()
    
    # Calculate Price Tightness
    range_pct = (df['High'] - df['Low']) / df['Close']
    avg_range = range_pct.rolling(window=window).mean()
    
    latest_vol = df['Volume'].iloc[-1]
    latest_range = range_pct.iloc[-1]
    
    is_whale = False
    signal = "NEUTRAL"
    
    # Anomaly: Volume > 1.5x Mean AND Range < 1.2x Mean
    if latest_vol > (avg_vol.iloc[-1] + 1.5 * vol_std.iloc[-1]):
        if latest_range < (avg_range.iloc[-1] * 1.2):
            is_whale = True
            # Accumulation vs Distribution
            if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
                signal = "ACCUMULATION"
            else:
                signal = "DISTRIBUTION"
                
    return is_whale, signal

def run_monte_carlo(df, days=30, simulations=500):
    """
    Institutional-grade Monte Carlo Simulation (Geometric Brownian Motion)
    """
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = float(df['Close'].iloc[-1])
    
    simulation_results = np.zeros((days + 1, simulations))
    simulation_results[0] = last_price
    
    for s in range(simulations):
        for d in range(1, days + 1):
            # Random Walk with Drift
            simulation_results[d, s] = simulation_results[d-1, s] * (1 + np.random.normal(mu, sigma))
            
    # Calculate Percentiles
    forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(days + 1)]
    p10 = np.percentile(simulation_results, 10, axis=1)
    p50 = np.percentile(simulation_results, 50, axis=1)
    p90 = np.percentile(simulation_results, 90, axis=1)
    
    # Calculate Probability of Upside
    final_prices = simulation_results[-1, :]
    prob_up = (np.sum(final_prices > last_price) / simulations) * 100

    return pd.DataFrame({'p10': p10, 'p50': p50, 'p90': p90}, index=forecast_dates), prob_up

def predict_long_term(df, days=365):
    """
    Predicts the 1-year price trajectory using Log-Normal Geometric Brownian Motion.
    Optimized for long-term investing horizons.
    """
    closes = df['Close'].values
    log_returns = np.diff(np.log(closes))
    
    # Yearly Drift and Volatility
    mu = np.mean(log_returns) * 252 
    sigma = np.std(log_returns) * np.sqrt(252)
    
    last_price = closes[-1]
    time = np.linspace(0, 1, days)
    
    # Expected Path (Drift)
    forecast = last_price * np.exp((mu - 0.5 * sigma**2) * time)
    
    # 95% Confidence Bounds
    upper = last_price * np.exp((mu - 0.5 * sigma**2) * time + 1.96 * sigma * np.sqrt(time))
    lower = last_price * np.exp((mu - 0.5 * sigma**2) * time - 1.96 * sigma * np.sqrt(time))
    
    # Calculate Probability of Upside (1-Year) using CDF
    from scipy.stats import norm
    mu_adj = mu - 0.5 * sigma**2
    # Probability that price in 1 year > last_price
    prob_up = norm.cdf(mu_adj / sigma) * 100 if sigma > 0 else 50.0
    
    return forecast, upper, lower, prob_up

# ==========================================
#  CLOUD PROFILE (auto-detected on Streamlit Community Cloud)
# ==========================================
def _is_cloud():
    """True when running on Streamlit Community Cloud / similar throttled host."""
    # Streamlit sets these env vars on its share platform
    return bool(os.environ.get("STREAMLIT_SHARING") or os.environ.get("HOSTNAME", "").startswith("streamlit"))


# ==========================================
#  TRAIN / PREDICT PHASE (refactored for caching)
# ==========================================
def _resolve_n_iters(cloud_iters, local_iters):
    """Return cloud-tuned iterations when on free tier, full otherwise."""
    return cloud_iters if _is_cloud() else local_iters


def _split_returns(X, y):
    """Return (y_return, latest_close) — converts price-target to %-return target."""
    current_close = X['Close_Lag1']
    y_return = ((y - current_close) / current_close) * 100
    y_return = y_return.replace([np.inf, -np.inf], 0).fillna(0)
    return y_return


def _safe_price(latest_close, y):
    """NaN-safe fallback for latest close."""
    if np.isnan(latest_close) or latest_close == 0:
        return float(y.iloc[-1])
    return float(latest_close)


def _apply_sentiment_to_price(pred_price, latest_close, sentiment_bias, strength):
    """Nudge a predicted price by `sentiment_bias * strength` percent."""
    if sentiment_bias == 0 or latest_close == 0:
        return pred_price
    return pred_price * (1 + (sentiment_bias * strength))


def compute_price_band(model_outputs, ensemble_pred, latest_close, base_band_pct=1.5):
    """
    Build an absolute-rupee price band from per-model predictions.

    Bayesian Ridge already returns a real std-dev (its own uncertainty).
    Other models use ensemble std-dev: if XGB says ₹850 and RF says ₹820,
    the band is wide. If they agree at ₹840, the band is narrow.

    `model_outputs` : list of (name, predicted_price)
    Returns (low_price, high_price, band_pct) — band_pct is the half-width as a %.
    """
    if not model_outputs or ensemble_pred is None or latest_close <= 0:
        # Fallback: fixed band if anything is missing
        return latest_close * (1 - base_band_pct/100), latest_close * (1 + base_band_pct/100), base_band_pct

    prices = np.array([p for _, p in model_outputs if p is not None and not np.isnan(p)])
    if len(prices) == 0:
        return latest_close * (1 - base_band_pct/100), latest_close * (1 + base_band_pct/100), base_band_pct

    # Use std-dev of model predictions as the band width, floored at 0.5%
    std_pct = max(0.5, float(np.std(prices)) / latest_close * 100)
    low = max(0.01, ensemble_pred - ensemble_pred * std_pct / 100)
    high = ensemble_pred + ensemble_pred * std_pct / 100
    return low, high, std_pct


# ---------- CONSENSUS (XGBoost + RandomForest + Ridge) ----------
# Cap a predicted daily return (% units) at +/- 4x the recent realized
# volatility. Without this guard, XGB/RF can extrapolate 30%+ moves on
# out-of-distribution feature vectors, e.g. ADANIPOWER at Rs.230
# previously returned Rs.164 (down) or Rs.297 (up) depending on which
# stale-cache features were served. 4x daily vol is roughly the 99%
# extreme under a normal assumption; anything beyond is model noise.
_CONSENSUS_VOL_CLAMP_MULT = 4.0


def _clamp_return(ret_pct: float, vol_pct: float) -> float:
    """Cap ret_pct at +/- clamp_mult * vol_pct (in same % units)."""
    if vol_pct is None or vol_pct <= 0 or np.isnan(vol_pct):
        return float(ret_pct)
    cap = _CONSENSUS_VOL_CLAMP_MULT * float(vol_pct)
    return float(np.clip(ret_pct, -cap, cap))


def _train_consensus_models(X, y):
    """
    Train XGBoost + RandomForest + Ridge in RETURN space (not raw price).

    Returning a price level directly is a bad target: a 200-row training
    window of "raw price" for a stock that drifts from Rs.40 to Rs.230
    teaches XGB to predict whatever value happens to be in the last row
    of its training window, and then the model wildly extrapolates on the
    out-of-distribution latest row.

    Returns artifacts dict with per-model sub-models and a stale-row list
    that the predict phase uses to recompute test scores against returns.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Convert targets to percent-returns on the train/test split
    close_train = X_train["Close_Lag1"]
    close_test = X_test["Close_Lag1"]
    ret_train = ((y_train - close_train) / close_train) * 100
    ret_test = ((y_test - close_test) / close_test) * 100
    ret_train = ret_train.replace([np.inf, -np.inf], 0).fillna(0)
    ret_test = ret_test.replace([np.inf, -np.inf], 0).fillna(0)

    n_iter = _resolve_n_iters(cloud_iters=120, local_iters=200)
    models = {
        "XGBoost": XGBRegressor(n_estimators=n_iter, learning_rate=0.04, max_depth=5,
                                subsample=0.8, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }

    preds = {}
    r2_scores = []
    for name, model in models.items():
        model.fit(X_train_sc, ret_train)
        p_test = model.predict(X_test_sc)
        # R² on RETURNS, not prices — a much more honest fit measure
        r2_scores.append(max(0.01, r2_score(ret_test, p_test)))
        preds[name] = p_test

    return {
        "models": models,
        "scaler": scaler,
        "r2": float(np.mean(r2_scores)),
    }


def _predict_consensus(artifacts, latest_row, sentiment_bias=0):
    """Run inference on already-trained consensus models. Returns (pred, r2, importances, price_band)."""
    models = artifacts["models"]
    scaler = artifacts["scaler"]
    latest_sc = scaler.transform(latest_row)

    # Per-model returns (in %), then clamped to +/- 4x recent realized vol
    raw_returns = {name: float(model.predict(latest_sc)[0]) for name, model in models.items()}
    recent_vol = float(latest_row["Volatility"].iloc[0]) if "Volatility" in latest_row.columns else 0.0
    clamped_returns = {name: _clamp_return(r, recent_vol) for name, r in raw_returns.items()}

    # Dynamic weighting — tilt toward XGBoost when news is extreme
    base_xgb_w, base_rf_w = 0.50, 0.35
    tilt = abs(sentiment_bias) * 0.1
    w_xgb = base_xgb_w + tilt
    w_rf = base_rf_w
    w_ridge = max(0, 1.0 - (w_xgb + w_rf))
    consensus_ret = (clamped_returns["XGBoost"] * w_xgb
                     + clamped_returns["RandomForest"] * w_rf
                     + clamped_returns["Ridge"] * w_ridge)

    # Convert return to price using the last known close
    latest_close = float(latest_row["Close_Lag1"].iloc[0]) if "Close_Lag1" in latest_row.columns else None
    if latest_close is None or latest_close <= 0:
        # Fallback: bail out to identity (no band can be built without a ref price)
        return 0.0, artifacts["r2"], pd.Series(dtype=float), (0.0, 0.0, 0.0)

    consensus_price = latest_close * (1 + consensus_ret / 100.0)

    # Sentiment multiplier (5% nudge on the price level)
    final_consensus = _apply_sentiment_to_price(consensus_price, latest_close, sentiment_bias, strength=0.05)

    # Feature importance from XGBoost
    importances = models["XGBoost"].feature_importances_
    feat_imp = pd.Series(importances, index=latest_row.columns).sort_values(ascending=False)

    # Price band from per-model DISAGREEMENT, in price-space
    model_outputs = []
    for name, ret_pct in clamped_returns.items():
        model_outputs.append((name, latest_close * (1 + ret_pct / 100.0)))
    low, high, band_pct = compute_price_band(model_outputs, final_consensus, latest_close)

    return final_consensus, artifacts["r2"], feat_imp, (low, high, band_pct)


# ---------- META-STACKED ENSEMBLE ----------
def _train_meta_stacker(X, y):
    """Train XGBoost + LightGBM + CatBoost -> Ridge meta-learner on %-returns."""
    current_close = X['Close_Lag1']
    y_return = _split_returns(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_return, test_size=0.15, shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    cat_iters = _resolve_n_iters(cloud_iters=80, local_iters=150)
    xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    lgb = LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
    cat = CatBoostRegressor(iterations=cat_iters, learning_rate=0.05, depth=5, random_state=42, verbose=0)

    xgb.fit(X_train_sc, y_train)
    lgb.fit(X_train_sc, y_train)
    cat.fit(X_train_sc, y_train)

    pred_xgb_test = xgb.predict(X_test_sc)
    pred_lgb_test = lgb.predict(X_test_sc)
    pred_cat_test = cat.predict(X_test_sc)
    meta_X_test = np.column_stack([pred_xgb_test, pred_lgb_test, pred_cat_test, X_test_sc])

    meta_learner = Ridge(alpha=10.0)
    meta_learner.fit(meta_X_test, y_test)

    r2 = float(max(0.01, r2_score(y_test, meta_learner.predict(meta_X_test))))

    return {
        "xgb": xgb, "lgb": lgb, "cat": cat,
        "meta_learner": meta_learner,
        "scaler": scaler,
        "current_close_series": current_close,
        "r2": r2,
    }


def _predict_meta_stacker(artifacts, latest_row, sentiment_bias=0):
    """Inference on trained meta-stacker. Returns (pred, r2, importances, price_band)."""
    xgb = artifacts["xgb"]
    lgb = artifacts["lgb"]
    cat = artifacts["cat"]
    meta_learner = artifacts["meta_learner"]
    scaler = artifacts["scaler"]

    latest_sc = scaler.transform(latest_row)
    latest_close = float(latest_row['Close_Lag1'].iloc[0])

    pred_xgb_latest = float(xgb.predict(latest_sc)[0])
    pred_lgb_latest = float(lgb.predict(latest_sc)[0])
    pred_cat_latest = float(cat.predict(latest_sc)[0])

    base_preds = np.array([[pred_xgb_latest, pred_lgb_latest, pred_cat_latest]])
    latest_2d = latest_sc.reshape(1, -1) if latest_sc.ndim == 1 else latest_sc
    meta_X_latest = np.hstack([base_preds, latest_2d])
    predicted_return = float(np.clip(meta_learner.predict(meta_X_latest)[0], -10.0, 10.0))

    adjusted_return = predicted_return + (sentiment_bias * 0.5)
    safe_close = _safe_price(latest_close, latest_row.iloc[:, 0])
    final_pred = safe_close * (1 + adjusted_return / 100.0)

    importances = (xgb.feature_importances_ + lgb.feature_importances_) / 2.0
    feat_imp = pd.Series(importances, index=latest_row.columns).sort_values(ascending=False)

    # Price band: convert each base model's predicted return to a price, then disagreement
    base_prices = [
        safe_close * (1 + pred_xgb_latest / 100.0),
        safe_close * (1 + pred_lgb_latest / 100.0),
        safe_close * (1 + pred_cat_latest / 100.0),
    ]
    model_outputs = [("XGB", base_prices[0]), ("LGB", base_prices[1]), ("CAT", base_prices[2])]
    low, high, band_pct = compute_price_band(model_outputs, final_pred, safe_close)

    return final_pred, artifacts["r2"], feat_imp, (low, high, band_pct)


# ---------- BAYESIAN RIDGE ----------
def _train_bayesian_ridge(X, y):
    """Train Bayesian Ridge on %-returns. Returns artifacts with std-return support."""
    y_return = _split_returns(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_return, test_size=0.15, shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = BayesianRidge()
    model.fit(X_train_sc, y_train)

    r2 = float(max(0.01, r2_score(y_test, model.predict(X_test_sc))))

    return {
        "model": model,
        "scaler": scaler,
        "r2": r2,
    }


def _predict_bayesian_ridge(artifacts, latest_row):
    """Inference on trained Bayesian Ridge. Returns (pred, r2, importances, price_band)."""
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    latest_sc = scaler.transform(latest_row)
    latest_close = float(latest_row['Close_Lag1'].iloc[0])

    pred_return, std_return = model.predict(latest_sc, return_std=True)
    pred_return_val = float(np.clip(pred_return[0], -10.0, 10.0))
    std_pct = float(abs(std_return[0]))

    safe_close = _safe_price(latest_close, latest_row.iloc[:, 0])
    final_pred = safe_close * (1 + pred_return_val / 100.0)

    importances = np.abs(model.coef_)
    feat_imp = pd.Series(importances, index=latest_row.columns).sort_values(ascending=False)

    # Bayesian gives a real uncertainty: ±std_return on the return, converted to absolute rupees
    low_price = max(0.01, safe_close * (1 + (pred_return_val - std_pct) / 100.0))
    high_price = safe_close * (1 + (pred_return_val + std_pct) / 100.0)
    return final_pred, artifacts["r2"], feat_imp, (low_price, high_price, std_pct)


# ==========================================
#  BACKWARDS-COMPATIBLE WRAPPERS (old API still works)
# ==========================================
def get_consensus_prediction(X, y, latest_row, sentiment_bias=0):
    """Wrapper: trains then predicts consensus in one call (used by callers without caching)."""
    artifacts = _train_consensus_models(X, y)
    pred, r2, feat_imp, _band = _predict_consensus(artifacts, latest_row, sentiment_bias)
    return pred, r2, feat_imp


def get_meta_stacked_prediction(X, y, latest_row, sentiment_bias=0):
    """Wrapper: trains then predicts meta-stacker in one call."""
    artifacts = _train_meta_stacker(X, y)
    pred, r2, feat_imp, _band = _predict_meta_stacker(artifacts, latest_row, sentiment_bias)
    return pred, r2, feat_imp


def get_bayesian_ridge_prediction(X, y, latest_row):
    """Wrapper: trains then predicts Bayesian Ridge in one call. Returns (pred, r2, imp, margin_error_pct)."""
    artifacts = _train_bayesian_ridge(X, y)
    pred, r2, feat_imp, (_low, _high, margin_pct) = _predict_bayesian_ridge(artifacts, latest_row)
    return pred, r2, feat_imp, margin_pct


def main():
    df = fetch_data(TICKER, START_DATE, END_DATE)
    X, y, feature_names, dates = prepare_features(df)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\n=== Hyperparameter Tuning ===")

    param_grids = {
        "Ridge": (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 50, 100, 500]}),
        "Lasso": (Lasso(max_iter=10000), {'alpha': [0.001, 0.01, 0.1, 1, 10]}),
        "SVR": (SVR(), {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1], 'kernel': ['rbf']}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42),
                             {'n_estimators': [50, 100], 'max_depth': [3, 5],
                              'learning_rate': [0.05, 0.1]}),
    }

    tuned_models = {}
    best_params_log = {}

    for name, (est, grid) in param_grids.items():
        model, params = tune_model(name, est, grid, X_train_sc, y_train)
        tuned_models[name] = model
        best_params_log[name] = params

    # Add non-tuned models
    tuned_models["Linear Regression"] = LinearRegression().fit(X_train_sc, y_train)
    tuned_models["Random Forest"] = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42).fit(X_train_sc, y_train)

    print("\n=== Model Evaluation ===")
    results = {}
    all_preds = {}

    for name, model in tuned_models.items():
        if not hasattr(model, 'coef_') and not isinstance(model, (SVR, RandomForestRegressor, GradientBoostingRegressor)):
            model.fit(X_train_sc, y_train)
        preds, metrics = evaluate_model(model, X_test_sc, y_test, name)
        results[name] = metrics
        all_preds[name] = preds.tolist()

    # Feature importances (Random Forest)
    rf = tuned_models["Random Forest"]
    importances = rf.feature_importances_.tolist()

    # Actual prices
    actual = y_test.values.tolist()
    test_dates = [str(d.date()) for d in dates_test]

    # Save output JSON for dashboard
    output = {
        'ticker': TICKER,
        'results': results,
        'predictions': all_preds,
        'actual': actual,
        'dates': test_dates,
        'features': feature_names,
        'importances': importances,
        'best_params': best_params_log
    }

    with open('/tmp/stock_results.json', 'w') as f:
        json.dump(output, f)

    print("\n=== Final Summary ===")
    summary = pd.DataFrame(results).T.sort_values('RMSE')
    print(summary.to_string())
    print(f"\nBest model: {summary.index[0]}")
    print("Results saved to /tmp/stock_results.json")
    return output

if __name__ == "__main__":
    main()