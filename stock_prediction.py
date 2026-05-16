import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = 'AAPL'
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
# ==========================================

def fetch_data(ticker, start, end):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data found.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def prepare_features(df):
    print("Engineering features...")
    df = df.copy()
    df['Target'] = df['Close'].shift(-1)
    
    # Base Features
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

def get_consensus_prediction(X, y, latest_row, sentiment_bias=0):
    """
    Enhanced Consensus Engine with Sentiment Integration & Dynamic Weighting
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    latest_sc = scaler.transform(latest_row)
    
    # Advanced Model Ensemble
    models = {
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.04, max_depth=6, subsample=0.8, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        "Ridge": Ridge(alpha=1.0)
    }
    
    preds = {}
    r2_scores = []
    
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        p_test = model.predict(X_test_sc)
        score = max(0.01, r2_score(y_test, p_test))
        r2_scores.append(score)
        preds[name] = model.predict(latest_sc)[0]
    
    # DYNAMIC WEIGHTING: Trust aggressive models (XGB) more if news is active
    base_xgb_w = 0.50
    base_rf_w = 0.35
    
    # If sentiment is extreme, tilt towards XGBoost
    tilt = abs(sentiment_bias) * 0.1
    w_xgb = base_xgb_w + tilt
    w_rf = base_rf_w
    w_ridge = max(0, 1.0 - (w_xgb + w_rf))
    
    consensus = (preds["XGBoost"] * w_xgb) + (preds["RandomForest"] * w_rf) + (preds["Ridge"] * w_ridge)
    
    # FINAL SENTIMENT ADJUSTMENT (The 5% Force Multiplier)
    final_consensus = consensus * (1 + (sentiment_bias * 0.05))
    
    return final_consensus, np.mean(r2_scores), models["XGBoost"].feature_importances_

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