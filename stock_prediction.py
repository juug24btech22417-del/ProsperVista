import yfinance as yf
import pandas as pd
import numpy as np
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

    # Momentum & RSI
    df['ROC5'] = df['Close'].pct_change(5) * 100
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26

    # TEMPORAL MEMORY (The Lags)
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['MACD_Lag1'] = df['MACD'].shift(1)
    df['Vol_Lag1'] = df['Volume'].shift(1)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    features = [
        'Open', 'High', 'Low', 'Volume',
        'MA7', 'MA21', 'MA50',
        'ROC5', 'RSI', 'MACD', 'Volatility',
        'Close_Lag1', 'Close_Lag2', 'RSI_Lag1', 'MACD_Lag1', 'Vol_Lag1'
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

def run_monte_carlo(df, days=30, simulations=500):
    """
    Institutional-grade Monte Carlo Simulation (Geometric Brownian Motion)
    """
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = float(df['Close'].iloc[-1])
    
    results = np.zeros((days + 1, simulations))
    results[0] = last_price
    
    for s in range(simulations):
        for d in range(1, days + 1):
            # Random Walk with Drift
            results[d, s] = results[d-1, s] * (1 + np.random.normal(mu, sigma))
            
    sim_df = pd.DataFrame(results)
    
    # Extract Confidence Bands
    forecast = pd.DataFrame({
        'p10': sim_df.quantile(0.1, axis=1),
        'p50': sim_df.quantile(0.5, axis=1),
        'p90': sim_df.quantile(0.9, axis=1)
    })
    
    return forecast

def get_consensus_prediction(X, y, latest_row):
    """
    Runs an institutional-grade ensemble (XGBoost, RF, Ridge) 
    and returns a weighted consensus prediction.
    """
    # Split for local validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    latest_sc = scaler.transform(latest_row)
    
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Ridge": Ridge(alpha=1.0)
    }
    
    preds = {}
    weights = {}
    r2_scores = []
    
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        p = model.predict(X_test_sc)
        score = max(0, r2_score(y_test, p))
        
        # Calculate current prediction
        current_p = model.predict(latest_sc)[0]
        preds[name] = current_p
        weights[name] = score
        r2_scores.append(score)
        
    # Weighted Average based on R2 performance
    total_weight = sum(weights.values())
    if total_weight == 0:
        consensus = sum(preds.values()) / len(preds)
        importances = [0] * len(X.columns)
    else:
        consensus = sum(preds[n] * (weights[n]/total_weight) for n in preds)
        # Use Random Forest importances as the representative impact
        importances = models["RandomForest"].feature_importances_
        
    return consensus, np.mean(r2_scores), importances

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