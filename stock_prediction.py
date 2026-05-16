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
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Momentum
    df['ROC5'] = df['Close'].pct_change(5) * 100
    df['ROC10'] = df['Close'].pct_change(10) * 100

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26

    # Bollinger Band width
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_width'] = (2 * bb_std) / (bb_mid + 1e-9)

    # Volume ratio
    df['Vol_MA10'] = df['Volume'].rolling(10).mean()
    df['Vol_ratio'] = df['Volume'] / (df['Vol_MA10'] + 1e-9)

    features = [
        'Open', 'Prev_Close', 'High', 'Low', 'Volume',
        'MA7', 'MA21', 'MA50',
        'ROC5', 'ROC10', 'Volatility', 'RSI', 'MACD',
        'BB_width', 'Vol_ratio'
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