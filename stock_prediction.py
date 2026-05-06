import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = 'AAPL'  # Example: AAPL, TSLA, INFY.NS
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d') # 2 years ago
END_DATE = datetime.now().strftime('%Y-%m-%d')
# ==========================================

def fetch_data(ticker, start, end):
    print(f"Fetching data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError("No data found for the given ticker and date range.")
    return df

def prepare_features(df):
    print("Preprocessing data and creating features...")
    # We want to predict the 'Close' price of the next day
    # Target variable: Close price shifted by -1
    df['Target'] = df['Close'].shift(-1)

    # Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()

    # Selection of features as per requirements
    features = ['Open', 'Prev_Close', 'High', 'Low', 'Volume', 'MA7', 'MA21']

    # Handle missing values caused by shifts and rolling windows
    df = df.dropna()

    X = df[features]
    y = df['Target']

    return X, y, features

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {model_name} Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return predictions, rmse, mae, r2

def main():
    try:
        # 1. Data Collection
        df = fetch_data(TICKER, START_DATE, END_DATE)

        # 2. Preprocessing
        X, y, feature_names = prepare_features(df)

        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. Model Training and Evaluation
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso()
        }

        results = {}
        all_predictions = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds, rmse, mae, r2 = evaluate_model(model, X_test_scaled, y_test, name)
            results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            all_predictions[name] = preds

        # 4. Visualization
        print("\nGenerating plots...")

        # Plot 1: Actual vs Predicted (using Linear Regression as primary)
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Price', color='blue')
        plt.plot(all_predictions["Linear Regression"], label='Predicted Price (Linear)', color='red', linestyle='--')
        plt.title(f'Actual vs Predicted Stock Price for {TICKER}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('actual_vs_predicted.png')
        plt.close()

        # Plot 2: Feature Importance (Coefficients of Linear Regression)
        lr_model = models["Linear Regression"]
        coefficients = lr_model.coef_

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, coefficients, color='skyblue')
        plt.title(f'Feature Importance (Linear Regression) for {TICKER}')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # 5. Summary Output
        print("\n" + "="*30)
        print("FINAL SUMMARY")
        print("="*30)
        summary_df = pd.DataFrame(results).T
        print(summary_df)
        print("\nPlots saved as 'actual_vs_predicted.png' and 'feature_importance.png'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
