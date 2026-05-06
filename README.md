# Prosper Vista | AI Stock Investment Advisor

Prosper Vista is an institutional-grade stock price prediction and investment decision support system. Built using a combination of Quantitative Analysis and Machine Learning, it transforms raw market data into actionable investment signals.

## 🚀 Project Overview
The goal of this project is to provide users with a data-driven approach to equity investing. Instead of relying on intuition, Prosper Vista uses **Linear Regression** and **Advanced Technical Indicators** to forecast the next day's closing price and provide a clear investment verdict: **BUY, SELL, or HOLD**.

## 🛠️ Technical Architecture

### 1. Data Pipeline
- **Source**: Real-time historical data fetched via `yfinance` API.
- **Preprocessing**: Handling of missing values, data normalization using `StandardScaler`, and train-test splitting (80/20).

### 2. Quantitative Feature Engineering
To improve prediction accuracy, the model doesn't just look at price. It incorporates:
- **Moving Averages**: 7-day and 21-day windows to identify short and medium-term trends.
- **RSI (Relative Strength Index)**: To detect overbought or oversold conditions.
- **Bollinger Bands**: To measure market volatility and price extremes.
- **Volume Analysis**: To validate the strength of the price movement.

### 3. Machine Learning Engine
The system implements and compares three regression models to ensure the best fit:
- **Linear Regression**: The core baseline for trend prediction.
- **Ridge Regression**: L2 Regularization to prevent overfitting.
- **Lasso Regression**: L1 Regularization for feature selection.

### 4. Investment Decision Logic
The "Investment Verdict" is generated based on:
- **Price Delta**: The difference between the predicted and current price.
- **Confidence Score**: The $R^2$ value of the model.
- **Verdict Criteria**: A `BUY` signal is only triggered if the predicted increase is significant AND the model confidence is high.

## 💻 Installation & Usage

### Prerequisites
- Python 3.10+

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ProsperVista.git
   cd ProsperVista
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📊 Key Features
- **Real-time Data**: Fetches latest NSE/BSE data on demand.
- **SaaS Dashboard**: Professional "Glassmorphism" UI for high-end data visualization.
- **Interactive Charts**: Candlestick charts with sticky hover tooltips.
- **Risk Management**: Provides suggested Entry, Target, and Stop-Loss prices.

---
*Developed as a Data Science Project for College Submission.*
