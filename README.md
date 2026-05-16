# 🔭 ProsperVista: AI-Driven Market Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**ProsperVista** is a next-generation equity analytics platform that bridges the gap between quantitative price data and qualitative market sentiment. By combining Machine Learning regression models with Real-Time NLP Sentiment Analysis, ProsperVista provides institutional-grade insights for retail investors.

---

## ✨ Key Features

### 🤖 Hybrid Prediction Engine (`stock_prediction.py`)
- **Advanced ML Models**: Switch between **Linear, Ridge, and Lasso** regression.
- **Feature Engineering**: Calculates **RSI**, **MACD**, **Bollinger Bands**, and **ROC Momentum**.
- **Automated Tuning**: Utilizes `GridSearchCV` and `TimeSeriesSplit` for high-precision price targets.

### 🧠 Sentiment Intelligence (`sentiment_engine.py`)
- **Real-Time NLP**: Fetches global news headlines via Yahoo Finance API and processes them using **VADER Sentiment Analysis**.
- **Market Mood Scoring**: Quantifies mood on a scale of -1.0 (Bearish) to +1.0 (Bullish).
- **Decision Bonus**: Predictions are automatically adjusted based on sentiment scores to improve real-world forecasting.

### ⚠️ Market Anomaly Detection
- Monitors major global tickers and highlights significant declines (>1.0% drop).
- Identifies the exact news trigger for market anomalies to explain "The Why" behind price volatility.

### 💎 Institutional UI/UX (`app.py` & `ui_elements.py`)
- **Glassmorphism Design**: A sleek, dark-themed interface with frosted-glass effects and premium typography.
- **Unified Analytics**: Interactive Plotly candlesticks, live watchlist sync, and feature-importance bars.
- **Fixed Intelligence Footer**: Persistent institutional-grade analytics sign-off.

---

## 🏛️ Project Structure

```text
ProsperVista/
├── app.py                  # 🚀 MAIN HUB: Terminal UI & global analysis flow
├── stock_prediction.py     # 🧠 BRAIN: Advanced ML models & technical indicators
├── sentiment_engine.py    # 📰 INTELLIGENCE: NLP news analysis & anomaly detection
├── watchlist_manager.py    # 📋 STATE: Persistent user watchlists & sidebar UI
├── ui_elements.py          # 🎨 DESIGN: Global CSS & premium UI components
├── watchlist.json          # 💾 STORAGE: JSON store for user stock data
└── requirements.txt        # ⚙️ SYSTEM: Project dependencies
```

---

## 🚀 Quick Start

### 1. Initialize Virtual Environment (VENV)
```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📈 Ticker Search Guide

| Market | Ticker Format | Example |
| :--- | :--- | :--- |
| **US Stocks** | Ticker Symbol | `AAPL`, `NVDA`, `TSLA` |
| **Indian NSE** | Ticker + `.NS` | `RELIANCE.NS`, `TATAPOWER.NS` |
| **Indian BSE** | 6-Digit Code + `.BO` | `500325.BO` (Reliance), `532540.BO` (TCS) |
| **Crypto** | Symbol + `-USD` | `BTC-USD`, `ETH-USD` |

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
<p align="center">
  <b>PROSPER VISTA © 2026 • INSTITUTIONAL GRADE EQUITY ANALYTICS</b><br>
  Built with ❤️ for Modern Investors
</p>
