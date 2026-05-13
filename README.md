# 🔭 ProsperVista: AI-Driven Market Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ProsperVista** is a next-generation equity analytics platform that bridges the gap between quantitative price data and qualitative market sentiment. By combining Machine Learning regression models with Real-Time NLP Sentiment Analysis, ProsperVista provides institutional-grade insights for retail investors.

---

## ✨ Key Features

### 🤖 Hybrid Prediction Engine
- **ML Models**: Switch between **Linear, Ridge, and Lasso** regression to forecast price movements based on historical OHLC data.
- **Dynamic Data Windows**: Analyze market trends from 1 to 5 years of historical depth.

### 🧠 Sentiment Intelligence
- **Real-Time NLP**: Fetches the latest global news headlines via Yahoo Finance API.
- **VADER Sentiment Analysis**: Quantifies "Market Mood" on a scale of -1.0 (Bearish) to +1.0 (Bullish).
- **Decision Bonus**: Predictions are automatically adjusted based on weighted news sentiment to improve real-world accuracy.

### ⚠️ Market Anomaly Detection
- Monitors major global tickers and highlights significant declines (>1.0% drop).
- Fetches the exact news trigger (trigger-reason) for market anomalies to explain "The Why" behind price drops.

### 💎 Institutional UI/UX
- **Glassmorphism Design**: A sleek, dark-themed interface with frosted-glass effects.
- **Real-Time Visuals**: Interactive Plotly charts and feature-importance breakdowns.
- **Fixed Intelligence Footer**: Persistent data attribution and market movers at the absolute bottom.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Custom CSS/Glassmorphism)
- **Data Source**: Yahoo Finance API (`yfinance`)
- **NLP Engine**: VADER Sentiment Analysis
- **Machine Learning**: Scikit-Learn (Linear Models, StandardScaler)
- **Visuals**: Plotly & Matplotlib

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ProsperVista.git
cd ProsperVista
```

### 2. Set Up Environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📈 Ticker Search Guide

To ensure high-precision data fetching, use the following formats:

| Market | Ticker Format | Example |
| :--- | :--- | :--- |
| **US Stocks** | Ticker Symbol | `AAPL`, `NVDA`, `TSLA` |
| **Indian NSE** | Ticker + `.NS` | `RELIANCE.NS`, `TATAMOTORS.NS` |
| **Indian BSE** | 6-Digit Code + `.BO` | `500325.BO` (Reliance), `532540.BO` (TCS) |
| **Crypto** | Symbol + `-USD` | `BTC-USD`, `ETH-USD` |

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built with ❤️ for Modern Investors
</p>
