# 📈 ProsperVista v3.0 — Next-Gen Quantitative Trading Terminal

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit 1.32+](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![fpdf2](https://img.shields.io/badge/fpdf2-2.8.7-FFD43B?style=for-the-badge&logo=pdf&logoColor=black)](https://pyfpdf.github.io/fpdf2/)
> **Full-spectrum institutional-grade trading terminal** — Featuring 9 high-performance analytical engines, real-time AI forecasting ensembles, sentiment NLP pipelines, Black-Scholes options Greeks, multi-strategy backtesting, portfolio simulation, and a dynamic PDF briefing builder. Designed with custom dark-space aesthetics, glassmorphism, and zero cloud database dependencies.

---

## ✨ Release Highlights: What's New in v3.0

This release elevates ProsperVista into a premium systematic research terminal with stunning styling, flawless layouts, and pixel-perfect responsiveness.

### 🎨 Premium Visual Styling & Layouts
* **Watchlist Button Alignment Fix**: Restructured sidebar watchlist rendering with custom column context routing. Replaced the clunky emoji minus with a sleek text-based mathematical minus (`−`) and injected target CSS padding overrides. Watchlist ticker names and delete buttons are now perfectly aligned horizontally on a single line with zero vertical wrapping.
* **Glassmorphic News Sentiment Cards**: Styled plain news feeds into interactive, light-absorbing slate tiles (`#161B22`) with glowing borders. Features 4px-left color accents for sentiment labels, transparent background badges, and horizontal hover-translation transitions (`translateX(4px)`).
* **Footer Overlap Resolution**: Increased main container scrolling buffers (`padding-bottom: 8rem !important;`), allowing all grid elements and action buttons to scroll cleanly and stay fully visible above the fixed footer.

### 📊 "Crazy" High-Quality PDF Research Briefings
* **Matplotlib Dark-Mode Telemetry Charts**: PDF exports now automatically draw and embed a customized dark financial chart (`#0B0E11` background, glowing neon blue price line, and a green 50-day moving average overlay) compiled dynamically from the last 90 days of the stock's actual telemetry.
* **Institutional Briefing Formats**: Generates executive-level reports containing quantitative metric panels, 14-day dynamic RSI readings, moving averages, expected upside percentiles, a 30-day Monte Carlo predictive risk weighting matrix, and formal legal disclaimers.
* **Zero Repo Pollution**: Engineered an automated temporary file cleanup routine using `tempfile` and in-memory byte buffers. The temporary PDF is loaded into memory, sent to the browser, and immediately destroyed from the local disk. No generated PDF files are ever left behind in your workspace folder.

---

## 🏛️ Comprehensive Architecture

```
ProsperVista/
├── app.py                  # 🚀 Terminal Hub — global navigation, state manager, & layout
├── dashboard_views.py      # 🎛️ UI Renderers for the 7 institutional analytics tabs
├── ui_elements.py          # 🎨 Design System — custom CSS, font interfaces, and typography
│
├── stock_prediction.py     # 🧠 ML Engine — XGBoost + RF ensemble, Monte Carlo, and Technical Indicators
├── sentiment_engine.py     # 📰 Sentiment Engine — VADER NLP, composite Fear & Greed, and anomalies
├── report_generator.py     # 📊 PDF Briefing Engine — matplotlib dark-mode charting, and fpdf2 grids
├── watchlist_manager.py    # 📋 Storage Controller — JSON-based persistent watchlist
│
├── portfolio_simulator.py  # 💼 Virtual Desk — virtual balance, trade logging, and live P&L tracking
├── options_engine.py       # 📊 Option Desk — Black-Scholes pricing model & custom payoff diagrams
├── risk_engine.py          # ⚡ Risk Desk — VaR, CVaR, stress testing, and portfolio Beta overlays
├── pattern_engine.py       # 🔍 Pattern Desk — Candlestick pattern scanner and Fibonacci retracements
├── correlation_engine.py   # 🔥 Matrix Desk — Dynamic coefficient heatmaps and diversification index
├── backtesting_engine.py   # 🧪 Lab Desk — Strategy backtest engine with comparative metrics
├── screener_engine.py      # 📡 Screen Desk — Dynamic multi-filter screening algorithms
│
├── watchlist.json          # 💾 Storage — Watchlist stock array
├── portfolio_data.json     # 💾 Storage — Paper trading logs (automatically generated)
└── requirements.txt        # ⚙️ Configuration — Pip dependency manifest
```

---

## 🚀 Quick Start

### 1. Clone & Initialize Virtual Environment
```bash
git clone https://github.com/yourusername/ProsperVista.git
cd ProsperVista

python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Core Dependencies
Ensure all quantitative, plotting, machine learning, and PDF generation libraries are installed:
```bash
pip install -r requirements.txt
```

### 3. Launch the Systematic Terminal
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your secure browser interface.

---

## 🔍 Analytical Module Guide

Use the polished **sidebar navigation buttons** (arranged in a clean, emoji-free list with zero overlap) to cycle through the terminal's 9 quantitative views:

| View | Module | Description |
| :--- | :--- | :--- |
| **Home** | [sentiment_engine.py](file:///e:/cloned-repos/ProsperVista/sentiment_engine.py) | Dynamic market anomaly tracker + composite Fear & Greed Index gauges. |
| **Portfolio** | [portfolio_simulator.py](file:///e:/cloned-repos/ProsperVista/portfolio_simulator.py) | Simulation trading engine with real-time margins, profit logs, and Kelly sizing advice. |
| **Options Greeks** | [options_engine.py](file:///e:/cloned-repos/ProsperVista/options_engine.py) | Black-Scholes Greeks calculators ($N(d_1)$, $N(d_2)$, Delta, Gamma) + interactive payoff charts. |
| **Risk Analytics** | [risk_engine.py](file:///e:/cloned-repos/ProsperVista/risk_engine.py) | Stress-testing metrics, Value-at-Risk (VaR), Conditional VaR (CVaR), and Beta coefficients. |
| **Patterns** | [pattern_engine.py](file:///e:/cloned-repos/ProsperVista/pattern_engine.py) | Scans for 15+ daily candlestick patterns, Fibonacci levels, and BB squeeze thresholds. |
| **Correlation** | [correlation_engine.py](file:///e:/cloned-repos/ProsperVista/correlation_engine.py) | Inter-asset price matrices with diversification scores and heatmaps. |
| **Backtesting** | [backtesting_engine.py](file:///e:/cloned-repos/ProsperVista/backtesting_engine.py) | System testing on 5 pre-built technical strategies with equity curves. |
| **Screener** | [screener_engine.py](file:///e:/cloned-repos/ProsperVista/screener_engine.py) | Dynamic stock filters matching growth, breakout, oversold, and quality criteria. |
| **Watchlist** | [watchlist_manager.py](file:///e:/cloned-repos/ProsperVista/watchlist_manager.py) | Premium grid cards featuring real-time prices, one-click analysis, and unified deletions. |

---

## 📈 Ticker Format Guide

The terminal queries high-frequency feeds using standard Yahoo Finance symbols:

* **US Equity Stocks**: Capitalized ticker only (e.g., `AAPL`, `NVDA`, `TSLA`).
* **Indian NSE Stocks**: Capitalized ticker with `.NS` suffix (e.g., `TATAPOWER.NS`, `RELIANCE.NS`, `INFY.NS`). If you type `RELIANCE` without a suffix, the terminal will automatically resolve it to `.NS` for seamless localized trading.
* **Indian BSE Stocks**: Six-digit BSE ID with `.BO` suffix (e.g., `500325.BO`).
* **Cryptocurrency Feeds**: Coin ticker with `-USD` suffix (e.g., `BTC-USD`, `ETH-USD`).
* **Market Indices**: Prefix with caret symbol `^` (e.g., `^NSEI` for Nifty 50, `^BSESN` for Sensex, `^GSPC` for S&P 500).

---

## ⚙️ System Requirements

* **Python Version**: 3.9 or higher (fully compatible with 3.10 and 3.11).
* **RAM**: 4GB minimum (8GB recommended for fitting machine learning ensembles and running Monte Carlo walks in memory).
* **Data Feed**: Steady internet connection required for high-speed Yahoo Finance REST fetching.
* **Operating System**: Windows (10/11), macOS (Intel/Apple Silicon), Linux (Ubuntu/Debian).

---
<p align="center">
  <b>PROSPER VISTA v3.0 © 2026 · QUANTITATIVE TERMINAL PLATFORM</b><br>
Built with ❤️ for Modern Investors 
</p>
