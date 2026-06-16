# 📈 ProsperVista v3.0 — Next-Gen Quantitative Trading Terminal

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit 1.32+](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google_Gemini-API_v1-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3-f55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)

> **Full-spectrum institutional-grade trading terminal** — Featuring 9 high-performance analytical engines, real-time AI forecasting ensembles, sentiment NLP pipelines, Black-Scholes options Greeks, multi-strategy backtesting, portfolio simulation, and a dynamic PDF briefing builder. Designed with custom dark-space aesthetics, glassmorphism, and zero cloud database dependencies.

---

## ✨ Release Highlights: What's New in v3.0

This release elevates ProsperVista into a premium systematic research terminal with stunning styling, flawless layouts, and pixel-perfect responsiveness.

### 🎨 Premium Visual Styling & Layouts
* **Symmetric 3x3 Control Desk Grid**: Realigned the terminal dashboard navigation menu into a clean, uniform 3x3 layout. All navigation buttons share identical sizing and wrap beautifully across all resolutions.
* **Watchlist Button Alignment Fix**: Restructured sidebar watchlist rendering with custom column context routing. Watchlist ticker names and delete buttons are now perfectly aligned horizontally.
* **Glassmorphic News Sentiment Cards**: Styled plain news feeds into interactive, light-absorbing slate tiles (`#161B22`) with glowing borders and horizontal hover-translation transitions.

### 🚀 Live IPO Intelligence & RAG Upgrades
* **Real-time Chittorgarh Feeds**: Replaced static mock databases with live scraping pipelines querying upcoming, ongoing, and completed Indian mainboard listings.
* **AI Prospectus Auditing**: Extracts Lot Size, P/E ratios, GMP levels, and subscription telemetry, feeding them to Gemini to render structural summaries and verdicts.
* **RAG Executive Document Overviews**: Automatically parses the first pages of uploaded 300+ page corporate documents to render instant identification and operational descriptions.

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
├── modules/                # 🧠 Quantitative AI Modules
│   ├── copilot/            # 💬 Neural Copilot - dynamic model routing & fallback queries
│   ├── ipo/                # 🚀 IPO Intelligence - Chittorgarh API scraper & scoring
│   ├── rag/                # 📖 RAG Research - semantic document indexing & summaries
│   └── news/               # 📰 AI News - Sentiment analysis & breadth indices
│
├── watchlist.json          # 💾 Storage — Watchlist stock array
├── portfolio_data.json     # 💾 Storage — Paper trading logs (automatically generated)
└── requirements.txt        # ⚙️ Configuration — Pip dependency manifest
```

---

## 🔑 AI Credentials Setup (Gemini & Groq API)

To unlock the Neural Copilot, RAG company research, AI news intelligence, and IPO prospectus summarization, you must configure your API keys. The system uses a resilient multi-provider strategy, automatically resolving keys from multiple environments.

> [!NOTE]
> If no Gemini API key is configured, the application remains fully functional by falling back to local regex-based prospectus parsers and TF-IDF semantic keyword lookups.

### 📝 Step 1: Create a `.env` File
In the root directory of the project, create a file named `.env`. Add your API keys using the exact variables below:

```env
# Google Gemini API Key (Required for Copilot, RAG summary, IPO Auditing)
GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"

# Groq API Key (Used as a fallback LLM provider if Gemini quota is reached)
GROQ_API_KEY="gsk_YourGroqApiKeyHere"
```

### 🔒 Alternative Credentials Injection
The terminal resolves keys in the following priority order:
1. **Local `.env` File**: Reads keys directly from `.env` in the root workspace.
2. **System Environment Variables**: Configured keys in your OS environment.
3. **Streamlit Secrets**: For deployments, create `.streamlit/secrets.toml` containing:
   ```toml
   GEMINI_API_KEY = "your-key"
   GROQ_API_KEY = "your-key"
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

Use the symmetric **control desk** to cycle through the terminal's 9 views:

| View | Location | Description |
| :--- | :--- | :--- |
| **Home** | `sentiment_engine.py` | Dynamic market anomaly tracker + composite Fear & Greed Index gauges. |
| **Charts** | `dashboard_views.py` | Technical indicator charts (RSI, Bollinger Bands, Moving Averages). |
| **Intraday** | `dashboard_views.py` | Real-time predictive charts with Bollinger Bands and volume profiling. |
| **Simulator** | `portfolio_simulator.py` | Paper trading logs with profit trackers and Kelly criterion calculators. |
| **Options & Risk** | `options_engine.py` / `risk_engine.py` | Black-Scholes payoff graphs and stress-testing indexes (VaR, CVaR). |
| **Watchlist** | `watchlist_manager.py` | Premium grid cards featuring real-time prices and one-click briefing triggers. |
| **IPOs** | `modules/ipo` | Live Chittorgarh tracking board, subscription telemetry, and demand scoring. |
| **RAG QA** | `modules/rag` | PDF vector parsing, source-cited queries, and automatic summaries. |
| **AI News** | `modules/news` | Global headlines sentiment metrics, sector trend heatmaps, and breadth indices. |

---

## 📈 Ticker Format Guide

The terminal queries high-frequency feeds using standard Yahoo Finance symbols:

* **US Equity Stocks**: Capitalized ticker only (e.g., `AAPL`, `NVDA`, `TSLA`).
* **Indian NSE Stocks**: Capitalized ticker with `.NS` suffix (e.g., `TATAPOWER.NS`, `RELIANCE.NS`, `INFY.NS`). Typing `RELIANCE` without a suffix automatically resolves it to `.NS` for convenience.
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
