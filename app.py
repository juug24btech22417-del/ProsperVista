# Prosper Vista v3.0.0 - Institutional Control Hub
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json
import textwrap
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load dotenv configuration if present
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

# MODULAR IMPORTS
from sentiment_engine import SentimentEngine
import watchlist_manager as wm
import stock_prediction as sp
import ui_elements as ui
import dashboard_views

# MODULES ENGINES IMPORTS
import modules.ipo.ipo_engine as ipo_engine
import modules.rag.rag_engine as rag_engine
import modules.news.news_engine as news_engine

# ==========================================
#  CACHING LAYER
# ==========================================
@st.cache_data(ttl=3600)
def get_cached_company_name(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info.get('longName', ticker)
    except:
        return ticker

@st.cache_data(ttl=1800)
def get_cached_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}


@st.cache_resource(show_spinner=False)
def _train_all_models(ticker, years):
    """
    Train all 5 base models in parallel for (ticker, years).
    Returns a dict of artifacts, one entry per model choice.

    Cached by Streamlit for 24h via resource cache. Resource cache holds
    fitted estimator objects in memory so subsequent calls are inference-only.
    """
    df_tuple = fetch_terminal_data(ticker, years)
    if df_tuple is None:
        return None
    df, name, _price = df_tuple
    X, y, _, _ = sp.prepare_features(df)
    if len(X) < 50:
        return None

    artifacts = {}
    # Parallel cold-start: CatBoost, LightGBM, XGBoost all release the GIL.
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_consensus = ex.submit(sp._train_consensus_models, X, y)
        f_meta = ex.submit(sp._train_meta_stacker, X, y)
        f_bayes = ex.submit(sp._train_bayesian_ridge, X, y)
        artifacts["consensus"] = f_consensus.result()
        artifacts["meta_stacker"] = f_meta.result()
        artifacts["bayesian"] = f_bayes.result()

    # Pre-compute scaler + latest row once for fast predict phase
    artifacts["_meta"] = {
        "ticker": ticker,
        "years": years,
        "sample_count": int(len(X)),
        "feature_columns": list(X.columns),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    return artifacts


def get_trained_models(ticker, years):
    """Public wrapper: returns the artifacts dict for a ticker or None."""
    return _train_all_models(ticker, years)


@st.cache_resource(show_spinner=False)
def _train_intraday_models(ticker, interval):
    """Intraday counterpart of _train_all_models — caches fitted models per (ticker, interval)."""
    try:
        df = sp.fetch_intraday_data(ticker, interval=interval, period="5d")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        X, y, _, _ = sp.prepare_intraday_features(df)
    except Exception:
        return None
    if len(X) < 50:
        return None

    artifacts = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_consensus = ex.submit(sp._train_consensus_models, X, y)
        f_meta = ex.submit(sp._train_meta_stacker, X, y)
        f_bayes = ex.submit(sp._train_bayesian_ridge, X, y)
        artifacts["consensus"] = f_consensus.result()
        artifacts["meta_stacker"] = f_meta.result()
        artifacts["bayesian"] = f_bayes.result()
    return artifacts


def _format_age(iso_ts):
    """Human-readable age from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        delta = datetime.now() - dt
        secs = int(delta.total_seconds())
        if secs < 60: return f"{secs}s ago"
        if secs < 3600: return f"{secs // 60}m ago"
        if secs < 86400: return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "unknown"


def _is_cloud_runtime():
    """True when running on Streamlit Community Cloud (used to show cloud-profile indicator)."""
    return bool(os.environ.get("STREAMLIT_SHARING") or os.environ.get("HOSTNAME", "").startswith("streamlit"))

# ==========================================
#  UI & CSS INJECTION
# ==========================================
def inject_ui():
    ui.inject_custom_css()

# ==========================================
#  INTEGRATED ANALYTICS HUB
# ==========================================
@st.cache_data(ttl=300)
def fetch_terminal_data(ticker, years=2):
    try:
        # Utilizing modular fetch
        start = (datetime.now() - pd.Timedelta(days=365*years)).strftime('%Y-%m-%d')
        end = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        df = sp.fetch_data(ticker, start, end)
        if df.empty: return None
        return df, get_cached_company_name(ticker), df['Close'].iloc[-1]
    except: return None

@st.cache_data(ttl=300)
def fetch_market_indices():
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NASDAQ": "^IXIC",
        "BTC/USD": "BTC-USD",
        "GOLD": "GC=F"
    }
    data = []
    try:
        syms = list(indices.values())
        bulk = yf.download(syms, period="2d", progress=False, threads=True)
        if bulk.empty: return data
        for name, sym in indices.items():
            try:
                if isinstance(bulk.columns, pd.MultiIndex):
                    close = bulk['Close'][sym].dropna()
                else:
                    close = bulk['Close'].dropna()
                if len(close) >= 2:
                    cur = close.iloc[-1]
                    prev = close.iloc[-2]
                    chg = ((cur - prev) / prev) * 100
                    data.append({"name": name, "cur": cur, "prev": prev, "chg": chg, "sym": sym})
            except: continue
    except: pass
    return data

@st.cache_data(ttl=300)
def fetch_sector_data():
    sectors = {
        "BANKING": "^NSEBANK",
        "IT": "^CNXIT",
        "PHARMA": "^CNXPHARMA",
        "AUTO": "^CNXAUTO",
        "METAL": "^CNXMETAL"
    }
    sector_data = []
    # Batch download all sector indices + EV stocks in one call
    ev_stocks = ["TATAMOTORS.NS", "M&M.NS", "OLECTRA.NS", "TVSMOTOR.NS"]
    all_syms = list(sectors.values()) + ev_stocks
    try:
        bulk = yf.download(all_syms, period="2d", progress=False, threads=True)
        if bulk.empty: return sector_data
        for name, sym in sectors.items():
            try:
                if isinstance(bulk.columns, pd.MultiIndex):
                    close = bulk['Close'][sym].dropna()
                else:
                    close = bulk['Close'].dropna()
                if len(close) >= 2:
                    s_cur = close.iloc[-1]
                    s_prev = close.iloc[-2]
                    s_chg = ((s_cur - s_prev) / s_prev) * 100
                    sector_data.append({"name": name, "chg": s_chg})
            except: continue
        # Custom EV Sector from batch data
        ev_chgs = []
        for s in ev_stocks:
            try:
                if isinstance(bulk.columns, pd.MultiIndex):
                    close = bulk['Close'][s].dropna()
                else:
                    continue
                if len(close) >= 2:
                    ev_chgs.append(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100)
            except: continue
        if ev_chgs:
            sector_data.append({"name": "EV ENERGY", "chg": sum(ev_chgs) / len(ev_chgs)})
    except: pass
    return sector_data


# Cached wrappers for sentiment engine (avoids re-fetching on every rerun)
@st.cache_data(ttl=180)
def _cached_fear_greed():
    engine = SentimentEngine()
    return engine.calculate_fear_greed()

@st.cache_data(ttl=120)
def _cached_market_movers():
    engine = SentimentEngine()
    return engine.get_market_movers()

def generate_research_report(name, ticker, price, target, chg, mood, sentiment_score, confidence):
    """
    Generates a stunning, high-contrast, vibrant HTML research document.
    """
    clr = "#00FF9D" if chg >= 0 else "#FF4B4B"
    mood_clr = "#00FF9D" if mood == "BULLISH" else "#FF4B4B" if mood == "BEARISH" else "#8B949E"
    grad = "linear-gradient(135deg, #00C875, #005030)" if chg >= 0 else "linear-gradient(135deg, #FF4B4B, #800000)"
    
    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background: #0B0E11; color: #C9D1D9; margin: 0; padding: 40px; }}
            .container {{ 
                max-width: 850px; margin: 0 auto; background: #161B22; 
                border: 1px solid #30363D; border-radius: 24px; overflow: hidden;
                box-shadow: 0 30px 60px rgba(0,0,0,0.5); 
            }}
            .top-banner {{ 
                background: {grad}; padding: 40px; text-align: center; color: white;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            .title {{ font-size: 38px; font-weight: 800; letter-spacing: -1.5px; margin: 0; }}
            .subtitle {{ opacity: 0.8; font-size: 13px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; margin-top: 10px; }}
            
            .content-area {{ padding: 40px; }}
            
            .badge {{ 
                display: inline-block; padding: 12px 35px; border-radius: 12px; 
                font-weight: 800; font-size: 28px; margin: 20px 0; 
                text-transform: uppercase; letter-spacing: 2px;
                box-shadow: 0 0 30px {clr}44; border: 2px solid {clr}; color: {clr};
                background: {clr}11;
            }}
            
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 30px 0; }}
            .stat-card {{ 
                background: #0D1117; border: 1px solid #30363D; border-radius: 16px; 
                padding: 15px; text-align: center; transition: 0.3s;
            }}
            .stat-card:hover {{ border-color: #58A6FF; transform: translateY(-3px); }}
            .stat-label {{ font-size: 9px; color: #8B949E; text-transform: uppercase; font-weight: 700; margin-bottom: 8px; letter-spacing: 1px; }}
            .stat-val {{ font-family: 'JetBrains Mono', monospace; font-size: 18px; color: #FFFFFF; font-weight: 700; }}
            
            .intel-box {{ 
                background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); 
                border-radius: 16px; padding: 25px; margin-top: 30px;
            }}
            .section-h {{ color: #58A6FF; font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 15px; }}
            .content-p {{ font-size: 15px; line-height: 1.7; color: #C9D1D9; margin: 0; }}
            
            .footer {{ 
                background: #0D1117; padding: 20px; border-top: 1px solid #30363D; 
                text-align: center; font-size: 11px; color: #485563; font-weight: 600;
            }}
            .highlight {{ color: {clr}; font-weight: 800; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="top-banner">
                <div class="title">Prosper Vista Analytics</div>
                <div class="subtitle">Institutional Research Intel</div>
            </div>
            
            <div class="content-area">
                <div style="text-align: center;">
                    <div style="font-size: 16px; color: #8B949E;">Research Profile for <span style="color:white; font-weight:700;">{name} ({ticker})</span></div>
                    <div class="badge">{'BUY' if chg > 1.2 else 'SELL' if chg < -1.2 else 'HOLD'}</div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">AI Target</div>
                        <div class="stat-val" style="color: {clr}">₹{target:,.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Confidence</div>
                        <div class="stat-val">{confidence:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Mood</div>
                        <div class="stat-val" style="color: {mood_clr}">{mood}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Entry Price</div>
                        <div class="stat-val">₹{price:,.2f}</div>
                    </div>
                </div>
                
                <div class="intel-box">
                    <div class="section-h">Ensemble Intelligence Audit</div>
                    <p class="content-p">
                        Our multi-model consensus (XGBoost & Random Forest) has identified a <span class="highlight">{chg:+.2f}%</span> potential move. 
                        The technical profile indicates strong <span class="highlight">{mood.lower()} momentum</span> anchored by a 
                        {confidence:.1f}% convergence across all institutional engines.
                    </p>
                </div>
                
                <div style="margin-top: 30px;">
                    <div class="section-h">Market Sentiment Pulse</div>
                    <p class="content-p" style="color: #8B949E;">
                        Sentiment score of <span style="color:{mood_clr}; font-weight:700;">{sentiment_score:+.2f}</span> detected. 
                        Global news clusters and social indicators are currently aligned with our predictive trajectory, 
                        showing increased volume interest in the {ticker} sector.
                    </p>
                </div>
            </div>
            
            <div class="footer">
                GENREATED BY PROSPER VISTA AI • {datetime.now().strftime('%B %d, %Y')} • CONFIDENTIAL RESEARCH
            </div>
        </div>
    </body>
    </html>
    """
    return report_html

# ==========================================
# MAIN TERMINAL APP
# ==========================================
def main():
    st.set_page_config(page_title="Prosper Vista", layout="wide")
    inject_ui()
    sentiment_engine = SentimentEngine()
    
    # Initialize session state variables early
    if 'current_ticker' not in st.session_state: st.session_state.current_ticker = "TATAPOWER.NS"
    if 'view_mode' not in st.session_state: st.session_state.view_mode = "home"
    if 'live_feed' not in st.session_state: st.session_state.live_feed = False
    if 'search_query' not in st.session_state: st.session_state.search_query = ""

    # GLOBAL DASHBOARD HEADER
    st.markdown(textwrap.dedent('''
        <div class="dashboard-header">
            <div class="dashboard-title">Prosper Vista</div>
            <div class="dashboard-desc">Next-Gen Market Intelligence</div>
            <div class="dashboard-long-desc">
                Advanced institutional-grade terminal for real-time market intelligence, 
                predictive forecasting, and sentiment analytics.
            </div>
        </div>
    '''), unsafe_allow_html=True)

    # 1. LIVE MARKET STRIP (DYNAMIC - CACHED)
    index_data = fetch_market_indices()
    index_html = '<div class="index-strip">'
    for idx in index_data:
        sym = idx['sym']
        cur = idx['cur']
        chg = idx['chg']
        name = idx['name']
        clr_class = "index-up" if chg >= 0 else "index-down"
        sign = "+" if chg >= 0 else ""
        prefix = "$" if "USD" in sym else "₹" if sym in ["^NSEI", "^BSESN", "GC=F"] else ""
        val_fmt = f"{prefix}{cur:,.2f}" if cur > 100 else f"{prefix}{cur:,.4f}"
        index_html += f'<div class="index-item">{name} <span class="index-value {clr_class}">{val_fmt} ({sign}{chg:.2f}%)</span></div>'
    index_html += '</div>'
    st.markdown(index_html, unsafe_allow_html=True)

    # 1.1 SECTOR LEADERBOARD (CACHED)
    sector_data = fetch_sector_data()
        
    if sector_data:
        # Sort by performance
        sector_data = sorted(sector_data, key=lambda x: x['chg'], reverse=True)
        
        with st.container():
            st.markdown('<div style="font-size:10px; color:#8B949E; text-transform:uppercase; margin-bottom:10px; letter-spacing:2px; text-align:center;">Sector Rotation Intelligence</div>', unsafe_allow_html=True)
            s_cols = st.columns(len(sector_data))
            for i, s in enumerate(sector_data):
                s_clr = "#00FF9D" if s['chg'] >= 0 else "#FF4B4B"
                with s_cols[i]:
                    st.markdown(f'''
                        <div style="background:#161B22; border:1px solid #30363D; padding:10px; border-radius:12px; text-align:center;">
                            <div style="font-size:9px; color:#8B949E; font-weight:700;">{s['name']}</div>
                            <div style="font-size:14px; color:{s_clr}; font-weight:800; font-family:'JetBrains Mono';">{s['chg']:+.2f}%</div>
                        </div>
                    ''', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # 1.2 TOP NAVIGATION DESKS
    nav_items = {
        "home": "🏠 Home",
        "analysis": "📊 Charts",
        "intraday": "⏱️ Intraday",
        "portfolio": "💼 Simulator",
        "options": "📐 Options & Risk",
        "watchlist": "🔍 Watchlist & Screener",
        "ipo": "🚀 IPOs",
        "rag": "📖 RAG QA",
        "news": "📰 AI News"
    }
    
    st.markdown('<div style="font-size:10px; color:#8B949E; text-transform:uppercase; margin-bottom:12px; letter-spacing:2px; text-align:center; font-weight:700;">Terminal Control Desks</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
    current_mode = st.session_state.get('view_mode', 'home')
    
    # 3x3 grid for uniform button sizes
    nav_keys = list(nav_items.keys())
    cols_per_row = 3
    for i in range(0, len(nav_keys), cols_per_row):
        row_keys = nav_keys[i:i+cols_per_row]
        row_cols = st.columns(cols_per_row)
        for c_idx, mode in enumerate(row_keys):
            label = nav_items[mode]
            is_active = (current_mode == mode)
            if row_cols[c_idx].button(label, key=f"top_nav_{mode}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.view_mode = mode
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown('<div style="margin-top:15px; margin-bottom:15px; border-bottom:1px solid #30363D;"></div>', unsafe_allow_html=True)

    # 2. Sidebar Controls
    st.sidebar.markdown(textwrap.dedent(f'''
        <div class="status-card">
            <div style="font-size:11px; font-weight:800; color:white; margin-bottom:12px; letter-spacing:1px; text-transform:uppercase;">Terminal Status</div>
            <div class="status-item">
                <span>System Link</span>
                <span><span class="status-dot"></span>Active</span>
            </div>
            <div class="status-item">
                <span>Neural Engine</span>
                <span style="color:white;">v2.1.0 Institutional</span>
            </div>
            <div class="status-item">
                <span>Market Session</span>
                <span style="color:#58A6FF;">{'OPEN' if datetime.now().weekday() < 5 else 'CLOSED'}</span>
            </div>
        </div>
    '''), unsafe_allow_html=True)
    
    if 'current_ticker' not in st.session_state: st.session_state.current_ticker = ""
    if 'view_mode' not in st.session_state: st.session_state.view_mode = "home"
    if 'live_feed' not in st.session_state: st.session_state.live_feed = False
    
    st.sidebar.markdown('<div style="font-size:10px; color:#8B949E; text-transform:uppercase; margin-bottom:15px; letter-spacing:1px; font-weight:800;">Command Search</div>', unsafe_allow_html=True)
    
    live_mode = st.sidebar.toggle("Live Data Engine (15s)", value=st.session_state.live_feed, help="Auto-refresh the terminal to simulate a live WebSocket feed.")
    st.session_state.live_feed = live_mode
    
    # Smart Ticker Entry
    raw_ticker = st.sidebar.text_input("Stock Ticker", 
                                     value=st.session_state.current_ticker if st.session_state.current_ticker else "TATAPOWER.NS",
                                     placeholder="e.g. RELIANCE, AAPL, BTC-USD",
                                     help="AI will automatically append .NS for Indian stocks if omitted.").upper()
    
    # Auto-Suffix Logic
    if raw_ticker and "." not in raw_ticker and "-" not in raw_ticker and len(raw_ticker) >= 3:
        # Check if it's a common Indian ticker pattern (Pure alphabetic)
        if raw_ticker.isalpha():
            processed_ticker = f"{raw_ticker}.NS"
        else:
            processed_ticker = raw_ticker
    else:
        processed_ticker = raw_ticker

    # Real-time Company Name Validation
    if processed_ticker:
        try:
            with st.sidebar:
                name_placeholder = st.empty()
                if processed_ticker != st.session_state.get('last_validated'):
                    c_name = get_cached_company_name(processed_ticker)
                    st.session_state.last_validated = processed_ticker
                    st.session_state.current_company_name = c_name
                
                name_placeholder.markdown(f'<div style="font-size:10px; color:#58A6FF; font-weight:700; margin-top:-10px; margin-bottom:15px;">{st.session_state.get("current_company_name", "")}</div>', unsafe_allow_html=True)
        except: pass

    years = st.sidebar.slider("Data Window", 1, 5, 2)
    model_choice = st.sidebar.selectbox("Model Engine", ["Meta Stacked Ensemble", "Bayesian Ridge (Honest)", "Elite Consensus (XGBoost+RF)", "Linear", "Ridge", "Lasso"])
    
    if st.sidebar.button("Analyze Market", key="main_analyze_btn", use_container_width=True):
        st.session_state.current_ticker = processed_ticker
        st.session_state.view_mode = "analysis"
        st.session_state.model_choice = model_choice
        st.rerun()

    if st.sidebar.button(" Intraday Terminal", key="intraday_desk_btn", use_container_width=True):
        st.session_state.current_ticker = processed_ticker
        st.session_state.view_mode = "intraday"
        st.session_state.model_choice = model_choice
        st.rerun()

    # Model status & manual retrain
    with st.sidebar.expander("Model status", expanded=False):
        cached = get_trained_models(processed_ticker, years) if processed_ticker else None
        if cached is None:
            st.caption("Cache: ❄️ COLD (no model trained for this ticker yet)")
        else:
            meta = cached.get("_meta", {})
            age = _format_age(meta.get("trained_at", ""))
            samples = meta.get("sample_count", "?")
            st.caption(f"Cache: ✅ WARM  •  Trained {age}")
            st.caption(f"Samples: {samples}  •  TTL 24h")
            st.caption("Cloud profile: ON" if _is_cloud_runtime() else "Cloud profile: OFF")
        if st.button("🔄 Retrain models", key="retrain_btn", use_container_width=True, help="Invalidates the 24h cache and retrains all models."):
            _train_all_models.clear()
            st.success("Cache cleared. Next click will retrain.")
            st.rerun()

    # Modular Watchlist Manager
    w = wm.load_watchlist()
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:10px; color:#8B949E; text-transform:uppercase; margin-bottom:10px; letter-spacing:1px;">Active Watchlist</div>', unsafe_allow_html=True)
    
    for t in w:
        c1, c2 = st.sidebar.columns([4, 1])
        if c1.button(t, key=f"s_{t}", use_container_width=True):
            st.session_state.current_ticker = t
            st.session_state.view_mode = "analysis"
            st.rerun()
        if c2.button("−", key=f"d_{t}", help=f"Remove {t}", use_container_width=True):
            wm.remove_from_watchlist(t)
            st.rerun()

    # 3. ANALYSIS LOGIC
    if st.session_state.current_ticker and st.session_state.view_mode == "analysis":
        sub_tab = st.radio("Charts Desk Mode", ["Technical Analysis", "Pattern Recognition", "Backtesting Suite", "Correlation Matrix"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if sub_tab == "Pattern Recognition":
            tk = st.session_state.current_ticker or "RELIANCE.NS"
            res = fetch_terminal_data(tk, years)
            if res:
                df, _, _ = res
                dashboard_views.render_patterns_view(df, tk)
            else:
                st.warning("Please analyze a stock first or enter a valid ticker.")
            ui.render_footer()
            return
        elif sub_tab == "Backtesting Suite":
            dashboard_views.render_backtest_view()
            ui.render_footer()
            return
        elif sub_tab == "Correlation Matrix":
            dashboard_views.render_correlation_view(w)
            ui.render_footer()
            return
            
        with st.spinner(f"Initializing High-Frequency Neural Feed & Aggregating Global Sentiment..."):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as executor:
                f_res = executor.submit(fetch_terminal_data, st.session_state.current_ticker, years)
                f_sent = executor.submit(sentiment_engine.get_news_sentiment, st.session_state.current_ticker)
                f_info = executor.submit(get_cached_ticker_info, st.session_state.current_ticker)

                res = f_res.result()
                sent = f_sent.result()
                info = f_info.result()

        if res:
            df, name, price = res
            s_score = sent.get('score', 0)

            # Auditing Temporal Patterns & Neural Consensus
            X, y, feature_names, dates = sp.prepare_features(df)
            choice = st.session_state.get('model_choice', "Elite Consensus (XGBoost+RF)")
            latest_row = X.iloc[[-1]]

            # Pre-calculate Monte Carlo for PDF Export
            mc_forecast, prob_up = sp.run_monte_carlo(df)
            is_w, w_type = sp.detect_whales(df)

            # ===== CACHED MODEL DISPATCH =====
            # Train-once, predict-many: heavy work is cached for 24h.
            artifacts = get_trained_models(st.session_state.current_ticker, years)
            st.session_state.model_artifacts = artifacts
            st.session_state.model_meta = artifacts.get("_meta", {}) if artifacts else {}

            if artifacts is None:
                st.warning("Not enough data to train the AI engine for this ticker.")
                return

            choice_to_artifact = {
                "Meta Stacked Ensemble": "meta_stacker",
                "Bayesian Ridge (Honest)": "bayesian",
                "Elite Consensus (XGBoost+RF)": "consensus",
            }

            if choice in choice_to_artifact:
                art = artifacts[choice_to_artifact[choice]]
                if choice == "Meta Stacked Ensemble":
                    pred, r2, importances, price_band = sp._predict_meta_stacker(art, latest_row, sentiment_bias=s_score)
                elif choice == "Bayesian Ridge (Honest)":
                    pred, r2, importances, price_band = sp._predict_bayesian_ridge(art, latest_row)
                    st.session_state.bayesian_margin = price_band[2]  # std_pct
                else:
                    pred, r2, importances, price_band = sp._predict_consensus(art, latest_row, sentiment_bias=s_score)
                st.session_state.price_band = price_band
            else:
                # Legacy Support (Linear/Ridge/Lasso): no caching, no price band
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LinearRegression, Ridge, Lasso
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X)
                model = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}[choice]
                model.fit(X_sc, y)
                pred = model.predict(scaler.transform(latest_row))[0]
                r2 = model.score(X_sc, y)
                importances = model.coef_ if hasattr(model, 'coef_') else [0]*len(feature_names)
                # Synthetic ±2% band for legacy models
                st.session_state.price_band = (pred * 0.98, pred * 1.02, 2.0)

            adj_pred = pred
            chg = ((adj_pred - price) / price) * 100
            
            # Header & Export
            h1, h2 = st.columns([4, 1])
            with h1:
                st.markdown(f'<h1 style="color:white; margin-bottom:0; font-size:42px;">{name}</h1><p style="color:#58A6FF; font-weight:600; letter-spacing:1px;">MARKET ANALYSIS • {st.session_state.current_ticker}</p>', unsafe_allow_html=True)
            with h2:
                st.markdown("<br>", unsafe_allow_html=True)
                try:
                    import report_generator
                    max_up = float(((mc_forecast['p90'].iloc[-1] - price) / price) * 100)
                    max_down = float(((mc_forecast['p10'].iloc[-1] - price) / price) * 100)
                    pdf_file = report_generator.generate_intelligence_report(
                        st.session_state.current_ticker, float(price), float(adj_pred), float(r2), 
                        str(sent.get("verdict", "NEUTRAL")), str(w_type if is_w else "STABLE"), 
                        float(prob_up), float(mc_forecast['p50'].iloc[-1]), max_up, max_down,
                        df=df
                    )
                    with open(pdf_file, "rb") as pdf:
                        pdf_data = pdf.read()
                    
                    import os
                    st.download_button(
                        label="Export PDF Briefing", 
                        data=pdf_data, 
                        file_name=f"report_{st.session_state.current_ticker}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf", 
                        mime="application/pdf", 
                        use_container_width=True
                    )
                    try:
                        os.remove(pdf_file)
                    except:
                        pass
                except Exception as e:
                    pass # Fail silently so UI doesn't break
            
            # 4. METRICS ROW
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            with m1: st.markdown(f'<div class="metric-card"><div class="metric-title">Current Price</div><div class="metric-val">₹{price:,.2f}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-card"><div class="metric-title">Target Close</div><div class="metric-val">₹{adj_pred:,.2f}</div></div>', unsafe_allow_html=True)
            with m3:
                clr = "#00FF9D" if chg >= 0 else "#FF4B4B"
                st.markdown(f'<div class="metric-card"><div class="metric-title">Exp. Change</div><div class="metric-val" style="color:{clr}">{chg:+.2f}%</div></div>', unsafe_allow_html=True)
            with m4:
                if choice == "Bayesian Ridge (Honest)":
                    margin_error = st.session_state.get('bayesian_margin', 0)
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Confidence (Margin)</div><div class="metric-val" style="font-size:15px;">{r2*100:.1f}% (±{margin_error:.1f}%)</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Confidence</div><div class="metric-val">{r2*100:.1f}%</div></div>', unsafe_allow_html=True)
            with m5:
                # Predicted Range — per-model absolute price band
                pb = st.session_state.get('price_band')
                if pb is not None:
                    low_p, high_p, band_pct = pb
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Pred. Range (±{band_pct:.1f}%)</div><div class="metric-val" style="font-size:15px;">₹{low_p:,.0f}–{high_p:,.0f}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Pred. Range</div><div class="metric-val" style="font-size:14px;">N/A</div></div>', unsafe_allow_html=True)
            with m6:
                mood = sent.get("verdict", "NEUTRAL")
                m_clr = "#00FF9D" if mood == "BULLISH" else "#FF4B4B" if mood == "BEARISH" else "#8B949E"
                st.markdown(f'<div class="metric-card"><div class="metric-title">Market Mood</div><div class="metric-val" style="color:{m_clr}">{mood}</div></div>', unsafe_allow_html=True)
            with m7:
                w_clr = "#00FF9D" if w_type == "ACCUMULATION" else "#FF4B4B" if w_type == "DISTRIBUTION" else "#8B949E"
                st.markdown(f'<div class="metric-card"><div class="metric-title">Whale Activity</div><div class="metric-val" style="color:{w_clr}; font-size:14px;">{w_type if is_w else "STABLE"}</div></div>', unsafe_allow_html=True)

            # 4.0 TRANSPARENCY FOOTER — what model, on how much data, how fresh
            model_meta = st.session_state.get('model_meta', {})
            sample_count = model_meta.get('sample_count', '?')
            trained_at = model_meta.get('trained_at', '')
            trained_age = _format_age(trained_at) if trained_at else 'unknown'
            st.markdown(
                f'<div style="font-size:11px; color:#8B949E; margin-top:8px; margin-bottom:4px; letter-spacing:0.5px;">'
                f'Model: <b style="color:#C9D1D9;">{choice}</b> &nbsp;•&nbsp; '
                f'Trained on <b style="color:#C9D1D9;">{sample_count}</b> samples &nbsp;•&nbsp; '
                f'Last trained <b style="color:#C9D1D9;">{trained_age}</b> &nbsp;•&nbsp; '
                f'Cache TTL 24h &nbsp;•&nbsp; '
                f'Range auto-sized from model disagreement'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 4.1 FUNDAMENTAL ROW
            st.markdown("<br>", unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns(4)
            with f1: st.markdown(f'<div class="metric-card" style="height:100px;"><div class="metric-title">P/E Ratio</div><div class="metric-val" style="font-size:18px;">{info.get("trailingPE", "N/A")}</div></div>', unsafe_allow_html=True)
            with f2: st.markdown(f'<div class="metric-card" style="height:100px;"><div class="metric-title">Market Cap</div><div class="metric-val" style="font-size:18px;">₹{info.get("marketCap", 0)/1e11:.1f}T</div></div>', unsafe_allow_html=True)
            with f3: st.markdown(f'<div class="metric-card" style="height:100px;"><div class="metric-title">Revenue Growth</div><div class="metric-val" style="font-size:18px; color:#58A6FF;">{info.get("revenueGrowth", 0)*100:+.1f}%</div></div>', unsafe_allow_html=True)
            with f4: st.markdown(f'<div class="metric-card" style="height:100px;"><div class="metric-title">Profit Margin</div><div class="metric-val" style="font-size:18px; color:#00FF9D;">{info.get("profitMargins", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)

            # 5. VERDICT BANNER
            v_type, v_class, v_msg = ("HOLD", "hold-box", "Neutral indicators. Market sentiment and ML forecast are balanced.")
            if chg > 1.2 and s_score > 0.05: v_type, v_class, v_msg = ("BUY", "buy-box", f"Bullish trend predicted with {mood} market sentiment.")
            elif chg < -1.2 and s_score < -0.05: v_type, v_class, v_msg = ("SELL", "sell-box", f"Bearish trend predicted with {mood} market sentiment.")
            
            target = adj_pred
            sl = price - (adj_pred - price) * 0.8
            
            st.markdown(textwrap.dedent(f'''
                <div class="verdict-box {v_class}">
                    <div class="verdict-main">{v_type}</div>
                    <div class="verdict-desc">{v_msg} Target: ₹{target:,.2f}</div>
                    <div class="trade-strip">
                        Entry: ₹{price:,.2f} | Target: ₹{target:,.2f} | Stop-Loss: ₹{sl:,.2f}
                    </div>
                </div>
            '''), unsafe_allow_html=True)
            
            # RESEARCH REPORT DOWNLOAD
            report_html = generate_research_report(name, st.session_state.current_ticker, price, target, chg, mood, s_score, r2*100)
            st.download_button(
                label="Download Institutional Research Brief",
                data=report_html,
                file_name=f"ProsperVista_{st.session_state.current_ticker}_Report.html",
                mime="text/html",
                use_container_width=True
            )

            # 6. CHARTS ROW
            cg, ci = st.columns([3, 1])
            with cg:
                fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), height=450)
                st.plotly_chart(fig, use_container_width=True)
            
            with ci:
                st.markdown("<div style='text-align:center; font-size:10px; color:#8B949E; text-transform:uppercase; margin-bottom:10px;'>Feature Impact</div>", unsafe_allow_html=True)
                impact_df = pd.DataFrame({'Feature': feature_names, 'Influence': importances}).sort_values('Influence')
                fig_i = go.Figure(go.Bar(x=impact_df['Influence'], y=impact_df['Feature'], orientation='h', marker=dict(color=impact_df['Influence'], colorscale='RdYlGn', cmid=0)))
                fig_i.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), height=400, xaxis_visible=True, yaxis_visible=True)
                st.plotly_chart(fig_i, use_container_width=True, config={'displayModeBar': False})

            # 7. MULTI-HORIZON INTELLIGENCE HUB
            with st.expander("Terminal Manual: How to Read the Neural Forecast", expanded=False):
                st.markdown("""
                    <div style="font-size:13px; color:#C9D1D9; line-height:1.6;">
                        <b>1. The Central Blue Line (Median Path)</b><br>
                        This represents the 'Neural Consensus.' It is the 'Center of Gravity' for where the stock is headed based on 500 simultaneous simulations.
                        <br><br>
                        <b>2. The Blue Fog (Institutional Band)</b><br>
                        This is the 90% Confidence Interval. Institutional traders look for price to stay within this 'Fog.' If the price breaks outside this zone, it signifies a major 'Black Swan' event or a massive trend shift.
                        <br><br>
                        <b>3. Probability of Profit</b><br>
                        Calculated by counting how many simulated futures end higher than today's price. A score above 60% is considered a strong 'Institutional Accumulation' signal.
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            t_short, t_long = st.tabs(["SHORT-TERM TRADING (30D)", "LONG-TERM INVESTING (1Y)"])
            
            with t_short:
                st.markdown("### Probabilistic Future Projection (30-Day Monte Carlo)")
                c_mc1, c_mc2 = st.columns([2.5, 1])
                
                with c_mc1:
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Scatter(x=mc_forecast.index, y=mc_forecast['p90'], mode='lines', line=dict(width=0), showlegend=False))
                    fig_mc.add_trace(go.Scatter(x=mc_forecast.index, y=mc_forecast['p10'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(88, 166, 255, 0.1)', name="90% Confidence Interval"))
                    fig_mc.add_trace(go.Scatter(x=mc_forecast.index, y=mc_forecast['p50'], mode='lines', line=dict(color='#58A6FF', width=3), name="Median Projection"))
                    fig_mc.update_layout(
                        template="plotly_dark", 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        height=400, 
                        margin=dict(l=0,r=0,t=40,b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_mc, use_container_width=True, config={'displayModeBar': False})
                    
                with c_mc2:
                    upside = ((mc_forecast['p90'].iloc[-1] - price) / price) * 100
                    downside = ((mc_forecast['p10'].iloc[-1] - price) / price) * 100
                    st.markdown(f'''
                        <div class="intel-box" style="margin-top:0;">
                            <div class="section-h">Risk Assessment</div>
                            <div style="margin-bottom:15px;">
                                <p style="font-size:10px; color:#8B949E; margin:0;">30-DAY MAX UPSIDE (P90)</p>
                                <p style="font-size:24px; color:#00FF9D; font-weight:800; margin:0;">{upside:+.2f}%</p>
                            </div>
                            <div>
                                <p style="font-size:10px; color:#8B949E; margin:0;">30-DAY MAX DRAWDOWN (P10)</p>
                                <p style="font-size:24px; color:#FF4B4B; font-weight:800; margin:0;">{downside:+.2f}%</p>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
            with t_long:
                st.markdown("### Institutional Growth Trajectory (1-Year Forecast)")
                lt_forecast, lt_upper, lt_lower, lt_prob = sp.predict_long_term(df)
                
                fig_lt = go.Figure()
                fig_lt.add_trace(go.Scatter(x=list(range(365)), y=lt_upper, mode='lines', line=dict(width=0), showlegend=False))
                fig_lt.add_trace(go.Scatter(x=list(range(365)), y=lt_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 157, 0.05)', name="Institutional Band"))
                fig_lt.add_trace(go.Scatter(x=list(range(365)), y=lt_forecast, mode='lines', line=dict(color='#00FF9D', width=4), name="Growth Path"))
                
                fig_lt.update_layout(
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    height=400, 
                    margin=dict(l=0,r=0,t=40,b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_lt, use_container_width=True, config={'displayModeBar': False})
                
                st.info("Investing Hub: This forecast uses annual drift and historical volatility to project the most mathematically likely price range over the next 365 days.")
                
                # LONG-TERM RISK AUDIT
                l_c1, l_c2, l_c3 = st.columns(3)
                with l_c1: st.markdown(f'<div class="metric-card"><div class="metric-title">1Y Exp. Upside</div><div class="metric-val" style="color:#00FF9D;">{((lt_forecast[-1]-price)/price)*100:+.1f}%</div></div>', unsafe_allow_html=True)
                with l_c2: st.markdown(f'<div class="metric-card"><div class="metric-title">1Y Drawdown Floor</div><div class="metric-val" style="color:#FF4B4B;">{((lt_lower[-1]-price)/price)*100:+.1f}%</div></div>', unsafe_allow_html=True)
                with l_c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Profit Probability</div><div class="metric-val" style="color:#58A6FF;">{lt_prob:.1f}%</div></div>', unsafe_allow_html=True)
            
            # 8. SENTIMENT INTELLIGENCE
            st.markdown("---")
            st.markdown("### Sentiment Intelligence & News Analysis")
            sc1, sc2 = st.columns([0.8, 3.2])
            with sc1:
                st.markdown(textwrap.dedent(f'''
                    <div class="sent-score-card">
                        <div class="metric-title">Overall Sentiment Score</div>
                        <div class="sent-big-num" style="color:{m_clr}">{s_score:+.2f}</div>
                    </div>
                '''), unsafe_allow_html=True)
            with sc2:
                for n in sent.get('news', [])[:8]:
                    tag_bg = "rgba(0, 200, 117, 0.1)" if n['sentiment'] == "BULLISH" else "rgba(255, 68, 68, 0.1)" if n['sentiment'] == "BEARISH" else "rgba(139, 148, 158, 0.1)"
                    st.markdown(textwrap.dedent(f'''
                        <div class="news-card">
                            <span class="news-sentiment-tag" style="color:{n['color']}; background:{tag_bg};">{n['sentiment']}</span>
                            <a href="{n['link']}" target="_blank" class="news-title-link">{n['title']}</a>
                            <div class="news-meta">{n['publisher']} &nbsp;•&nbsp; {n['time']}</div>
                        </div>
                    '''), unsafe_allow_html=True)

    elif st.session_state.current_ticker and st.session_state.view_mode == "intraday":
        with st.spinner(f"Initializing High-Frequency Intraday Neural Feed..."):
            try:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as executor:
                    f_df = executor.submit(sp.fetch_intraday_data, st.session_state.current_ticker, "5m", "5d")
                    f_sent = executor.submit(sentiment_engine.get_news_sentiment, st.session_state.current_ticker)
                    
                    df = f_df.result()
                    sent = f_sent.result()
                    
                price = df['Close'].iloc[-1]
                
                # Header
                st.markdown(f'<h1 style="color:white; margin-bottom:0; font-size:42px;"> INTRADAY DESK</h1><p style="color:#00FF9D; font-weight:600; letter-spacing:1px;">HIGH-FREQUENCY TERMINAL • {st.session_state.current_ticker} (5M)</p>', unsafe_allow_html=True)
                
                st.markdown("""
                <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
                  <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
                    🎓 Intraday Terminal Guide: What to Check
                  </div>
                  <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
                    <ul style="margin: 0; padding-left: 20px;">
                      <li><b>Intraday Neural Telemetry</b>: Real-time neural consensus forecast matching short-horizon intraday signals.</li>
                      <li><b>VWAP Divergence</b>: Evaluates whether the current price is trading above or below the Volume Weighted Average Price (VWAP) indicating strong volume-backed price trends.</li>
                      <li><b>Micro-Whale Flow</b>: Detects rapid high-frequency volume clusters suggesting institutional algorithmic activity inside the order book.</li>
                    </ul>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                X, y, feature_names, dates = sp.prepare_intraday_features(df)
                if len(X) > 0:
                    choice = st.session_state.get('model_choice', "Elite Consensus (XGBoost+RF)")
                    latest_row = X.iloc[[-1]]

                    # Intraday path uses its own cache key (interval + period differ from daily)
                    intraday_artifacts = _train_intraday_models(st.session_state.current_ticker, "5m")
                    if intraday_artifacts is None:
                        st.warning("Could not train intraday models for this ticker.")
                        return

                    choice_to_artifact = {
                        "Meta Stacked Ensemble": "meta_stacker",
                        "Bayesian Ridge (Honest)": "bayesian",
                        "Elite Consensus (XGBoost+RF)": "consensus",
                    }

                    if choice in choice_to_artifact:
                        art = intraday_artifacts[choice_to_artifact[choice]]
                        if choice == "Meta Stacked Ensemble":
                            pred, r2, _imp, price_band = sp._predict_meta_stacker(art, latest_row, sentiment_bias=sent.get('score', 0))
                        elif choice == "Bayesian Ridge (Honest)":
                            pred, r2, _imp, price_band = sp._predict_bayesian_ridge(art, latest_row)
                            st.session_state.bayesian_margin = price_band[2]
                        else:
                            pred, r2, _imp, price_band = sp._predict_consensus(art, latest_row, sentiment_bias=sent.get('score', 0))
                        st.session_state.intraday_price_band = price_band
                    else:
                        # Legacy Support (Linear/Ridge/Lasso): no caching, no price band
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.linear_model import LinearRegression, Ridge, Lasso
                        scaler = StandardScaler()
                        X_sc = scaler.fit_transform(X)
                        model = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}[choice]
                        model.fit(X_sc, y)
                        pred = model.predict(scaler.transform(latest_row))[0]
                        r2 = model.score(X_sc, y)
                        st.session_state.intraday_price_band = (pred * 0.98, pred * 1.02, 2.0)
                        
                    is_w, w_type = sp.detect_micro_whales(df)
                    vwap = X['VWAP'].iloc[-1]
                    
                    # 1 & 2 & 4. New Metrics: Squeeze, Pivots, Alpha
                    pivots = sp.calculate_intraday_pivots(df)
                    is_squeeze = sp.detect_volatility_squeeze(df)
                    has_alpha, stk_ret, idx_ret = sp.calculate_alpha_divergence(st.session_state.current_ticker, df)
                    
                    # Layout Top Row
                    i1, i2, i3, i4, i5 = st.columns(5)
                    with i1: st.markdown(f'<div class="metric-card"><div class="metric-title">Current Price</div><div class="metric-val">₹{price:,.2f}</div></div>', unsafe_allow_html=True)
                    with i2:
                        if choice == "Bayesian Ridge (Honest)":
                            margin_error = st.session_state.get('bayesian_margin', 0)
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Target (Next 5m)</div><div class="metric-val" style="font-size:15px;">₹{pred:,.2f} (±{margin_error:.1f}%)</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Target (Next 5m)</div><div class="metric-val">₹{pred:,.2f}</div></div>', unsafe_allow_html=True)

                    vwap_clr = "#00FF9D" if price >= vwap else "#FF4B4B"
                    with i3: st.markdown(f'<div class="metric-card"><div class="metric-title">Current VWAP</div><div class="metric-val" style="color:{vwap_clr}">₹{vwap:,.2f}</div></div>', unsafe_allow_html=True)

                    w_clr = "#00FF9D" if w_type == "ACCUMULATION" else "#FF4B4B" if w_type == "DISTRIBUTION" else "#8B949E"
                    with i4: st.markdown(f'<div class="metric-card"><div class="metric-title">Micro-Whale Detect</div><div class="metric-val" style="color:{w_clr}">{w_type if is_w else "STABLE"}</div></div>', unsafe_allow_html=True)
                    with i5:
                        ipb = st.session_state.get('intraday_price_band')
                        if ipb is not None:
                            ilow, ihigh, iband_pct = ipb
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Pred. Range (±{iband_pct:.2f}%)</div><div class="metric-val" style="font-size:14px;">₹{ilow:,.2f}–{ihigh:,.2f}</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Pred. Range</div><div class="metric-val" style="font-size:14px;">N/A</div></div>', unsafe_allow_html=True)
                    
                    # Elite Features Row
                    st.markdown("<br>", unsafe_allow_html=True)
                    e1, e2, e3 = st.columns(3)
                    sqz_text = " BREAKOUT IMMINENT" if is_squeeze else "Normal Range"
                    sqz_clr = "#FF9900" if is_squeeze else "#8B949E"
                    with e1: st.markdown(f'<div class="metric-card"><div class="metric-title">Volatility Squeeze</div><div class="metric-val" style="color:{sqz_clr}; font-size:18px;">{sqz_text}</div></div>', unsafe_allow_html=True)
                    
                    alpha_text = f"YES (+{stk_ret-idx_ret:.1f}%)" if has_alpha else "NO"
                    alpha_clr = "#58A6FF" if has_alpha else "#8B949E"
                    with e2: st.markdown(f'<div class="metric-card"><div class="metric-title">Alpha Divergence (RS)</div><div class="metric-val" style="color:{alpha_clr}; font-size:18px;">{alpha_text}</div></div>', unsafe_allow_html=True)
                    
                    # 3. Execution Matrix Math (Risk/Reward)
                    sl = pivots['S1'] if price > pivots['P'] else pivots['S2']
                    tp = pivots['R1'] if price > pivots['P'] else pivots['P']
                    if price > sl:
                        rr = (tp - price) / (price - sl) if (price - sl) > 0 else 0
                        rr_text = f"1:{rr:.1f}"
                    else:
                        rr_text = "N/A"
                    with e3: st.markdown(f'<div class="metric-card"><div class="metric-title">Auto R:R Matrix</div><div style="font-size:12px; color:#ccc; margin-top:5px;">SL: ₹{sl:.2f} &nbsp;|&nbsp; TP: ₹{tp:.2f}</div><div class="metric-val" style="font-size:18px; color:#00FF9D;">{rr_text}</div></div>', unsafe_allow_html=True)
                    st.markdown("### Real-Time 5M Chart & Liquidity Zones")
                    
                    # Align chart data perfectly with the dropped NaNs from the AI engine
                    plot_dates = dates[-100:]
                    plot_df = df.loc[plot_dates]
                    plot_vwap = X['VWAP'].iloc[-100:]
                    
                    # Convert dates to strings to remove overnight/weekend gaps on the chart
                    plot_dates_str = plot_dates.strftime('%b %d %H:%M')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=plot_dates_str, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=plot_dates_str, y=plot_vwap, mode='lines', line=dict(color='#00FF9D', width=2), name="VWAP"))
                    
                    # Add Liquidity Pivot Lines
                    fig.add_hline(y=pivots['R1'], line_dash="dash", line_color="#FF4B4B", annotation_text="Resistance 1", annotation_position="top left")
                    fig.add_hline(y=pivots['P'], line_dash="dot", line_color="#58A6FF", annotation_text="Daily Pivot", annotation_position="top left")
                    fig.add_hline(y=pivots['S1'], line_dash="dash", line_color="#00FF9D", annotation_text="Support 1", annotation_position="bottom left")
                    
                    fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), height=500, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Intraday Manual / Legend
                    with st.expander(" How to Read This Chart & Trade", expanded=False):
                        st.markdown("""
                        **The Intraday Chart is your institutional battle map. Here is what the overlays mean:**
                        
                        *   <span style="border-bottom: 3px solid #00FF9D; display: inline-block; width: 30px; margin-bottom: 4px; margin-right: 5px;"></span> **Solid Green Line (VWAP):** The *Volume-Weighted Average Price*. This is the most important line for day traders. If the price is **above** VWAP, institutions are buying (Bullish). If it is **below** VWAP, they are selling (Bearish). Never buy if the price is below VWAP.
                        *   <span style="border-bottom: 3px dotted #58A6FF; display: inline-block; width: 30px; margin-bottom: 4px; margin-right: 5px;"></span> **Blue Dotted Line (Daily Pivot):** The mathematical center of gravity for today. The price will naturally act like a magnet to this line. If the price is hovering here, the market is undecided.
                        *   <span style="border-bottom: 3px dashed #FF4B4B; display: inline-block; width: 30px; margin-bottom: 4px; margin-right: 5px;"></span> **Red Dashed Line (Resistance 1):** The primary ceiling. This is where a massive wall of sellers is sitting. If the price breaks *above* this line with high volume, it is a massive breakout signal.
                        *   <span style="border-bottom: 3px dashed #00FF9D; display: inline-block; width: 30px; margin-bottom: 4px; margin-right: 5px;"></span> **Green Dashed Line (Support 1):** The primary floor. This is where a massive wall of buyers is sitting. If the price breaks *below* this line, it's a breakdown signal.
                        
                        **How to use the Auto R:R Matrix:**
                        The matrix automatically calculates a safe trade setup based on these lines. It tells you where to place your Stop Loss (SL) to protect your capital, and where to place your Take Profit (TP) to cash out.
                        """, unsafe_allow_html=True)
                    
                    # Initialize Paper Trading Session States
                    st.session_state.setdefault('paper_balance', 1000000.0)
                    st.session_state.setdefault('paper_trades', [])
                    st.session_state.setdefault('trade_history', [])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("###  Interactive Day Trading Desk")
                
                    
                    col_p1, col_p2 = st.columns([1, 1])
                    
                    with col_p1:
                        st.markdown(f'<div class="metric-card" style="margin-bottom: 15px;">'
                                    f'<div class="metric-title">Simulator Balance (Cash)</div>'
                                    f'<div class="metric-val" style="color: #00FF9D; font-size: 24px;">₹{st.session_state.paper_balance:,.2f}</div>'
                                    '</div>', unsafe_allow_html=True)
                        
                        # Order Entry Panel
                        with st.container(border=True):
                            st.markdown("#####  Order Entry Panel")
                            t_action = st.radio("Action", ["BUY (Long)", "SELL (Short)"], horizontal=True, key="trade_action_radio")
                            t_qty = st.number_input("Quantity", min_value=1, value=100, step=10, key="trade_qty_input")
                            
                            # Estimate cost/margin
                            est_val = price * t_qty
                            st.markdown(f"<div style='font-size:12px; color:#8b949e; margin-bottom:10px;'>Estimated Value: ₹{est_val:,.2f}</div>", unsafe_allow_html=True)
                            
                            # Buy/Sell targets from pivots
                            sl_val = pivots['S1'] if price > pivots['P'] else pivots['S2']
                            tp_val = pivots['R1'] if price > pivots['P'] else pivots['P']
                            
                            if st.button(" EXECUTE ORDER", use_container_width=True):
                                order_type = "BUY" if "BUY" in t_action else "SELL"
                                
                                # Validate funds if buying
                                if est_val > st.session_state.paper_balance:
                                    st.error("Insufficient funds in simulator balance!")
                                else:
                                    # Add to active positions
                                    st.session_state.paper_trades.append({
                                        'ticker': st.session_state.current_ticker,
                                        'type': order_type,
                                        'entry': price,
                                        'qty': t_qty,
                                        'sl': sl_val,
                                        'tp': tp_val,
                                        'timestamp': datetime.now().strftime('%H:%M:%S')
                                    })
                                    # Deduct cash as collateral/cost
                                    st.session_state.paper_balance -= est_val
                                    st.success(f"Executed {order_type} {t_qty} shares at ₹{price:.2f}!")
                                    st.rerun()
                                    
                    with col_p2:
                        st.markdown('<div style="margin-bottom: 8px; font-weight: bold; color: #8b949e;"> LIVE SYSTEM LOGS</div>', unsafe_allow_html=True)
                        # Fetch and print Live Signal Feed
                        sig_feed = sp.get_intraday_signals(df)
                        
                        feed_html = '<div style="background-color: #0d1117; padding: 15px; border-radius: 8px; border: 1px solid #30363d; font-family: monospace; height: 260px; overflow-y: auto; font-size: 12px; color: #c9d1d9; line-height: 1.5;">'
                        if sig_feed:
                            # Print in reverse (newest first)
                            for s in reversed(sig_feed):
                                feed_html += f'<div style="margin-bottom: 8px;"><span style="color: #8b949e;">[{s["time"]}]</span> <span style="color: {s["color"]}; font-weight: bold;">{s["msg"]}</span></div>'
                        else:
                            feed_html += '<div style="color: #8b949e; text-align: center; margin-top: 100px;">Awaiting next 5m tick for signals...</div>'
                        feed_html += '</div>'
                        st.markdown(feed_html, unsafe_allow_html=True)
                        
                    # Active Positions / Portfolio
                    st.markdown("<br>####  Open Positions", unsafe_allow_html=True)
                    active_pos = st.session_state.paper_trades
                    if not active_pos:
                        st.info("No active open positions. Place an order above to start trading.")
                    else:
                        for idx, pos in enumerate(active_pos):
                            # Resolve price for this specific position's ticker to avoid cross-ticker P&L leaks
                            if pos['ticker'] == st.session_state.current_ticker:
                                pos_price = price
                            else:
                                # Fetch and cache prices for non-active tickers for 60s to maintain elite terminal speed
                                st.session_state.setdefault('price_cache', {})
                                cache = st.session_state.price_cache
                                now_ts = time.time()
                                cache_entry = cache.get(pos['ticker'])
                                
                                if cache_entry and (now_ts - cache_entry['time'] < 60):
                                    pos_price = cache_entry['price']
                                else:
                                    try:
                                        pos_df = yf.download(pos['ticker'], interval="5m", period="1d", progress=False)
                                        if isinstance(pos_df.columns, pd.MultiIndex):
                                            pos_df.columns = pos_df.columns.droplevel(1)
                                        pos_price = float(pos_df['Close'].iloc[-1])
                                        cache[pos['ticker']] = {'price': pos_price, 'time': now_ts}
                                    except:
                                        pos_price = pos['entry'] # Fallback to entry to prevent UI breaks
                            
                            # Calculate dynamic profit
                            if pos['type'] == "BUY":
                                profit = (pos_price - pos['entry']) * pos['qty']
                                roi = (pos_price - pos['entry']) / pos['entry'] * 100
                            else:
                                profit = (pos['entry'] - pos_price) * pos['qty']
                                roi = (pos['entry'] - pos_price) / pos['entry'] * 100
                                
                            prof_clr = "#00FF9D" if profit >= 0 else "#FF4B4B"
                            prof_sign = "+" if profit >= 0 else ""
                            
                            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([1, 1, 1, 1.5, 1])
                            with p_col1: st.markdown(f"<div style='padding-top:5px;'><b>{pos['ticker']}</b> ({pos['type']})</div>", unsafe_allow_html=True)
                            with p_col2: st.markdown(f"<div style='padding-top:5px;'>Qty: {pos['qty']}</div>", unsafe_allow_html=True)
                            with p_col3: st.markdown(f"<div style='padding-top:5px;'>Entry: ₹{pos['entry']:.2f}</div>", unsafe_allow_html=True)
                            with p_col4: st.markdown(f"<div style='padding-top:5px;'>P&L: <span style='color:{prof_clr}; font-weight:bold;'>{prof_sign}₹{profit:,.2f} ({roi:+.2f}%)</span></div>", unsafe_allow_html=True)
                            with p_col5:
                                if st.button("Close", key=f"close_{idx}_{pos['timestamp']}", use_container_width=True):
                                    # Credit back simulator balance (margin + profit)
                                    st.session_state.paper_balance += (pos['entry'] * pos['qty']) + profit
                                    
                                    # Log to history
                                    st.session_state.trade_history.append({
                                        'ticker': pos['ticker'],
                                        'type': pos['type'],
                                        'entry': pos['entry'],
                                        'exit': price,
                                        'qty': pos['qty'],
                                        'pnl': profit,
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    # Remove position
                                    st.session_state.paper_trades.pop(idx)
                                    st.success("Position closed successfully!")
                                    st.rerun()
                        
                        # Trade History Log
                        if st.session_state.trade_history:
                            with st.expander(" Closed Trades Log"):
                                st.dataframe(pd.DataFrame(st.session_state.trade_history))
                else:
                    st.warning("Not enough 5m data available to run the AI engine.")
            except Exception as e:
                st.error(f"Intraday fetch error: {str(e)}")

    elif st.session_state.view_mode == "watchlist":
        sub_tab = st.radio("Watchlist Desk Mode", ["Watchlist Monitor", "Market Screener Grid"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if sub_tab == "Market Screener Grid":
            dashboard_views.render_screener_view()
            ui.render_footer()
            return
            
        # 1. EXECUTIVE HEALTH CHECK
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            h1, h2, h3 = st.columns(3)
            
            # Aggregate Intelligence
            total_upside = 0
            whales_detected = 0
            top_ticker = "NONE"
            top_val = -999
            
            with st.spinner("Compiling Executive Portfolio Intelligence..."):
                if w:
                    try:
                        bulk = yf.download(w, period="2d", progress=False, threads=True)
                        for t in w:
                            try:
                                if isinstance(bulk.columns, pd.MultiIndex):
                                    hist = bulk.xs(t, level=1, axis=1)
                                else:
                                    hist = bulk
                                if not hist.empty and len(hist) >= 2:
                                    hist_df = pd.DataFrame(index=hist.index)
                                    hist_df['Close'] = hist['Close']
                                    hist_df['Volume'] = hist['Volume']
                                    hist_df['Open'] = hist['Open']
                                    hist_df['High'] = hist['High']
                                    hist_df['Low'] = hist['Low']
                                    
                                    is_w, w_type = sp.detect_whales(hist_df)
                                    if is_w and w_type == "ACCUMULATION": whales_detected += 1
                                    
                                    c_p = hist['Close'].iloc[-1]
                                    p_p = hist['Close'].iloc[-2]
                                    diff = ((c_p - p_p) / p_p) * 100
                                    total_upside += diff
                                    if diff > top_val: 
                                        top_val = diff
                                        top_ticker = t
                            except: continue
                    except: pass
            
            avg_health = total_upside / len(w) if w else 0
            h_clr = "#00FF9D" if avg_health >= 0 else "#FF4B4B"
            
            with h1:
                st.markdown(f'<div style="text-align:center;"><div class="metric-title">Portfolio Health</div><div style="font-size:24px; font-weight:800; color:{h_clr}">{avg_health:+.2f}%</div></div>', unsafe_allow_html=True)
            with h2:
                st.markdown(f'<div style="text-align:center;"><div class="metric-title">Whale Activity</div><div style="font-size:24px; font-weight:800; color:#58A6FF;">{whales_detected} Assets</div></div>', unsafe_allow_html=True)
            with h3:
                st.markdown(f'<div style="text-align:center;"><div class="metric-title">Neural Top Pick</div><div style="font-size:24px; font-weight:800; color:#00FF9D;">{top_ticker}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Market Intelligence")
        search_q = st.text_input("Search or Add Ticker (e.g. TATAPOWER.NS, AAPL, BTC-USD)", 
                                 value=st.session_state.search_query,
                                 placeholder="Type ticker and press Enter...")
        
        st.session_state.search_query = search_q.upper()
        search_q = st.session_state.search_query

        if search_q:
            filtered_w = [t for t in w if search_q in t]
            if not filtered_w:
                if st.button(f"Add '{search_q}' to Watchlist"):
                    wm.add_to_watchlist(search_q)
                    st.session_state.search_query = ""
                    st.rerun()
            display_w = filtered_w if filtered_w else [search_q]
        else: display_w = w

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Batch download display list prices
        watchlist_details = {}
        if display_w:
            try:
                bulk_display = yf.download(display_w, period="2d", progress=False, threads=True)
                for t in display_w:
                    try:
                        if isinstance(bulk_display.columns, pd.MultiIndex):
                            hist = bulk_display.xs(t, level=1, axis=1)
                        else:
                            hist = bulk_display
                        if not hist.empty and len(hist) >= 2:
                            p = hist['Close'].iloc[-1]
                            prev_close = hist['Close'].iloc[-2]
                            c = p - prev_close
                            pct = (c / prev_close) * 100
                            watchlist_details[t] = {
                                "price": p,
                                "change": c,
                                "pct": pct
                            }
                    except: pass
            except: pass
            
        col_list = st.columns(3)
        for i, t in enumerate(display_w):
            with col_list[i % 3]:
                try:
                    if t in watchlist_details:
                        det = watchlist_details[t]
                        p = det["price"]
                        c = det["change"]
                        pct = det["pct"]
                        clr = "#3FB950" if c >= 0 else "#F85149"
                        st.markdown(textwrap.dedent(f'''
                            <div class="stock-box" style="margin-bottom:10px; border-radius:8px; padding:20px; background:#161B22; border:1px solid #30363D;">
                                <div class="stock-ticker" style="font-size:10px; color:#8B949E; text-transform:uppercase;">{t}</div>
                                <div class="stock-price" style="font-size:28px; font-weight:800; color:#FFFFFF; margin:5px 0;">₹{p:,.2f}</div>
                                <div class="stock-chg" style="color:{clr}; font-size:13px; font-weight:700;">{c:+.2f} ({pct:+.2f}%)</div>
                            </div>
                        '''), unsafe_allow_html=True)
                        
                        # Connected buttons underneath with space downside
                        st.markdown('<div style="margin-top:5px;"></div>', unsafe_allow_html=True)
                        btn_c1, btn_c2 = st.columns([7, 1])
                        with btn_c1:
                            if st.button(f"Analyze {t.split('.')[0]}", key=f"lt_ana_{t}", use_container_width=True):
                                st.session_state.current_ticker = t
                                st.session_state.view_mode = "analysis"
                                st.rerun()
                        with btn_c2:
                            if st.button("−", key=f"lt_rm_{t}", help=f"Remove {t} from Watchlist", use_container_width=True):
                                wm.remove_from_watchlist(t)
                                st.rerun()
                except: pass

    elif st.session_state.view_mode == "radar":
        st.markdown('<h1 style="color:white; margin-bottom:0; font-size:42px;"> INTRADAY MARKET RADAR</h1><p style="color:#00FF9D; font-weight:600; letter-spacing:1px;">REAL-TIME MULTI-TICKER BATTLEGROUND SCANNER</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not w:
            st.info("Your Watchlist is empty. Add stocks in the Watchlist tab to view them on the Radar!")
        else:
            # Table Header
            table_html = textwrap.dedent("""\
            <style>
            .radar-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
                text-align: left;
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                overflow: hidden;
            }
            .radar-table th {
                background-color: #161b22;
                color: #8b949e;
                padding: 12px 15px;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
                border-bottom: 1px solid #30363d;
            }
            .radar-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #21262d;
                color: #c9d1d9;
            }
            .radar-table tr:hover {
                background-color: #1f242c;
            }
            .badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                display: inline-block;
            }
            </style>
            <table class="radar-table">
                <thead>
                    <tr>
                        <th>Asset Ticker</th>
                        <th>Price</th>
                        <th>Change</th>
                        <th>VWAP Pulse</th>
                        <th>Volatility Squeeze</th>
                        <th>Whale Flow</th>
                    </tr>
                </thead>
                <tbody>
            """)
            
            with st.spinner("Initiating Intraday Radar sweeps across watchlist..."):
                for t in w:
                    try:
                        # 2d 5m data is enough to calculate VWAP and Squeeze and is extremely fast!
                        df_intra = sp.fetch_intraday_data(t, interval="5m", period="2d")
                        metrics = sp.get_radar_metrics(df_intra)
                        
                        price_val = f"₹{metrics['price']:,.2f}"
                        chg_clr = "#00FF9D" if metrics['change_pct'] >= 0 else "#FF4B4B"
                        chg_sign = "+" if metrics['change_pct'] >= 0 else ""
                        chg_val = f"<span style='color:{chg_clr}; font-weight:bold;'>{chg_sign}{metrics['change_pct']:.2f}%</span>"
                        
                        vwap_badge = f"<span class='badge' style='background-color:{metrics['vwap_color']}15; color:{metrics['vwap_color']}; border: 1px solid {metrics['vwap_color']}30;'>{metrics['vwap_status']}</span>"
                        sqz_badge = f"<span class='badge' style='background-color:{metrics['squeeze_color']}15; color:{metrics['squeeze_color']}; border: 1px solid {metrics['squeeze_color']}30;'>{metrics['squeeze_status']}</span>"
                        whale_badge = f"<span class='badge' style='background-color:{metrics['whale_color']}15; color:{metrics['whale_color']}; border: 1px solid {metrics['whale_color']}30;'>{metrics['whale_status']}</span>"
                        
                        table_html += textwrap.dedent(f"""\
                        <tr>
                            <td style="font-weight:bold; color:#ffffff;">{t}</td>
                            <td>{price_val}</td>
                            <td>{chg_val}</td>
                            <td>{vwap_badge}</td>
                            <td>{sqz_badge}</td>
                            <td>{whale_badge}</td>
                        </tr>
                        """)
                    except Exception as e:
                        table_html += textwrap.dedent(f"""\
                        <tr>
                            <td style="font-weight:bold; color:#8b949e;">{t}</td>
                            <td colspan="5" style="color:#8b949e; font-style:italic;">Failed to gather intraday telemetry ({str(e)})</td>
                        </tr>
                        """)
            
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Action quick-jump panel
            st.markdown("<br>### Direct Tactical Deployment", unsafe_allow_html=True)
            cols = st.columns(len(w) if len(w) <= 4 else 4)
            for idx, t in enumerate(w[:8]):
                with cols[idx % len(cols)]:
                    if st.button(f"Deploy to {t}", key=f"radar_jump_{t}", use_container_width=True):
                        st.session_state.current_ticker = t
                        st.session_state.view_mode = "intraday"
                        st.rerun()

    elif st.session_state.view_mode == "home":
        # Renders the premium Sentiment / Fear & Greed page
        st.markdown(textwrap.dedent('''
            <div class="dashboard-header">
                <div class="dashboard-title">Fear & Greed Index</div>
                <div class="dashboard-long-desc">A real-time sentiment tool aggregating market momentum, volatility, and social sentiment patterns to capture shifting emotional states and potential inflection points.</div>
            </div>
        '''), unsafe_allow_html=True)
        dashboard_views.render_fear_greed(_cached_fear_greed())
        
        st.markdown(textwrap.dedent('''
            <div class="dashboard-header">
                <div class="dashboard-title">Market Anomalies (Significant Declines)</div>
                <div class="dashboard-long-desc">Real-time tracking of high-beta and institutional assets experiencing anomalous downward pressure and volume distress.</div>
            </div>
        '''), unsafe_allow_html=True)
        movers = _cached_market_movers()
        m_cols = st.columns(4)
        for i, m in enumerate(movers[:8]):
            with m_cols[i % 4]:
                st.markdown(textwrap.dedent(f'''
                    <div class="anomaly-card">
                        <div class="anomaly-ticker">{m['ticker']}</div>
                        <div class="anomaly-name">{m['name']}</div>
                        <div class="anomaly-change">{m['change']:.2f}%</div>
                        <div class="anomaly-reason">{m['reason']}</div>
                    </div>
                '''), unsafe_allow_html=True)
                if st.button(f"Analyze {m['ticker']}", key=f"ana_m_{m['ticker']}", use_container_width=True):
                    st.session_state.current_ticker = m['ticker']
                    st.session_state.view_mode = "analysis"
                    st.rerun()

    elif st.session_state.view_mode == "portfolio":
        dashboard_views.render_portfolio_view()
        
    elif st.session_state.view_mode == "options":
        sub_tab = st.radio("Options Desk Mode", ["Options Greeks Analysis", "Risk Analytics Dashboard"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        if sub_tab == "Options Greeks Analysis":
            dashboard_views.render_options_view()
        else:
            dashboard_views.render_risk_view(st.session_state.current_ticker or "RELIANCE.NS")
            
    elif st.session_state.view_mode == "ipo":
        ipo_engine.render_ipo_dashboard()
        
    elif st.session_state.view_mode == "rag":
        rag_engine.render_rag_interface()
        
    elif st.session_state.view_mode == "news":
        news_engine.render_news_intelligence_panel()

    else:
        # LANDING PAGE (Fallback to Home)
        st.markdown(textwrap.dedent('''
            <div class="dashboard-header">
                <div class="dashboard-title">Fear & Greed Index</div>
                <div class="dashboard-long-desc">A real-time sentiment tool aggregating market momentum, volatility, and social sentiment patterns to capture shifting emotional states and potential inflection points.</div>
            </div>
        '''), unsafe_allow_html=True)
        dashboard_views.render_fear_greed(_cached_fear_greed())
        
        st.markdown(textwrap.dedent('''
            <div class="dashboard-header">
                <div class="dashboard-title">Market Anomalies (Significant Declines)</div>
                <div class="dashboard-long-desc">Real-time tracking of high-beta and institutional assets experiencing anomalous downward pressure and volume distress.</div>
            </div>
        '''), unsafe_allow_html=True)
        movers = _cached_market_movers()
        m_cols = st.columns(4)
        for i, m in enumerate(movers[:8]):
            with m_cols[i % 4]:
                st.markdown(textwrap.dedent(f'''
                    <div class="anomaly-card">
                        <div class="anomaly-ticker">{m['ticker']}</div>
                        <div class="anomaly-name">{m['name']}</div>
                        <div class="anomaly-change">{m['change']:.2f}%</div>
                        <div class="anomaly-reason">{m['reason']}</div>
                    </div>
                '''), unsafe_allow_html=True)
                if st.button(f"Analyze {m['ticker']}", key=f"ana_m_{m['ticker']}", use_container_width=True):
                    st.session_state.current_ticker = m['ticker']
                    st.session_state.view_mode = "analysis"
                    st.rerun()

    # === FLOATING AI CO-PILOT ASSISTANT ===
    import modules.copilot.copilot as copilot
    
    st.markdown('<div class="floating-assistant-container"></div>', unsafe_allow_html=True)
    if st.button("💬", key="floating_copilot_trigger_btn", help="Open Neural Copilot"):
        st.session_state.show_copilot = not st.session_state.get('show_copilot', False)
        st.rerun()
    
    if st.session_state.get('show_copilot', False):
        st.markdown('<div class="floating-copilot-marker"></div>', unsafe_allow_html=True)
        with st.container():
            c_close1, c_close2 = st.columns([5, 1])
            with c_close2:
                if st.button("❌", key="close_copilot_panel_btn", help="Close Copilot"):
                    st.session_state.show_copilot = False
                    st.rerun()
            with c_close1:
                st.markdown('<div class="copilot-header" style="font-size:14px; font-weight:800; color:#58A6FF; padding-top:4px;">💬 Neural Copilot</div>', unsafe_allow_html=True)
                
            copilot.render_copilot_panel()

    # FINAL FOOTER
    ui.render_footer()
    
    # LIVE FEED LOOP
    if st.session_state.get('live_feed', False) and st.session_state.current_ticker and st.session_state.view_mode == "analysis":
        time.sleep(15)
        st.rerun()

if __name__ == "__main__":
    main()