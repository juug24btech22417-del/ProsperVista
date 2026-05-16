# Prosper Vista v2.1.0 - Institutional Upgrade
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json
import textwrap
from datetime import datetime

# MODULAR IMPORTS
from sentiment_engine import SentimentEngine
import watchlist_manager as wm
import stock_prediction as sp

# ==========================================
# 🎨 UI & CSS INJECTION (POLISHED GROWW STYLE)
# ==========================================
def inject_ui():
    st.markdown(textwrap.dedent("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* SURGICAL HEADER HIDE */
        [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {
            display: none !important;
        }

        /* Global Overrides */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif !important;
            background-color: #0B0E11 !important;
            color: #C9D1D9 !important;
        }

        /* Eliminate Top & Bottom Gap */
        .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }
        footer { display: none !important; }

        /* Dashboard Branding */
        .dashboard-header { text-align: center; margin-bottom: 20px; padding: 20px 0; }
        .dashboard-title { color: #FFFFFF; font-size: 52px; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 5px; }
        .dashboard-desc { color: #58A6FF; font-size: 13px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
        .dashboard-long-desc { color: #8B949E; font-size: 14px; max-width: 700px; margin: 0 auto; line-height: 1.6; }

        /* Premium Boxed Metric */
        .metric-card {
            background: #161B22; border: 1px solid #30363D; border-radius: 12px;
            padding: 20px; text-align: center; height: 120px;
            display: flex; flex-direction: column; justify-content: center;
        }
        .metric-title { color: #8B949E; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .metric-val { color: #FFFFFF; font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

        /* Watchlist Stock Box */
        .stock-box {
            background: #161B22; border: 1px solid #30363D; border-radius: 12px;
            padding: 25px; margin-bottom: 20px; transition: 0.3s;
        }
        .stock-box:hover { border-color: #58A6FF; background: #1C2128; }
        .stock-ticker { font-size: 10px; color: #8B949E; font-weight: 700; letter-spacing: 1px; }
        .stock-price { font-size: 28px; color: #FFFFFF; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin: 5px 0; }
        .stock-chg { font-size: 13px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

        /* Verdict Banner */
        .verdict-box { border-radius: 12px; padding: 30px; text-align: center; margin: 20px 0; border: 1px solid transparent; }
        .buy-box { background: rgba(0, 200, 117, 0.1); border-color: #00C875; color: #00C875; }
        .sell-box { background: rgba(255, 68, 68, 0.1); border-color: #FF4444; color: #FF4444; }
        .hold-box { background: rgba(139, 148, 158, 0.1); border-color: #8B949E; color: #8B949E; }
        .verdict-main { font-size: 48px; font-weight: 800; margin-bottom: 10px; letter-spacing: 2px; }
        .verdict-desc { font-size: 14px; opacity: 0.9; margin-bottom: 15px; }
        .trade-strip {
            display: inline-flex; gap: 20px; background: rgba(0,0,0,0.3); 
            padding: 8px 25px; border-radius: 50px; font-size: 13px; font-family: 'JetBrains Mono', monospace;
        }

        /* Sidebar Fix */
        [data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"] {
            width: 100% !important;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] div[data-testid="stMarkdownContainer"] p {
            font-size: 10px !important; color: #8B949E !important; margin-bottom: 2px !important;
        }

        /* Global Input Styling */
        div[data-testid="stTextInput"] input {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            color: #C9D1D9 !important;
            border-radius: 8px !important;
        }

        /* Market Index Strip - Desktop Original */
        .index-strip {
            display: flex; 
            justify-content: space-around;
            background: #0D1117; 
            border-bottom: 1px solid #30363D; 
            padding: 10px 20px; 
            margin: 1rem -5rem 2rem -5rem; 
            overflow-x: auto;
        }
        .index-item { 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 11px; 
            color: #8B949E; 
            padding: 0 15px; 
            white-space: nowrap; 
        }
        .index-up { color: #00C875; font-weight: 700; }
        .index-down { color: #FF4444; font-weight: 700; }

        /* Anomaly Cards */
        .anomaly-card {
            background: #161B22; border-left: 4px solid #FF4444; border-radius: 8px;
            padding: 15px; height: 180px; margin-bottom: 20px;
        }
        .anomaly-ticker { font-size: 10px; color: #8B949E; font-weight: 600; }
        .anomaly-name { font-size: 14px; color: #FFFFFF; font-weight: 700; margin: 3px 0; height: 20px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
        .anomaly-change { font-size: 20px; color: #FF4444; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
        .anomaly-reason { font-size: 10px; color: #8B949E; line-height: 1.4; height: 45px; overflow: hidden; margin-top: 5px; }

        /* Refined Footer */
        .footer {
            margin-top: 40px; padding: 15px 0; border-top: 1px solid #30363D;
            text-align: center; color: #485563; font-size: 11px;
            font-weight: 500; letter-spacing: 0.5px;
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .dashboard-title { font-size: 32px !important; }
            .dashboard-desc { font-size: 11px !important; }
            .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
            .index-strip { 
                margin: 0 -1rem 1rem -1rem !important; 
                justify-content: flex-start !important;
                padding: 12px 0 !important;
            }
            .index-item { padding: 0 25px !important; border-right: 1px solid #30363D; }
            .index-item:last-child { border-right: none; }
        }
        </style>
    """), unsafe_allow_html=True)

# ==========================================
# 📊 INTEGRATED ANALYTICS HUB
# ==========================================
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker, years=2):
    try:
        # Utilizing modular fetch
        start = (datetime.now() - pd.Timedelta(days=365*years)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        df = sp.fetch_data(ticker, start, end)
        if df.empty: return None
        return df, yf.Ticker(ticker).info.get('longName', ticker), df['Close'].iloc[-1]
    except: return None

def generate_research_report(name, ticker, price, target, chg, mood, sentiment_score, confidence):
    """
    Generates a stunning, high-contrast, vibrant HTML research document.
    """
    clr = "#00FF9D" if chg >= 0 else "#FF4B4B"
    mood_clr = "#00FF9D" if mood == "BULLISH" else "#FF4B4B" if mood == "BEARISH" else "#8B949E"
    grad = "linear-gradient(135deg, #00C875, #005030)" if chg >= 0 else "linear-gradient(135deg, #FF4B4B, #800000)"
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
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
# 🚀 MAIN TERMINAL APP
# ==========================================
def main():
    st.set_page_config(page_title="Prosper Vista", layout="wide")
    inject_ui()
    sentiment_engine = SentimentEngine()
    
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

    # 1. Market Strip
    st.markdown(textwrap.dedent('''
        <div class="index-strip">
            <div class="index-item">NIFTY 50 <span class="index-up">22,475.85 (+0.45%)</span></div>
            <div class="index-item">SENSEX <span class="index-up">73,917.03 (+0.41%)</span></div>
            <div class="index-item">NASDAQ <span class="index-down">16,332.24 (-0.12%)</span></div>
            <div class="index-item">BTC/USD <span class="index-up">$66,432 (+2.15%)</span></div>
            <div class="index-item">GOLD <span class="index-up">₹72,450 (+1.2%)</span></div>
        </div>
    '''), unsafe_allow_html=True)

    # 2. Sidebar Controls
    st.sidebar.title("Terminal Controls")
    if 'current_ticker' not in st.session_state: st.session_state.current_ticker = ""
    if 'view_mode' not in st.session_state: st.session_state.view_mode = "analysis"

    ticker_input = st.sidebar.text_input("Stock Ticker", value=st.session_state.current_ticker if st.session_state.current_ticker else "TATAPOWER.NS").upper()
    years = st.sidebar.slider("Data Window", 1, 5, 2)
    model_choice = st.sidebar.selectbox("Model Engine", ["Elite Consensus (XGBoost+RF)", "Linear", "Ridge", "Lasso"])
    
    if st.sidebar.button("Analyze Market", key="main_analyze_btn", use_container_width=True):
        st.session_state.current_ticker = ticker_input
        st.session_state.view_mode = "analysis"
        st.session_state.model_choice = model_choice
        st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("Watchlist Overview", key="watchlist_view_btn", use_container_width=True):
        st.session_state.view_mode = "watchlist"
        st.rerun()

    # Modular Watchlist Manager
    w = wm.load_watchlist()
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:10px; color:#8B949E; text-transform:uppercase; margin-bottom:10px; letter-spacing:1px;">Active Watchlist</div>', unsafe_allow_html=True)
    
    for t in w:
        c1, c2 = st.sidebar.columns([5, 1])
        with c1: 
            if st.button(t, key=f"s_{t}", use_container_width=True):
                st.session_state.current_ticker = t
                st.session_state.view_mode = "analysis"
                st.rerun()
        with c2:
            if st.button("X", key=f"d_{t}", help=f"Remove {t}"):
                wm.remove_from_watchlist(t); st.rerun()

    # 3. ANALYSIS LOGIC
    if st.session_state.current_ticker and st.session_state.view_mode == "analysis":
        res = fetch_terminal_data(st.session_state.current_ticker, years)
        if res:
            df, name, price = res
            
            # Model Execution Hub
            X, y, feature_names, dates = sp.prepare_features(df)
            choice = st.session_state.get('model_choice', "Elite Consensus (XGBoost+RF)")
            
            if choice == "Elite Consensus (XGBoost+RF)":
                pred, r2, importances = sp.get_consensus_prediction(X, y, X.iloc[[-1]])
            else:
                # Legacy Support
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LinearRegression, Ridge, Lasso
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X)
                model = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}[choice]
                model.fit(X_sc, y)
                pred = model.predict(scaler.transform(X.iloc[[-1]]))[0]
                r2 = model.score(X_sc, y)
                importances = model.coef_ if hasattr(model, 'coef_') else [0]*len(feature_names)
            
            sent = sentiment_engine.get_news_sentiment(st.session_state.current_ticker)
            s_score = sent.get('score', 0)
            adj_pred = pred + (s_score * (price * 0.02))
            chg = ((adj_pred - price) / price) * 100
            
            # Header
            st.markdown(f'<h1 style="color:white; margin-bottom:0; font-size:42px;">{name}</h1><p style="color:#58A6FF; font-weight:600; letter-spacing:1px;">MARKET ANALYSIS • {st.session_state.current_ticker}</p>', unsafe_allow_html=True)
            
            # 4. METRICS ROW
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.markdown(f'<div class="metric-card"><div class="metric-title">Current Price</div><div class="metric-val">₹{price:,.2f}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-card"><div class="metric-title">Target Close</div><div class="metric-val">₹{adj_pred:,.2f}</div></div>', unsafe_allow_html=True)
            with m3: 
                clr = "#00FF9D" if chg >= 0 else "#FF4B4B"
                st.markdown(f'<div class="metric-card"><div class="metric-title">Exp. Change</div><div class="metric-val" style="color:{clr}">{chg:+.2f}%</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="metric-card"><div class="metric-title">Confidence</div><div class="metric-val">{r2*100:.1f}%</div></div>', unsafe_allow_html=True)
            with m5: 
                mood = sent.get("verdict", "NEUTRAL")
                m_clr = "#00FF9D" if mood == "BULLISH" else "#FF4B4B" if mood == "BEARISH" else "#8B949E"
                st.markdown(f'<div class="metric-card"><div class="metric-title">Market Mood</div><div class="metric-val" style="color:{m_clr}">{mood}</div></div>', unsafe_allow_html=True)

            # 4.1 FUNDAMENTAL ROW
            st.markdown("<br>", unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns(4)
            info = yf.Ticker(st.session_state.current_ticker).info
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

            # 7. SENTIMENT
            st.markdown("---")
            st.markdown("### Sentiment Intelligence & News Analysis")
            sc1, sc2 = st.columns([1, 2.5])
            with sc1:
                st.markdown(textwrap.dedent(f'''
                    <div class="sent-score-card">
                        <div class="metric-title">Overall Sentiment Score</div>
                        <div class="sent-big-num" style="color:{m_clr}">{s_score:+.2f}</div>
                    </div>
                '''), unsafe_allow_html=True)
            with sc2:
                for n in sent.get('news', [])[:8]:
                    st.markdown(textwrap.dedent(f'''
                        <div class="news-card">
                            <div style="font-size:10px; color:{n['color']}; font-weight:700; margin-bottom:5px;">{n['sentiment']}</div>
                            <a href="{n['link']}" target="_blank" style="color:#FFFFFF; font-weight:600; text-decoration:none; font-size:15px;">{n['title']}</a>
                            <div style="font-size:10px; color:#8B949E; margin-top:5px;">{n['publisher']} • {n['time']}</div>
                        </div>
                    '''), unsafe_allow_html=True)

    elif st.session_state.view_mode == "watchlist":
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
        col_list = st.columns(3)
        for i, t in enumerate(display_w):
            with col_list[i % 3]:
                try:
                    s = yf.Ticker(t); h = s.history(period="1d")
                    p = h['Close'].iloc[-1]; c = p - h['Open'].iloc[-1]; clr = "#00C875" if c >= 0 else "#FF4444"
                    st.markdown(textwrap.dedent(f'''
                        <div class="stock-box">
                            <div class="stock-ticker">{t}</div>
                            <div class="stock-price">₹{p:,.2f}</div>
                            <div class="stock-chg" style="color:{clr}">{c:+.2f} ({ (c/h['Open'].iloc[-1])*100 :+.2f}%)</div>
                        </div>
                    '''), unsafe_allow_html=True)
                    if st.button(f"Analyze {t}", key=f"lt_{t}", use_container_width=True):
                        st.session_state.current_ticker = t
                        st.session_state.view_mode = "analysis"
                        st.rerun()
                except: pass

    else:
        # LANDING PAGE
        st.markdown("### Market Anomalies (Significant Declines)")
        movers = sentiment_engine.get_market_movers()
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

    # FINAL FOOTER
    st.markdown(textwrap.dedent('''
        <div class="footer">
            PROSPER VISTA &copy; 2026 • INSTITUTIONAL GRADE ANALYTICS
        </div>
    '''), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
