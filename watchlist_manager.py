import streamlit as st
import json
import os
import yfinance as yf
import pandas as pd

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except:
            return ["RELIANCE.NS", "TCS.NS", "NVDA", "AAPL"]
    return ["RELIANCE.NS", "TCS.NS", "NVDA", "AAPL"]

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f)

def add_to_watchlist(ticker):
    watchlist = load_watchlist()
    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        save_watchlist(watchlist)
        return True
    return False

def remove_from_watchlist(ticker):
    watchlist = load_watchlist()
    if ticker in watchlist:
        watchlist.remove(ticker)
        save_watchlist(watchlist)
        return True
    return False

def render_watchlist_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Market Watchlist")
    
    # Add new stock
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        new_ticker = st.text_input("Add to Watchlist", placeholder="e.g. INF_Y.NS", label_visibility="collapsed").upper()
    with col2:
        if st.button("➕"):
            if add_to_watchlist(new_ticker):
                st.toast(f"Added {new_ticker}")
                st.rerun()

    watchlist = load_watchlist()
    
    selected_ticker = None
    for ticker in watchlist:
        col_t, col_r = st.sidebar.columns([4, 1])
        with col_t:
            if st.button(f"🔍 {ticker}", key=f"btn_{ticker}", use_container_width=True):
                selected_ticker = ticker
        with col_r:
            if st.button("🗑️", key=f"del_{ticker}"):
                remove_from_watchlist(ticker)
                st.rerun()
    
    return selected_ticker

def render_watchlist_grid(watchlist):
    if not watchlist:
        st.info("Watchlist Empty.")
        return

    st.markdown("### 📋 Watchlist")
    
    for ticker in watchlist:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty: continue
            
            price = hist['Close'].iloc[-1]
            prev_close = hist['Open'].iloc[-1]
            change = price - prev_close
            pct_change = (change / prev_close) * 100
            
            color = "#3FB950" if change >= 0 else "#F85149"
            
            st.markdown(f'''
                <div class="glass-card" style="border-left: 3px solid {color}; padding: 12px; margin-bottom: 10px;">
                    <div style="font-size: 11px; color: #8B949E; font-weight: 700;">{ticker}</div>
                    <div style="font-size: 18px; font-weight: 700; color: #FFFFFF; margin: 4px 0;">₹{price:.2f}</div>
                    <div style="color: {color}; font-family: 'JetBrains Mono'; font-size: 12px;">
                        {pct_change:+.2f}%
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            if st.button(f"Analyze", key=f"grid_btn_{ticker}", use_container_width=True):
                st.session_state.current_ticker = ticker
                st.session_state.view_mode = "analysis"
                st.rerun()
        except Exception:
            pass
