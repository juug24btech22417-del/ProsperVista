import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Prosper Vista | AI Stock Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TOTAL UI OVERRIDE: Professional SaaS Dashboard CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #05070a !important;
        color: #e1e1e1 !important;
    }

    /* Hide Streamlit's default elements */
    header, footer, .stAppElementContainer {
        background-color: #05070a !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Professional Dashboard Layout */
    .dashboard-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        padding: 10px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(22, 28, 35, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border: 1px solid rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }

    /* Metric Styling */
    .metric-box {
        text-align: center;
        padding: 15px;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }

    /* Verdict Styling */
    .verdict-banner {
        grid-column: span 4;
        text-align: center;
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .verdict-buy {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        color: #4ade80;
        box-shadow: 0 0 30px rgba(74, 222, 128, 0.2);
    }
    .verdict-sell {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        color: #f87171;
        box-shadow: 0 0 30px rgba(248, 113, 113, 0.2);
    }
    .verdict-hold {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: #94a3b8;
        box-shadow: 0 0 30px rgba(148, 163, 184, 0.1);
    }
    .verdict-text {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 4px;
        margin: 0;
    }
    .verdict-subtext {
        font-size: 16px;
        opacity: 0.8;
        margin-top: 10px;
    }

    .strategy-pill {
        background: rgba(0,0,0,0.3);
        padding: 12px 20px;
        border-radius: 50px;
        display: inline-block;
        font-size: 13px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 15px;
        color: #fff;
    }

    /* Chart Container */
    .chart-container {
        grid-column: span 3;
        background: rgba(22, 28, 35, 0.5);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .sidebar-panel {
        grid-column: span 1;
        background: rgba(22, 28, 35, 0.5);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# ANALYTICS ENGINE
# ==========================================

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return ma + (std * 2), ma - (std * 2)

def fetch_and_prepare_data(ticker, days=730):
    df = yf.download(ticker, period=f"{days//30}mo")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df.iloc[:, :5]

    close = df['Close']
    # Ensure we have 1D arrays by flattening any 2D arrays coming from yfinance
    def flatten(col):
        if isinstance(col, (pd.Series, np.ndarray)):
            return np.array(col).flatten()
        return col

    data = pd.DataFrame({
        'Close': flatten(close),
        'Open': flatten(df['Open']),
        'High': flatten(df['High']),
        'Low': flatten(df['Low']),
        'Volume': flatten(df['Volume'])
    }, index=df.index)
    data['MA7'] = close.rolling(window=7).mean()
    data['MA21'] = close.rolling(window=21).mean()
    data['RSI'] = calculate_rsi(close)
    upper, lower = calculate_bollinger_bands(close)
    data['Upper_Band'], data['Lower_Band'] = upper, lower
    data['Target'] = data['Close'].shift(-1)
    data['Prev_Close'] = data['Close'].shift(1)
    data = data.bfill().ffill()

    features = ['Open', 'Prev_Close', 'High', 'Low', 'Volume', 'MA7', 'MA21', 'RSI', 'Upper_Band', 'Lower_Band']
    X, y = data[features].iloc[:-1], data['Target'].iloc[:-1]
    return X, y, features, data[features].iloc[[-1]], data['Close'].iloc[-1], df

def run_ml_pipeline(X, y, latest_row, model_type="Linear"):
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}[model_type]
    model.fit(X_train_scaled, y_train)

    prediction = model.predict(scaler.transform(latest_row))[0]
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
    r2 = r2_score(y_test, model.predict(X_test_scaled))
    return prediction, rmse, r2, model.coef_

# ==========================================
# MAIN DASHBOARD
# ==========================================

def main():
    # Use sidebar for input only
    st.sidebar.title("Controls")
    ticker = st.sidebar.text_input("Stock Ticker", value="TATAPOWER.NS").upper()
    years = st.sidebar.slider("Data Window", 1, 5, 2)
    model_choice = st.sidebar.selectbox("Model Engine", ["Linear", "Ridge", "Lasso"])
    run_btn = st.sidebar.button("Analyze Market")

    if run_btn:
        with st.spinner("Computing..."):
            res = fetch_and_prepare_data(ticker, days=years*365)
            if not res:
                st.error("Ticker not found.")
                return

            X, y, features, latest_row, current_price, raw_df = res
            pred_price, rmse, r2, coeffs = run_ml_pipeline(X, y, latest_row, model_choice)
            pct_change = ((pred_price - current_price) / current_price) * 100

            # Decision Logic
            if pct_change > 0.5 and r2 > 0.6:
                verdict, v_css, msg = "BUY", "verdict-buy", f"Bullish trend predicted. Target: ₹{pred_price:.2f}"
                strategy = f"Entry: ₹{current_price:.2f} | Target: ₹{pred_price:.2f} | Stop-Loss: ₹{current_price - rmse:.2f}"
            elif pct_change < -0.5 and r2 > 0.6:
                verdict, v_css, msg = "SELL", "verdict-sell", f"Bearish trend predicted. Target: ₹{pred_price:.2f}"
                strategy = "High risk identified. Avoid entry or consider exiting positions."
            else:
                verdict, v_css, msg = "HOLD", "verdict-hold", "Market neutrality detected or low model confidence."
                strategy = "No strong signal. Maintain current positions."

            # RENDER CUSTOM HTML DASHBOARD

            # Construct Metrics HTML as a single string to avoid Streamlit rendering glitches
            metrics_html = f'''
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px;">
                <div class="glass-card metric-box">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">₹{current_price:.2f}</div>
                </div>
                <div class="glass-card metric-box">
                    <div class="metric-label">Predicted Price</div>
                    <div class="metric-value">₹{pred_price:.2f}</div>
                </div>
                <div class="glass-card metric-box">
                    <div class="metric-label">Expected Change</div>
                    <div class="metric-value" style="color: {'#4ade80' if pct_change > 0 else '#f87171'}">{pct_change:+.2f}%</div>
                </div>
                <div class="glass-card metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{r2*100:.1f}%</div>
                </div>
            </div>
            '''

            # Construct Verdict HTML
            verdict_html = f'''
            <div class="verdict-banner {v_css}" style="text-align: center; padding: 40px; border-radius: 20px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1);">
                <div class="verdict-text" style="font-size: 42px; font-weight: 800; letter-spacing: 4px; margin: 0;">{verdict}</div>
                <div class="verdict-subtext" style="font-size: 16px; opacity: 0.8; margin-top: 10px;">{msg}</div>
                <div class="strategy-pill" style="background: rgba(0,0,0,0.3); padding: 12px 20px; border-radius: 50px; display: inline-block; font-size: 13px; border: 1px solid rgba(255,255,255,0.1); margin-top: 15px; color: #fff;">{strategy}</div>
            </div>
            '''

            st.markdown(metrics_html, unsafe_allow_html=True)
            st.markdown(verdict_html, unsafe_allow_html=True)

            # Chart and Influence section
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = go.Figure(data=[go.Candlestick(
                    x=raw_df.index, open=raw_df['Open'], high=raw_df['High'],
                    low=raw_df['Low'], close=raw_df['Close'], name="Price"
                )])
                fig.update_layout(
                    template="plotly_dark", xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(gridcolor='#1f2937'), xaxis=dict(gridcolor='#1f2937')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label" style="text-align:center">Feature Impact</div>', unsafe_allow_html=True)
                feat_df = pd.DataFrame({'Feature': features, 'Influence': coeffs})
                fig_feat = px.bar(feat_df, x='Influence', y='Feature', orientation='h',
                                 template="plotly_dark", color='Influence', color_continuous_scale='RdYlGn')
                fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     margin=dict(l=0, r=0, t=0, b=0), height=300)
                st.plotly_chart(fig_feat, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Data Table
            with st.expander("Detailed Market Data"):
                st.dataframe(raw_df.tail(100), use_container_width=True)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 100px 0;">
                <h1 style="color: white; font-weight: 700; font-size: 48px;">Prosper Vista</h1>
                <p style="color: #8b949e; font-size: 18px;">Institutional-Grade Predictive Analytics for Equity Markets.</p>
                <div style="margin-top: 20px; color: #4b5563;">Configure the ticker in the sidebar to generate analysis.</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
