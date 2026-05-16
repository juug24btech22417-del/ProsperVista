import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* Minimal Header Hide (Don't block sidebar toggle) */
        header, [data-testid="stAppDeployButton"] {
            visibility: hidden !important;
            height: 0 !important;
        }

        /* Global Overrides */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', sans-serif !important;
            background-color: #0B0E11 !important;
            color: #C9D1D9 !important;
        }

        /* Eliminate Top Space */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }

        /* Ticker-style Monospace for Numbers */
        .metric-value, .verdict-text, .footer-text, .news-tag, .index-value {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Technical Trading Cards */
        .glass-card {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 4px !important;
            padding: 20px !important;
            transition: border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: #58A6FF !important;
        }

        /* Market Index Strip */
        .index-strip {
            display: flex;
            gap: 20px;
            background: #0D1117;
            border-bottom: 1px solid #30363D;
            padding: 8px 20px;
            margin: 0rem -5rem 2rem -5rem;
            overflow-x: auto;
            white-space: nowrap;
        }
        .index-item {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #8B949E;
        }
        .index-value {
            color: #FFFFFF;
            font-weight: 700;
            margin-left: 5px;
        }
        .index-up { color: #3FB950; }
        .index-down { color: #F85149; }

        /* Metric Styling */
        .metric-box {
            text-align: left;
            border-left: 3px solid #30363D;
            padding-left: 15px;
        }
        .metric-label {
            font-size: 10px;
            color: #8B949E;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
            font-weight: 700;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #FFFFFF;
        }

        /* Verdict Banners */
        .verdict-banner {
            text-align: center;
            padding: 30px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .verdict-buy {
            background: rgba(35, 134, 54, 0.1) !important;
            color: #3FB950 !important;
            border: 1px solid #238636 !important;
        }
        .verdict-sell {
            background: rgba(218, 54, 51, 0.1) !important;
            color: #F85149 !important;
            border: 1px solid #DA3633 !important;
        }
        .verdict-hold {
            background: rgba(48, 54, 61, 0.5) !important;
            color: #8B949E !important;
            border: 1px solid #30363D !important;
        }

        /* Footer */
        .site-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #161B22;
            border-top: 1px solid #30363D;
            padding: 10px 0;
            text-align: center;
            z-index: 9999;
        }
        
        .stApp { padding-bottom: 60px !important; }
        </style>
    """, unsafe_allow_html=True)

def render_market_strip():
    st.markdown('''
        <div class="index-strip">
            <div class="index-item">NIFTY 50 <span class="index-value index-up">22,475.85 (+0.45%)</span></div>
            <div class="index-item">SENSEX <span class="index-value index-up">73,917.03 (+0.41%)</span></div>
            <div class="index-item">NASDAQ <span class="index-value index-down">16,332.24 (-0.12%)</span></div>
            <div class="index-item">S&P 500 <span class="index-value index-up">5,222.68 (+0.16%)</span></div>
            <div class="index-item">GOLD <span class="index-value index-up">₹72,450 (+1.2%)</span></div>
        </div>
    ''', unsafe_allow_html=True)

def render_footer():
    st.markdown(f'''
    <div class="site-footer">
        <div class="footer-text" style="color: #8B949E; font-size: 10px; letter-spacing: 1px;">
            PROSPER VISTA &copy; 2026 | INSTITUTIONAL EQUITY ANALYTICS | 
            <span style="opacity: 0.6;">Data: Yahoo Finance</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
