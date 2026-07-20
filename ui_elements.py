import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* Minimal Header Hide (Don't block sidebar toggle) */
        header {
            background-color: transparent !important;
        }
        [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] {
            display: none !important;
        }

        /* Global Overrides */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', sans-serif !important;
            background-color: #0B0E11 !important;
            color: #C9D1D9 !important;
        }

        /* Custom styled buttons globally */
        .stButton > button, .stDownloadButton > button {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            color: #C9D1D9 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover,
        .stButton > button:focus, .stDownloadButton > button:focus,
        .stButton > button:active, .stDownloadButton > button:active {
            border-color: #58A6FF !important;
            color: #58A6FF !important;
            background-color: #1C2128 !important;
            box-shadow: 0 0 15px rgba(88, 166, 255, 0.15) !important;
        }

        /* Eliminate Top & Bottom Gap */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 8rem !important;
        }
        footer {
            display: none !important;
        }

        /* Ticker-style Monospace for Numbers */
        .metric-value, .verdict-text, .footer-text, .news-tag, .index-value, .metric-val, .metric-val-num, .anomaly-change {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Technical Trading / Glass Cards */
        .glass-card {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 8px !important;
            padding: 20px !important;
            transition: border-color 0.25s ease, transform 0.25s ease;
        }
        .glass-card:hover {
            border-color: #58A6FF !important;
            transform: translateY(-2px);
        }

        /* Watchlist Stock Box */
        .stock-box {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 12px !important;
            padding: 25px !important;
            margin-bottom: 20px !important;
            transition: all 0.3s ease !important;
        }
        .stock-box:hover {
            border-color: #58A6FF !important;
            background: #1C2128 !important;
            transform: translateY(-2px);
        }
        .stock-ticker {
            font-size: 10px !important;
            color: #8B949E !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
        }
        .stock-price {
            font-size: 28px !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            font-family: 'JetBrains Mono', monospace !important;
            margin: 5px 0 !important;
        }
        .stock-chg {
            font-size: 13px !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Market Index Strip */
        .index-strip {
            display: flex;
            justify-content: space-around;
            background: #0D1117;
            border-bottom: 1px solid #30363D;
            padding: 10px 20px;
            margin: 0rem -5rem 2rem -5rem;
            overflow-x: auto;
            white-space: nowrap;
        }
        .index-item {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #8B949E;
            padding: 0 15px;
            white-space: nowrap;
        }
        .index-value {
            font-weight: 700;
            margin-left: 5px;
        }
        .index-up { color: #00C875; }
        .index-down { color: #FF4444; }

        /* Metric Styling */
        .metric-card {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
            padding: 15px 5px !important;
            border-radius: 16px !important;
            text-align: center !important;
            height: 110px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            transition: border-color 0.25s ease !important;
        }
        .metric-card:hover {
            border-color: #58A6FF !important;
        }
        .metric-title, .metric-label {
            color: #8B949E !important;
            font-size: 8px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            margin-bottom: 8px !important;
        }
        .metric-val, .metric-value {
            color: #FFFFFF !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', monospace !important;
            line-height: 1.2 !important;
            text-align: center !important;
        }
        .metric-box {
            text-align: left;
            border-left: 3px solid #30363D;
            padding-left: 15px;
        }

        /* Metric Grid — CSS native grid, sidesteps Streamlit column collapse on mobile */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 8px 0;
        }

        /* Performance Card Styling */
        .performance-card {
            background-color: #0D1117;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
        }
        .perf-title {
            color: #FFFFFF;
            font-size: 20px;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 20px;
        }
        .perf-row {
            margin-bottom: 24px;
        }
        .perf-labels {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 6px;
        }
        .lbl-small {
            font-size: 11px;
            color: #8B949E;
            display: block;
            text-transform: uppercase;
        }
        .val-mono {
            font-size: 16px;
            font-weight: 700;
            color: #FFFFFF;
            font-family: 'JetBrains Mono', monospace;
        }
        .lbl-center {
            font-size: 12px;
            color: #8B949E;
            font-weight: 600;
            padding-bottom: 2px;
        }
        .slider-track-container {
            position: relative;
            height: 16px;
        }
        .slider-track {
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(90deg, #FF4B4B 0%, #FFA500 50%, #00FF9D 100%);
        }
        .slider-pointer {
            position: absolute;
            top: 6px;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-bottom: 8px solid #FFFFFF;
        }
        .returns-list {
            margin-top: 20px;
        }
        .ret-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #21262D;
        }
        .ret-item:last-child {
            border-bottom: none;
        }
        .ret-item span:first-child {
            font-size: 14px;
            color: #C9D1D9;
            font-weight: 500;
        }
        .ret-val {
            font-size: 14px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        /* ── Analyst Price Target Card ── */
        .analyst-card {
            background: #0D1117;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px 24px;
            margin: 16px 0;
        }
        .analyst-title {
            font-size: 16px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 16px;
        }
        .consensus-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 14px;
        }
        .analyst-targets-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 12px;
        }
        .analyst-target-item {
            background: #161B22;
            border: 1px solid #21262D;
            border-radius: 8px;
            padding: 10px 14px;
        }
        .analyst-target-label {
            font-size: 10px;
            color: #8B949E;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }
        .analyst-target-val {
            font-size: 16px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .analyst-upside {
            font-size: 11px;
            margin-top: 3px;
        }

        /* ── Pattern Alert Card ── */
        .pattern-alerts-card {
            background: #0D1117;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px 24px;
            margin: 16px 0;
        }
        .pattern-alerts-title {
            font-size: 16px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 14px;
        }
        .pattern-alert-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid #21262D;
        }
        .pattern-alert-item:last-child { border-bottom: none; }
        .pattern-signal-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .pattern-name {
            font-size: 14px;
            font-weight: 600;
            color: #C9D1D9;
            flex: 1;
        }
        .pattern-strength {
            font-size: 12px;
            color: #FFA500;
            letter-spacing: 1px;
        }
        .pattern-date {
            font-size: 11px;
            color: #8B949E;
            font-family: 'JetBrains Mono', monospace;
        }
        .overall-signal-badge {
            display: inline-block;
            padding: 5px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 14px;
        }

        /* ── Peer Comparison Table ── */
        .peer-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }
        .peer-table th {
            background: #161B22;
            color: #8B949E;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #30363D;
        }
        .peer-table td {
            padding: 10px 14px;
            border-bottom: 1px solid #21262D;
            color: #C9D1D9;
        }
        .peer-table .peer-active-row td {
            background: rgba(88, 166, 255, 0.07);
            color: #FFFFFF;
            font-weight: 600;
        }
        .peer-table tr:last-child td { border-bottom: none; }

        /* Verdict Banners */
        .verdict-box, .verdict-banner {
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
            border: 1px solid transparent;
        }
        .buy-box, .verdict-buy {
            background: rgba(0, 200, 117, 0.1) !important;
            border-color: #00C875 !important;
            color: #00C875 !important;
        }
        .sell-box, .verdict-sell {
            background: rgba(255, 68, 68, 0.1) !important;
            border-color: #FF4444 !important;
            color: #FF4444 !important;
        }
        .hold-box, .verdict-hold {
            background: rgba(139, 148, 158, 0.1) !important;
            border-color: #8B949E !important;
            color: #8B949E !important;
        }
        .verdict-main {
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 10px;
            letter-spacing: 2px;
        }
        .verdict-desc {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 15px;
        }
        .trade-strip {
            display: inline-flex;
            gap: 20px;
            background: rgba(0,0,0,0.3);
            padding: 8px 25px;
            border-radius: 50px;
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* === STAT CARDS === */
        .stat-card-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0; }
        .stat-card { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 14px; text-align: center; transition: all 0.25s; }
        .stat-card:hover { border-color: #58A6FF; transform: translateY(-2px); }
        .stat-card-label { font-size: 9px; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-weight: 700; }
        .stat-card-value { font-size: 18px; font-weight: 800; color: #FFF; font-family: 'JetBrains Mono', monospace; }

        /* === GLOBAL DASHBOARD HEADER === */
        .dashboard-header {
            border-left: 4px solid #58A6FF;
            padding-left: 20px;
            margin: 15px 0 25px 0;
            background: linear-gradient(90deg, rgba(22, 27, 34, 0.4) 0%, rgba(22, 27, 34, 0) 100%);
            padding-top: 10px;
            padding-bottom: 10px;
            border-radius: 0 8px 8px 0;
        }
        .dashboard-title {
            font-size: 38px !important;
            font-weight: 800 !important;
            line-height: 1.1 !important;
            background: linear-gradient(90deg, #58A6FF 0%, #00FF9D 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            letter-spacing: -0.5px !important;
            margin-bottom: 2px !important;
        }
        .dashboard-desc {
            font-size: 13px !important;
            font-weight: 700 !important;
            color: #00FF9D !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            margin-bottom: 8px !important;
        }
        .dashboard-long-desc {
            font-size: 14px !important;
            color: #8B949E !important;
            max-width: 900px !important;
            line-height: 1.5 !important;
            font-weight: 400 !important;
        }

        /* === MODULE HEADER === */
        .module-header { font-size: 11px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; padding: 8px 0; margin-bottom: 15px; border-bottom: 1px solid #30363D; }

        /* === TABLES === */
        .pv-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
        .pv-table thead th { background: #0D1117; color: #8B949E; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; padding: 10px 12px; text-align: left; border-bottom: 2px solid #30363D; }
        .pv-table tbody td { padding: 10px 12px; border-bottom: 1px solid #161B22; color: #C9D1D9; }
        .pv-table tbody tr:hover { background: #161B22; }
        .pv-table .positive { color: #00C875; font-weight: 700; }
        .pv-table .negative { color: #FF4444; font-weight: 700; }

        /* === PATTERN / TRADE / GREEK CARDS === */
        .pattern-card { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 14px; margin-bottom: 8px; border-left: 3px solid #58A6FF; }
        .pattern-card.bullish { border-left-color: #00C875; }
        .pattern-card.bearish { border-left-color: #FF4444; }
        .pattern-name { font-size: 14px; font-weight: 700; color: #FFF; margin-bottom: 4px; }
        .pattern-signal { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
        .pattern-desc { font-size: 11px; color: #8B949E; margin-top: 4px; }

        .trade-card { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .trade-card .ticker { font-weight: 700; color: #FFF; font-size: 14px; }
        .trade-card .side-long { color: #00C875; font-size: 10px; font-weight: 700; }
        .trade-card .side-short { color: #FF4444; font-size: 10px; font-weight: 700; }
        .trade-card .pnl { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }

        .greeks-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 15px 0; }
        .greek-card { background: #0D1117; border: 1px solid #30363D; border-radius: 8px; padding: 14px; text-align: center; }
        .greek-symbol { font-size: 22px; font-weight: 800; color: #58A6FF; font-family: 'JetBrains Mono', monospace; }
        .greek-name { font-size: 9px; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
        .greek-value { font-size: 16px; font-weight: 700; color: #FFF; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }

        .stress-bar { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
        .stress-scenario { font-size: 13px; font-weight: 600; color: #FFF; margin-bottom: 4px; }
        .stress-desc { font-size: 10px; color: #8B949E; margin-bottom: 8px; }
        .stress-impact { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #FF4444; }
        .stress-fill { height: 6px; border-radius: 3px; background: #FF4444; margin-top: 6px; }

        .fib-level { display: flex; justify-content: space-between; padding: 6px 12px; border-bottom: 1px solid #161B22; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
        .fib-ratio { color: #58A6FF; font-weight: 700; }
        .fib-price { color: #FFF; }

        .fg-score { font-size: 52px; font-weight: 800; font-family: 'JetBrains Mono', monospace; text-align: center; margin-top: 10px; }
        .fg-label { font-size: 14px; font-weight: 700; text-align: center; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px; }
        .fg-component { display: flex; justify-content: space-between; font-size: 11px; color: #8B949E; padding: 6px 0; border-bottom: 1px solid #161B22; }
        .fg-component-bar { height: 4px; border-radius: 2px; margin-top: 4px; background: linear-gradient(90deg,#FF4444 0%,#FFB000 25%,#8B949E 50%,#58A6FF 75%,#00C875 100%); }

        /* Anomaly Cards */
        .anomaly-card {
            background: #161B22 !important;
            border-left: 4px solid #FF4444 !important;
            border-radius: 8px !important;
            padding: 15px !important;
            height: 180px !important;
            margin-bottom: 20px !important;
        }
        .anomaly-ticker {
            font-size: 10px !important;
            color: #8B949E !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
        }
        .anomaly-name {
            font-size: 14px !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            margin: 3px 0 !important;
            height: 20px !important;
            overflow: hidden !important;
            white-space: nowrap !important;
            text-overflow: ellipsis !important;
        }
        .anomaly-change {
            font-size: 20px !important;
            color: #FF4444 !important;
            font-weight: 800 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        .anomaly-reason {
            font-size: 10px !important;
            color: #8B949E !important;
            line-height: 1.4 !important;
            height: 45px !important;
            overflow: hidden !important;
            margin-top: 5px !important;
        }

        /* === SIDEBAR === */
        [data-testid="stSidebar"] {
            background-color: #0D1117 !important;
            border-right: 1px solid #30363D !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #161B22 0%, #0D1117 100%) !important;
            border: 1px solid #30363D !important;
            color: #C9D1D9 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            font-size: 10px !important;
            padding: 12px !important;
            margin-top: 10px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #58A6FF !important;
            box-shadow: 0 0 20px rgba(88, 166, 255, 0.15) !important;
            transform: translateY(-2px) !important;
            color: #58A6FF !important;
        }

        /* Watchlist Buttons inside Columns in Sidebar */
        [data-testid="stSidebar"] [data-testid="column"] .stButton > button,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button {
            width: 100% !important;
            height: 38px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin-top: 0px !important;
            font-size: 11px !important;
            border-radius: 6px !important;
            letter-spacing: 0.5px !important;
            text-transform: none !important;
            background: #161B22 !important;
        }
        [data-testid="stSidebar"] [data-testid="column"] .stButton > button:hover,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
            border-color: #58A6FF !important;
            box-shadow: 0 0 10px rgba(88, 166, 255, 0.1) !important;
            transform: none !important;
        }

        .status-card {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
            padding: 15px !important;
            border-radius: 12px !important;
            margin-bottom: 25px !important;
        }
        .status-item {
            display: flex !important;
            justify-content: space-between !important;
            font-size: 10px !important;
            color: #8B949E !important;
            margin-bottom: 5px !important;
        }
        .status-dot {
            height: 6px !important;
            width: 6px !important;
            background: #00C875 !important;
            border-radius: 50% !important;
            display: inline-block !important;
            box-shadow: 0 0 10px #00C875 !important;
            margin-right: 5px !important;
        }

        /* Inputs */
        div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div {
            background-color: #161B22 !important; border: 1px solid #30363D !important;
            color: #C9D1D9 !important; border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stTextInput"] input:focus, div[data-testid="stSelectbox"] > div:focus-within {
            border-color: #58A6FF !important;
            box-shadow: 0 0 15px rgba(88, 166, 255, 0.2) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { background: #0D1117; border-bottom: 1px solid #30363D; gap: 0; }
        .stTabs [data-baseweb="tab"] { background: transparent !important; color: #8B949E !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important; padding: 12px 20px !important; border-bottom: 2px solid transparent !important; }
        .stTabs [aria-selected="true"] { color: #58A6FF !important; border-bottom: 2px solid #58A6FF !important; }

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

        /* Sentiment Intelligence & News Cards */
        .sent-score-card {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 10px !important;
            padding: 16px 20px !important;
            text-align: center !important;
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
            transition: all 0.3s ease !important;
        }
        .sent-score-card:hover {
            border-color: #58A6FF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 20px rgba(88, 166, 255, 0.12) !important;
        }
        .sent-big-num {
            font-size: 38px !important;
            font-weight: 800 !important;
            font-family: 'JetBrains Mono', monospace !important;
            margin-top: 6px !important;
        }
        .news-card {
            background: transparent !important;
            border: none !important;
            border-left: 3px solid #58A6FF !important;
            border-bottom: 1px solid #21262D !important;
            border-radius: 0 !important;
            padding: 4px 0 8px 14px !important;
            margin-bottom: 10px !important;
            transition: all 0.25s ease !important;
            display: block !important;
            text-decoration: none !important;
        }
        .news-card:hover {
            border-left-color: #00FF9D !important;
            transform: translateX(4px) !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        .news-sentiment-tag {
            font-size: 9px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            padding: 3px 8px !important;
            border-radius: 4px !important;
            display: inline-block !important;
            margin-bottom: 8px !important;
        }
        .news-title-link {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            text-decoration: none !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
            display: block !important;
            margin-bottom: 6px !important;
            transition: color 0.2s ease !important;
        }
        .news-title-link:hover {
            color: #58A6FF !important;
        }
        .news-meta {
            font-size: 10px !important;
            color: #8B949E !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* === MOBILE RESPONSIVENESS AND OVERLAP CORRECTION === */
        @media (max-width: 768px) {
            /* Fixes for Performance, Analyst, and Pattern Cards on Mobile */
            .performance-card, .analyst-card, .pattern-alerts-card {
                padding: 16px !important;
                background-color: #0D1117 !important;
                border: 1px solid #30363D !important;
                width: 100% !important;
                box-sizing: border-box !important;
                border-radius: 12px !important;
                display: block !important;
            }
            .perf-labels, .ret-item {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                justify-content: space-between !important;
                align-items: center !important;
                width: 100% !important;
            }
            .perf-labels > div {
                width: auto !important;
                flex: 0 1 auto !important;
            }
            .analyst-targets-grid {
                display: flex !important;
                flex-direction: column !important;
                gap: 12px !important;
            }
            .val-mono, .ret-val {
                display: inline-block !important;
                text-align: right !important;
            }
            .lbl-small, .ret-item span:first-child {
                display: inline-block !important;
            }
            .lbl-center {
                display: inline-block !important;
                text-align: center !important;
                flex: 1 !important;
            }
            .slider-track-container, .slider-track, .returns-list {
                width: 100% !important;
            }

            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
            }
            .dashboard-title {
                font-size: 26px !important;
            }
            .dashboard-desc {
                font-size: 10px !important;
                letter-spacing: 1px !important;
            }
            .dashboard-long-desc {
                font-size: 12px !important;
            }
            .index-strip {
                margin: 0rem 0rem 1.5rem 0rem !important;
                padding: 10px 5px !important;
            }
            .stat-card-row {
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)) !important;
                gap: 8px !important;
            }
            .stat-card {
                padding: 10px 5px !important;
            }
            .stat-card-value {
                font-size: 14px !important;
            }
            .metric-card {
                height: auto !important;
                min-height: 90px !important;
                padding: 10px 5px !important;
            }
            .metric-grid {
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 8px !important;
            }
            .metric-val, .metric-value {
                font-size: 15px !important;
            }
            .greeks-grid {
                grid-template-columns: repeat(3, 1fr) !important;
                gap: 6px !important;
            }
            .greek-card {
                padding: 8px 4px !important;
            }
            .greek-symbol {
                font-size: 16px !important;
            }
            .greek-value {
                font-size: 12px !important;
            }
            .anomaly-card {
                height: auto !important;
                min-height: 180px !important;
            }
            .anomaly-reason {
                height: auto !important;
                max-height: 80px !important;
            }
            .site-footer {
                position: relative !important;
                margin-top: 40px !important;
                border-top: 1px solid #30363D !important;
            }
            .stApp {
                padding-bottom: 20px !important;
            }
        }

        @media (max-width: 480px) {
            .greeks-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }
            .dashboard-title {
                font-size: 22px !important;
            }
            .verdict-main {
                font-size: 32px !important;
            }
        }
        
        /* === FLOATING CHAT WIDGET === */
        /* Styled Streamlit Button targeted by next-sibling wrapper */
        div[data-testid="element-container"]:has(.floating-assistant-container) + div[data-testid="element-container"] button {
            position: fixed !important;
            bottom: 25px !important;
            right: 25px !important;
            width: 60px !important;
            height: 60px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #58A6FF 0%, #00FF9D 100%) !important;
            border: none !important;
            color: #0B0E11 !important;
            font-size: 26px !important;
            box-shadow: 0 8px 32px rgba(0, 255, 157, 0.4) !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 999999 !important;
        }
        div[data-testid="element-container"]:has(.floating-assistant-container) + div[data-testid="element-container"] button:hover {
            transform: scale(1.15) rotate(5deg) !important;
            box-shadow: 0 12px 40px rgba(0, 255, 157, 0.6) !important;
        }
        
        /* Styled Streamlit Container targeted by sibling wrapper */
        div[data-testid="element-container"]:has(.floating-copilot-marker) + div[data-testid="element-container"] {
            position: fixed !important;
            bottom: 95px !important;
            right: 25px !important;
            width: 420px !important;
            height: 600px !important;
            background: #0d1117ea !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid #30363d !important;
            border-radius: 20px !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.8) !important;
            z-index: 999998 !important;
            padding: 20px !important;
            overflow: hidden !important;
            display: block !important;
        }
        div[data-testid="element-container"]:has(.floating-copilot-marker) + div[data-testid="element-container"] > div[data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        
        .copilot-header {
            flex-shrink: 0 !important;
            margin-bottom: 12px !important;
        }
        
        .copilot-messages-container {
            flex-grow: 1 !important;
            overflow-y: auto !important;
            margin-bottom: 12px !important;
            padding-right: 5px !important;
        }
        
        .copilot-input-container {
            flex-shrink: 0 !important;
        }

        /* Make Streamlit Chat Messages inside the Copilot container extremely compact and premium */
        .copilot-messages-container [data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 4px 0px !important;
        }
        .copilot-messages-container [data-testid="stChatMessageAvatar"] {
            width: 20px !important;
            height: 20px !important;
        }
        .copilot-messages-container [data-testid="stChatMessageContent"] {
            font-size: 12px !important;
            padding: 8px 12px !important;
            border-radius: 12px !important;
            background-color: #161b2299 !important;
            border: 1px solid #30363D !important;
        }
        
        @media (max-width: 480px) {
            .floating-copilot-window {
                width: calc(100% - 40px) !important;
                right: 20px !important;
                left: 20px !important;
                height: 70vh !important;
            }
        }

        /* Top Navigation Menu Uniformity & Auto-wrapping Flex Grid */
        .top-nav-container [data-testid="stHorizontalBlock"],
        .top-nav-container .stHorizontalBlock {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        .top-nav-container [data-testid="column"] {
            min-width: 110px !important;
            width: auto !important;
            flex: 1 1 110px !important;
            margin: 0 !important;
        }
        .top-nav-container [data-testid="column"] button {
            height: 52px !important;
            min-height: 52px !important;
            font-size: 13px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            padding: 4px 8px !important;
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown(f'''
    <div class="site-footer">
        <div class="footer-text" style="color: #8B949E; font-size: 10px; letter-spacing: 1px;">
            PROSPER VISTA &copy; 2026 | INSTITUTIONAL EQUITY ANALYTICS | 
            <span style="opacity: 0.6;">Data: Yahoo Finance</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def render_stat_card(label, value, color="#FFFFFF"):
    return f'<div class="stat-card"><div class="stat-card-label">{label}</div><div class="stat-card-value" style="color:{color}">{value}</div></div>'

def render_stat_row(cards):
    html = '<div class="stat-card-row">'
    for c in cards:
        html += render_stat_card(c[0], c[1], c[2] if len(c) > 2 else "#FFFFFF")
    return html + '</div>'

def render_pattern_card(pattern):
    sig = pattern.get('signal', 'NEUTRAL')
    css = sig.lower() if sig in ['BULLISH','BEARISH','NEUTRAL'] else 'neutral'
    clr = '#3FB950' if sig=='BULLISH' else '#F85149' if sig=='BEARISH' else '#8B949E'
    st_str = '[' + str(pattern.get('strength', 1)) + '/5]'
    return f'<div class="pattern-card {css}"><div class="pattern-name">{pattern.get("pattern","")}</div><div class="pattern-signal" style="color:{clr}">{sig} {st_str}</div><div class="pattern-desc">{pattern.get("description","")}</div></div>'

def render_stress_bar(scenario, max_shock=-50):
    shock = abs(scenario.get('shock_pct', 0))
    width = min(100, (shock / abs(max_shock)) * 100)
    return f'<div class="stress-bar"><div class="stress-scenario">{scenario["scenario"]}</div><div class="stress-desc">{scenario["description"]}</div><div style="display:flex;justify-content:space-between"><div class="stress-impact">₹{scenario["impact"]:,.0f}</div><div style="font-size:12px;color:#F85149;font-weight:700">{scenario["shock_pct"]}%</div></div><div style="background:#30363D;border-radius:3px"><div class="stress-fill" style="width:{width}%"></div></div></div>'

def render_greek_card(symbol, name, value):
    return f'<div class="greek-card"><div class="greek-symbol">{symbol}</div><div class="greek-name">{name}</div><div class="greek-value">{value}</div></div>'

def render_trade_card(trade):
    pnl = trade.get('net_pnl', trade.get('pnl', 0))
    clr = '#3FB950' if pnl >= 0 else '#F85149'
    side_cls = 'side-long' if trade.get('side') == 'LONG' else 'side-short'
    return f'<div class="trade-card"><div><div class="ticker">{trade.get("ticker","")}</div><div class="{side_cls}">{trade.get("side","LONG")}</div></div><div style="text-align:right"><div class="pnl" style="color:{clr}">₹{pnl:+,.2f}</div><div style="font-size:10px;color:#8B949E">{trade.get("pnl_pct",0):+.2f}%</div></div></div>'

def render_fib_levels(levels):
    html = ''
    for lv in levels:
        tc = '#3FB950' if lv['type']=='SUPPORT' else '#F85149'
        html += f'<div class="fib-level"><span class="fib-ratio">{lv["ratio"]}</span><span class="fib-price">₹{lv["price"]:,.2f}</span><span style="color:{tc};font-size:10px;font-weight:700">{lv["type"]}</span></div>'
    return html