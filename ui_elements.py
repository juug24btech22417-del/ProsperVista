import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"], footer { display: none !important; }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #0D0E15 !important;
        color: #C9D1D9 !important;
    }
    .block-container { padding-top: 0 !important; padding-bottom: 60px !important; max-width: 100% !important; }

    /* === INDEX STRIP === */
    .index-strip {
        display: flex; gap: 0; background: #0A0B10;
        border-bottom: 1px solid #1E2030;
        padding: 10px 120px 10px 30px; overflow-x: auto;
        white-space: nowrap; margin: 0 -5rem 1.5rem -5rem;
    }
    .index-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px; color: #8892B0;
        padding: 0 16px; border-right: 1px solid #1E2030;
        white-space: nowrap;
    }
    .index-item:last-child { border-right: none; }
    .index-up { color: #00E676; font-weight: 700; }
    .index-down { color: #FF4B4B; font-weight: 700; }

    /* === SECTOR CHIPS === */
    .sector-strip {
        display: flex; gap: 10px; justify-content: center;
        flex-wrap: wrap; margin-bottom: 25px;
    }
    .sector-chip {
        background: #12151E; border: 1px solid #1E2030;
        border-radius: 8px; padding: 12px 20px;
        text-align: center; min-width: 100px;
        transition: border-color 0.2s;
    }
    .sector-chip:hover { border-color: #58A6FF; }
    .sector-label { font-size: 9px; color: #8892B0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 6px; }
    .sector-val { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; }

    /* === ANOMALY CARDS === */
    .anomaly-card {
        background: #12151E; border: 1px solid #1E2030;
        border-left: 4px solid #FF4B4B;
        border-radius: 8px; padding: 18px;
        height: 180px; overflow: hidden;
        transition: border-color 0.2s;
    }
    .anomaly-card:hover { border-color: #FF6B6B; }
    .anomaly-ticker { font-size: 9px; color: #8892B0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .anomaly-name { font-size: 14px; color: #FFFFFF; font-weight: 700; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .anomaly-chg { font-size: 26px; color: #FF4B4B; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-bottom: 8px; }
    .anomaly-reason { font-size: 11px; color: #6B7280; line-height: 1.5; }

    /* === METRIC CARDS === */
    .metric-card {
        background: #12151E; border: 1px solid #1E2030;
        border-radius: 10px; padding: 16px 12px;
        text-align: center; transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #58A6FF; }
    .metric-label { font-size: 9px; color: #8892B0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 8px; }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 800; color: #FFFFFF; }

    /* === VERDICT BANNERS === */
    .verdict-buy { background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.3); border-radius: 10px; padding: 28px; text-align: center; }
    .verdict-sell { background: rgba(255,75,75,0.08); border: 1px solid rgba(255,75,75,0.3); border-radius: 10px; padding: 28px; text-align: center; }
    .verdict-hold { background: rgba(139,146,176,0.08); border: 1px solid rgba(139,146,176,0.3); border-radius: 10px; padding: 28px; text-align: center; }

    /* === STAT CARDS (modules) === */
    .stat-card-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0; }
    .stat-card { background: #12151E; border: 1px solid #1E2030; border-radius: 10px; padding: 14px; text-align: center; transition: all 0.2s; }
    .stat-card:hover { border-color: #58A6FF; transform: translateY(-2px); }
    .stat-card-label { font-size: 9px; color: #8892B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-weight: 700; }
    .stat-card-value { font-size: 18px; font-weight: 800; color: #FFF; font-family: 'JetBrains Mono', monospace; }

    /* === MODULE HEADER === */
    .module-header { font-size: 11px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; padding: 8px 0; margin-bottom: 15px; border-bottom: 1px solid #1E2030; }

    /* === TABLES === */
    .pv-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .pv-table thead th { background: #0A0B10; color: #8892B0; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; padding: 10px 12px; text-align: left; border-bottom: 2px solid #1E2030; }
    .pv-table tbody td { padding: 10px 12px; border-bottom: 1px solid #12151E; color: #C9D1D9; }
    .pv-table tbody tr:hover { background: #12151E; }
    .pv-table .positive { color: #00E676; font-weight: 700; }
    .pv-table .negative { color: #FF4B4B; font-weight: 700; }

    /* === PATTERN / TRADE / GREEK CARDS === */
    .pattern-card { background: #12151E; border: 1px solid #1E2030; border-radius: 8px; padding: 14px; margin-bottom: 8px; border-left: 3px solid #58A6FF; }
    .pattern-card.bullish { border-left-color: #00E676; }
    .pattern-card.bearish { border-left-color: #FF4B4B; }
    .pattern-name { font-size: 14px; font-weight: 700; color: #FFF; margin-bottom: 4px; }
    .pattern-signal { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .pattern-desc { font-size: 11px; color: #8892B0; margin-top: 4px; }

    .trade-card { background: #12151E; border: 1px solid #1E2030; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .trade-card .ticker { font-weight: 700; color: #FFF; font-size: 14px; }
    .trade-card .side-long { color: #00E676; font-size: 10px; font-weight: 700; }
    .trade-card .side-short { color: #FF4B4B; font-size: 10px; font-weight: 700; }
    .trade-card .pnl { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }

    .greeks-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 15px 0; }
    .greek-card { background: #0A0B10; border: 1px solid #1E2030; border-radius: 10px; padding: 14px; text-align: center; }
    .greek-symbol { font-size: 22px; font-weight: 800; color: #58A6FF; font-family: 'JetBrains Mono', monospace; }
    .greek-name { font-size: 9px; color: #8892B0; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .greek-value { font-size: 16px; font-weight: 700; color: #FFF; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }

    .stress-bar { background: #12151E; border: 1px solid #1E2030; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
    .stress-scenario { font-size: 13px; font-weight: 600; color: #FFF; margin-bottom: 4px; }
    .stress-desc { font-size: 10px; color: #8892B0; margin-bottom: 8px; }
    .stress-impact { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #FF4B4B; }
    .stress-fill { height: 6px; border-radius: 3px; background: #FF4B4B; margin-top: 6px; }

    .fib-level { display: flex; justify-content: space-between; padding: 6px 12px; border-bottom: 1px solid #12151E; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .fib-ratio { color: #58A6FF; font-weight: 700; }
    .fib-price { color: #FFF; }

    .fg-score { font-size: 52px; font-weight: 800; font-family: 'JetBrains Mono', monospace; text-align: center; margin-top: 10px; }
    .fg-label { font-size: 14px; font-weight: 700; text-align: center; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px; }
    .fg-component { display: flex; justify-content: space-between; font-size: 11px; color: #8892B0; padding: 6px 0; border-bottom: 1px solid #12151E; }
    .fg-component-bar { height: 4px; border-radius: 2px; margin-top: 4px; background: linear-gradient(90deg,#FF4B4B 0%,#FFB000 25%,#8892B0 50%,#58A6FF 75%,#00E676 100%); }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] { background-color: #0A0B10 !important; border-right: 1px solid #1E2030 !important; }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%; background: #12151E !important; border: 1px solid #1E2030 !important;
        color: #8892B0 !important; border-radius: 6px !important;
        font-weight: 600 !important; font-size: 11px !important;
        text-transform: uppercase !important; letter-spacing: 1px !important;
        padding: 10px !important; margin-top: 4px !important; transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover { border-color: #58A6FF !important; color: #58A6FF !important; }

    /* Inputs */
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div {
        background-color: #12151E !important; border: 1px solid #1E2030 !important;
        color: #C9D1D9 !important; border-radius: 6px !important;
    }

    /* Native metric overrides */
    [data-testid="metric-container"] { background: #12151E; border: 1px solid #1E2030; border-radius: 10px; padding: 14px 12px !important; }
    [data-testid="metric-container"] label { font-size: 9px !important; color: #8892B0 !important; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 18px !important; font-weight: 800 !important; color: #FFF !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #0A0B10; border-bottom: 1px solid #1E2030; gap: 0; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #8892B0 !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important; padding: 12px 20px !important; border-bottom: 2px solid transparent !important; }
    .stTabs [aria-selected="true"] { color: #58A6FF !important; border-bottom: 2px solid #58A6FF !important; }

    /* Footer */
    .site-footer { position: fixed; bottom: 0; left: 0; width: 100%; background: #0A0B10; border-top: 1px solid #1E2030; padding: 10px 0; text-align: center; z-index: 9999; }
    </style>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="site-footer">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3D4460;letter-spacing:1px;">
        PROSPER VISTA v3.0 &copy; 2026 &nbsp;|&nbsp; INSTITUTIONAL EQUITY ANALYTICS &nbsp;|&nbsp;
        <span style="opacity:0.5;">Data: Yahoo Finance</span>
      </span>
    </div>
    """, unsafe_allow_html=True)


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
    clr = '#00E676' if sig=='BULLISH' else '#FF4B4B' if sig=='BEARISH' else '#8892B0'
    st_str = '★' * pattern.get('strength', 1)
    return f'<div class="pattern-card {css}"><div class="pattern-name">{pattern.get("pattern","")}</div><div class="pattern-signal" style="color:{clr}">{sig} {st_str}</div><div class="pattern-desc">{pattern.get("description","")}</div></div>'

def render_stress_bar(scenario, max_shock=-50):
    shock = abs(scenario.get('shock_pct', 0))
    width = min(100, (shock / abs(max_shock)) * 100)
    return f'<div class="stress-bar"><div class="stress-scenario">{scenario["scenario"]}</div><div class="stress-desc">{scenario["description"]}</div><div style="display:flex;justify-content:space-between"><div class="stress-impact">₹{scenario["impact"]:,.0f}</div><div style="font-size:12px;color:#FF4B4B;font-weight:700">{scenario["shock_pct"]}%</div></div><div style="background:#1E2030;border-radius:3px"><div class="stress-fill" style="width:{width}%"></div></div></div>'

def render_greek_card(symbol, name, value):
    return f'<div class="greek-card"><div class="greek-symbol">{symbol}</div><div class="greek-name">{name}</div><div class="greek-value">{value}</div></div>'

def render_trade_card(trade):
    pnl = trade.get('net_pnl', trade.get('pnl', 0))
    clr = '#00E676' if pnl >= 0 else '#FF4B4B'
    side_cls = 'side-long' if trade.get('side') == 'LONG' else 'side-short'
    return f'<div class="trade-card"><div><div class="ticker">{trade.get("ticker","")}</div><div class="{side_cls}">{trade.get("side","LONG")}</div></div><div style="text-align:right"><div class="pnl" style="color:{clr}">₹{pnl:+,.2f}</div><div style="font-size:10px;color:#8892B0">{trade.get("pnl_pct",0):+.2f}%</div></div></div>'

def render_fib_levels(levels):
    html = ''
    for lv in levels:
        tc = '#00E676' if lv['type']=='SUPPORT' else '#FF4B4B'
        html += f'<div class="fib-level"><span class="fib-ratio">{lv["ratio"]}</span><span class="fib-price">₹{lv["price"]:,.2f}</span><span style="color:{tc};font-size:10px;font-weight:700">{lv["type"]}</span></div>'
    return html
