# ProsperVista v3.0 — Dashboard View Renderers
# Renders UI for all 7 new modules to keep app.py clean

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import textwrap

import portfolio_simulator as ps
import options_engine as oe
import risk_engine as re_eng
import pattern_engine as pe
import correlation_engine as ce
import backtesting_engine as be
import screener_engine as se
import ui_elements as ui


def render_portfolio_view():
    """Render the Portfolio Simulator dashboard."""
    st.markdown('<div class="module-header">Virtual Portfolio Simulator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Portfolio Simulator Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>Asset Allocation</b>: Compares holdings across capital weights to ensure optimal diversification and reduce systematic exposure.</li>
          <li><b>Cash Reserve Balance</b>: Monitors dynamic purchasing power to deploy capital tactically during key market corrections.</li>
          <li><b>Simulated PnL Metrics</b>: Tracks unrealized gains or losses and win-loss ratios to quantify the statistical expectancy of your trade setups.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    summary = ps.get_portfolio_summary()
    pv = summary["portfolio_value"]
    ic = summary["initial_capital"]
    tr = summary["total_return_pct"]
    tr_clr = "#3FB950" if tr >= 0 else "#F85149"
    cards = [
        ("Portfolio Value", f"₹{pv:,.0f}", tr_clr),
        ("Cash Balance", f"₹{summary['cash_balance']:,.0f}", "#FFFFFF"),
        ("Unrealized P&L", f"₹{summary['total_unrealized_pnl']:+,.0f}", "#3FB950" if summary['total_unrealized_pnl'] >= 0 else "#F85149"),
        ("Realized P&L", f"₹{summary['total_realized_pnl']:+,.0f}", "#3FB950" if summary['total_realized_pnl'] >= 0 else "#F85149"),
        ("Win Rate", f"{summary['win_rate']}%", "#58A6FF"),
        ("Total Return", f"{tr:+.2f}%", tr_clr),
    ]
    st.markdown(ui.render_stat_row(cards), unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["OPEN TRADE", "POSITIONS", "TRADE JOURNAL"])
    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker = st.text_input("Ticker", "RELIANCE.NS", key="port_ticker").upper()
            qty = st.number_input("Quantity", 1, 10000, 10, key="port_qty")
        with c2:
            entry = st.number_input("Entry Price (₹)", 1.0, 100000.0, 100.0, key="port_entry")
            sl = st.number_input("Stop Loss (₹)", 0.0, 100000.0, 0.0, key="port_sl")
        with c3:
            tp = st.number_input("Take Profit (₹)", 0.0, 100000.0, 0.0, key="port_tp")
            side = st.selectbox("Side", ["LONG", "SHORT"], key="port_side")
        if st.button("Open Position", use_container_width=True, key="port_open"):
            result = ps.open_position(ticker, qty, entry, sl or None, tp or None, side)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Opened {side} position: {qty}x {ticker} @ ₹{entry}")
                st.rerun()

    with t2:
        positions = ps.get_open_positions()
        if not positions:
            st.info("No open positions. Open a trade to get started.")
        for pos in positions:
            clr = "#3FB950" if pos["pnl_pct"] >= 0 else "#F85149"
            st.markdown(f'''<div class="trade-card slide-in">
                <div><div class="ticker">{pos["ticker"]}</div>
                <div class="side-{'long' if pos['side']=='LONG' else 'short'}">{pos["side"]} • {pos["quantity"]} shares</div></div>
                <div style="text-align:right;">
                <div class="pnl" style="color:{clr}">₹{pos["unrealized_pnl"]:+,.2f}</div>
                <div style="font-size:10px;color:#8B949E;">Entry ₹{pos["entry_price"]:,.2f} → ₹{pos["current_price"]:,.2f}</div></div>
            </div>''', unsafe_allow_html=True)
            if st.button(f"Close #{pos['id']}", key=f"close_{pos['id']}"):
                ps.close_position(pos['id'], pos['current_price'])
                st.rerun()

    with t3:
        trades = ps.get_closed_trades()
        if not trades:
            st.info("No closed trades yet.")
        for t in reversed(trades[-20:]):
            st.markdown(ui.render_trade_card(t), unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Reset Portfolio (₹10L)", key="port_reset"):
        ps.reset_portfolio()
        st.rerun()


def render_options_view():
    """Render the Options Greeks Calculator dashboard."""
    st.markdown('<div class="module-header">Options Greeks Calculator (Black-Scholes)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Options Greeks Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>Delta (Δ)</b>: Measures sensitivity to stock price moves. A Delta of 0.60 means the option price is expected to rise by ₹0.60 for every ₹1.00 increase in the underlying stock. (Also used as a proxy for the probability of expiring in-the-money).</li>
          <li><b>Gamma (Γ)</b>: Measures the acceleration of Delta. High Gamma means Delta changes rapidly on stock moves, indicating high speed risk near the strike price.</li>
          <li><b>Theta (Θ)</b>: The daily time-decay rate. A Theta of -1.50 means your option loses ₹1.50 in value every single day, even if the stock price doesn't move.</li>
          <li><b>Vega (V)</b>: Volatility sensitivity. A Vega of 2.40 means the option price rises by ₹2.40 for every 1% increase in Implied Volatility (IV).</li>
          <li><b>Rho (ρ)</b>: Interest rate sensitivity. Measures price change for a 1% change in risk-free rates.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Stock Price (₹)", 1.0, 100000.0, 1000.0, key="opt_s")
        K = st.number_input("Strike Price (₹)", 1.0, 100000.0, 1000.0, key="opt_k")
    with c2:
        T = st.number_input("Time to Expiry (days)", 1, 365, 30, key="opt_t") / 365.0
        r = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 6.5, key="opt_r") / 100.0
    with c3:
        sigma = st.number_input("Volatility (%)", 1.0, 200.0, 25.0, key="opt_sig") / 100.0
        opt_type = st.selectbox("Option Type", ["call", "put"], key="opt_type")

    # Live auto-calculation on input change (no button needed)
    greeks = oe.calculate_greeks(S, K, T, r, sigma, opt_type)
    symbols = [("Δ", "Delta", greeks["delta"]), ("Γ", "Gamma", greeks["gamma"]),
               ("Θ", "Theta", greeks["theta"]), ("V", "Vega", greeks["vega"]),
               ("ρ", "Rho", greeks["rho"])]
    html = '<div class="greeks-grid">'
    for sym, name, val in symbols:
        html += ui.render_greek_card(sym, name, f"{val:.4f}")
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(f'<div class="stat-card" style="text-align:center;margin:15px 0;"><div class="stat-card-label">Fair Value ({opt_type.upper()})</div><div class="stat-card-value" style="color:#58A6FF;">₹{greeks["price"]:.4f}</div></div>', unsafe_allow_html=True)

    # Payoff Diagram
    payoff_df, premium = oe.generate_payoff_data(S, K, T, r, sigma, opt_type, "buy")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=payoff_df['stock_price'], y=payoff_df['payoff'], mode='lines', line=dict(color='#58A6FF', width=3), name='P&L'))
    fig.add_hline(y=0, line_dash="dash", line_color="#30363D")
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=30,b=0), title="Payoff at Expiration")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_risk_view(ticker=""):
    """Render the Risk Analytics dashboard."""
    st.markdown('<div class="module-header">Risk Analytics Engine</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Risk Analytics Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>Value-at-Risk (VaR)</b>: Quantifies maximum expected loss over a specific horizon at a given confidence interval (95% or 99%). A 95% historical VaR of 3% means there is a 5% chance of losing 3% or more in a single session.</li>
          <li><b>Maximum Drawdown (MDD)</b>: Measures the worst peak-to-trough historical drop. High MDD signals vulnerability to sudden tail-risk shocks.</li>
          <li><b>Beta Telemetry</b>: Systematic volatility index relative to benchmark. A Beta of 1.25 indicates 25% higher volatility compared to market index.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    risk_ticker = st.text_input("Ticker for Risk Analysis", value=ticker or "RELIANCE.NS", key="risk_ticker").upper()

    if st.button("Run Risk Analysis", use_container_width=True, key="risk_run"):
        with st.spinner("Computing VaR, CVaR, Drawdown..."):
            var_data = re_eng.calculate_all_var(risk_ticker)
            dd_stats = re_eng.max_drawdown_stats(risk_ticker)
            risk_adj = re_eng.risk_adjusted_metrics(risk_ticker)

        cards = [
            ("Historical VaR (95%)", f"{var_data['historical_var']:.2f}%", "#F85149"),
            ("Parametric VaR", f"{var_data['parametric_var']:.2f}%", "#F85149"),
            ("Monte Carlo VaR", f"{var_data['monte_carlo_var']:.2f}%", "#F85149"),
            ("Max Drawdown", f"{dd_stats['max_drawdown']:.1f}%", "#F85149"),
            ("Beta", f"{risk_adj['beta']:.3f}", "#58A6FF"),
            ("Jensen's Alpha", f"{risk_adj['jensens_alpha']:.2f}%", "#3FB950" if risk_adj['jensens_alpha'] > 0 else "#F85149"),
        ]
        st.markdown(ui.render_stat_row(cards), unsafe_allow_html=True)

        # Drawdown Chart
        dd_df = re_eng.calculate_drawdown_series(risk_ticker)
        if not dd_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dd_df['date'], y=dd_df['drawdown'], fill='tozeroy', fillcolor='rgba(248,81,73,0.2)', line=dict(color='#F85149', width=2), name='Drawdown'))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=30,b=0), title="Drawdown Analysis")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Stress Test
        st.markdown("### Stress Test Scenarios")
        stress = re_eng.run_stress_test(1000000)
        for s in stress[:5]:
            st.markdown(ui.render_stress_bar(s), unsafe_allow_html=True)


def render_patterns_view(df=None, ticker=""):
    """Render the Pattern Recognition dashboard."""
    st.markdown('<div class="module-header">Technical Pattern Recognition</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Pattern Recognition Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>Candlestick Pattern Reversals</b>: Automatically scans technical feeds to detect exhaustion patterns like Hammers or Bullish Engulfing bars indicating structural pivots.</li>
          <li><b>Support & Resistance channels</b>: Detects congestion price floors and ceiling zones where volume clusters create natural price boundaries.</li>
          <li><b>Fibonacci Levels</b>: Identifies retracement bounds (e.g., 38.2%, 61.8%) where corrective flows frequently lose momentum and resume their primary trend.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if df is None or df.empty:
        st.info("Run an analysis first to see pattern detection.")
        return

    report = pe.generate_pattern_report(df)
    sig = report['overall_signal']
    sig_clr = "#3FB950" if sig == "BULLISH" else "#F85149" if sig == "BEARISH" else "#8B949E"
    st.markdown(f'<div style="text-align:center;margin:15px 0;"><span style="font-size:28px;font-weight:800;color:{sig_clr};">{sig}</span><br><span style="font-size:11px;color:#8B949E;">OVERALL TECHNICAL SIGNAL • {report["bullish_patterns"]} Bullish / {report["bearish_patterns"]} Bearish</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("#### Candlestick Patterns Detected")
        if report['patterns']:
            for p in report['patterns'][:8]:
                st.markdown(ui.render_pattern_card(p), unsafe_allow_html=True)
        else:
            st.info("No significant patterns detected in recent candles.")
    with c2:
        st.markdown("#### Fibonacci Retracement")
        if report['fibonacci']:
            st.markdown(ui.render_fib_levels(report['fibonacci']), unsafe_allow_html=True)
        sr = report['support_resistance']
        if sr['support']:
            st.markdown("#### Support Levels")
            for s in sr['support'][:3]:
                st.markdown(f'<div class="fib-level"><span style="color:#3FB950;">SUPPORT</span><span class="fib-price">₹{s:,.2f}</span></div>', unsafe_allow_html=True)
        if sr['resistance']:
            st.markdown("#### Resistance Levels")
            for r in sr['resistance'][:3]:
                st.markdown(f'<div class="fib-level"><span style="color:#F85149;">RESISTANCE</span><span class="fib-price">₹{r:,.2f}</span></div>', unsafe_allow_html=True)

    # BB Squeeze
    bb = report['bb_squeeze']
    squeeze_clr = "#FFB000" if bb['in_squeeze'] else "#3FB950"
    st.markdown(f'<div class="stat-card" style="margin:15px 0;"><div class="stat-card-label">Bollinger Squeeze</div><div class="stat-card-value" style="color:{squeeze_clr};font-size:16px;">{"SQUEEZE ACTIVE — Breakout Imminent (" + str(bb["squeeze_duration"]) + " days)" if bb["in_squeeze"] else "No Squeeze — Normal Volatility"}</div></div>', unsafe_allow_html=True)


def render_correlation_view(watchlist):
    """Render the Correlation Heatmap dashboard."""
    st.markdown('<div class="module-header">Correlation & Diversification Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Correlation Matrix Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>Pearson Correlation Coef. (-1.0 to +1.0)</b>: A value of +1.0 indicates perfect synchronized movement, whereas -1.0 implies a perfect negative correlation (natural hedge). Pairs below 0.3 are optimal for constructing diversified portfolios.</li>
          <li><b>Diversification Index</b>: Aggregates overall cross-correlation metrics. Higher scores suggest minimal factor overlap, preventing catastrophic systemic portfolio drawdowns.</li>
          <li><b>Beta Index</b>: Measures isolated systematic risk coefficients against benchmarks to identify capital hedges.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tickers = st.text_input("Tickers (comma-separated)", ",".join(watchlist[:6]) if watchlist else "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS", key="corr_tickers")
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    if st.button("Build Correlation Matrix", use_container_width=True, key="corr_build") and len(ticker_list) >= 2:
        with st.spinner("Fetching correlation data..."):
            corr_matrix = ce.build_correlation_matrix(ticker_list)
            div_score = ce.portfolio_diversification_score(ticker_list)

        if not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdYlGn', zmid=0, text=np.round(corr_matrix.values, 2), texttemplate='%{text}', textfont={"size": 11}))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            cards = [
                ("Diversification Score", f"{div_score['score']:.0f}/100", "#3FB950" if div_score['score'] > 60 else "#F85149"),
                ("Avg Correlation", f"{div_score['avg_correlation']:.3f}", "#FFFFFF"),
                ("Most Correlated", f"{div_score['most_correlated'][0][:6]}/{div_score['most_correlated'][1][:6]}", "#FFB000"),
            ]
            st.markdown(ui.render_stat_row(cards), unsafe_allow_html=True)

    # Beta Calculator
    st.markdown("### Beta Calculator")
    bc1, bc2 = st.columns(2)
    with bc1:
        beta_ticker = st.text_input("Stock", "TATAMOTORS.NS", key="beta_t")
    with bc2:
        bench = st.selectbox("Benchmark", ["^NSEI", "^GSPC", "^IXIC"], key="beta_bench")
    if st.button("Calculate Beta", key="beta_calc"):
        beta = ce.calculate_beta(beta_ticker, bench)
        st.markdown(ui.render_stat_row([("Beta", f"{beta['beta']:.3f}", "#58A6FF"), ("R²", f"{beta['r_squared']:.3f}", "#FFFFFF"), ("Alpha", f"{beta['alpha']:.2f}%", "#3FB950" if beta['alpha'] > 0 else "#F85149")]), unsafe_allow_html=True)


def render_backtest_view():
    """Render the Backtesting Engine dashboard with persistent results and simultaneous comparison table."""
    st.markdown('<div class="module-header">Strategy Backtesting Engine</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Backtesting Desk Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>CAGR vs Buy & Hold</b>: Compares strategy annualized compound growth against passive holding. Superior strategy performance indicates positive alpha.</li>
          <li><b>Max Strategy Drawdown</b>: Verifies peak-to-trough drawdowns under historic stress periods to gauge risk-reward viability.</li>
          <li><b>Win Rate & Profit Factor</b>: A profit factor > 1.25 is required to establish positive expectancy in live institutional execution environments.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        bt_ticker = st.text_input("Ticker", "TATAPOWER.NS", key="bt_ticker").upper()
    with c2:
        bt_strategy = st.selectbox("Strategy", list(be.STRATEGIES.keys()), key="bt_strat")
    with c3:
        bt_period = st.selectbox("Backtest Period", ["1y", "2y", "3y", "5y"], index=1, key="bt_period")

    # Initialise session state keys for persistent backtest results
    if "bt_result_cache" not in st.session_state:
        st.session_state.bt_result_cache = None
    if "bt_comparison_cache" not in st.session_state:
        st.session_state.bt_comparison_cache = None
    if "bt_last_key" not in st.session_state:
        st.session_state.bt_last_key = ""

    current_key = f"{bt_ticker}_{bt_strategy}_{bt_period}"

    if st.button("Run Backtesting Suite", use_container_width=True, key="bt_run") or st.session_state.bt_last_key == current_key:
        if st.session_state.bt_last_key != current_key:
            with st.spinner(f"Running full backtesting suite on {bt_ticker}..."):
                st.session_state.bt_result_cache = be.run_backtest(bt_ticker, bt_strategy, bt_period)
                st.session_state.bt_comparison_cache = be.compare_strategies(bt_ticker, bt_period)
                st.session_state.bt_last_key = current_key

        result = st.session_state.bt_result_cache
        comparison = st.session_state.bt_comparison_cache

        if result and "error" in result:
            st.error(result["error"])
            return

        if result:
            m = result["metrics"]
            ret_clr = "#3FB950" if m["total_return"] > 0 else "#F85149"
            alpha_clr = "#3FB950" if m["alpha"] > 0 else "#F85149"
            cards = [
                ("Total Return", f"{m['total_return']:+.2f}%", ret_clr),
                ("CAGR", f"{m['cagr']:.2f}%", ret_clr),
                ("Max Drawdown", f"{m['max_drawdown']:.1f}%", "#F85149"),
                ("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}", "#58A6FF"),
                ("Win Rate", f"{m['win_rate']}%", "#FFFFFF"),
                ("Alpha vs B&H", f"{m['alpha']:+.2f}%", alpha_clr),
            ]
            st.markdown(ui.render_stat_row(cards), unsafe_allow_html=True)

            # Equity Curve
            eq = result["equity_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq['date'], y=eq['equity'], mode='lines', line=dict(color='#58A6FF', width=2), fill='tozeroy', fillcolor='rgba(88,166,255,0.1)', name='Strategy'))
            fig.add_hline(y=m['initial_capital'], line_dash="dash", line_color="#30363D")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=30,b=0), title=f"Equity Growth Path ({bt_strategy})")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Trade Log
            with st.expander(f"Trade Log ({m['total_trades']} trades)", expanded=False):
                for t in result["trades"][-20:]:
                    clr = "#3FB950" if t['pnl'] > 0 else "#F85149"
                    st.markdown(f'<div class="trade-card"><div><div class="ticker">{t["side"]}</div><div style="font-size:10px;color:#8B949E;">{t["entry_date"]} → {t["exit_date"]}</div></div><div style="text-align:right;"><div class="pnl" style="color:{clr}">₹{t["pnl"]:+,.0f}</div><div style="font-size:10px;color:#8B949E;">₹{t["entry_price"]:,.0f} → ₹{t["exit_price"]:,.0f}</div></div></div>', unsafe_allow_html=True)

        # Strategy Comparison
        if comparison is not None and not comparison.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:16px;font-weight:700;color:#fff;margin-bottom:12px;">Cross-Strategy Performance Matrix</div>', unsafe_allow_html=True)
            st.dataframe(comparison[["strategy", "total_return", "cagr", "max_drawdown", "sharpe_ratio", "win_rate", "profit_factor", "buy_hold_return", "alpha"]].style.format({"total_return": "{:+.2f}%", "cagr": "{:.2f}%", "max_drawdown": "{:.1f}%", "sharpe_ratio": "{:.2f}", "win_rate": "{:.1f}%", "profit_factor": "{:.2f}", "buy_hold_return": "{:.2f}%", "alpha": "{:+.2f}%"}), use_container_width=True)


def render_screener_view():
    """Render the Market Screener dashboard."""
    st.markdown('<div class="module-header">Market Screener</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
      <div style="font-size: 13px; color: #58A6FF; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
        Market Screener Guide: What to Check
      </div>
      <div style="font-size: 13px; color: #C9D1D9; line-height: 1.6;">
        <ul style="margin: 0; padding-left: 20px;">
          <li><b>Volume Ratio</b>: A ratio > 1.0 indicates the asset is trading on higher-than-average volume. This indicates high institutional interest or news-driven moves.</li>
          <li><b>RSI (Relative Strength Index)</b>: Values below 35 indicate oversold conditions (potential reversal candidates), while values above 65 signify overbought conditions (potential profit-taking targets).</li>
          <li><b>From 52W High</b>: High-growth momentum stocks tend to hover within 0% to -10% of their 52-week highs. Values close to 0% signify breakout candidates.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        screen_name = st.selectbox("Pre-built Screen", list(se.PREBUILT_SCREENS.keys()), key="scr_name")
        st.caption(se.PREBUILT_SCREENS[screen_name]["description"])
    with c2:
        universe = st.selectbox("Stock Universe", list(se.UNIVERSES.keys()), key="scr_uni")

    if st.button("Run Screen", use_container_width=True, key="scr_run"):
        progress = st.progress(0, text="Scanning...")
        def update_progress(current, total):
            progress.progress(current / total, text=f"Scanning {current}/{total}...")
        with st.spinner(""):
            results = se.run_screen(screen_name, universe=universe, progress_callback=update_progress)
        progress.empty()

        if not results:
            st.warning("No stocks matched the filter criteria.")
            return

        st.success(f"Found {len(results)} matching stocks")
        # Build results table
        rows = ""
        for r in results:
            chg_clr = "positive" if r['change_1d'] >= 0 else "negative"
            rows += f"<tr><td style='font-weight:700;color:#FFFFFF;'>{r['ticker']}</td><td>₹{r['price']:,.2f}</td><td class='{chg_clr}'>{r['change_1d']:+.2f}%</td><td>{r['rsi']:.0f}</td><td>{r['volume_ratio']:.1f}x</td><td class='{chg_clr}'>{r['change_20d']:+.1f}%</td><td>{r['from_52w_high']:.0f}%</td></tr>"

        st.markdown(f'''<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; width: 100%;"><table class="pv-table"><thead><tr>
            <th>Ticker</th><th>Price</th><th>1D Chg</th><th>RSI</th><th>Vol Ratio</th><th>20D Chg</th><th>From 52W High</th>
        </tr></thead><tbody>{rows}</tbody></table></div>''', unsafe_allow_html=True)

        # Quick analyze buttons
        st.markdown("#### Quick Analyze")
        btn_cols = st.columns(min(5, len(results)))
        for i, r in enumerate(results[:5]):
            with btn_cols[i]:
                if st.button(r['ticker'].split('.')[0], key=f"scr_ana_{r['ticker']}"):
                    st.session_state.current_ticker = r['ticker']
                    st.session_state.view_mode = "analysis"
                    st.rerun()


def render_fear_greed(fg_data):
    """Render the Fear & Greed Index widget. Accepts pre-computed data dict."""
    fg = fg_data
    st.markdown(f'''
        <div style="text-align:center; margin:20px 0;">
            <div class="fg-score" style="color:{fg['color']}">{fg['score']:.0f}</div>
            <div class="fg-label" style="color:{fg['color']}">{fg['label']}</div>
            <div style="margin-top:15px;">
                <div class="fg-component-bar" style="width:80%;margin:0 auto;"></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    for comp, val in fg['components'].items():
        bar_clr = "#F85149" if val < 40 else "#FFB000" if val < 60 else "#3FB950"
        st.markdown(f'''<div class="fg-component">
            <span>{comp.upper()}</span>
            <span style="color:{bar_clr};font-weight:700;">{val:.0f}</span>
        </div>''', unsafe_allow_html=True)
