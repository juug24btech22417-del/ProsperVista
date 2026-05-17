# ProsperVista v3.0 — Risk Analytics Engine
# VaR, CVaR, Stress Testing, Drawdown Analysis — all computed locally

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta

# ==========================================
# VALUE AT RISK (VaR) — Three Methods
# ==========================================

def historical_var(returns, confidence=0.95, horizon=1):
    """Historical simulation VaR."""
    if returns is None or len(returns) < 10:
        return 0.0
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var = abs(sorted_returns[index]) * np.sqrt(horizon)
    return round(var * 100, 4)

def parametric_var(returns, confidence=0.95, horizon=1):
    """Parametric (Gaussian) VaR."""
    if returns is None or len(returns) < 10:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    z_score = norm.ppf(1 - confidence)
    var = abs(mu + z_score * sigma) * np.sqrt(horizon)
    return round(var * 100, 4)

def monte_carlo_var(returns, confidence=0.95, horizon=1, simulations=10000):
    """Monte Carlo simulation VaR."""
    if returns is None or len(returns) < 10:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    simulated = np.random.normal(mu, sigma, simulations) * np.sqrt(horizon)
    sorted_sim = np.sort(simulated)
    index = int((1 - confidence) * simulations)
    var = abs(sorted_sim[index])
    return round(var * 100, 4)

def calculate_all_var(ticker, period="1y", confidence=0.95):
    """Calculate VaR using all three methods for a ticker."""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        returns = df['Close'].pct_change().dropna().values
    except Exception:
        returns = np.array([])

    return {
        "historical_var": historical_var(returns, confidence),
        "parametric_var": parametric_var(returns, confidence),
        "monte_carlo_var": monte_carlo_var(returns, confidence),
        "confidence": confidence * 100,
        "data_points": len(returns)
    }

# ==========================================
# CONDITIONAL VaR (CVaR / Expected Shortfall)
# ==========================================

def calculate_cvar(returns, confidence=0.95):
    """CVaR: Average loss beyond the VaR threshold."""
    if returns is None or len(returns) < 10:
        return 0.0
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    tail_losses = sorted_returns[:index]
    cvar = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else 0
    return round(cvar * 100, 4)

# ==========================================
# STRESS TESTING (Historical Crash Scenarios)
# ==========================================

CRASH_SCENARIOS = {
    "2008 Financial Crisis": {"description": "Lehman collapse, -38% peak drawdown", "shock": -0.38, "duration_days": 180},
    "COVID Crash (2020)": {"description": "Pandemic sell-off, -34% in 33 days", "shock": -0.34, "duration_days": 33},
    "Dot-com Bubble (2000)": {"description": "Tech bubble burst, -49% over 2 years", "shock": -0.49, "duration_days": 500},
    "Flash Crash (2010)": {"description": "Algorithmic cascade, -9% intraday", "shock": -0.09, "duration_days": 1},
    "Brexit (2016)": {"description": "UK EU referendum shock, -8%", "shock": -0.08, "duration_days": 3},
    "India Demonetization (2016)": {"description": "₹500/₹1000 ban, -6% Nifty", "shock": -0.06, "duration_days": 5},
    "Russia-Ukraine War (2022)": {"description": "Geopolitical risk spike, -12%", "shock": -0.12, "duration_days": 30},
}

def run_stress_test(current_value, scenarios=None):
    """
    Run stress test on portfolio/position value against crash scenarios.
    Returns projected impact for each scenario.
    """
    if scenarios is None:
        scenarios = CRASH_SCENARIOS
    results = []
    for name, params in scenarios.items():
        impact = current_value * params["shock"]
        stressed_value = current_value + impact
        results.append({
            "scenario": name, "description": params["description"],
            "shock_pct": round(params["shock"] * 100, 1),
            "impact": round(impact, 2), "stressed_value": round(stressed_value, 2),
            "duration_days": params["duration_days"]
        })
    return sorted(results, key=lambda x: x["impact"])

# ==========================================
# DRAWDOWN ANALYSIS
# ==========================================

def calculate_drawdown_series(ticker, period="2y"):
    """Calculate full drawdown series for charting."""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        prices = df['Close']
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max * 100
        return pd.DataFrame({
            "date": df.index, "price": prices.values,
            "rolling_max": rolling_max.values, "drawdown": drawdown.values
        })
    except Exception:
        return pd.DataFrame()

def max_drawdown_stats(ticker, period="2y"):
    """Get max drawdown statistics."""
    dd_df = calculate_drawdown_series(ticker, period)
    if dd_df.empty:
        return {"max_drawdown": 0, "max_dd_date": "N/A", "recovery_days": 0}
    max_dd_idx = dd_df['drawdown'].idxmin()
    max_dd = dd_df.loc[max_dd_idx, 'drawdown']
    max_dd_date = dd_df.loc[max_dd_idx, 'date']
    # Estimate recovery
    post_dd = dd_df.loc[max_dd_idx:]
    recovered = post_dd[post_dd['drawdown'] >= -0.5]
    recovery_days = len(post_dd) - len(recovered) if not recovered.empty else len(post_dd)
    return {
        "max_drawdown": round(max_dd, 2),
        "max_dd_date": str(max_dd_date)[:10] if hasattr(max_dd_date, 'strftime') else str(max_dd_date),
        "recovery_days": recovery_days,
        "current_drawdown": round(dd_df['drawdown'].iloc[-1], 2)
    }

# ==========================================
# RISK-ADJUSTED RETURNS
# ==========================================

def risk_adjusted_metrics(ticker, benchmark="^NSEI", period="1y"):
    """
    Calculate Information Ratio, Treynor Ratio, Jensen's Alpha.
    """
    try:
        stock_df = yf.download(ticker, period=period, progress=False)
        bench_df = yf.download(benchmark, period=period, progress=False)
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.droplevel(1)
        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.droplevel(1)
        stock_ret = stock_df['Close'].pct_change().dropna()
        bench_ret = bench_df['Close'].pct_change().dropna()
        # Align dates
        common = stock_ret.index.intersection(bench_ret.index)
        stock_ret = stock_ret[common]
        bench_ret = bench_ret[common]
        if len(stock_ret) < 20:
            return {"information_ratio": 0, "treynor_ratio": 0, "jensens_alpha": 0, "beta": 0}
    except Exception:
        return {"information_ratio": 0, "treynor_ratio": 0, "jensens_alpha": 0, "beta": 0}

    rf_daily = 0.065 / 252
    excess_stock = stock_ret - rf_daily
    excess_bench = bench_ret - rf_daily
    # Beta
    cov_matrix = np.cov(excess_stock, excess_bench)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1
    # Treynor Ratio
    treynor = (excess_stock.mean() * 252) / beta if beta != 0 else 0
    # Jensen's Alpha
    alpha = (stock_ret.mean() - rf_daily) - beta * (bench_ret.mean() - rf_daily)
    alpha_annual = alpha * 252
    # Information Ratio
    tracking_error = (stock_ret - bench_ret).std() * np.sqrt(252)
    info_ratio = ((stock_ret.mean() - bench_ret.mean()) * 252) / tracking_error if tracking_error > 0 else 0

    return {
        "information_ratio": round(info_ratio, 3),
        "treynor_ratio": round(treynor * 100, 3),
        "jensens_alpha": round(alpha_annual * 100, 3),
        "beta": round(beta, 3)
    }

# ==========================================
# COMPREHENSIVE RISK REPORT
# ==========================================

def generate_risk_report(ticker, portfolio_value=None):
    """Generate a full risk report for a ticker."""
    var_data = calculate_all_var(ticker)
    dd_stats = max_drawdown_stats(ticker)
    risk_adj = risk_adjusted_metrics(ticker)
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        returns = df['Close'].pct_change().dropna().values
        cvar = calculate_cvar(returns)
    except Exception:
        cvar = 0
    stress = run_stress_test(portfolio_value or 1000000)
    return {
        "var": var_data, "cvar": cvar, "drawdown": dd_stats,
        "risk_adjusted": risk_adj, "stress_test": stress
    }
