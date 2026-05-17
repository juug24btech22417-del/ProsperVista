# ProsperVista v3.0 — Options Greeks Calculator
# Black-Scholes pricing with full Greeks suite — no paid API needed

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# ==========================================
# BLACK-SCHOLES CORE
# ==========================================

def _d1(S, K, T, r, sigma):
    """Calculate d1 parameter for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r, sigma):
    """Calculate d2 parameter for Black-Scholes."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price.
    S: Current stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (annual, decimal)
    sigma: Volatility (annual, decimal)
    """
    if T <= 0:
        if option_type == "call":
            return max(0, S - K)
        return max(0, K - S)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ==========================================
# FULL GREEKS SUITE
# ==========================================

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate all Greeks for an option.
    Returns dict with: delta, gamma, theta, vega, rho, price
    """
    if T <= 0 or sigma <= 0:
        price = max(0, S - K) if option_type == "call" else max(0, K - S)
        return {"price": round(price, 4), "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    price = black_scholes_price(S, K, T, r, sigma, option_type)

    # DELTA: Rate of change of option price w.r.t. underlying
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # GAMMA: Rate of change of delta w.r.t. underlying (same for call & put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # THETA: Time decay (per day)
    common_theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta = (common_theta - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (common_theta + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # VEGA: Sensitivity to volatility (per 1% move)
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100

    # RHO: Sensitivity to interest rate (per 1% move)
    if option_type == "call":
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    else:
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    return {
        "price": round(price, 4), "delta": round(delta, 4), "gamma": round(gamma, 6),
        "theta": round(theta, 4), "vega": round(vega, 4), "rho": round(rho, 4)
    }

# ==========================================
# IMPLIED VOLATILITY SOLVER
# ==========================================

def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Calculate Implied Volatility using Brent's method.
    Returns IV as a decimal (e.g., 0.25 = 25%).
    """
    if T <= 0 or market_price <= 0:
        return 0.0
    try:
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return round(iv, 4)
    except (ValueError, RuntimeError):
        return 0.0

# ==========================================
# PAYOFF DIAGRAMS
# ==========================================

def calculate_payoff(S_range, K, premium, option_type="call", side="buy"):
    """
    Calculate P&L payoff at expiration for charting.
    S_range: array of stock prices at expiration
    K: strike price
    premium: option premium paid/received
    side: 'buy' or 'sell'
    """
    if option_type == "call":
        intrinsic = np.maximum(S_range - K, 0)
    else:
        intrinsic = np.maximum(K - S_range, 0)
    if side == "buy":
        payoff = intrinsic - premium
    else:
        payoff = premium - intrinsic
    return payoff

def generate_payoff_data(S, K, T, r, sigma, option_type="call", side="buy"):
    """Generate payoff curve data for Plotly charting."""
    premium = black_scholes_price(S, K, T, r, sigma, option_type)
    price_range = np.linspace(S * 0.7, S * 1.3, 200)
    payoff = calculate_payoff(price_range, K, premium, option_type, side)
    return pd.DataFrame({
        "stock_price": price_range, "payoff": payoff,
        "breakeven": np.zeros(len(price_range))
    }), premium

# ==========================================
# GREEKS SURFACE (Strike x Expiry)
# ==========================================

def generate_greeks_surface(S, r, sigma, greek="delta", option_type="call"):
    """
    Generate 2D surface data for a selected Greek across strikes and expiries.
    Returns data suitable for Plotly heatmap.
    """
    strikes = np.linspace(S * 0.8, S * 1.2, 20)
    expiries = np.linspace(0.01, 1.0, 20)  # 1 week to 1 year
    z_data = np.zeros((len(expiries), len(strikes)))
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            greeks = calculate_greeks(S, K, T, r, sigma, option_type)
            z_data[i, j] = greeks.get(greek, 0)
    return strikes, expiries, z_data

# ==========================================
# OPTIONS CHAIN BUILDER (from yfinance)
# ==========================================

def get_options_chain(ticker):
    """
    Fetch options chain from yfinance (free, may be limited).
    Returns calls and puts DataFrames.
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None, None, []
        # Get nearest expiration
        chain = stock.option_chain(expirations[0])
        return chain.calls, chain.puts, list(expirations)
    except Exception:
        return None, None, []

# ==========================================
# STRATEGY ANALYZER
# ==========================================

def analyze_strategy(legs):
    """
    Analyze a multi-leg options strategy.
    legs: list of dicts with keys: type (call/put), side (buy/sell), strike, premium, quantity
    Returns payoff data for the combined strategy.
    """
    price_range = np.linspace(
        min(l["strike"] for l in legs) * 0.7,
        max(l["strike"] for l in legs) * 1.3,
        300
    )
    total_payoff = np.zeros(len(price_range))
    total_cost = 0

    for leg in legs:
        payoff = calculate_payoff(price_range, leg["strike"], leg["premium"], leg["type"], leg["side"])
        total_payoff += payoff * leg.get("quantity", 1)
        if leg["side"] == "buy":
            total_cost += leg["premium"] * leg.get("quantity", 1)
        else:
            total_cost -= leg["premium"] * leg.get("quantity", 1)

    max_profit = np.max(total_payoff)
    max_loss = np.min(total_payoff)
    breakevens = price_range[np.where(np.diff(np.sign(total_payoff)))[0]]

    return {
        "price_range": price_range, "payoff": total_payoff,
        "max_profit": round(max_profit, 2), "max_loss": round(max_loss, 2),
        "total_cost": round(total_cost, 2),
        "breakevens": [round(b, 2) for b in breakevens],
        "risk_reward": round(abs(max_profit / max_loss), 2) if max_loss != 0 else float('inf')
    }

import yfinance as yf
