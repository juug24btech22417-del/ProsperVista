# ProsperVista v3.0 — Portfolio Simulator Engine
# Storage: JSON file-based (no database required)

import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

PORTFOLIO_FILE = "portfolio_data.json"

def _load_portfolio_data():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "positions": [], "closed_trades": [],
        "cash_balance": 1000000.0, "initial_capital": 1000000.0,
        "created_at": datetime.now().isoformat()
    }

def _save_portfolio_data(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def open_position(ticker, quantity, entry_price, stop_loss=None, take_profit=None, side="LONG"):
    data = _load_portfolio_data()
    cost = entry_price * quantity
    if cost > data["cash_balance"]:
        return {"error": f"Insufficient funds. Need ₹{cost:,.2f}, have ₹{data['cash_balance']:,.2f}"}
    position = {
        "id": len(data["positions"]) + len(data["closed_trades"]) + 1,
        "ticker": ticker.upper(), "side": side, "quantity": quantity,
        "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
        "opened_at": datetime.now().isoformat(), "status": "OPEN"
    }
    data["positions"].append(position)
    data["cash_balance"] -= cost
    _save_portfolio_data(data)
    return position

def close_position(position_id, exit_price):
    data = _load_portfolio_data()
    for i, pos in enumerate(data["positions"]):
        if pos["id"] == position_id and pos["status"] == "OPEN":
            if pos["side"] == "LONG":
                pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
            else:
                pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
                pnl_pct = ((pos["entry_price"] - exit_price) / pos["entry_price"]) * 100
            commission = (pos["entry_price"] * pos["quantity"] + exit_price * pos["quantity"]) * 0.0005
            net_pnl = pnl - commission
            closed_trade = {
                **pos, "exit_price": exit_price, "closed_at": datetime.now().isoformat(),
                "gross_pnl": round(pnl, 2), "commission": round(commission, 2),
                "net_pnl": round(net_pnl, 2), "pnl_pct": round(pnl_pct, 2), "status": "CLOSED"
            }
            data["closed_trades"].append(closed_trade)
            data["positions"].pop(i)
            data["cash_balance"] += (exit_price * pos["quantity"]) - commission
            _save_portfolio_data(data)
            return closed_trade
    return {"error": f"Position {position_id} not found or already closed"}

def get_open_positions():
    data = _load_portfolio_data()
    positions = []
    for pos in data["positions"]:
        if pos["status"] != "OPEN":
            continue
        try:
            ticker_data = yf.Ticker(pos["ticker"])
            hist = ticker_data.history(period="1d")
            current_price = float(hist['Close'].iloc[-1]) if not hist.empty else pos["entry_price"]
        except Exception:
            current_price = pos["entry_price"]
        if pos["side"] == "LONG":
            unrealized_pnl = (current_price - pos["entry_price"]) * pos["quantity"]
            pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
        else:
            unrealized_pnl = (pos["entry_price"] - current_price) * pos["quantity"]
            pnl_pct = ((pos["entry_price"] - current_price) / pos["entry_price"]) * 100
        positions.append({
            **pos, "current_price": round(current_price, 2),
            "unrealized_pnl": round(unrealized_pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "market_value": round(current_price * pos["quantity"], 2)
        })
    return positions

def get_closed_trades():
    data = _load_portfolio_data()
    return data.get("closed_trades", [])

def get_portfolio_summary():
    data = _load_portfolio_data()
    positions = get_open_positions()
    closed = data.get("closed_trades", [])
    total_market_value = sum(p["market_value"] for p in positions)
    total_unrealized = sum(p["unrealized_pnl"] for p in positions)
    total_realized = sum(t["net_pnl"] for t in closed)
    portfolio_value = data["cash_balance"] + total_market_value
    winning_trades = [t for t in closed if t["net_pnl"] > 0]
    losing_trades = [t for t in closed if t["net_pnl"] <= 0]
    win_rate = (len(winning_trades) / len(closed) * 100) if closed else 0
    avg_win = np.mean([t["net_pnl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t["net_pnl"] for t in losing_trades])) if losing_trades else 1
    gross_profit = sum(t["net_pnl"] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t["net_pnl"] for t in losing_trades)) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_return = ((portfolio_value - data["initial_capital"]) / data["initial_capital"]) * 100
    return {
        "portfolio_value": round(portfolio_value, 2), "cash_balance": round(data["cash_balance"], 2),
        "initial_capital": data["initial_capital"], "total_market_value": round(total_market_value, 2),
        "total_unrealized_pnl": round(total_unrealized, 2), "total_realized_pnl": round(total_realized, 2),
        "total_return_pct": round(total_return, 2), "total_trades": len(closed),
        "win_rate": round(win_rate, 1), "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2), "open_positions": len(positions),
        "positions": positions, "closed_trades": closed
    }

def reset_portfolio(initial_capital=1000000.0):
    data = {
        "positions": [], "closed_trades": [],
        "cash_balance": initial_capital, "initial_capital": initial_capital,
        "created_at": datetime.now().isoformat()
    }
    _save_portfolio_data(data)
    return data

def calculate_risk_metrics(returns_series):
    if returns_series is None or len(returns_series) < 2:
        return {"sharpe_ratio": 0, "sortino_ratio": 0, "max_drawdown": 0,
                "calmar_ratio": 0, "volatility_annual": 0, "var_95": 0}
    returns = returns_series.dropna()
    risk_free_rate = 0.065
    mean_daily = returns.mean()
    std_daily = returns.std()
    annual_return = mean_daily * 252
    annual_vol = std_daily * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    calmar = annual_return / abs(max_dd / 100) if max_dd != 0 else 0
    var_95 = np.percentile(returns, 5) * 100
    return {
        "sharpe_ratio": round(sharpe, 2), "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 2), "calmar_ratio": round(calmar, 2),
        "volatility_annual": round(annual_vol * 100, 2), "var_95": round(var_95, 2),
        "annual_return": round(annual_return * 100, 2)
    }

def get_portfolio_returns(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df['Close'].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)

def kelly_criterion(win_rate, avg_win, avg_loss):
    if avg_loss <= 0 or win_rate <= 0:
        return {"kelly_pct": 0, "half_kelly_pct": 0, "recommendation": "Insufficient data"}
    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    half_kelly = max(0, kelly / 2)
    if kelly <= 0:
        rec = "NO POSITION — Negative expectancy"
    elif half_kelly < 0.05:
        rec = "MINIMAL — Very small position only"
    elif half_kelly < 0.15:
        rec = "CONSERVATIVE — Standard position size"
    elif half_kelly < 0.25:
        rec = "MODERATE — Above-average conviction"
    else:
        rec = "AGGRESSIVE — High conviction trade"
    return {
        "kelly_pct": round(kelly * 100, 2), "half_kelly_pct": round(half_kelly * 100, 2),
        "recommendation": rec, "win_loss_ratio": round(win_loss_ratio, 2)
    }

def generate_equity_curve():
    data = _load_portfolio_data()
    closed = data.get("closed_trades", [])
    if not closed:
        return pd.DataFrame({"date": [datetime.now()], "equity": [data["initial_capital"]]})
    equity = data["initial_capital"]
    curve_data = [{"date": data.get("created_at", datetime.now().isoformat()), "equity": equity}]
    for trade in closed:
        equity += trade["net_pnl"]
        curve_data.append({"date": trade.get("closed_at", datetime.now().isoformat()), "equity": round(equity, 2)})
    df = pd.DataFrame(curve_data)
    df['date'] = pd.to_datetime(df['date'])
    return df
