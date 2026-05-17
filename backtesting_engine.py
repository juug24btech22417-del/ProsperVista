# ProsperVista v3.0 — Backtesting Engine
# Test trading strategies on historical data with full performance metrics

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STRATEGY DEFINITIONS
# ==========================================

def strategy_ma_crossover(df, fast_period=10, slow_period=50):
    """
    Moving Average Crossover Strategy.
    BUY when fast MA crosses above slow MA.
    SELL when fast MA crosses below slow MA.
    """
    df = df.copy()
    df['fast_ma'] = df['Close'].rolling(fast_period).mean()
    df['slow_ma'] = df['Close'].rolling(slow_period).mean()
    df = df.dropna()
    
    signals = pd.Series(0, index=df.index)
    signals[df['fast_ma'] > df['slow_ma']] = 1
    signals[df['fast_ma'] <= df['slow_ma']] = -1
    
    return signals, df, f"MA Crossover ({fast_period}/{slow_period})"

def strategy_rsi(df, period=14, oversold=30, overbought=70):
    """
    RSI Mean Reversion Strategy.
    BUY when RSI < oversold threshold.
    SELL when RSI > overbought threshold.
    """
    df = df.copy()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=period)
    df = df.dropna()
    
    signals = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        if rsi < oversold and position <= 0:
            position = 1
        elif rsi > overbought and position >= 0:
            position = -1
        signals.iloc[i] = position
    
    return signals, df, f"RSI ({period}, {oversold}/{overbought})"

def strategy_macd(df, fast=12, slow=26, signal=9):
    """
    MACD Signal Line Crossover Strategy.
    BUY when MACD crosses above signal line.
    SELL when MACD crosses below signal line.
    """
    df = df.copy()
    df['MACD'] = df['Close'].ewm(span=fast).mean() - df['Close'].ewm(span=slow).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
    df = df.dropna()
    
    signals = pd.Series(0, index=df.index)
    signals[df['MACD'] > df['MACD_Signal']] = 1
    signals[df['MACD'] <= df['MACD_Signal']] = -1
    
    return signals, df, f"MACD ({fast}/{slow}/{signal})"

def strategy_bollinger(df, period=20, num_std=2):
    """
    Bollinger Band Breakout Strategy.
    BUY when price touches lower band (mean reversion).
    SELL when price touches upper band.
    """
    df = df.copy()
    df['BB_Mid'] = df['Close'].rolling(period).mean()
    df['BB_Upper'] = df['BB_Mid'] + num_std * df['Close'].rolling(period).std()
    df['BB_Lower'] = df['BB_Mid'] - num_std * df['Close'].rolling(period).std()
    df = df.dropna()
    
    signals = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        if price <= df['BB_Lower'].iloc[i] and position <= 0:
            position = 1
        elif price >= df['BB_Upper'].iloc[i] and position >= 0:
            position = -1
        signals.iloc[i] = position
    
    return signals, df, f"Bollinger ({period}, {num_std}σ)"

def strategy_multi_indicator(df):
    """
    Multi-Indicator Confluence Strategy.
    BUY when MACD > Signal AND (RSI < 45 OR Price > MA50).
    SELL when MACD < Signal AND (RSI > 55 OR Price < MA50).
    """
    df = df.copy()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    
    signals = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        macd = df['MACD'].iloc[i]
        macd_sig = df['MACD_Signal'].iloc[i]
        price = df['Close'].iloc[i]
        ma50 = df['MA50'].iloc[i]
        
        if macd > macd_sig and (rsi < 45 or price > ma50) and position <= 0:
            position = 1
        elif macd < macd_sig and (rsi > 55 or price < ma50) and position >= 0:
            position = -1
        signals.iloc[i] = position
    
    return signals, df, "Multi-Indicator Confluence"

STRATEGIES = {
    "MA Crossover": strategy_ma_crossover,
    "RSI Mean Reversion": strategy_rsi,
    "MACD Signal": strategy_macd,
    "Bollinger Bands": strategy_bollinger,
    "Multi-Indicator": strategy_multi_indicator
}

# ==========================================
# BACKTESTING ENGINE
# ==========================================

def run_backtest(ticker, strategy_name="MA Crossover", period="2y", initial_capital=1000000, commission=0.001):
    """
    Run a full backtest on a given strategy.
    
    Returns:
        dict with performance metrics, equity curve, and trade log
    """
    # Fetch data
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 60:
            return {"error": "Insufficient data for backtesting"}
    except Exception as e:
        return {"error": f"Data fetch failed: {str(e)}"}
    
    # Get strategy signals
    strategy_func = STRATEGIES.get(strategy_name, strategy_ma_crossover)
    signals, df, label = strategy_func(df)
    
    # Simulate trades
    capital = initial_capital
    position = 0  # 0 = flat, 1 = long, -1 = short
    shares = 0
    entry_price = 0
    trades = []
    equity_curve = []
    
    for i in range(1, len(signals)):
        date = signals.index[i]
        price = float(df['Close'].iloc[i])
        signal = signals.iloc[i]
        prev_signal = signals.iloc[i-1]
        
        # Signal change = trade
        if signal != prev_signal:
            # Close existing position
            if position != 0 and shares > 0:
                exit_value = shares * price
                comm = exit_value * commission
                if position == 1:
                    pnl = (price - entry_price) * shares - comm
                else:
                    pnl = (entry_price - price) * shares - comm
                capital += exit_value - comm if position == 1 else (2 * entry_price * shares - exit_value) - comm
                trades.append({
                    "entry_date": str(entry_date)[:10], "exit_date": str(date)[:10],
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": round(entry_price, 2), "exit_price": round(price, 2),
                    "shares": shares, "pnl": round(pnl, 2),
                    "pnl_pct": round((pnl / (entry_price * shares)) * 100, 2)
                })
                shares = 0
                position = 0
            
            # Open new position
            if signal == 1:
                shares = int(capital * 0.95 / price)  # Use 95% of capital
                if shares > 0:
                    entry_price = price
                    entry_date = date
                    capital -= shares * price * (1 + commission)
                    position = 1
            elif signal == -1:
                shares = int(capital * 0.95 / price)
                if shares > 0:
                    entry_price = price
                    entry_date = date
                    position = -1
        
        # Track equity
        if position == 1:
            equity = capital + shares * price
        elif position == -1:
            equity = capital + shares * (2 * entry_price - price)
        else:
            equity = capital
        
        equity_curve.append({"date": date, "equity": round(equity, 2)})
    
    # Close any remaining position at last price
    if position != 0 and shares > 0:
        last_price = float(df['Close'].iloc[-1])
        if position == 1:
            capital += shares * last_price
        else:
            capital += shares * (2 * entry_price - last_price)
    
    # Calculate performance metrics
    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {"error": "No trades generated"}
    
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df['returns'] = eq_df['equity'].pct_change()
    
    final_equity = eq_df['equity'].iloc[-1]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # CAGR
    days = (eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Max Drawdown
    rolling_max = eq_df['equity'].cummax()
    drawdown = (eq_df['equity'] - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()
    
    # Win/Loss stats
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(winning) / len(trades) * 100) if trades else 0
    
    avg_win = np.mean([t['pnl'] for t in winning]) if winning else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losing])) if losing else 1
    
    gross_profit = sum(t['pnl'] for t in winning) if winning else 0
    gross_loss = abs(sum(t['pnl'] for t in losing)) if losing else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe Ratio
    returns = eq_df['returns'].dropna()
    sharpe = (returns.mean() * 252 - 0.065) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Buy & Hold comparison
    bh_return = ((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0])) - 1) * 100
    
    metrics = {
        "strategy": label, "ticker": ticker,
        "initial_capital": initial_capital, "final_equity": round(final_equity, 2),
        "total_return": round(total_return, 2), "cagr": round(cagr, 2),
        "max_drawdown": round(max_dd, 2), "sharpe_ratio": round(sharpe, 2),
        "total_trades": len(trades), "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "buy_hold_return": round(bh_return, 2),
        "alpha": round(total_return - bh_return, 2)
    }
    
    return {
        "metrics": metrics, "equity_curve": eq_df,
        "trades": trades, "drawdown": drawdown.values.tolist()
    }

# ==========================================
# STRATEGY COMPARISON
# ==========================================

def compare_strategies(ticker, period="2y"):
    """Run all strategies on the same ticker and compare performance."""
    results = []
    for name in STRATEGIES:
        result = run_backtest(ticker, name, period)
        if "error" not in result:
            results.append(result["metrics"])
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values("total_return", ascending=False)
