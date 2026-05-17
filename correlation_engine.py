# ProsperVista v3.0 — Correlation & Heatmap Engine
# Cross-asset correlation analysis, Beta calculation, Cointegration

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CORRELATION MATRIX
# ==========================================

def build_correlation_matrix(tickers, period="6mo"):
    """
    Build correlation matrix from a list of tickers.
    Returns DataFrame suitable for heatmap plotting.
    """
    if not tickers or len(tickers) < 2:
        return pd.DataFrame()
    
    close_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                close_data[ticker] = df['Close']
        except Exception:
            continue
    
    if len(close_data) < 2:
        return pd.DataFrame()
    
    prices_df = pd.DataFrame(close_data).dropna()
    returns_df = prices_df.pct_change().dropna()
    
    return returns_df.corr()

# ==========================================
# ROLLING CORRELATION
# ==========================================

def rolling_correlation(ticker1, ticker2, period="1y", window=30):
    """
    Calculate rolling correlation between two assets.
    Returns DataFrame with date and correlation values.
    """
    try:
        df1 = yf.download(ticker1, period=period, progress=False)
        df2 = yf.download(ticker2, period=period, progress=False)
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.droplevel(1)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.droplevel(1)
        
        r1 = df1['Close'].pct_change().dropna()
        r2 = df2['Close'].pct_change().dropna()
        
        common = r1.index.intersection(r2.index)
        r1 = r1[common]
        r2 = r2[common]
        
        rolling_corr = r1.rolling(window).corr(r2).dropna()
        
        return pd.DataFrame({
            "date": rolling_corr.index,
            "correlation": rolling_corr.values
        })
    except Exception:
        return pd.DataFrame()

# ==========================================
# BETA CALCULATION
# ==========================================

def calculate_beta(ticker, benchmark="^NSEI", period="1y"):
    """
    Calculate stock beta relative to a benchmark index.
    Beta > 1: More volatile than market
    Beta < 1: Less volatile than market
    Beta < 0: Inversely correlated
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
        
        common = stock_ret.index.intersection(bench_ret.index)
        stock_ret = stock_ret[common]
        bench_ret = bench_ret[common]
        
        if len(stock_ret) < 20:
            return {"beta": 1.0, "r_squared": 0, "alpha": 0}
        
        cov = np.cov(stock_ret, bench_ret)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1
        
        # R-squared from regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(bench_ret, stock_ret)
        
        return {
            "beta": round(beta, 3),
            "r_squared": round(r_value**2, 3),
            "alpha": round(intercept * 252 * 100, 3),  # Annualized alpha %
            "p_value": round(p_value, 4)
        }
    except Exception:
        return {"beta": 1.0, "r_squared": 0, "alpha": 0, "p_value": 1.0}

# ==========================================
# SECTOR CORRELATION
# ==========================================

SECTOR_TICKERS = {
    "BANKING": "^NSEBANK",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "GOLD": "GC=F",
    "OIL": "CL=F"
}

def sector_correlation_matrix(period="6mo"):
    """Build correlation matrix across major sectors and asset classes."""
    return build_correlation_matrix(list(SECTOR_TICKERS.values()), period)

# ==========================================
# COINTEGRATION TEST (Pair Trading)
# ==========================================

def test_cointegration(ticker1, ticker2, period="1y"):
    """
    Engle-Granger cointegration test for pair trading.
    Cointegrated pairs tend to revert to a mean spread.
    """
    try:
        df1 = yf.download(ticker1, period=period, progress=False)
        df2 = yf.download(ticker2, period=period, progress=False)
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.droplevel(1)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.droplevel(1)
        
        p1 = df1['Close'].dropna()
        p2 = df2['Close'].dropna()
        common = p1.index.intersection(p2.index)
        p1 = p1[common].values
        p2 = p2[common].values
        
        if len(p1) < 30:
            return {"cointegrated": False, "p_value": 1.0, "spread_zscore": 0}
        
        # Simple OLS regression
        slope, intercept, _, _, _ = stats.linregress(p2, p1)
        spread = p1 - (slope * p2 + intercept)
        
        # ADF test on spread (simplified)
        spread_diff = np.diff(spread)
        spread_lag = spread[:-1]
        if np.std(spread_lag) == 0:
            return {"cointegrated": False, "p_value": 1.0, "spread_zscore": 0}
        
        slope_adf, _, _, p_value, _ = stats.linregress(spread_lag, spread_diff)
        
        # Current Z-score of spread
        zscore = (spread[-1] - np.mean(spread)) / np.std(spread) if np.std(spread) > 0 else 0
        
        return {
            "cointegrated": p_value < 0.05,
            "p_value": round(p_value, 4),
            "spread_zscore": round(zscore, 3),
            "hedge_ratio": round(slope, 4),
            "signal": "LONG SPREAD" if zscore < -2 else "SHORT SPREAD" if zscore > 2 else "NO SIGNAL",
            "spread_data": spread.tolist()[-60:]  # Last 60 days
        }
    except Exception:
        return {"cointegrated": False, "p_value": 1.0, "spread_zscore": 0}

# ==========================================
# DIVERSIFICATION SCORE
# ==========================================

def portfolio_diversification_score(tickers, period="6mo"):
    """
    Calculate a diversification score for a portfolio.
    Score 0-100: 100 = perfectly diversified, 0 = highly concentrated.
    """
    corr_matrix = build_correlation_matrix(tickers, period)
    if corr_matrix.empty or len(corr_matrix) < 2:
        return {"score": 0, "avg_correlation": 0, "most_correlated": ("N/A", "N/A", 0)}
    
    # Average absolute correlation (excluding diagonal)
    n = len(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    avg_corr = np.abs(corr_matrix.values[mask]).mean()
    
    # Diversification score: lower avg correlation = better diversification
    score = max(0, min(100, (1 - avg_corr) * 100))
    
    # Find most correlated pair
    corr_values = corr_matrix.values.copy()
    np.fill_diagonal(corr_values, 0)
    max_idx = np.unravel_index(np.abs(corr_values).argmax(), corr_values.shape)
    most_corr_pair = (corr_matrix.index[max_idx[0]], corr_matrix.columns[max_idx[1]])
    most_corr_val = corr_values[max_idx]
    
    return {
        "score": round(score, 1),
        "avg_correlation": round(avg_corr, 3),
        "most_correlated": (most_corr_pair[0], most_corr_pair[1], round(most_corr_val, 3))
    }
