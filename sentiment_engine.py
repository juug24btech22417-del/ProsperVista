import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_news_sentiment(self, ticker):
        """
        Fetches news for a given ticker using yfinance and analyzes sentiment.
        Returns a dictionary with sentiment score and a list of news items.
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return {"score": 0, "verdict": "NEUTRAL", "news": []}

            scored_news = []
            total_compound = 0
            
            for item in news:
                content = item.get('content', {})
                title = content.get('title', '')
                if not title: continue # Skip if no title

                # Analyze title sentiment
                sentiment = self.analyzer.polarity_scores(title)
                compound = sentiment['compound']
                total_compound += compound
                
                # Categorize sentiment
                if compound >= 0.05:
                    label = "BULLISH"
                    color = "#4ade80"
                elif compound <= -0.05:
                    label = "BEARISH"
                    color = "#f87171"
                else:
                    label = "NEUTRAL"
                    color = "#94a3b8"
                
                # Extract links - try different possible paths
                link = content.get('canonicalUrl', {}).get('url', '#')
                if link == '#':
                    link = content.get('clickThroughUrl', {}).get('url', '#')

                scored_news.append({
                    "title": title,
                    "publisher": content.get('provider', {}).get('displayName', 'Unknown'),
                    "link": link,
                    "time": content.get('pubDate', 'Recently'),
                    "sentiment": label,
                    "sentiment_score": compound,
                    "color": color
                })

            valid_count = len(scored_news)
            avg_score = total_compound / valid_count if valid_count > 0 else 0
            
            # Overall Verdict
            if avg_score >= 0.05:
                verdict = "BULLISH"
            elif avg_score <= -0.05:
                verdict = "BEARISH"
            else:
                verdict = "NEUTRAL"

            return {
                "score": avg_score,
                "verdict": verdict,
                "news": scored_news[:10]  # Return top 10 news items
            }
        except Exception as e:
            print(f"Error fetching sentiment: {e}")
            return {"score": 0, "verdict": "NEUTRAL", "news": []}

    def get_market_movers(self):
        """
        Fetches significant market movers (Declines) and their triggers.
        Uses batch download for speed, then only fetches details for decliners.
        """
        try:
            tickers = [
                "TSLA", "AAPL", "NVDA", "MSFT", "AMD", "META", "GOOGL", # US Tech
                "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "TATAMOTORS.NS", "ZOMATO.NS" # Indian Market
            ]
            movers = []
            
            # Single batch download for all tickers
            bulk = yf.download(tickers, period="2d", progress=False, threads=True)
            if bulk.empty: return movers
            
            # Identify decliners from batch data
            decliners = []
            for t in tickers:
                try:
                    if isinstance(bulk.columns, pd.MultiIndex):
                        close = bulk['Close'][t].dropna()
                    else:
                        close = bulk['Close'].dropna()
                    if len(close) < 2: continue
                    change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
                    if change < -1.0:
                        decliners.append((t, change))
                except: continue
            
            # Only fetch .info and .news for actual decliners (much fewer calls)
            def _fetch_decliner_details(ticker_change):
                t, change = ticker_change
                try:
                    s = yf.Ticker(t)
                    info = s.info
                    company_name = info.get('longName', t)
                    news = s.news
                    reason = news[0].get('content', {}).get('title', 'No recent news found.') if news else "Market Volatility"
                    return {"ticker": t, "name": company_name, "change": change, "reason": reason}
                except:
                    return {"ticker": t, "name": t, "change": change, "reason": "Market Volatility"}
            
            if decliners:
                with ThreadPoolExecutor(max_workers=min(6, len(decliners))) as executor:
                    results = list(executor.map(_fetch_decliner_details, decliners))
                movers = sorted(results, key=lambda x: x['change'])
            
            return movers
        except Exception as e:
            print(f"Error in market movers: {e}")
            return []

    def calculate_fear_greed(self):
        """
        Calculates a real-time Fear & Greed Index (0-100) using a composite of:
        - Market Momentum (Distance of major assets from 125-day SMA)
        - Price Strength (Aggregate RSI)
        - Market Volatility (Historical Volatility vs Average)
        - News Sentiment (Polarity compound score of recent headlines)
        
        Optimized: Single batch download for all price data, parallel news fetch.
        """
        try:
            components = {}
            import numpy as np
            
            ref_tickers = ["RELIANCE.NS", "TSLA", "AAPL"]
            
            # Single batch download for ALL price data (150 days covers all needs)
            bulk = yf.download(ref_tickers, period="150d", progress=False, threads=True)
            if bulk.empty:
                raise ValueError("No data from batch download")
            
            # 1. Market Momentum Component (needs 125-day SMA)
            momentum_scores = []
            for t in ref_tickers:
                try:
                    if isinstance(bulk.columns, pd.MultiIndex):
                        close = bulk['Close'][t].dropna()
                    else:
                        close = bulk['Close'].dropna()
                    if len(close) >= 125:
                        sma = close.rolling(window=125).mean().iloc[-1]
                        current = close.iloc[-1]
                        dist_pct = ((current - sma) / sma) * 100
                        score = max(0, min(100, 50 + dist_pct * 5))
                        momentum_scores.append(score)
                except: pass
            components["Market Momentum"] = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 50.0

            # 2. Price Strength Component (RSI — needs 14+ days)
            rsi_scores = []
            for t in ref_tickers:
                try:
                    if isinstance(bulk.columns, pd.MultiIndex):
                        close = bulk['Close'][t].dropna()
                    else:
                        close = bulk['Close'].dropna()
                    if len(close) >= 14:
                        delta = close.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                        rs = gain / loss if loss != 0 else 0
                        rsi = 100 - (100 / (1 + rs)) if loss != 0 else 100
                        rsi_scores.append(rsi)
                except: pass
            components["Price Strength"] = sum(rsi_scores) / len(rsi_scores) if rsi_scores else 50.0

            # 3. Market Volatility Component (needs 20+ days)
            vol_scores = []
            for t in ref_tickers:
                try:
                    if isinstance(bulk.columns, pd.MultiIndex):
                        close = bulk['Close'][t].dropna()
                    else:
                        close = bulk['Close'].dropna()
                    if len(close) >= 20:
                        returns = close.pct_change().dropna()
                        realized_vol = returns.std() * np.sqrt(252) * 100
                        vol_scores.append(max(0, min(100, 100 - realized_vol * 1.5)))
                except: pass
            components["Market Volatility"] = sum(vol_scores) / len(vol_scores) if vol_scores else 50.0

            # 4. News Sentiment Component (parallel fetch)
            def _fetch_sentiment(t):
                try:
                    res = self.get_news_sentiment(t)
                    return (res["score"] + 1) * 50
                except: return 50.0
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                sent_scores = list(executor.map(_fetch_sentiment, ref_tickers))
            components["News Sentiment"] = sum(sent_scores) / len(sent_scores) if sent_scores else 50.0

            # Compute composite score
            final_score = sum(components.values()) / len(components)
            
            # Map score to label & color
            if final_score < 25:
                label = "Extreme Fear"
                color = "#FF4B4B"
            elif final_score < 45:
                label = "Fear"
                color = "#FF7C7C"
            elif final_score < 55:
                label = "Neutral"
                color = "#94a3b8"
            elif final_score < 75:
                label = "Greed"
                color = "#4ade80"
            else:
                label = "Extreme Greed"
                color = "#00FF9D"

            return {
                "score": final_score,
                "label": label,
                "color": color,
                "components": components
            }
        except Exception as e:
            print(f"Error calculating fear & greed: {e}")
            return {
                "score": 50.0,
                "label": "Neutral",
                "color": "#94a3b8",
                "components": {
                    "Market Momentum": 50.0,
                    "Price Strength": 50.0,
                    "Market Volatility": 50.0,
                    "News Sentiment": 50.0
                }
            }

if __name__ == "__main__":
    # Test block
    engine = SentimentEngine()
    result = engine.get_news_sentiment("TSLA")
    print(f"Overall Verdict: {result['verdict']} ({result['score']:.2f})")
    for n in result['news']:
        print(f"[{n['sentiment']}] {n['title']}")