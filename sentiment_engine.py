import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd

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
        Includes full company names and news headlines.
        """
        try:
            # Expanded monitoring list across sectors
            tickers = [
                "TSLA", "AAPL", "NVDA", "MSFT", "AMD", "META", "GOOGL", # US Tech
                "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "TATAMOTORS.NS", "ZOMATO.NS" # Indian Market
            ]
            movers = []
            
            for t in tickers:
                s = yf.Ticker(t)
                hist = s.history(period="2d")
                if len(hist) < 2: continue
                
                # Fetch full company name for better UI
                info = s.info
                company_name = info.get('longName', t)
                
                change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                
                # Report those down significantly
                if change < -1.0: 
                    news = s.news
                    # Note: You can replace this with NewsAPI.org call here if you have an API Key:
                    # news = my_news_api_client.get_everything(q=company_name)
                    
                    reason = news[0].get('content', {}).get('title', 'No recent news found.') if news else "Market Volatility"
                    movers.append({
                        "ticker": t,
                        "name": company_name,
                        "change": change,
                        "reason": reason
                    })
            
            return sorted(movers, key=lambda x: x['change'])
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
        """
        try:
            components = {}
            import numpy as np
            
            # 1. Market Momentum Component
            momentum_scores = []
            for t in ["RELIANCE.NS", "TSLA", "AAPL"]:
                try:
                    s = yf.Ticker(t)
                    hist = s.history(period="150d")
                    if len(hist) >= 125:
                        sma = hist['Close'].rolling(window=125).mean().iloc[-1]
                        current = hist['Close'].iloc[-1]
                        dist_pct = ((current - sma) / sma) * 100
                        score = max(0, min(100, 50 + dist_pct * 5))
                        momentum_scores.append(score)
                except Exception:
                    pass
            components["Market Momentum"] = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 50.0

            # 2. Price Strength Component (RSI)
            rsi_scores = []
            for t in ["RELIANCE.NS", "TSLA", "AAPL"]:
                try:
                    s = yf.Ticker(t)
                    hist = s.history(period="30d")
                    if len(hist) >= 14:
                        delta = hist['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                        rs = gain / loss if loss != 0 else 0
                        rsi = 100 - (100 / (1 + rs)) if loss != 0 else 100
                        rsi_scores.append(rsi)
                except Exception:
                    pass
            components["Price Strength"] = sum(rsi_scores) / len(rsi_scores) if rsi_scores else 50.0

            # 3. Market Volatility Component
            vol_scores = []
            for t in ["RELIANCE.NS", "TSLA", "AAPL"]:
                try:
                    s = yf.Ticker(t)
                    hist = s.history(period="30d")
                    if len(hist) >= 20:
                        returns = hist['Close'].pct_change().dropna()
                        realized_vol = returns.std() * np.sqrt(252) * 100
                        vol_scores.append(max(0, min(100, 100 - realized_vol * 1.5)))
                except Exception:
                    pass
            components["Market Volatility"] = sum(vol_scores) / len(vol_scores) if vol_scores else 50.0

            # 4. News Sentiment Component
            sent_scores = []
            for t in ["RELIANCE.NS", "TSLA", "AAPL"]:
                try:
                    res = self.get_news_sentiment(t)
                    score = (res["score"] + 1) * 50
                    sent_scores.append(score)
                except Exception:
                    pass
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