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

if __name__ == "__main__":
    # Test block
    engine = SentimentEngine()
    result = engine.get_news_sentiment("TSLA")
    print(f"Overall Verdict: {result['verdict']} ({result['score']:.2f})")
    for n in result['news']:
        print(f"[{n['sentiment']}] {n['title']}")
