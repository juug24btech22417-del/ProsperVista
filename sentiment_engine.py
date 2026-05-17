# ProsperVista v3.0 — Enhanced Sentiment Intelligence Engine
# Multi-source sentiment, Fear & Greed Index, Sentiment Trends

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Custom financial lexicon for better accuracy
        self._add_financial_lexicon()
    
    def _add_financial_lexicon(self):
        """Add finance-specific terms to VADER for improved accuracy."""
        financial_terms = {
            'bullish': 2.5, 'bearish': -2.5, 'rally': 2.0, 'crash': -3.0,
            'surge': 2.5, 'plunge': -3.0, 'soar': 2.5, 'tank': -2.5,
            'breakout': 2.0, 'breakdown': -2.0, 'upgrade': 2.0, 'downgrade': -2.5,
            'outperform': 2.0, 'underperform': -2.0, 'buy': 1.5, 'sell': -1.5,
            'overweight': 1.5, 'underweight': -1.5, 'accumulate': 1.5,
            'dividend': 1.5, 'buyback': 2.0, 'restructuring': -1.0,
            'bankruptcy': -3.5, 'default': -3.0, 'fraud': -3.5, 'scandal': -3.0,
            'profit': 2.0, 'loss': -2.0, 'revenue': 1.0, 'debt': -1.0,
            'growth': 1.5, 'decline': -1.5, 'expansion': 1.5, 'contraction': -1.5,
            'innovation': 1.5, 'disruption': 0.5, 'partnership': 1.5,
            'acquisition': 1.0, 'merger': 0.5, 'layoff': -2.0, 'layoffs': -2.0,
            'beat': 1.5, 'miss': -1.5, 'exceeds': 1.5, 'disappoints': -2.0,
            'record': 1.5, 'all-time high': 2.5, 'all-time low': -2.5,
            'volatile': -0.5, 'uncertainty': -1.0, 'risk': -0.5, 'momentum': 1.0,
            'recession': -2.5, 'inflation': -1.0, 'recovery': 1.5, 'stimulus': 1.5,
            'tariff': -1.0, 'sanction': -1.5, 'war': -2.5, 'peace': 1.5
        }
        self.analyzer.lexicon.update(financial_terms)

    def get_news_sentiment(self, ticker):
        """
        Enhanced news sentiment with weighted scoring.
        Title + content keywords analyzed for deeper sentiment.
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return {"score": 0, "verdict": "NEUTRAL", "news": [], "trend": "STABLE"}

            scored_news = []
            total_compound = 0
            weights = []
            
            for idx, item in enumerate(news):
                content = item.get('content', {})
                title = content.get('title', '')
                if not title:
                    continue

                # Recency weighting: newer articles get more weight
                recency_weight = max(0.5, 1.0 - (idx * 0.05))
                
                # Analyze title sentiment
                sentiment = self.analyzer.polarity_scores(title)
                compound = sentiment['compound']
                
                # Boost extreme sentiments (strong signals matter more)
                if abs(compound) > 0.5:
                    compound *= 1.2
                
                total_compound += compound * recency_weight
                weights.append(recency_weight)
                
                # Categorize
                if compound >= 0.05:
                    label = "BULLISH"
                    color = "#4ade80"
                elif compound <= -0.05:
                    label = "BEARISH"
                    color = "#f87171"
                else:
                    label = "NEUTRAL"
                    color = "#94a3b8"
                
                link = content.get('canonicalUrl', {}).get('url', '#')
                if link == '#':
                    link = content.get('clickThroughUrl', {}).get('url', '#')

                scored_news.append({
                    "title": title,
                    "publisher": content.get('provider', {}).get('displayName', 'Unknown'),
                    "link": link,
                    "time": content.get('pubDate', 'Recently'),
                    "sentiment": label,
                    "sentiment_score": round(compound, 3),
                    "color": color,
                    "weight": round(recency_weight, 2)
                })

            total_weight = sum(weights) if weights else 1
            avg_score = total_compound / total_weight if total_weight > 0 else 0
            
            # Clamp to [-1, 1]
            avg_score = max(-1.0, min(1.0, avg_score))
            
            # Verdict with confidence levels
            if avg_score >= 0.25:
                verdict = "STRONGLY BULLISH"
            elif avg_score >= 0.05:
                verdict = "BULLISH"
            elif avg_score <= -0.25:
                verdict = "STRONGLY BEARISH"
            elif avg_score <= -0.05:
                verdict = "BEARISH"
            else:
                verdict = "NEUTRAL"

            # Sentiment distribution
            bullish_count = sum(1 for n in scored_news if n['sentiment'] == 'BULLISH')
            bearish_count = sum(1 for n in scored_news if n['sentiment'] == 'BEARISH')
            neutral_count = sum(1 for n in scored_news if n['sentiment'] == 'NEUTRAL')

            return {
                "score": round(avg_score, 3),
                "verdict": verdict,
                "news": scored_news[:10],
                "distribution": {
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                    "neutral": neutral_count
                },
                "confidence": round(abs(avg_score) * 100, 1)
            }
        except Exception as e:
            print(f"Error fetching sentiment: {e}")
            return {"score": 0, "verdict": "NEUTRAL", "news": [], "distribution": {"bullish": 0, "bearish": 0, "neutral": 0}}

    # ==========================================
    # FEAR & GREED INDEX (Composite)
    # ==========================================
    
    def calculate_fear_greed(self):
        """
        Calculate a composite Fear & Greed Index (0-100).
        Components:
        1. Market Momentum (NIFTY vs 125-day MA)
        2. Market Breadth (advances vs declines approximation)
        3. Volatility (India VIX proxy)
        4. Safe Haven demand (Gold vs equity)
        5. Sentiment score from news
        
        Returns score 0 (Extreme Fear) to 100 (Extreme Greed).
        """
        components = {}
        
        # 1. MARKET MOMENTUM
        try:
            nifty = yf.download("^NSEI", period="6mo", progress=False)
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.droplevel(1)
            if not nifty.empty:
                current = float(nifty['Close'].iloc[-1])
                ma125 = float(nifty['Close'].rolling(125).mean().iloc[-1])
                momentum = min(100, max(0, ((current / ma125) - 0.95) * 1000))
                components["momentum"] = round(momentum, 1)
        except Exception:
            components["momentum"] = 50

        # 2. VOLATILITY (inverse — high vol = fear)
        try:
            vix = yf.download("^INDIAVIX", period="1mo", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.droplevel(1)
            if not vix.empty:
                current_vix = float(vix['Close'].iloc[-1])
                # VIX 10-15 = extreme greed, 30+ = extreme fear
                vol_score = max(0, min(100, 100 - (current_vix - 10) * 4))
                components["volatility"] = round(vol_score, 1)
            else:
                components["volatility"] = 50
        except Exception:
            components["volatility"] = 50

        # 3. SAFE HAVEN (Gold performance vs Nifty)
        try:
            gold = yf.download("GC=F", period="1mo", progress=False)
            if isinstance(gold.columns, pd.MultiIndex):
                gold.columns = gold.columns.droplevel(1)
            if not gold.empty and not nifty.empty:
                gold_ret = (float(gold['Close'].iloc[-1]) / float(gold['Close'].iloc[0]) - 1) * 100
                nifty_ret = (float(nifty['Close'].iloc[-1]) / float(nifty['Close'].iloc[-20]) - 1) * 100
                # If gold outperforms equity = fear
                safe_haven = max(0, min(100, 50 + (nifty_ret - gold_ret) * 5))
                components["safe_haven"] = round(safe_haven, 1)
            else:
                components["safe_haven"] = 50
        except Exception:
            components["safe_haven"] = 50

        # 4. MARKET BREADTH (Nifty vs Bank Nifty divergence as proxy)
        try:
            bank_nifty = yf.download("^NSEBANK", period="1mo", progress=False)
            if isinstance(bank_nifty.columns, pd.MultiIndex):
                bank_nifty.columns = bank_nifty.columns.droplevel(1)
            if not bank_nifty.empty:
                bn_ret = (float(bank_nifty['Close'].iloc[-1]) / float(bank_nifty['Close'].iloc[0]) - 1) * 100
                breadth = max(0, min(100, 50 + bn_ret * 3))
                components["breadth"] = round(breadth, 1)
            else:
                components["breadth"] = 50
        except Exception:
            components["breadth"] = 50

        # 5. NEWS SENTIMENT
        try:
            sent = self.get_news_sentiment("^NSEI")
            sent_score = max(0, min(100, 50 + sent['score'] * 50))
            components["sentiment"] = round(sent_score, 1)
        except Exception:
            components["sentiment"] = 50

        # COMPOSITE INDEX (equal weighted)
        if components:
            composite = np.mean(list(components.values()))
        else:
            composite = 50

        # Label
        if composite >= 80:
            label = "EXTREME GREED"
            color = "#00FF9D"
        elif composite >= 60:
            label = "GREED"
            color = "#4ade80"
        elif composite >= 40:
            label = "NEUTRAL"
            color = "#94a3b8"
        elif composite >= 20:
            label = "FEAR"
            color = "#f87171"
        else:
            label = "EXTREME FEAR"
            color = "#FF4444"

        return {
            "score": round(composite, 1),
            "label": label,
            "color": color,
            "components": components,
            "timestamp": datetime.now().isoformat()
        }

    # ==========================================
    # MARKET MOVERS
    # ==========================================

    def get_market_movers(self):
        """Fetches significant market movers (Declines) and their triggers."""
        try:
            tickers = [
                "TSLA", "AAPL", "NVDA", "MSFT", "AMD", "META", "GOOGL",
                "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "TATAMOTORS.NS", "ZOMATO.NS"
            ]
            movers = []
            
            for t in tickers:
                s = yf.Ticker(t)
                hist = s.history(period="2d")
                if len(hist) < 2:
                    continue
                
                info = s.info
                company_name = info.get('longName', t)
                change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                
                if change < -1.0: 
                    news = s.news
                    reason = news[0].get('content', {}).get('title', 'No recent news found.') if news else "Market Volatility"
                    movers.append({
                        "ticker": t, "name": company_name,
                        "change": change, "reason": reason
                    })
            
            return sorted(movers, key=lambda x: x['change'])
        except Exception as e:
            print(f"Error in market movers: {e}")
            return []

    # ==========================================
    # SECTOR SENTIMENT AGGREGATOR
    # ==========================================

    def get_sector_sentiment(self):
        """Aggregate sentiment by sector for sector rotation signals."""
        sectors = {
            "Technology": ["INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
            "Auto": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
            "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
            "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "NTPC.NS"]
        }
        
        sector_scores = {}
        for sector, tickers in sectors.items():
            scores = []
            for ticker in tickers[:2]:  # Limit to 2 per sector for speed
                try:
                    sent = self.get_news_sentiment(ticker)
                    scores.append(sent['score'])
                except Exception:
                    continue
            
            if scores:
                avg = np.mean(scores)
                if avg >= 0.05:
                    verdict = "BULLISH"
                    color = "#00FF9D"
                elif avg <= -0.05:
                    verdict = "BEARISH"
                    color = "#FF4444"
                else:
                    verdict = "NEUTRAL"
                    color = "#8B949E"
                
                sector_scores[sector] = {
                    "score": round(avg, 3),
                    "verdict": verdict,
                    "color": color
                }
        
        return sector_scores


if __name__ == "__main__":
    engine = SentimentEngine()
    result = engine.get_news_sentiment("TSLA")
    print(f"Overall Verdict: {result['verdict']} ({result['score']:.2f})")
    for n in result['news']:
        print(f"[{n['sentiment']}] {n['title']}")
    
    fg = engine.calculate_fear_greed()
    print(f"\nFear & Greed Index: {fg['score']:.1f} ({fg['label']})")
    for comp, val in fg['components'].items():
        print(f"  {comp}: {val}")
