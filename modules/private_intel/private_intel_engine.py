"""
Private Company Intelligence Engine
Provides sentiment, peer proxy, valuation estimation, and IPO watch
for companies with no public stock price data.
"""

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

_vader = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
#  KNOWN PRIVATE COMPANY DATABASE
#  Last known funding round valuations (USD billions) and peer tickers
# ─────────────────────────────────────────────────────────────────────────────
PRIVATE_CO_DB = {
    "spacex": {
        "display_name": "SpaceX (Space Exploration Technologies Corp.)",
        "sector": "Aerospace & Defense",
        "valuation_b": 350,
        "funding_stage": "Series N+",
        "founded": 2002,
        "peers": ["RKLB", "ASTS", "LMT", "RTX"],
        "ipo_probability": "moderate",
        "notes": "Last valued at ~$350B in Dec 2024 tender offer."
    },
    "openai": {
        "display_name": "OpenAI",
        "sector": "Artificial Intelligence",
        "valuation_b": 157,
        "funding_stage": "Series F",
        "founded": 2015,
        "peers": ["MSFT", "GOOGL", "META", "AMZN"],
        "ipo_probability": "moderate",
        "notes": "Raised $6.6B at $157B valuation in Oct 2024. IPO discussions ongoing."
    },
    "stripe": {
        "display_name": "Stripe",
        "sector": "Financial Technology",
        "valuation_b": 65,
        "funding_stage": "Series I",
        "founded": 2010,
        "peers": ["PYPL", "SQ", "ADYEN.AS", "V"],
        "ipo_probability": "high",
        "notes": "Last valued at $65B in Mar 2023. IPO expected soon."
    },
    "databricks": {
        "display_name": "Databricks",
        "sector": "Cloud & Data Analytics",
        "valuation_b": 62,
        "funding_stage": "Series J",
        "founded": 2013,
        "peers": ["SNOW", "MDB", "DDOG", "PLTR"],
        "ipo_probability": "high",
        "notes": "Raised $10B at $62B valuation in Dec 2024."
    },
    "anthropic": {
        "display_name": "Anthropic",
        "sector": "Artificial Intelligence",
        "valuation_b": 61,
        "funding_stage": "Series E",
        "founded": 2021,
        "peers": ["GOOGL", "AMZN", "MSFT", "META"],
        "ipo_probability": "low",
        "notes": "Raised $7.5B at ~$61B valuation in 2024."
    },
    "bytedance": {
        "display_name": "ByteDance (TikTok parent)",
        "sector": "Social Media & Entertainment",
        "valuation_b": 220,
        "funding_stage": "Late Stage",
        "founded": 2012,
        "peers": ["META", "SNAP", "GOOGL", "PINS"],
        "ipo_probability": "low",
        "notes": "Est. ~$220B. Regulatory hurdles in US delay IPO plans."
    },
    "shein": {
        "display_name": "Shein",
        "sector": "E-Commerce & Fast Fashion",
        "valuation_b": 66,
        "funding_stage": "Late Stage",
        "founded": 2012,
        "peers": ["AMZN", "PDD", "BABA", "JD"],
        "ipo_probability": "moderate",
        "notes": "London IPO filing pending regulatory approval."
    },
    "epic games": {
        "display_name": "Epic Games",
        "sector": "Gaming & Metaverse",
        "valuation_b": 32,
        "funding_stage": "Series N",
        "founded": 1991,
        "peers": ["RBLX", "EA", "ATVI", "TTWO"],
        "ipo_probability": "low",
        "notes": "Last valued at ~$32B. Tim Sweeney has resisted going public."
    },
    "fanatics": {
        "display_name": "Fanatics",
        "sector": "Sports Commerce",
        "valuation_b": 31,
        "funding_stage": "Late Stage",
        "founded": 2011,
        "peers": ["EBAY", "DKS", "NKE"],
        "ipo_probability": "moderate",
        "notes": "Valued at ~$31B. Expanding into sports betting and cards."
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_db(company_name: str):
    """Return DB entry for a known private company, or None."""
    key = company_name.lower().strip()
    for db_key, data in PRIVATE_CO_DB.items():
        if db_key in key or key in db_key:
            return data
    return None


def _fetch_news_sentiment(company_name: str) -> dict:
    """
    Fetch Yahoo Finance news for a company name query.
    Returns sentiment score, verdict, and top headlines.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}&newsCount=10"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return {"score": 0, "verdict": "NEUTRAL", "headlines": []}

        news_items = resp.json().get("news", [])
        if not news_items:
            return {"score": 0, "verdict": "NEUTRAL", "headlines": []}

        total = 0
        headlines = []
        for item in news_items[:8]:
            title = item.get("title", "")
            if not title:
                continue
            score = _vader.polarity_scores(title)["compound"]
            total += score
            if score >= 0.05:
                label, color = "BULLISH", "#4ade80"
            elif score <= -0.05:
                label, color = "BEARISH", "#f87171"
            else:
                label, color = "NEUTRAL", "#94a3b8"
            headlines.append({
                "title": title,
                "label": label,
                "color": color,
                "score": score,
                "link": item.get("link", "#"),
            })

        avg = total / len(headlines) if headlines else 0
        if avg >= 0.05:
            verdict = "BULLISH"
        elif avg <= -0.05:
            verdict = "BEARISH"
        else:
            verdict = "NEUTRAL"

        return {"score": round(avg, 3), "verdict": verdict, "headlines": headlines}

    except Exception:
        return {"score": 0, "verdict": "NEUTRAL", "headlines": []}


def _fetch_peer_data(peer_tickers: list) -> list:
    """
    Fetch live price + 1-day change for each peer ticker.
    Returns list of dicts with ticker, name, price, change_pct.
    """
    peers = []
    for ticker in peer_tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None)
            prev_close = getattr(info, "previous_close", None)
            if price and prev_close and prev_close > 0:
                chg = ((price - prev_close) / prev_close) * 100
            else:
                chg = 0.0
            name = yf.Ticker(ticker).info.get("shortName", ticker)
            peers.append({
                "ticker": ticker,
                "name": name,
                "price": round(price, 2) if price else None,
                "change_pct": round(chg, 2),
            })
        except Exception:
            peers.append({"ticker": ticker, "name": ticker, "price": None, "change_pct": 0.0})
    return peers


def _estimate_valuation(db_entry: dict, sentiment_score: float) -> dict:
    """
    Apply a naive sentiment-adjusted growth multiplier to the known valuation.
    """
    base = db_entry.get("valuation_b")
    if not base:
        return {"low": None, "mid": None, "high": None}

    # Small sentiment adjustment: +/- 10% based on mood
    adjustment = 1 + (sentiment_score * 0.10)
    mid = base * adjustment
    return {
        "low": round(mid * 0.88, 1),
        "mid": round(mid, 1),
        "high": round(mid * 1.12, 1),
        "currency": "USD",
    }


def _ipo_badge(probability: str) -> dict:
    mapping = {
        "high": {"label": "HIGH", "color": "#4ade80", "emoji": "🟢"},
        "moderate": {"label": "MODERATE", "color": "#facc15", "emoji": "🟡"},
        "low": {"label": "LOW", "color": "#f87171", "emoji": "🔴"},
    }
    return mapping.get(probability, mapping["low"])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_private_intel(company_name: str) -> dict:
    """
    Entry point. Given a company name string, returns a full intel dict.

    Returns:
        {
          "display_name": str,
          "sector": str,
          "sentiment": {score, verdict, headlines},
          "peers": [{ticker, name, price, change_pct}, ...],
          "valuation": {low, mid, high, currency},
          "ipo": {label, color, emoji},
          "ipo_probability": str,
          "funding_stage": str,
          "founded": int,
          "notes": str,
          "known": bool,   # True if in our DB, False if unknown private co
        }
    """
    db = _lookup_db(company_name)
    sentiment = _fetch_news_sentiment(company_name)

    if db:
        peers = _fetch_peer_data(db.get("peers", []))
        valuation = _estimate_valuation(db, sentiment["score"])
        ipo = _ipo_badge(db.get("ipo_probability", "low"))
        return {
            "display_name": db["display_name"],
            "sector": db.get("sector", "N/A"),
            "sentiment": sentiment,
            "peers": peers,
            "valuation": valuation,
            "ipo": ipo,
            "ipo_probability": db.get("ipo_probability", "low"),
            "funding_stage": db.get("funding_stage", "N/A"),
            "founded": db.get("founded", "N/A"),
            "notes": db.get("notes", ""),
            "known": True,
        }
    else:
        # Unknown private company — do best-effort with sentiment only
        return {
            "display_name": company_name.title(),
            "sector": "Private / Unlisted",
            "sentiment": sentiment,
            "peers": [],
            "valuation": {"low": None, "mid": None, "high": None},
            "ipo": _ipo_badge("low"),
            "ipo_probability": "low",
            "funding_stage": "Unknown",
            "founded": "N/A",
            "notes": "This company is not in our private company database. Showing sentiment analysis only.",
            "known": False,
        }
