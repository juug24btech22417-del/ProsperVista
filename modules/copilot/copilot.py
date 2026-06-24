import streamlit as st
import google.generativeai as genai
import yfinance as yf
import os
import re

# Load dotenv configuration if present
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

def get_ai_client_key(api_key=None):
    """
    Checks if a Gemini or Groq API key is present in environment, secrets, or parameter.
    Returns (key, provider) where provider is 'gemini' or 'groq'.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
            
    if key:
        if key.startswith("gsk_"):
            return key, "groq"
        else:
            return key, "gemini"
    return None, None

def get_gemini_client(api_key=None):
    key, provider = get_ai_client_key(api_key)
    return key is not None

def get_best_gemini_model():
    """
    Finds the best available Gemini flash model dynamically.
    Defaults to 'gemini-2.0-flash' if listing fails.
    """
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Look for flash models
        flash_models = [m for m in models if "flash" in m.lower()]
        if flash_models:
            gemini_flash = [m for m in flash_models if m.startswith("models/gemini-")]
            if gemini_flash:
                best = gemini_flash[0]
                if best.startswith("models/"):
                    return best[7:]
                return best
        # Fallbacks
        for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
            for m in models:
                if fallback in m:
                    return fallback
    except Exception:
        pass
    return "gemini-2.0-flash"

def query_groq(prompt, api_key, system_instruction):
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-specdec",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            # Try fallback model if llama-3.3 is not supported/offline
            if "model" in response.text:
                payload["model"] = "llama-3.1-8b-instant"
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
            return f"Groq API Error: {response.text}"
    except Exception as e:
        return f"Groq Connection Error: {str(e)}"

def extract_tickers(text):
    """
    Finds potential tickers in the user's query.
    Looks for capital letters (3-10 chars) optionally followed by .NS or .BO.
    """
    # Pattern to match tickers like TCS, RELIANCE, AAPL, TATAPOWER.NS, BTC-USD
    pattern = r'\b[A-Z]{3,10}(?:\.[A-Z]{2})?\b|\b[A-Z]{2,5}-[A-Z]{3,5}\b'
    candidates = re.findall(pattern, text.upper())
    
    normalized = []
    for c in candidates:
        # Avoid common words that look like tickers
        if c in ["BUY", "SELL", "HOLD", "THE", "AND", "FOR", "NIFTY", "SENSEX", "USD", "INR"]:
            continue
        # Auto-append .NS for Indian stocks if pure alphabetical and not already suffixed
        if c.isalpha() and len(c) >= 3 and "." not in c:
            normalized.append(f"{c}.NS")
        else:
            normalized.append(c)
            
    return list(set(normalized))

@st.cache_data(ttl=1800)
def get_cached_copilot_ticker_info(ticker):
    try:
        t = yf.Ticker(ticker)
        # Pre-fetch news alongside info to save requests
        return {"info": t.info, "news": t.news}
    except:
        return {"info": {}, "news": []}

@st.cache_data(ttl=600)
def get_cached_copilot_ticker_history(ticker, period="5d"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except:
        return None

def get_ticker_context(ticker):
    """
    Fetches real-time context for a ticker to ground the AI's response.
    """
    try:
        data = get_cached_copilot_ticker_info(ticker)
        info = data["info"]
        name = info.get('longName', ticker)
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('navPrice')
        
        # Get historical returns for short-term trend
        hist = get_cached_copilot_ticker_history(ticker, period="5d")
        trend = "STABLE"
        if hist is not None and len(hist) >= 2:
            pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            trend = f"UP {pct:+.2f}%" if pct > 0.5 else f"DOWN {pct:+.2f}%" if pct < -0.5 else "SIDEWAYS"
            
        metrics = {
            "name": name,
            "ticker": ticker,
            "price": f"₹{price:,.2f}" if price else "N/A",
            "pe": info.get('trailingPE', 'N/A'),
            "market_cap": f"₹{info.get('marketCap', 0)/1e11:.2f}T" if info.get('marketCap') else "N/A",
            "margin": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A",
            "trend_5d": trend,
            "volume_avg": info.get('averageVolume', 'N/A')
        }
        
        # Add recent headlines
        headlines = []
        news = data["news"]
        if news:
            for item in news[:3]:
                title = item.get('content', {}).get('title')
                if title:
                    headlines.append(title)
        metrics["headlines"] = headlines
        
        return metrics
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def build_system_context(user_prompt):
    """
    Scans the prompt, fetches stock metrics, and creates grounding context.
    """
    tickers = extract_tickers(user_prompt)
    if not tickers:
        return ""
        
    context_str = "\n\n[REAL-TIME MARKET CONTEXT FOR YOUR ATTENTION]:\n"
    for ticker in tickers[:3]: # Limit to 3 tickers to preserve context window
        context = get_ticker_context(ticker)
        if "error" in context:
            continue
        context_str += f"- **{context['name']} ({context['ticker']})**:\n"
        context_str += f"  - Current Price: {context['price']}\n"
        context_str += f"  - Trailing P/E: {context['pe']}\n"
        context_str += f"  - Market Cap: {context['market_cap']}\n"
        context_str += f"  - Profit Margin: {context['margin']}\n"
        context_str += f"  - 5-Day Trend: {context['trend_5d']}\n"
        if context.get("headlines"):
            context_str += f"  - Recent News Headlines:\n"
            for h in context["headlines"]:
                context_str += f"    * {h}\n"
    return context_str

def render_copilot_panel():
    """
    Renders the Copilot interface inside a Streamlit container.
    """
    # 1. API Key Check & Input
    key, provider = get_ai_client_key()
    # Hydrate session state from the persisted key on first render so the
    # user only has to type their key once — survives app restarts.
    if "gemini_api_key" not in st.session_state or not st.session_state.gemini_api_key:
        try:
            import user_settings
            persisted = user_settings.get_api_key()
            if persisted:
                st.session_state.gemini_api_key = persisted
        except Exception:
            pass
    user_key = st.session_state.get("gemini_api_key", "") # Keep using the same state key for compatibility

    if not key and not user_key:
        st.markdown('<div style="font-size: 11px; color: #8B949E; margin-bottom: 8px;">Enter Gemini or Groq API Key to unlock Neural Copilot:</div>', unsafe_allow_html=True)
        user_key = st.text_input("Enter API Key to unlock Copilot:", type="password", key="api_key_copilot_input", label_visibility="collapsed")
        if user_key:
            st.session_state.gemini_api_key = user_key
            # Persist so the user doesn't have to re-enter on next app open
            try:
                import user_settings
                user_settings.set_api_key(user_key)
            except Exception:
                pass
            st.rerun()
        st.markdown('<div style="font-size: 10px; color: #58A6FF; margin-top: 5px;">💡 Supports Google Gemini keys and Groq keys (starting with gsk_). Key is stored locally in user_settings.json.</div>', unsafe_allow_html=True)
        return

    # If we have a persisted/env key but no explicit session key, sync the
    # session state so the rest of the panel sees it.
    if not user_key and key:
        st.session_state.gemini_api_key = key

    active_key = user_key or key
    active_provider = "groq" if active_key.startswith("gsk_") else "gemini"

    # "Forget key" affordance — wipes the persisted key (env/st.secrets
    # remain untouched, since the user didn't put those there).
    if user_key and not key:
        if st.button("🗝️ Forget saved key", key="copilot_forget_key", help="Remove the API key saved in user_settings.json. You'll be asked for it again next time."):
            try:
                import user_settings
                user_settings.clear_api_key()
            except Exception:
                pass
            st.session_state.gemini_api_key = ""
            st.rerun()

    # Configure API with user key if provided and provider is Gemini
    if active_provider == "gemini":
        genai.configure(api_key=active_key)
        
    # 2. Initialize Chat History
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = [
            {"role": "assistant", "content": "Welcome to ProsperVista's Neural Copilot. Ask me anything about Indian stocks, compare companies, explain metrics, or analyze market trends!"}
        ]
        
    # 3. Render Scrollable Messages
    st.markdown('<div class="copilot-messages-container">', unsafe_allow_html=True)
    for msg in st.session_state.copilot_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)
            
    # 4. Render Fixed Inline Chat Input inside the Floating Window
    st.markdown('<div class="copilot-input-container">', unsafe_allow_html=True)
    with st.form(key="copilot_chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            prompt = st.text_input("Message...", placeholder="Message Copilot...", label_visibility="collapsed", key="copilot_text_prompt")
        with col_btn:
            submit = st.form_submit_button("➔")
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Handle Submission
    if submit and prompt:
        # Add user message to state
        st.session_state.copilot_messages.append({"role": "user", "content": prompt})
        
        # Grounding & Generation
        try:
            # Fetch grounding data
            grounding_context = build_system_context(prompt)
            
            system_instruction = (
                "You are ProsperVista's Neural Copilot, an elite AI quantitative research assistant. "
                "You help users make informed decisions by explaining metrics, comparing stock portfolios, and summarizing trends. "
                "Ground your answers in the real-time context provided by system alerts. If data is not provided, specify that you are using "
                "historical data. Be concise, professional, and format numbers cleanly."
            )
            
            # Compile full prompt
            full_prompt = prompt
            if grounding_context:
                full_prompt += grounding_context
                
            if active_provider == "gemini":
                # Setup model
                model = genai.GenerativeModel(get_best_gemini_model(),
                    system_instruction=system_instruction
                )
                response = model.generate_content(full_prompt)
                response_text = response.text
            else:
                # Query Groq
                response_text = query_groq(full_prompt, active_key, system_instruction)
                
            st.session_state.copilot_messages.append({"role": "assistant", "content": response_text})
            st.rerun()
        except Exception as e:
            st.error(f"Error generating response: {e}")
