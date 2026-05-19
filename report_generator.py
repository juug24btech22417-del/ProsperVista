from fpdf import FPDF
import datetime
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

class PDFReport(FPDF):
    def header(self):
        # Full page dark background (#0B0E11)
        self.set_fill_color(11, 14, 17)
        self.rect(0, 0, 210, 297, 'F')
        
        # Top glowing neon accent bar (#58A6FF)
        self.set_fill_color(88, 166, 255)
        self.rect(0, 0, 210, 4, 'F')
        
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(255, 255, 255) # Crisp White
        self.cell(0, 15, 'PROSPER VISTA', 0, 1, 'L')
        
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(139, 148, 158) # Elegant Slate Gray
        self.cell(0, 4, 'QUANTITATIVE EQUITY RESEARCH | SYSTEMATIC INTELLIGENCE BRIEFING', 0, 1, 'L')
        self.ln(4)
        
        # Subtle horizontal divider line (#30363D) - placed safely at Y=34 to prevent overlap
        self.set_draw_color(48, 54, 61)
        self.line(15, 34, 195, 34)
        self.ln(6)

    def footer(self):
        self.set_y(-18)
        
        # Footer Divider Line (#30363D)
        self.set_draw_color(48, 54, 61)
        self.line(15, 276, 195, 276)
        
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(139, 148, 158)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, f'CONFIDENTIAL BRIEFING | GENERATED ON: {now_str} | SECURE PORT: localhost:8501 | PAGE {self.page_no()}', 0, 0, 'C')

def generate_intelligence_report(ticker, current_price, target_price, confidence, sentiment, whale_activity, prob_up, p50, max_up, max_down, df=None):
    pdf = PDFReport()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    # 1. ASSET SPECIFICATION HEADER
    pdf.set_xy(15, 38)
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(88, 166, 255) # Glowing Neon Blue (#58A6FF)
    pdf.cell(0, 8, f'TELEMETRY AUDIT: {ticker}', 0, 1)
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(139, 148, 158) # Slate Gray
    pdf.cell(0, 5, f'Security Telemetry: {ticker} (NSE / US Equity Feed) | Model Type: Elite Neural Consensus Model', 0, 1)
    
    # 2. GRID OF CORE METRICS (Glassmorphic dark panels - #161B22 background with #30363D borders)
    pdf.set_fill_color(22, 27, 34)
    pdf.set_draw_color(48, 54, 61)
    
    # Grid Cell 1: Current Price
    pdf.rect(15, 54, 57, 22, 'DF')
    pdf.set_xy(17, 56)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(53, 4, 'CURRENT PRICE', 0, 1)
    pdf.set_xy(17, 61)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(53, 8, f'INR {current_price:,.2f}', 0, 1)
    
    # Grid Cell 2: Neural Target
    pdf.rect(76, 54, 57, 22, 'DF')
    pdf.set_xy(78, 56)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(53, 4, 'NEURAL TARGET', 0, 1)
    pdf.set_xy(78, 61)
    chg = ((target_price - current_price) / current_price) * 100
    chg_clr = (0, 200, 117) if chg >= 0 else (255, 68, 68) # Neon Green vs Neon Red
    pdf.set_text_color(*chg_clr)
    pdf.set_font('Helvetica', 'B', 13) # Slightly compact to fit percent cleanly
    pdf.cell(53, 8, f'INR {target_price:,.2f} ({chg:+.2f}%)', 0, 1)
    
    # Grid Cell 3: Neural Confidence
    pdf.rect(138, 54, 57, 22, 'DF')
    pdf.set_xy(140, 56)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(53, 4, 'MODEL CONFIDENCE', 0, 1)
    pdf.set_xy(140, 61)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(53, 8, f'{confidence*100:.1f}% R-Sqr', 0, 1)
    
    # Grid Cell 4: Sentiment Mood
    pdf.rect(15, 80, 57, 20, 'DF')
    pdf.set_xy(17, 82)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(53, 4, 'SENTIMENT MOOD', 0, 1)
    pdf.set_xy(17, 87)
    pdf.set_font('Helvetica', 'B', 11)
    mood_clr = (0, 200, 117) if sentiment == "BULLISH" else (255, 68, 68) if sentiment == "BEARISH" else (139, 148, 158)
    pdf.set_text_color(*mood_clr)
    pdf.cell(53, 6, f'{sentiment}', 0, 1)
    
    # Grid Cell 5: Whale Activity
    pdf.rect(76, 80, 57, 20, 'DF')
    pdf.set_xy(78, 82)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(53, 4, 'WHALE ACTIVITY FLOW', 0, 1)
    pdf.set_xy(78, 87)
    pdf.set_font('Helvetica', 'B', 11)
    w_clr = (0, 200, 117) if "ACCUMULATION" in whale_activity else (255, 68, 68) if "DISTRIBUTION" in whale_activity else (139, 148, 158)
    pdf.set_text_color(*w_clr)
    pdf.cell(53, 6, f'{whale_activity}', 0, 1)
    
    # Grid Cell 6: Dynamic RSI (14D)
    pdf.rect(138, 80, 57, 20, 'DF')
    pdf.set_xy(140, 82)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(53, 4, 'DYNAMIC RSI (14D)', 0, 1)
    pdf.set_xy(140, 87)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(255, 255, 255)
    
    # Calculate RSI & SMA
    rsi_val = 50.0
    sma_50_val = 0.0
    if df is not None and len(df) >= 14:
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])
        except:
            pass
    if df is not None and len(df) >= 50:
        try:
            sma_50_val = float(df['Close'].rolling(window=50).mean().iloc[-1])
        except:
            pass
            
    pdf.cell(53, 6, f'{rsi_val:.1f} ({"Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral Zone"})', 0, 1)
    
    # 3. 90-DAY PERFORMANCE CHART SECTION (Y=104 to Y=172)
    chart_path = None
    if df is not None and not df.empty:
        try:
            # Matplotlib Dark Theme Line Chart matching PDF dark aesthetic
            fig, ax = plt.subplots(figsize=(7.5, 3.0), facecolor='#0B0E11')
            ax.set_facecolor('#0B0E11')
            
            prices = df['Close'].tail(90)
            dates = prices.index
            
            # Plot the line with glowing cyan color
            ax.plot(dates, prices.values, color='#58A6FF', linewidth=2.0, label='Price Feed')
            
            # Horizontal lines for 50-day SMA if available
            if len(df) >= 50:
                sma_prices = df['Close'].rolling(window=50).mean().tail(90)
                ax.plot(sma_prices.index, sma_prices.values, color='#00FF9D', linestyle='--', linewidth=1.2, label='50-day SMA')
                
            ax.set_title(f"{ticker} - 90-Day Technical Feed & Telemetry Trend", color='#FFFFFF', fontsize=9, fontweight='bold', pad=8)
            ax.tick_params(colors='#8B949E', labelsize=7)
            ax.grid(color='#21262D', linestyle=':', linewidth=0.5)
            
            # Stylize borders
            for spine in ax.spines.values():
                spine.set_color('#30363D')
                
            ax.legend(facecolor='#0B0E11', edgecolor='#30363D', labelcolor='#FFFFFF', fontsize=7, loc='upper left')
            plt.tight_layout()
            
            # Save chart to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                chart_path = tmp_img.name
            fig.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            
            # Insert chart into PDF (Absolute Y-positioning to prevent overlap!)
            pdf.image(chart_path, x=15, y=104, w=180, h=68)
        except Exception as e:
            pdf.set_xy(15, 104)
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(255, 68, 68)
            pdf.cell(0, 10, f'Warning: Technical performance chart telemetry could not render ({str(e)}).', 0, 1)

    # 4. PREDICTIVE RISK ASSESSMENT (Monte Carlo Panels - Absolute positioning Y=176 to Y=214)
    pdf.set_xy(15, 176)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(88, 166, 255) # Glowing Neon Blue (#58A6FF)
    pdf.cell(0, 8, '30-DAY PREDICTIVE RISK ASSESSMENT', 0, 1)
    
    # Draw Risk Matrix Panel
    pdf.set_fill_color(22, 27, 34)
    pdf.set_draw_color(48, 54, 61)
    pdf.rect(15, 184, 180, 34, 'DF')
    
    # Left column of risk assessment
    pdf.set_xy(18, 186)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(85, 4, 'MONTE CARLO TARGETS', 0, 1)
    
    pdf.set_xy(18, 192)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(85, 5, f'Median Landing Zone (P50): INR {p50:,.2f}', 0, 1)
    
    pdf.set_xy(18, 198)
    pdf.cell(85, 5, f'Max Structural Upside (P90): {max_up:+.2f}%', 0, 1)
    
    pdf.set_xy(18, 204)
    pdf.set_text_color(255, 68, 68) # High contrast Red
    pdf.cell(85, 5, f'Max Structural Drawdown (P10): {max_down:+.2f}%', 0, 1)
    
    # Right column of risk assessment
    pdf.set_xy(105, 186)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(85, 4, 'RISK PROFILE MATRIX', 0, 1)
    
    pdf.set_xy(105, 192)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(85, 5, f'Probability of Profit (Win Rate): {prob_up:.1f}%', 0, 1)
    
    pdf.set_xy(105, 198)
    risk_rating = "CONSERVATIVE" if max_down > -5 else "MODERATE" if max_down > -15 else "HIGHLY SPECULATIVE"
    risk_clr = (0, 200, 117) if risk_rating == "CONSERVATIVE" else (217, 119, 6) if risk_rating == "MODERATE" else (255, 68, 68)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(48, 5, 'Portfolio Risk Weighting: ', 0, 0)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*risk_clr)
    pdf.cell(40, 5, f'{risk_rating}', 0, 1)
    
    pdf.set_xy(105, 204)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(85, 5, f'50-day SMA Telemetry Overlay: ' + (f'INR {sma_50_val:,.2f}' if sma_50_val > 0 else 'N/A'), 0, 1)
    
    # 5. INSTITUTIONAL DISCLAIMER & STAMP (Y=224 onwards)
    pdf.set_xy(15, 224)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(139, 148, 158) # Elegant muted slate color
    pdf.multi_cell(180, 3.8, "Disclaimer: This Quantitative Equity Research Briefing is generated automatically by Prosper Vista's algorithmic engine. The mathematical projections, including Monte Carlo landing zones and Elite Neural Consensus targets, represent statistical outcomes rather than guarantees of capital gains. Past performance is not indicative of future returns. Systematic modeling carries inherent volatility risk. All figures are based on non-adjusted market terminal data.")
    
    # Clean up temporary chart file if generated
    if chart_path and os.path.exists(chart_path):
        try:
            os.remove(chart_path)
        except:
            pass
            
    # Save the PDF to a secure system temporary directory instead of the local repo workspace
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        filename = tmp_pdf.name
        
    pdf.output(filename)
    return filename
