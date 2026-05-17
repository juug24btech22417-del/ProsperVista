from fpdf import FPDF
import datetime
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

class PDFReport(FPDF):
    def header(self):
        # Top branding accent bar
        self.set_fill_color(9, 105, 218) # Institutional Navy Blue
        self.rect(0, 0, 210, 4, 'F')
        
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(22, 27, 34) # Slate Black
        self.cell(0, 15, 'PROSPER VISTA', 0, 1, 'L')
        
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(100, 110, 120)
        self.cell(0, 4, 'QUANTITATIVE EQUITY RESEARCH | SYSTEMATIC INTELLIGENCE BRIEFING', 0, 1, 'L')
        self.ln(4)
        
        # Subtle horizontal divider line
        self.set_draw_color(208, 215, 222)
        self.line(15, 32, 195, 32)
        self.ln(6)

    def footer(self):
        self.set_y(-18)
        self.set_draw_color(208, 215, 222)
        self.line(15, 279, 195, 279)
        
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(140, 150, 160)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, f'CONFIDENTIAL BRIEFING | GENERATED ON: {now_str} | SECURE PORT: localhost:8501 | PAGE {self.page_no()}', 0, 0, 'C')

def generate_intelligence_report(ticker, current_price, target_price, confidence, sentiment, whale_activity, prob_up, p50, max_up, max_down, df=None):
    pdf = PDFReport()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    # 1. ASSET SPECIFICATION HEADER
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(9, 105, 218)
    pdf.cell(0, 8, f'TELEMETRY AUDIT: {ticker}', 0, 1)
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(0, 5, f'Security Telemetry: {ticker} (NSE / US Equity Feed) | Model Type: Elite Neural Consensus Model', 0, 1)
    pdf.ln(4)
    
    # 2. GRID OF CORE METRICS (Rendered in beautiful high-contrast panels)
    # Row 1
    pdf.set_fill_color(246, 248, 250) # Light gray panel background
    pdf.set_draw_color(208, 215, 222) # Soft border
    
    # Grid Cell 1: Current Price
    pdf.rect(15, 52, 57, 24, 'DF')
    pdf.set_xy(17, 54)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(53, 4, 'CURRENT PRICE', 0, 1)
    pdf.set_xy(17, 60)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 27, 34)
    pdf.cell(53, 8, f'INR {current_price:,.2f}', 0, 1)
    
    # Grid Cell 2: Target Price
    pdf.rect(76, 52, 57, 24, 'DF')
    pdf.set_xy(78, 54)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(53, 4, 'NEURAL TARGET', 0, 1)
    pdf.set_xy(78, 60)
    pdf.set_font('Helvetica', 'B', 14)
    chg = ((target_price - current_price) / current_price) * 100
    chg_clr = (9, 105, 218) if chg >= 0 else (209, 36, 47) # Navy vs Red
    pdf.set_text_color(*chg_clr)
    pdf.cell(53, 8, f'INR {target_price:,.2f} ({chg:+.2f}%)', 0, 1)
    
    # Grid Cell 3: Neural Confidence
    pdf.rect(138, 52, 57, 24, 'DF')
    pdf.set_xy(140, 54)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(53, 4, 'MODEL CONFIDENCE', 0, 1)
    pdf.set_xy(140, 60)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 27, 34)
    pdf.cell(53, 8, f'{confidence*100:.1f}% R-Sqr', 0, 1)
    
    pdf.ln(24) # Move down past grid row 1
    
    # Row 2
    # Grid Cell 4: Market Mood
    pdf.rect(15, 80, 57, 20, 'DF')
    pdf.set_xy(17, 82)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(53, 4, 'SENTIMENT MOOD', 0, 1)
    pdf.set_xy(17, 88)
    pdf.set_font('Helvetica', 'B', 11)
    mood_clr = (26, 127, 55) if sentiment == "BULLISH" else (209, 36, 47) if sentiment == "BEARISH" else (87, 96, 106)
    pdf.set_text_color(*mood_clr)
    pdf.cell(53, 6, f'{sentiment}', 0, 1)
    
    # Grid Cell 5: Whale Activity
    pdf.rect(76, 80, 57, 20, 'DF')
    pdf.set_xy(78, 82)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(53, 4, 'WHALE ACTIVITY FLOW', 0, 1)
    pdf.set_xy(78, 88)
    pdf.set_font('Helvetica', 'B', 11)
    w_clr = (26, 127, 55) if "ACCUMULATION" in whale_activity else (209, 36, 47) if "DISTRIBUTION" in whale_activity else (87, 96, 106)
    pdf.set_text_color(*w_clr)
    pdf.cell(53, 6, f'{whale_activity}', 0, 1)
    
    # Grid Cell 6: Dynamic RSI (Calculated from DataFrame)
    pdf.rect(138, 80, 57, 20, 'DF')
    pdf.set_xy(140, 82)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(53, 4, 'DYNAMIC RSI (14D)', 0, 1)
    pdf.set_xy(140, 88)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(22, 27, 34)
    
    # Calculate RSI
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
    
    pdf.ln(18)
    
    # 3. 90-DAY PERFORMANCE CHART SECTION
    chart_path = None
    if df is not None and not df.empty:
        try:
            # Matplotlib Dark Theme Line Chart
            fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor='#0B0E11')
            ax.set_facecolor('#0B0E11')
            
            prices = df['Close'].tail(90)
            dates = prices.index
            
            # Plot the line with glowing cyan color
            ax.plot(dates, prices.values, color='#58A6FF', linewidth=2.0, label='Price Feed')
            
            # Horizontal lines for 50-day SMA if available
            if len(df) >= 50:
                sma_prices = df['Close'].rolling(window=50).mean().tail(90)
                ax.plot(sma_prices.index, sma_prices.values, color='#00FF9D', linestyle='--', linewidth=1.2, label='50-day SMA')
                
            ax.set_title(f"{ticker} - 90-Day Technical Feed & Telemetry Trend", color='#FFFFFF', fontsize=10, fontweight='bold', pad=8)
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
            
            # Insert chart into PDF
            pdf.image(chart_path, x=15, y=107, w=180, h=768 / 11) # Perfect proportional height
            pdf.ln(80)
        except Exception as e:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(209, 36, 47)
            pdf.cell(0, 10, f'Warning: Technical performance chart telemetry could not render ({str(e)}).', 0, 1)
            pdf.ln(10)
    else:
        pdf.ln(10)
        
    # 4. PREDICTIVE RISK ASSESSMENT (Monte Carlo Panels)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(9, 105, 218)
    pdf.cell(0, 8, '30-DAY PREDICTIVE RISK ASSESSMENT', 0, 1)
    
    pdf.rect(15, 196, 180, 36, 'DF')
    
    # Left column of risk assessment
    pdf.set_xy(18, 198)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(85, 4, 'MONTE CARLO TARGETS', 0, 1)
    
    pdf.set_xy(18, 203)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(22, 27, 34)
    pdf.cell(85, 5, f'Median Landing Zone (P50): INR {p50:,.2f}', 0, 1)
    
    pdf.set_xy(18, 208)
    pdf.cell(85, 5, f'Max Structural Upside (P90): {max_up:+.2f}%', 0, 1)
    
    pdf.set_xy(18, 213)
    pdf.set_text_color(209, 36, 47)
    pdf.cell(85, 5, f'Max Structural Drawdown (P10): {max_down:+.2f}%', 0, 1)
    
    # Right column of risk assessment
    pdf.set_xy(105, 198)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(87, 96, 106)
    pdf.cell(85, 4, 'RISK PROFILE MATRIX', 0, 1)
    
    pdf.set_xy(105, 203)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(22, 27, 34)
    pdf.cell(85, 5, f'Probability of Profit (Win Rate): {prob_up:.1f}%', 0, 1)
    
    pdf.set_xy(105, 208)
    risk_rating = "CONSERVATIVE" if max_down > -5 else "MODERATE" if max_down > -15 else "HIGHLY SPECULATIVE"
    risk_clr = (26, 127, 55) if risk_rating == "CONSERVATIVE" else (217, 119, 6) if risk_rating == "MODERATE" else (209, 36, 47)
    pdf.cell(45, 5, 'Portfolio Risk Weighting: ', 0, 0)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*risk_clr)
    pdf.cell(40, 5, f'{risk_rating}', 0, 1)
    
    pdf.set_xy(105, 213)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(22, 27, 34)
    pdf.cell(85, 5, f'50-day SMA Telemetry Overlay: ' + (f'INR {sma_50_val:,.2f}' if sma_50_val > 0 else 'N/A'), 0, 1)
    
    pdf.ln(28)
    
    # 5. INSTITUTIONAL DISCLAIMER & STAMP
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(140, 150, 160)
    pdf.multi_cell(0, 3.5, "Disclaimer: This Quantitative Equity Research Briefing is generated automatically by Prosper Vista's algorithmic engine. The mathematical projections, including Monte Carlo landing zones and Elite Neural Consensus targets, represent statistical outcomes rather than guarantees of capital gains. Past performance is not indicative of future returns. Systematic modeling carries inherent volatility risk. All figures are based on non-adjusted market terminal data.")
    
    # Clean up temporary chart file if generated
    if chart_path and os.path.exists(chart_path):
        try:
            os.remove(chart_path)
        except:
            pass
            
    # Save the PDF to a secure system temporary directory instead of the local repo workspace!
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        filename = tmp_pdf.name
        
    pdf.output(filename)
    return filename
