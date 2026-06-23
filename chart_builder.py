"""
chart_builder.py — TradingView-style chart construction for Prosper Vista.

Pure functions: no Streamlit dependencies. Takes OHLCV DataFrame + indicator
set, returns a plotly Figure. Used by app.py for both daily and intraday views.

Indicators are computed via the `ta` library (already a project dependency).
Daily VWAP is computed manually since `ta` doesn't include a daily VWAP.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta


# ==========================================
#  THEME
# ==========================================
COLOR_UP = "#00FF9D"
COLOR_DOWN = "#FF4B4B"
COLOR_GRID = "#1F242C"
COLOR_AXIS = "#8B949E"
COLOR_BG = "rgba(0,0,0,0)"

# Distinct colors for moving averages so multiple can stack without colliding
SMA_COLORS = {"SMA 20": "#58A6FF", "SMA 50": "#FFA500", "SMA 200": "#BC8F8F"}
EMA_COLORS = {"EMA 9": "#DDA0DD", "EMA 21": "#7FFFD4"}


# ==========================================
#  INDICATOR COMPUTATION
# ==========================================
def _safe_series(series, fill_value=np.nan):
    """Return a clean pandas Series with NaN for warmup rows."""
    return pd.Series(series, dtype="float64").fillna(fill_value)


def compute_overlay(df, name):
    """
    Compute overlay (price-chart) indicator traces.

    Returns: list of (x, y, name, color, dash, fill, line_width) tuples.
    Caller converts to go.Scatter traces.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    x = df.index

    if name in SMA_COLORS:
        w = int(name.split()[1])
        s = ta.trend.sma_indicator(close, window=w)
        return [(x, _safe_series(s), name, SMA_COLORS[name], "solid", None, 1.5)]

    if name in EMA_COLORS:
        w = int(name.split()[1])
        s = ta.trend.ema_indicator(close, window=w)
        return [(x, _safe_series(s), name, EMA_COLORS[name], "solid", None, 1.5)]

    if name == "Bollinger (20, 2σ)":
        u = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
        l = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
        m = ta.volatility.bollinger_mavg(close, window=20)
        # Caller renders upper + lower as fill, middle as dashed line
        return [
            (x, _safe_series(u), "BB Upper", COLOR_AXIS, "solid", "tonexty", 1.0),
            (x, _safe_series(l), "BB Lower", COLOR_AXIS, "solid", None, 1.0),
            (x, _safe_series(m), "BB Mid (20)", "#FFA500", "dash", None, 1.0),
        ]

    if name == "VWAP Daily":
        # Daily VWAP: cumsum(typical_price * volume) / cumsum(volume)
        # Use a per-day reset to make it match TradingView's intraday-style anchor
        # for daily timeframe (each day's VWAP is the typical price itself, so we
        # compute a rolling anchored VWAP from the start of the visible window).
        tp = (high + low + close) / 3.0
        vwap = (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        return [(x, _safe_series(vwap), "VWAP", COLOR_UP, "solid", None, 1.5)]

    return []


def compute_subplot(df, name):
    """
    Compute subplot (separate pane) indicator traces.

    Returns: (subplot_kind, list_of_trace_dicts) where subplot_kind is a string
    the caller can use to pick axis configuration. Each trace dict has:
        x, y, name, color, mode ('lines' or 'bars'), fill (None / 'tozeroy' / 'tonexty')
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    x = df.index

    if name == "RSI 14":
        s = ta.momentum.rsi(close, window=14)
        return ("rsi", [
            {"x": x, "y": _safe_series(s), "name": "RSI 14", "color": "#DDA0DD",
             "mode": "lines", "fill": None, "ref": (30, 70)},
        ])

    if name == "MACD (12,26,9)":
        macd = ta.trend.macd(close, window_slow=26, window_fast=12)
        sig = ta.trend.macd_signal(close, window_slow=26, window_fast=12, window_sign=9)
        hist = ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9)
        # Histogram bars colored by sign
        hist_colors = [COLOR_UP if (v is not None and v >= 0) else COLOR_DOWN
                       for v in hist]
        return ("macd", [
            {"x": x, "y": _safe_series(macd), "name": "MACD", "color": "#58A6FF",
             "mode": "lines", "fill": None, "ref": None},
            {"x": x, "y": _safe_series(sig), "name": "Signal", "color": "#FFA500",
             "mode": "lines", "fill": None, "ref": None},
            {"x": x, "y": _safe_series(hist), "name": "Hist", "color": COLOR_UP,
             "mode": "bars", "fill": None, "ref": None, "marker_colors": hist_colors},
        ])

    if name == "Stochastic RSI":
        k = ta.momentum.stoch_rsi(close, window=14, smooth1=3, smooth2=3)
        d = ta.momentum.stoch(close, high, low, window=14, smooth_window=3)
        return ("stoch", [
            {"x": x, "y": _safe_series(k), "name": "StochRSI K", "color": "#58A6FF",
             "mode": "lines", "fill": None, "ref": (20, 80)},
            {"x": x, "y": _safe_series(d), "name": "Stoch D", "color": "#FFA500",
             "mode": "lines", "fill": None, "ref": None},
        ])

    if name == "ATR 14":
        s = ta.volatility.average_true_range(high, low, close, window=14)
        return ("atr", [
            {"x": x, "y": _safe_series(s), "name": "ATR 14", "color": "#DDA0DD",
             "mode": "lines", "fill": None, "ref": None},
        ])

    if name == "OBV":
        s = ta.volume.on_balance_volume(close, volume)
        return ("obv", [
            {"x": x, "y": _safe_series(s), "name": "OBV", "color": "#58A6FF",
             "mode": "lines", "fill": None, "ref": None},
        ])

    return (None, [])


# ==========================================
#  CHART BUILDERS
# ==========================================
def _make_axis(side="right", is_price=False, is_pct=False, extra=None):
    """Build a styled y-axis config."""
    cfg = dict(
        side=side,
        gridcolor=COLOR_GRID,
        zeroline=False,
        showline=False,
        color=COLOR_AXIS,
        tickfont=dict(size=10, color=COLOR_AXIS),
    )
    if is_price:
        cfg["tickprefix"] = "₹"
        cfg["tickformat"] = ",.2f"
    if is_pct:
        cfg["ticksuffix"] = "%"
        cfg["tickformat"] = ".1f"
    if extra:
        cfg.update(extra)
    return cfg


def _make_xaxis(extra=None):
    cfg = dict(
        gridcolor=COLOR_GRID,
        zeroline=False,
        showline=False,
        color=COLOR_AXIS,
        tickfont=dict(size=10, color=COLOR_AXIS),
        type="category",  # avoid weekend/holiday gap artifacts
    )
    if extra:
        cfg.update(extra)
    return cfg

def _add_overlay_traces(fig, df, indicators, row=1, always_overlays=None):
    """Add indicator overlay traces to the price subplot."""
    all_overlays = list(indicators) if indicators else []
    if always_overlays:
        all_overlays = all_overlays + always_overlays

    for spec in all_overlays:
        if isinstance(spec, dict) and spec.get("type") == "indicator":
            name = spec["name"]
        elif isinstance(spec, str):
            name = spec
        else:
            continue

        traces = compute_overlay(df, name)
        if not traces:
            continue

        if name == "Bollinger (20, 2σ)":
            # Render upper as fill, lower as outline, middle as dashed
            upper_x, upper_y, _, _, _, _, _ = traces[0]
            lower_x, lower_y, _, _, _, _, _ = traces[1]
            mid_x, mid_y, _, _, dash, _, w = traces[2]
            fig.add_trace(go.Scatter(
                x=upper_x, y=upper_y, mode="lines",
                line=dict(width=0, color=COLOR_AXIS),
                name="BB Upper", showlegend=True,
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=lower_x, y=lower_y, mode="lines",
                line=dict(width=0, color=COLOR_AXIS),
                fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
                name="BB Lower", showlegend=True,
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=mid_x, y=mid_y, mode="lines",
                line=dict(width=w, color="#FFA500", dash=dash),
                name="BB Mid (20)", showlegend=True,
            ), row=row, col=1)
        else:
            for x, y, nm, color, dash, fill, lw in traces:
                if fill == "tonexty":
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode="lines",
                        line=dict(width=lw, color=color),
                        fill=fill, fillcolor="rgba(88,166,255,0.08)",
                        name=nm,
                    ), row=row, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode="lines",
                        line=dict(width=lw, color=color, dash=dash),
                        name=nm,
                    ), row=row, col=1)


def _add_subplot_traces(fig, df, indicators, row):
    """Add subplot indicator traces to the given subplot row."""
    for name in (indicators or []):
        kind, traces = compute_subplot(df, name)
        for t in traces:
            color = t["color"]
            if t.get("marker_colors"):
                marker = dict(color=t["marker_colors"])
                fig.add_trace(go.Bar(
                    x=t["x"], y=t["y"], name=t["name"],
                    marker=marker, showlegend=(kind in ("macd",)),
                ), row=row, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=t["x"], y=t["y"], mode=t["mode"],
                    line=dict(width=1.4, color=color),
                    fill=t.get("fill"),
                    name=t["name"],
                ), row=row, col=1)
            # Reference lines (RSI 30/70, Stoch 20/80)
            if t.get("ref"):
                lo, hi = t["ref"]
                fig.add_hline(y=lo, line=dict(color=COLOR_AXIS, width=0.5, dash="dot"),
                              row=row, col=1)
                fig.add_hline(y=hi, line=dict(color=COLOR_AXIS, width=0.5, dash="dot"),
                              row=row, col=1)


def _add_volume(fig, df, row):
    """Add volume subplot. Bars colored by candle direction."""
    close = df["Close"]
    open_ = df["Open"]
    colors = np.where(close >= open_, COLOR_UP, COLOR_DOWN)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker=dict(color=colors, line=dict(width=0)),
        name="Volume", showlegend=False,
        hovertemplate="<b>%{x}</b><br>Vol: %{y:,.0f}<extra></extra>",
    ), row=row, col=1)


def _add_candlesticks(fig, df, row=1):
    """Add OHLCV candlesticks to the given subplot row."""
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color=COLOR_UP, width=1), fillcolor=COLOR_UP),
        decreasing=dict(line=dict(color=COLOR_DOWN, width=1), fillcolor=COLOR_DOWN),
        name="Price",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "O: ₹%{open:,.2f}<br>"
            "H: ₹%{high:,.2f}<br>"
            "L: ₹{low:,.2f}<br>"
            "C: ₹%{close:,.2f}<extra></extra>"
        ),
    ), row=row, col=1)


def _add_hline(fig, y, color, label, row=1):
    """Add a horizontal reference line to a subplot."""
    if y is None or np.isnan(y):
        return
    fig.add_hline(
        y=float(y),
        line=dict(color=color, width=1, dash="dash"),
        annotation_text=label, annotation_position="top left",
        annotation_font=dict(color=color, size=10),
        row=row, col=1,
    )


def _categorical_x(df):
    """Convert datetime index to category strings (no weekend gaps)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.strftime("%Y-%m-%d %H:%M") if df.index.inferred_type == "datetime64" else df.index.strftime("%Y-%m-%d")
    return df.index.astype(str)


def build_daily_chart(df, indicators=None, height=600):
    """
    Build a TradingView-style daily chart.

    df: OHLCV DataFrame (datetime index, columns Open/High/Low/Close/Volume).
    indicators: iterable of indicator name strings (subset of available).
    height: chart height in pixels.
    """
    indicators = set(indicators or [])
    overlays = {n for n in indicators if n in {
        "SMA 20", "SMA 50", "SMA 200", "EMA 9", "EMA 21", "Bollinger (20, 2σ)", "VWAP Daily"
    }}
    subplot_inds = {n for n in indicators if n in {
        "RSI 14", "MACD (12,26,9)", "Stochastic RSI", "ATR 14", "OBV"
    }}

    # Subplot rows: price (always), volume (always), indicator (only if any subplot inds)
    rows = 2 + (1 if subplot_inds else 0)
    row_heights = [0.55, 0.20, 0.25] if subplot_inds else [0.70, 0.30]
    if subplot_inds:
        row_heights = [0.55, 0.20, 0.25]
    # If 2 subplot indicators, split indicator row 50/50
    if len(subplot_inds) >= 2:
        rows = 3
        row_heights = [0.50, 0.20, 0.30]
    # Re-do rows for 1-subplot case: price/volume/ind = 3 rows total
    if subplot_inds and rows == 3:
        pass  # already correct
    elif subplot_inds and rows == 2:
        # bug guard — subplot_inds present but rows didn't grow
        rows = 3
        row_heights = [0.55, 0.20, 0.25]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # 1) Candlesticks
    _add_candlesticks(fig, df, row=1)

    # 2) Overlays on price
    _add_overlay_traces(fig, df, overlays, row=1)

    # 3) Volume
    _add_volume(fig, df, row=2)

    # 4) Subplot indicators
    if subplot_inds:
        _add_subplot_traces(fig, df, subplot_inds, row=3)

    # Layout / styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=0, r=60, t=10, b=30),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(size=10, color=COLOR_AXIS),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    # Right-side price scale on top subplot, generic on others
    fig.update_yaxes(_make_axis(is_price=True), row=1, col=1)
    fig.update_yaxes(_make_axis(), row=2, col=1)
    if subplot_inds:
        fig.update_yaxes(_make_axis(), row=3, col=1)

    # X-axes: hide tick labels on non-bottom rows
    fig.update_xaxes(_make_xaxis(), row=1, col=1)
    fig.update_xaxes(_make_xaxis({"showticklabels": False}), row=2, col=1)
    if subplot_inds:
        fig.update_xaxes(_make_xaxis({"showticklabels": False}), row=3, col=1)
    else:
        fig.update_xaxes(_make_xaxis(), row=2, col=1)

    return fig


def build_intraday_chart(df, indicators=None, height=600, always_overlays=None):
    """
    Build a TradingView-style intraday chart.

    df: OHLCV DataFrame.
    indicators: iterable of indicator name strings.
    always_overlays: list of {"type": "hline", "y": float, "color": str, "label": str}
                     for VWAP / pivot lines that are always rendered.
    """
    indicators = set(indicators or [])
    overlays = {n for n in indicators if n in {
        "SMA 20", "SMA 50", "SMA 200", "EMA 9", "EMA 21", "Bollinger (20, 2σ)"
    }}
    subplot_inds = {n for n in indicators if n in {
        "RSI 14", "MACD (12,26,9)", "Stochastic RSI", "ATR 14", "OBV"
    }}

    rows = 2 + (1 if subplot_inds else 0)
    row_heights = [0.55, 0.20, 0.25] if subplot_inds else [0.70, 0.30]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    _add_candlesticks(fig, df, row=1)
    _add_overlay_traces(fig, df, overlays, row=1)

    # Always-on intraday overlays (VWAP + pivots)
    if always_overlays:
        for spec in always_overlays:
            if spec.get("type") == "hline":
                _add_hline(fig, spec["y"], spec["color"], spec["label"], row=1)
            elif spec.get("type") == "vwap":
                vwap = spec["data"]
                fig.add_trace(go.Scatter(
                    x=df.index, y=vwap, mode="lines",
                    line=dict(color=COLOR_UP, width=2),
                    name="VWAP",
                ), row=1, col=1)

    _add_volume(fig, df, row=2)
    if subplot_inds:
        _add_subplot_traces(fig, df, subplot_inds, row=3)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=0, r=60, t=10, b=30),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(size=10, color=COLOR_AXIS),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_yaxes(_make_axis(is_price=True), row=1, col=1)
    fig.update_yaxes(_make_axis(), row=2, col=1)
    if subplot_inds:
        fig.update_yaxes(_make_axis(), row=3, col=1)

    fig.update_xaxes(_make_xaxis(), row=1, col=1)
    fig.update_xaxes(_make_xaxis({"showticklabels": False}), row=2, col=1)
    if subplot_inds:
        fig.update_xaxes(_make_xaxis({"showticklabels": False}), row=3, col=1)
    else:
        fig.update_xaxes(_make_xaxis(), row=2, col=1)

    return fig
