"""
Qullamaggie-style momentum screener:
  - Relative strength rank vs. benchmark (SPY) over multiple lookbacks
  - ADR% (Average Daily Range %) — his standard volatility/position-sizing metric
  - Distance from 52-week high
  - Tight consolidation + breakout detection
  - Episodic Pivot detection (gap + volume surge on a catalyst-style move)
"""
import numpy as np
import pandas as pd


def adr_pct(df: pd.DataFrame, window: int = 20) -> float:
    """
    Average Daily Range %, Qullamaggie's core volatility measure:
    mean of (High/Low - 1) * 100 over the window. Used to size stops
    (stop distance is typically a fraction of ADR%).
    """
    if len(df) < window:
        window = len(df)
    if window < 2:
        return float("nan")
    recent = df.tail(window)
    daily_range_pct = (recent["High"] / recent["Low"] - 1) * 100
    return round(daily_range_pct.mean(), 2)


def pct_change_lookback(df: pd.DataFrame, days: int) -> float:
    if len(df) <= days:
        return float("nan")
    latest = df["Close"].iloc[-1]
    past = df["Close"].iloc[-1 - days]
    if past == 0 or pd.isna(past):
        return float("nan")
    return round((latest / past - 1) * 100, 2)


def pct_from_52w_high(df: pd.DataFrame) -> float:
    lookback = min(len(df), 252)
    high = df["High"].tail(lookback).max()
    latest = df["Close"].iloc[-1]
    if high == 0 or pd.isna(high):
        return float("nan")
    return round((latest / high - 1) * 100, 2)


def volume_ratio(df: pd.DataFrame, window: int = 20) -> float:
    """Latest volume vs its trailing average — spots volume surges."""
    if len(df) < window + 1:
        return float("nan")
    avg_vol = df["Volume"].iloc[-(window + 1):-1].mean()
    latest_vol = df["Volume"].iloc[-1]
    if avg_vol == 0 or pd.isna(avg_vol):
        return float("nan")
    return round(latest_vol / avg_vol, 2)


def is_tight_consolidation(df: pd.DataFrame, window: int = 10, max_range_pct: float = 8.0) -> bool:
    """
    Flags a tight base: the high-low range over the last `window` days is
    within max_range_pct of the window's average price. Used as the setup
    for a breakout entry (Qullamaggie's 10/20-day tight flag).
    """
    if len(df) < window:
        return False
    recent = df.tail(window)
    range_pct = (recent["High"].max() / recent["Low"].min() - 1) * 100
    return range_pct <= max_range_pct


def is_breakout_today(df: pd.DataFrame, window: int = 10) -> bool:
    """True if today's close breaks above the prior `window`-day high, on above-average volume."""
    if len(df) < window + 2:
        return False
    prior_high = df["High"].iloc[-(window + 1):-1].max()
    today_close = df["Close"].iloc[-1]
    vol_ratio = volume_ratio(df, window=20)
    return bool(today_close > prior_high and (pd.isna(vol_ratio) or vol_ratio >= 1.3))


def is_episodic_pivot(df: pd.DataFrame, gap_threshold_pct: float = 10.0, vol_multiple: float = 2.0) -> bool:
    """
    Gap-up on a catalyst with a volume surge — Bonde/Qullamaggie's "EP" setup.
    Approximates the gap as today's open vs. yesterday's close, since actual
    news/earnings-date data isn't pulled here.
    """
    if len(df) < 21:
        return False
    prev_close = df["Close"].iloc[-2]
    today_open = df["Open"].iloc[-1]
    if prev_close == 0 or pd.isna(prev_close):
        return False
    gap_pct = (today_open / prev_close - 1) * 100
    vol_ratio = volume_ratio(df, window=20)
    return bool(gap_pct >= gap_threshold_pct and not pd.isna(vol_ratio) and vol_ratio >= vol_multiple)


def compute_relative_strength_rank(price_data: dict[str, pd.DataFrame], lookback: int = 63) -> pd.Series:
    """
    Percentile rank (0-100) of each ticker's `lookback`-day return vs. the rest
    of the universe. 90+ = top decile, matches Qullamaggie's RS-line style filter.
    """
    returns = {}
    for ticker, df in price_data.items():
        r = pct_change_lookback(df, lookback)
        if not pd.isna(r):
            returns[ticker] = r
    series = pd.Series(returns)
    if series.empty:
        return series
    return (series.rank(pct=True) * 100).round(1)


def build_momentum_screen(price_data: dict[str, pd.DataFrame], min_rs_rank: float = 80.0) -> pd.DataFrame:
    """
    Builds the full momentum candidate table: RS rank, ADR%, returns over
    multiple windows, distance from 52w high, and flags for breakout / EP setups.
    """
    rs_rank = compute_relative_strength_rank(price_data, lookback=63)

    rows = []
    for ticker, df in price_data.items():
        if ticker not in rs_rank.index:
            continue
        rows.append({
            "Ticker": ticker,
            "RS Rank": rs_rank[ticker],
            "1D %": pct_change_lookback(df, 1),
            "5D %": pct_change_lookback(df, 5),
            "1M %": pct_change_lookback(df, 21),
            "3M %": pct_change_lookback(df, 63),
            "ADR%": adr_pct(df),
            "Vol Ratio": volume_ratio(df),
            "% From 52W High": pct_from_52w_high(df),
            "Tight Base": is_tight_consolidation(df),
            "Breakout Today": is_breakout_today(df),
            "Episodic Pivot": is_episodic_pivot(df),
            "Last Close": round(df["Close"].iloc[-1], 2),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out[out["RS Rank"] >= min_rs_rank].sort_values("RS Rank", ascending=False)
    return out.reset_index(drop=True)
