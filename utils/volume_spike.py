"""
Volume-spike screen: flags a stock if ANY single day within a recent window
hit a volume threshold — not the average, not the total, but each individual
day checked on its own. This catches a one-day volume spike (e.g. an
earnings pop, a news-driven surge) that would get diluted away if you
instead looked at average or rolling-sum volume over the same window.

Example: "at least one single day in the last 9 bars had volume >= 9,000,000"
means day t-1 OR t-2 OR ... OR t-9 individually cleared 9M — a stock with
one huge day and eight quiet days still qualifies, even though its average
volume over the window might be well under 9M.
"""
import pandas as pd


def any_day_volume_spike(df: pd.DataFrame, window: int = 9, threshold: float = 9_000_000) -> bool:
    """
    True if any single day's Volume within the last `window` bars is
    >= `threshold`. Checks each day individually (not an average/sum).
    """
    if len(df) == 0 or "Volume" not in df.columns:
        return False
    recent = df["Volume"].tail(window)
    return bool((recent >= threshold).any())


def days_with_volume_spike(df: pd.DataFrame, window: int = 9, threshold: float = 9_000_000) -> pd.Series:
    """
    Returns the individual days (within the last `window` bars) whose
    Volume met/exceeded `threshold`, as a Series of {date: volume}.
    Empty Series if none qualify.
    """
    if len(df) == 0 or "Volume" not in df.columns:
        return pd.Series(dtype=float)
    recent = df["Volume"].tail(window)
    return recent[recent >= threshold]


def build_volume_spike_screen(price_data: dict[str, pd.DataFrame], window: int = 9,
                               threshold: float = 9_000_000) -> pd.DataFrame:
    """
    Scans the full universe and returns a table of tickers that had at
    least one single-day volume spike within the lookback window, with
    details on which day(s) triggered it.
    """
    rows = []
    for ticker, df in price_data.items():
        spikes = days_with_volume_spike(df, window=window, threshold=threshold)
        if spikes.empty:
            continue

        most_recent_spike_date = spikes.index[-1]
        most_recent_spike_vol = spikes.iloc[-1]
        days_ago = len(df.tail(window)) - list(df.tail(window).index).index(most_recent_spike_date) - 1

        rows.append({
            "Ticker": ticker,
            "Spike Count (in window)": len(spikes),
            "Max Volume in Window": int(spikes.max()),
            "Most Recent Spike Date": most_recent_spike_date.strftime("%Y-%m-%d")
            if hasattr(most_recent_spike_date, "strftime") else str(most_recent_spike_date),
            "Most Recent Spike Volume": int(most_recent_spike_vol),
            "Days Ago": days_ago,
            "Last Close": round(df["Close"].iloc[-1], 2) if len(df) else None,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Max Volume in Window", ascending=False).reset_index(drop=True)
