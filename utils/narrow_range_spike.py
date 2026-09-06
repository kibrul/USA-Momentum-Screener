"""
Narrow-range volume spike screen: flags a stock that had a single-day
volume spike within a recent window, WHERE the price barely moved that
day — a possible "quiet accumulation/distribution" signal (large size
traded without pushing price around), as opposed to a spike that comes
with a big directional move (which the regular volume_spike.py / momentum
episodic-pivot logic already catches).

"Narrow range" here means the day's Open-to-Close % change stayed within
a tight band (default -1.9% to +1.9%) — NOT the High-Low range (that's
what utils/momentum.py's is_tight_consolidation / ADR% already cover).

Two modes, since "narrow range" and "the spike" can refer to either the
same day or different days:

  - "spike_day": the day that had the volume spike is ALSO the narrow-range
    day (high volume, but that specific day didn't move much).
  - "most_recent_day": a volume spike happened at some point in the last
    `window` days, and separately, the MOST RECENT day is currently sitting
    in a narrow range — i.e. "the spike already happened, and now it's
    gone quiet again," a classic pre-breakout compression read.
"""
import pandas as pd

DEFAULT_WINDOW = 9
DEFAULT_VOLUME_THRESHOLD = 9_000_000
DEFAULT_MAX_ABS_PCT = 1.9  # narrow-range band: Open-to-Close % change within [-1.9%, +1.9%]


def open_to_close_pct(df: pd.DataFrame, idx: int) -> float:
    """Open-to-Close % change for the row at integer position `idx` in df."""
    row = df.iloc[idx]
    if row["Open"] == 0 or pd.isna(row["Open"]) or pd.isna(row["Close"]):
        return float("nan")
    return (row["Close"] / row["Open"] - 1) * 100


def is_narrow_range_day(df: pd.DataFrame, idx: int, max_abs_pct: float = DEFAULT_MAX_ABS_PCT) -> bool:
    """True if the day at position `idx` had an Open-to-Close % change within [-max_abs_pct, +max_abs_pct]."""
    pct = open_to_close_pct(df, idx)
    if pd.isna(pct):
        return False
    return abs(pct) <= max_abs_pct


def find_spike_day_narrow_range(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                                 volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
                                 max_abs_pct: float = DEFAULT_MAX_ABS_PCT) -> dict | None:
    """
    Mode "spike_day": scans the last `window` days for a day that BOTH had
    volume >= threshold AND was itself a narrow-range day. Returns details
    of the most recent such day, or None if no day qualifies.
    """
    if len(df) < 1 or "Volume" not in df.columns:
        return None

    recent = df.tail(window)
    for pos in range(len(recent) - 1, -1, -1):  # most recent first
        row = recent.iloc[pos]
        if row["Volume"] >= volume_threshold:
            global_idx = len(df) - len(recent) + pos
            pct = open_to_close_pct(df, global_idx)
            if pd.notna(pct) and abs(pct) <= max_abs_pct:
                return {
                    "date": recent.index[pos],
                    "volume": row["Volume"],
                    "open_to_close_pct": round(pct, 2),
                    "days_ago": len(recent) - pos - 1,
                }
    return None


def find_recent_spike_then_quiet(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                                  volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
                                  max_abs_pct: float = DEFAULT_MAX_ABS_PCT) -> dict | None:
    """
    Mode "most_recent_day": checks whether (a) ANY day in the last `window`
    days had volume >= threshold, AND (b) the MOST RECENT day (today/last
    close) is itself a narrow-range day. The spike day and the narrow-range
    day can be different days. Returns details if both conditions hold.
    """
    if len(df) < 2 or "Volume" not in df.columns:
        return None

    recent = df.tail(window)
    spike_mask = recent["Volume"] >= volume_threshold
    if not spike_mask.any():
        return None

    last_idx = len(df) - 1
    last_pct = open_to_close_pct(df, last_idx)
    if pd.isna(last_pct) or abs(last_pct) > max_abs_pct:
        return None

    spike_rows = recent[spike_mask]
    most_recent_spike_date = spike_rows.index[-1]
    most_recent_spike_volume = spike_rows["Volume"].iloc[-1]
    days_since_spike = list(recent.index).index(df.index[last_idx]) - list(recent.index).index(most_recent_spike_date)

    return {
        "date": df.index[last_idx],
        "last_day_open_to_close_pct": round(last_pct, 2),
        "spike_date": most_recent_spike_date,
        "spike_volume": most_recent_spike_volume,
        "days_since_spike": days_since_spike,
    }


def build_narrow_range_spike_screen(price_data: dict[str, pd.DataFrame], mode: str = "spike_day",
                                     window: int = DEFAULT_WINDOW,
                                     volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
                                     max_abs_pct: float = DEFAULT_MAX_ABS_PCT) -> pd.DataFrame:
    """
    Scans the full universe for the narrow-range volume spike pattern.
    mode: "spike_day" (spike day itself was narrow-range) or
          "most_recent_day" (spike happened recently, most recent day is narrow-range now).
    """
    finder = find_spike_day_narrow_range if mode == "spike_day" else find_recent_spike_then_quiet

    rows = []
    for ticker, df in price_data.items():
        result = finder(df, window=window, volume_threshold=volume_threshold, max_abs_pct=max_abs_pct)
        if result is None:
            continue

        if mode == "spike_day":
            rows.append({
                "Ticker": ticker,
                "Spike Date": result["date"].strftime("%Y-%m-%d") if hasattr(result["date"], "strftime") else str(result["date"]),
                "Days Ago": result["days_ago"],
                "Volume on Spike Day": int(result["volume"]),
                "Open→Close % (Spike Day)": result["open_to_close_pct"],
                "Last Close": round(df["Close"].iloc[-1], 2),
            })
        else:
            rows.append({
                "Ticker": ticker,
                "Most Recent Date": result["date"].strftime("%Y-%m-%d") if hasattr(result["date"], "strftime") else str(result["date"]),
                "Open→Close % (Most Recent Day)": result["last_day_open_to_close_pct"],
                "Spike Date": result["spike_date"].strftime("%Y-%m-%d") if hasattr(result["spike_date"], "strftime") else str(result["spike_date"]),
                "Spike Volume": int(result["spike_volume"]),
                "Days Since Spike": result["days_since_spike"],
                "Last Close": round(df["Close"].iloc[-1], 2),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_col = "Volume on Spike Day" if mode == "spike_day" else "Spike Volume"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)
