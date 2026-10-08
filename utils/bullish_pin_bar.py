"""
Bullish Pin Bar pattern: within the last N trading days (default 3), finds
any single day whose candle matches a bullish pin bar / hammer shape:

  - Small real body located near the TOP of the day's range (minimal upper wick)
  - Long lower wick: at least two-thirds of the day's total High-Low range
  - That day's Volume >= the prior day's Volume

Per-bar geometry:
    total_range = High - Low
    body_top    = max(Open, Close)
    body_bottom = min(Open, Close)
    body_size   = body_top - body_bottom
    upper_wick  = High - body_top      (distance from the high down to the body)
    lower_wick  = body_bottom - Low    (distance from the body down to the low)

"Near the top" is enforced by capping upper_wick as a % of the range — a
small upper wick means the body sits right up against the day's high.
"Long lower wick >= two-thirds" is the defining pin-bar/hammer rejection
of the day's low. A small-body cap is included too, though it's largely
implied once lower_wick >= 2/3 leaves little room for body + upper wick.
"""
import pandas as pd

DEFAULT_WINDOW = 3
DEFAULT_MIN_LOWER_WICK_PCT = 66.67   # at least two-thirds of total range
DEFAULT_MAX_UPPER_WICK_PCT = 10.0    # keeps the body near the top (minimal upper wick)
DEFAULT_MAX_BODY_PCT = 33.0          # small real body
DEFAULT_REQUIRE_GREEN_BODY = True    # Close > Open, for a strictly "bullish" (green) pin bar
DEFAULT_REQUIRE_VOLUME_CONFIRMATION = True  # day's Volume >= prior day's Volume


def candle_metrics(row: pd.Series):
    """Returns geometry percentages for one OHLC bar, or None if the bar is degenerate (High == Low, or NaNs)."""
    o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
    if any(pd.isna(v) for v in (o, h, l, c)):
        return None
    total_range = h - l
    if total_range <= 0:
        return None

    body_top = max(o, c)
    body_bottom = min(o, c)
    body_size = body_top - body_bottom
    upper_wick = h - body_top
    lower_wick = body_bottom - l

    return {
        "total_range": total_range,
        "body_pct": body_size / total_range * 100,
        "upper_wick_pct": upper_wick / total_range * 100,
        "lower_wick_pct": lower_wick / total_range * 100,
        "is_green": c > o,
    }


def is_bullish_pin_bar(row: pd.Series, min_lower_wick_pct: float = DEFAULT_MIN_LOWER_WICK_PCT,
                        max_upper_wick_pct: float = DEFAULT_MAX_UPPER_WICK_PCT,
                        max_body_pct: float = DEFAULT_MAX_BODY_PCT,
                        require_green_body: bool = DEFAULT_REQUIRE_GREEN_BODY) -> bool:
    m = candle_metrics(row)
    if m is None:
        return False
    if m["lower_wick_pct"] < min_lower_wick_pct:
        return False
    if m["upper_wick_pct"] > max_upper_wick_pct:
        return False
    if m["body_pct"] > max_body_pct:
        return False
    if require_green_body and not m["is_green"]:
        return False
    return True


def find_bullish_pin_bar_in_window(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                                    min_lower_wick_pct: float = DEFAULT_MIN_LOWER_WICK_PCT,
                                    max_upper_wick_pct: float = DEFAULT_MAX_UPPER_WICK_PCT,
                                    max_body_pct: float = DEFAULT_MAX_BODY_PCT,
                                    require_green_body: bool = DEFAULT_REQUIRE_GREEN_BODY,
                                    require_volume_confirmation: bool = DEFAULT_REQUIRE_VOLUME_CONFIRMATION):
    """
    Scans the last `window` trading days for a day matching the bullish pin
    bar shape AND (if require_volume_confirmation) whose Volume >= the
    immediately prior day's Volume. Returns details of the MOST RECENT
    matching day within the window, or None if no day qualifies.
    """
    if len(df) < window + 1:  # +1 so every candidate day still has a prior day to compare volume against
        return None

    recent = df.tail(window + 1)  # one extra day at the front purely for the first candidate's volume comparison
    n = len(recent)

    for pos in range(n - 1, n - 1 - window, -1):  # most recent candidate day first
        if pos < 1:
            break
        row = recent.iloc[pos]
        prev_row = recent.iloc[pos - 1]

        if require_volume_confirmation:
            if pd.isna(row.get("Volume")) or pd.isna(prev_row.get("Volume")):
                continue
            if row["Volume"] < prev_row["Volume"]:
                continue

        if is_bullish_pin_bar(row, min_lower_wick_pct, max_upper_wick_pct, max_body_pct, require_green_body):
            m = candle_metrics(row)
            return {
                "date": recent.index[pos],
                "days_ago": n - 1 - pos,
                "open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"],
                "volume": row["Volume"], "prev_volume": prev_row["Volume"],
                "lower_wick_pct": round(m["lower_wick_pct"], 1),
                "upper_wick_pct": round(m["upper_wick_pct"], 1),
                "body_pct": round(m["body_pct"], 1),
            }
    return None


def build_bullish_pin_bar_screen(price_data: dict[str, pd.DataFrame], window: int = DEFAULT_WINDOW,
                                  min_lower_wick_pct: float = DEFAULT_MIN_LOWER_WICK_PCT,
                                  max_upper_wick_pct: float = DEFAULT_MAX_UPPER_WICK_PCT,
                                  max_body_pct: float = DEFAULT_MAX_BODY_PCT,
                                  require_green_body: bool = DEFAULT_REQUIRE_GREEN_BODY,
                                  require_volume_confirmation: bool = DEFAULT_REQUIRE_VOLUME_CONFIRMATION) -> pd.DataFrame:
    rows = []
    for ticker, df in price_data.items():
        result = find_bullish_pin_bar_in_window(
            df, window=window, min_lower_wick_pct=min_lower_wick_pct,
            max_upper_wick_pct=max_upper_wick_pct, max_body_pct=max_body_pct,
            require_green_body=require_green_body, require_volume_confirmation=require_volume_confirmation,
        )
        if result is None:
            continue
        rows.append({
            "Ticker": ticker,
            "Date": result["date"].strftime("%Y-%m-%d") if hasattr(result["date"], "strftime") else str(result["date"]),
            "Days Ago": result["days_ago"],
            "Open": round(result["open"], 2), "High": round(result["high"], 2),
            "Low": round(result["low"], 2), "Close": round(result["close"], 2),
            "Volume": int(result["volume"]), "Prev Day Volume": int(result["prev_volume"]),
            "Lower Wick %": result["lower_wick_pct"], "Upper Wick %": result["upper_wick_pct"],
            "Body %": result["body_pct"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Lower Wick %", ascending=False).reset_index(drop=True)
