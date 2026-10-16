"""
Gap-up screen: finds stocks that gapped up by at least `min_gap_pct` (default 9%) in a single day on big
volume, on ANY day in the last `window` trading days (default 21, about one month), or on the latest bar only
(the "current day" check).

Definitions
  gap %   = (Open / previous Close - 1) * 100            (the opening gap, not the whole-day move)
  RVOL    = that day's Volume / average Volume of the (up to) 20 days BEFORE it
  "big high volume" = RVOL >= min_rvol (default 2.0x)  AND  Volume >= min_volume shares (default 0 = off)

A day needs at least 10 prior bars of volume history to get an RVOL; days without it are skipped.

Current day: while the US market is open, today's bar is unfinished, so its volume is only the volume so far
(RVOL reads low until the close). `partial_bar_date` lets the caller skip the volume tests for that one
unfinished bar, so a stock that gapped at the open still shows up; its gap % is already final at the open.

Pure price/volume geometry, so there is nothing market-specific; it is used by the USA app.
"""
import numpy as np
import pandas as pd

DEFAULT_WINDOW = 21            # trading days, about one month
DEFAULT_MIN_GAP_PCT = 9.0
DEFAULT_MIN_RVOL = 2.0
DEFAULT_MIN_VOLUME = 0         # shares; 0 = no absolute minimum
AVG_VOLUME_DAYS = 20
MIN_BASELINE_DAYS = 10


def find_gap_ups(df: pd.DataFrame,
                 window: int = DEFAULT_WINDOW,
                 min_gap_pct: float = DEFAULT_MIN_GAP_PCT,
                 min_rvol: float = DEFAULT_MIN_RVOL,
                 min_volume: float = DEFAULT_MIN_VOLUME,
                 latest_only: bool = False,
                 partial_bar_date=None) -> list:
    """All qualifying gap days in the scan range, oldest first (each a dict). Empty list if none."""
    if any(col not in df.columns for col in ("Open", "Close", "Volume")) or len(df) < 2:
        return []

    o = df["Open"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)
    n = len(df)
    start = n - 1 if latest_only else max(1, n - window)
    partial_date = pd.Timestamp(partial_bar_date).date() if partial_bar_date is not None else None

    found = []
    for t in range(start, n):
        prev_close, day_open = c[t - 1], o[t]
        if not (prev_close > 0) or not (day_open > 0) or np.isnan(c[t]):
            continue  # bad / missing data
        gap_pct = (day_open / prev_close - 1) * 100
        if gap_pct < min_gap_pct:
            continue

        baseline = v[max(0, t - AVG_VOLUME_DAYS):t]
        baseline = baseline[~np.isnan(baseline)]
        avg_vol = baseline.mean() if baseline.size >= MIN_BASELINE_DAYS else float("nan")
        day_vol = v[t]
        rvol = day_vol / avg_vol if (avg_vol > 0 and not np.isnan(day_vol)) else float("nan")

        is_partial = partial_date is not None and pd.Timestamp(df.index[t]).date() == partial_date
        if not is_partial:
            if np.isnan(rvol) or rvol < min_rvol:
                continue
            if np.isnan(day_vol) or day_vol < min_volume:
                continue
        elif np.isnan(avg_vol):
            continue  # nothing to compare against even for the unfinished bar

        found.append({
            "date": df.index[t],
            "days_ago": n - 1 - t,
            "gap_pct": round(gap_pct, 2),
            "open": round(day_open, 2),
            "prev_close": round(prev_close, 2),
            "close": round(c[t], 2),
            "day_change_pct": round((c[t] / day_open - 1) * 100, 2),
            "volume": day_vol,
            "avg_volume": avg_vol,
            "rvol": round(rvol, 2) if not np.isnan(rvol) else float("nan"),
            "unfinished_bar": is_partial,
        })
    return found


def build_gap_up_screen(price_data: dict, **params) -> pd.DataFrame:
    """One row per ticker (its most recent qualifying gap day); `params` go straight to find_gap_ups()."""
    rows = []
    for ticker, df in price_data.items():
        hits = find_gap_ups(df, **params)
        if not hits:
            continue
        g = hits[-1]  # most recent qualifying day
        last_close = float(df["Close"].iloc[-1])
        d = g["date"]
        rows.append({
            "Ticker": ticker,
            "Gap Date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
            "Days Ago": g["days_ago"],
            "Gap %": g["gap_pct"],
            "Day Change % (Open→Close)": g["day_change_pct"],
            "Volume": int(g["volume"]) if not np.isnan(g["volume"]) else None,
            "Avg Vol (20d)": int(g["avg_volume"]) if not np.isnan(g["avg_volume"]) else None,
            "RVOL": g["rvol"],
            "Prev Close": g["prev_close"],
            "Gap-Day Open": g["open"],
            "Gap-Day Close": g["close"],
            "Since Gap %": round((last_close / g["close"] - 1) * 100, 2) if g["close"] > 0 else float("nan"),
            "Gap Days (in range)": len(hits),
            "Last Close": round(last_close, 2),
            "Unfinished Bar": g["unfinished_bar"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Days Ago", "Gap %"], ascending=[True, False]).reset_index(drop=True)


def gap_chart_data(df: pd.DataFrame, gap_date, prev_close: float, bars: int = 60) -> pd.DataFrame:
    """Close for the last `bars` bars plus a flat 'Pre-gap close' line from the gap day on (does price hold the gap?)."""
    out = pd.DataFrame({"Close": df["Close"]}).tail(bars)
    out["Pre-gap Close"] = np.where(out.index >= pd.Timestamp(gap_date), prev_close, np.nan)
    return out
