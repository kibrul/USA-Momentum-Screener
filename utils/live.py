"""
Live intraday layer for a WATCHLIST (not the full market — see README).

Uses yfinance's lightweight `fast_info` per ticker, which hits a much
cheaper endpoint than downloading full OHLCV bars, so polling every
1-5 minutes for a watchlist of dozens of names is reasonable on free data.

Note: yfinance quotes are typically real-time-ish during market hours but
are NOT guaranteed exchange-direct real-time — treat as "near real-time,"
not tick-accurate. Good enough for a watchlist monitor, not for execution.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

NY_TZ = ZoneInfo("America/New_York")


def is_market_open(now: datetime | None = None) -> bool:
    """
    Rough US market-hours check: Mon-Fri, 9:30-16:00 America/New_York.
    Does NOT account for market holidays (July 4th, Thanksgiving, etc.) —
    on those days this will incorrectly report "open." Good enough for a
    "should I bother auto-refreshing" gate, not for anything critical.
    """
    now = now.astimezone(NY_TZ) if now else datetime.now(NY_TZ)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def fetch_live_snapshot(tickers: list[str]) -> pd.DataFrame:
    """
    Pulls a lightweight current snapshot per ticker: last price, previous
    close, day high/low, day volume, and 10-day average volume.
    One fast_info call per ticker — fine for watchlists up to ~50-100 names,
    not meant for scanning thousands.
    """
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            last_price = info.get("last_price") or info.get("lastPrice")
            prev_close = info.get("previous_close") or info.get("previousClose")
            day_high = info.get("day_high") or info.get("dayHigh")
            day_low = info.get("day_low") or info.get("dayLow")
            day_volume = info.get("last_volume") or info.get("lastVolume") or info.get("regularMarketVolume")
            avg_vol_10d = info.get("ten_day_average_volume") or info.get("tenDayAverageVolume")

            if last_price is None or prev_close in (None, 0):
                continue

            pct_change = (last_price / prev_close - 1) * 100
            rvol = (day_volume / avg_vol_10d) if (day_volume and avg_vol_10d) else float("nan")
            pct_from_high = (last_price / day_high - 1) * 100 if day_high else float("nan")
            pct_from_low = (last_price / day_low - 1) * 100 if day_low else float("nan")
            at_day_high = bool(day_high and last_price >= day_high * 0.999)

            rows.append({
                "Ticker": t,
                "Last Price": round(last_price, 2),
                "% Change": round(pct_change, 2),
                "RVOL (vs 10d avg)": round(rvol, 2) if pd.notna(rvol) else None,
                "% From Day High": round(pct_from_high, 2) if pd.notna(pct_from_high) else None,
                "% From Day Low": round(pct_from_low, 2) if pd.notna(pct_from_low) else None,
                "At Day High": at_day_high,
                "Breakout Now": bool(at_day_high and pd.notna(rvol) and rvol >= 1.5),
                "Intraday EP": bool(pct_change >= 10 and pd.notna(rvol) and rvol >= 2.0),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def compute_live_breadth(snapshot: pd.DataFrame) -> dict:
    """Quick breadth read across the current watchlist snapshot."""
    if snapshot.empty:
        return {"up_4pct": 0, "down_4pct": 0, "avg_rvol": float("nan"), "count": 0}

    up_4 = (snapshot["% Change"] >= 4).sum()
    down_4 = (snapshot["% Change"] <= -4).sum()
    avg_rvol = snapshot["RVOL (vs 10d avg)"].dropna().mean() if snapshot["RVOL (vs 10d avg)"].notna().any() else float("nan")

    return {
        "up_4pct": int(up_4),
        "down_4pct": int(down_4),
        "avg_rvol": round(avg_rvol, 2) if pd.notna(avg_rvol) else None,
        "count": len(snapshot),
    }
