"""
"Tendon" pattern: within a rolling window of the 9-day Simple Moving
Average (MA9 = SMA(Close, 9), matching "MA(9, close, 0)"), find at least
one V/U-shaped decline-then-recovery swing, followed AFTERWARD by a flat,
sideways consolidation in the MA line — the shape in the reference chart:
rise -> peak -> rounded trough -> recovery to a new high -> flat tail.

Detection approach, run on the trailing `window` bars (default 63 ≈ 3
trading months) of MA9:
  1. Find the trough — the single lowest MA9 point in the window.
  2. Confirm it's a REAL dip, not noise: the highest point before the
     trough must be at least `min_decline_pct` above it (the down-leg),
     and the highest point after the trough must be at least
     `min_recovery_pct` above it (the up-leg / recovery peak).
  3. Confirm there's room left for a consolidation tail: the recovery
     peak must occur before the trailing `consolidation_window` bars
     begin (so the order is decline -> trough -> recovery -> sideways,
     not recovery happening right at the end with no flat tail after it).
  4. Confirm that trailing zone is actually flat: its range as a % of
     its average must be within `max_consolidation_range_pct`.
"""
import numpy as np
import pandas as pd

MA_PERIOD = 9
DEFAULT_WINDOW = 63                          # ~3 months of trading days
DEFAULT_MIN_DECLINE_PCT = 8.0                # minimum drop into the trough, from the prior high
DEFAULT_MIN_RECOVERY_PCT = 8.0               # minimum rise out of the trough, to the recovery peak
DEFAULT_CONSOLIDATION_WINDOW = 12            # trailing bars checked for flatness
DEFAULT_MAX_CONSOLIDATION_RANGE_PCT = 5.0    # max (max-min)/mean of MA within the consolidation zone
MIN_TROUGH_MARGIN = 5                        # trough needs at least this many bars before it in the window


def compute_ma9(df: pd.DataFrame) -> pd.Series:
    return df["Close"].rolling(MA_PERIOD).mean()


def detect_tendon_pattern(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                           min_decline_pct: float = DEFAULT_MIN_DECLINE_PCT,
                           min_recovery_pct: float = DEFAULT_MIN_RECOVERY_PCT,
                           consolidation_window: int = DEFAULT_CONSOLIDATION_WINDOW,
                           max_consolidation_range_pct: float = DEFAULT_MAX_CONSOLIDATION_RANGE_PCT):
    """Returns a details dict if the pattern is found in the trailing `window` bars, else None."""
    ma9 = compute_ma9(df).dropna()
    if len(ma9) < window:
        return None

    recent_ma = ma9.tail(window)
    values = recent_ma.values
    n = len(values)

    trough_pos = int(np.argmin(values))

    # Need enough room before the trough for a real decline, and enough
    # room after it for both a recovery AND a trailing consolidation zone.
    if trough_pos < MIN_TROUGH_MARGIN or (n - trough_pos) < (MIN_TROUGH_MARGIN + consolidation_window):
        return None

    trough_value = values[trough_pos]
    pre_segment = values[:trough_pos + 1]
    pre_peak_value = pre_segment.max()
    pre_peak_pos = int(np.argmax(pre_segment))

    post_segment = values[trough_pos:]
    post_peak_value = post_segment.max()
    post_peak_pos = trough_pos + int(np.argmax(post_segment))

    if trough_value <= 0 or pre_peak_value <= 0:
        return None

    decline_pct = (pre_peak_value - trough_value) / pre_peak_value * 100
    recovery_pct = (post_peak_value - trough_value) / trough_value * 100

    if decline_pct < min_decline_pct or recovery_pct < min_recovery_pct:
        return None

    # Consolidation zone = bars from the recovery peak onward (capped to the
    # most recent `consolidation_window` bars of that post-peak stretch) —
    # this ties the flat zone to actually being AFTER the peak, rather than
    # an arbitrary trailing slice of the whole window that might not align
    # with where the peak actually falls.
    days_since_peak = n - 1 - post_peak_pos
    min_days_after_peak = max(3, consolidation_window // 2)
    if days_since_peak < min_days_after_peak:
        return None  # peak too close to the window's end — no real tail to judge flatness on

    post_peak_segment = values[post_peak_pos:]
    consolidation_zone = post_peak_segment[-consolidation_window:] if len(post_peak_segment) >= consolidation_window else post_peak_segment

    zone_mean = consolidation_zone.mean()
    if zone_mean <= 0:
        return None
    zone_range_pct = (consolidation_zone.max() - consolidation_zone.min()) / zone_mean * 100

    if zone_range_pct > max_consolidation_range_pct:
        return None

    return {
        "trough_date": recent_ma.index[trough_pos],
        "pre_peak_date": recent_ma.index[pre_peak_pos],
        "post_peak_date": recent_ma.index[post_peak_pos],
        "decline_pct": round(decline_pct, 2),
        "recovery_pct": round(recovery_pct, 2),
        "consolidation_range_pct": round(zone_range_pct, 2),
        "days_since_post_peak": n - 1 - post_peak_pos,
        "ma9_series": recent_ma,  # returned so the app can plot it for visual confirmation
    }


def build_tendon_screen(price_data: dict[str, pd.DataFrame], window: int = DEFAULT_WINDOW,
                         min_decline_pct: float = DEFAULT_MIN_DECLINE_PCT,
                         min_recovery_pct: float = DEFAULT_MIN_RECOVERY_PCT,
                         consolidation_window: int = DEFAULT_CONSOLIDATION_WINDOW,
                         max_consolidation_range_pct: float = DEFAULT_MAX_CONSOLIDATION_RANGE_PCT):
    """Scans the universe and returns (results_df, {ticker: match_dict}) for matches."""
    rows = []
    matches = {}
    for ticker, df in price_data.items():
        result = detect_tendon_pattern(
            df, window=window, min_decline_pct=min_decline_pct, min_recovery_pct=min_recovery_pct,
            consolidation_window=consolidation_window, max_consolidation_range_pct=max_consolidation_range_pct,
        )
        if result is None:
            continue
        matches[ticker] = result
        rows.append({
            "Ticker": ticker,
            "Trough Date": result["trough_date"].strftime("%Y-%m-%d") if hasattr(result["trough_date"], "strftime") else str(result["trough_date"]),
            "Recovery Peak Date": result["post_peak_date"].strftime("%Y-%m-%d") if hasattr(result["post_peak_date"], "strftime") else str(result["post_peak_date"]),
            "Decline %": result["decline_pct"],
            "Recovery %": result["recovery_pct"],
            "Consolidation Range %": result["consolidation_range_pct"],
            "Days Since Peak": result["days_since_post_peak"],
            "Last Close": round(df["Close"].iloc[-1], 2),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Consolidation Range %").reset_index(drop=True)
    return out, matches
