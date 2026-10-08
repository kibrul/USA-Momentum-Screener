"""
"Anticipation" pattern: a 3-9 day straight-up run, followed by a very narrow
sideways consolidation lasting 3-9 days that is still in progress (it ends on
the latest bar). The tight coil near the highs is what anticipates the next move.

Two conditions, matching the reference chart:

  1. RUN: bullish candles going almost straight up for 3-9 days.
     Strict (default): every run candle is green AND makes a higher high and a
     higher low than the candle before it. Relaxed: each candle just closes
     above the prior close (allows an occasional red-bodied candle).

  2. CONSOLIDATION: the most recent 3-9 candles (after the run's peak bar)
     move sideways in a very narrow box:
        range %        = (highest High / lowest Low - 1) * 100  <= max_range_pct
        drop from peak = (peak Close - lowest Low) / peak Close * 100
                                                          <= max_drop_from_peak_pct
     The second check keeps the coil up near the highs, so "run, crash, then go
     flat" is not mistaken for the pattern.

The consolidation must run right up to the latest bar (the setup is live now).
The run must END at the peak bar: the candle right after the peak must not
continue the run. Consolidations containing zero-volume (no-trade) days or
zero-price rows are ignored, since a frozen price is not a real base (this
matters for thinly traded DSE stocks).

Pure price geometry, so it is currency-agnostic; the same file is used by the
USA and DSE apps. Self-contained: it does not import any other utils module.
"""
import numpy as np
import pandas as pd

DEFAULT_MIN_RUN_DAYS = 3
DEFAULT_MAX_RUN_DAYS = 9
DEFAULT_MIN_CONSOLIDATION_DAYS = 3
DEFAULT_MAX_CONSOLIDATION_DAYS = 9
DEFAULT_MAX_RANGE_PCT = 2.5            # "very narrow": whole box, highest High vs lowest Low
DEFAULT_MAX_DROP_FROM_PEAK_PCT = 3.0   # box must stay near the highs
DEFAULT_STRICT_RUN = True


def _run_bar_qualifies(o, h, l, c, i, strict):
    """Does bar i count as an 'up' bar of the run (judged against bar i-1)?"""
    if i < 1:
        return False
    if strict:
        return c[i] > o[i] and h[i] > h[i - 1] and l[i] > l[i - 1]
    return c[i] > c[i - 1]


def _run_length_ending_at(o, h, l, c, p, strict, cap):
    """Length of the unbroken up-bar streak ending at bar p (stops counting just past `cap`)."""
    k = 0
    i = p
    while i >= 1 and _run_bar_qualifies(o, h, l, c, i, strict):
        k += 1
        i -= 1
        if k > cap:
            break
    return k


def detect_anticipation(df: pd.DataFrame,
                        min_run_days: int = DEFAULT_MIN_RUN_DAYS,
                        max_run_days: int = DEFAULT_MAX_RUN_DAYS,
                        min_consolidation_days: int = DEFAULT_MIN_CONSOLIDATION_DAYS,
                        max_consolidation_days: int = DEFAULT_MAX_CONSOLIDATION_DAYS,
                        max_range_pct: float = DEFAULT_MAX_RANGE_PCT,
                        max_drop_from_peak_pct: float = DEFAULT_MAX_DROP_FROM_PEAK_PCT,
                        strict_run: bool = DEFAULT_STRICT_RUN):
    """Returns a details dict if the setup is live on the latest bar, else None."""
    if any(col not in df.columns for col in ("Open", "High", "Low", "Close")):
        return None

    df = df.tail(max_run_days + max_consolidation_days + 3)  # only recent bars can matter
    n = len(df)
    if n < min_run_days + min_consolidation_days + 2:
        return None

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else None
    last = n - 1

    # Longest consolidation first, so the reported base is the full one.
    for cons in range(min(max_consolidation_days, n - 2), min_consolidation_days - 1, -1):
        p = last - cons  # peak bar = last bar of the run
        if p < 1:
            continue

        run_len = _run_length_ending_at(o, h, l, c, p, strict_run, max_run_days)
        if run_len < min_run_days or run_len > max_run_days:
            continue

        # The run must END at the peak: if the next candle still continues it, a shorter
        # consolidation (peak one bar later) is the right reading and is tried on its own.
        if _run_bar_qualifies(o, h, l, c, p + 1, strict_run):
            continue

        box_h, box_l = h[p + 1:], l[p + 1:]
        if np.isnan(box_h).any() or np.isnan(box_l).any() or (box_l <= 0).any():
            continue  # bad data rows
        if v is not None and (np.nan_to_num(v[p + 1:]) <= 0).any():
            continue  # no-trade days: a frozen price is not a real base

        hi, lo = box_h.max(), box_l.min()
        range_pct = (hi / lo - 1) * 100
        if range_pct > max_range_pct:
            continue

        drop_pct = (c[p] - lo) / c[p] * 100 if c[p] > 0 else float("nan")
        if np.isnan(drop_pct) or drop_pct > max_drop_from_peak_pct:
            continue

        start_ref = p - run_len  # bar just before the run began
        return {
            "peak_date": df.index[p],
            "run_days": run_len,
            "run_gain_pct": round((c[p] / c[start_ref] - 1) * 100, 2) if c[start_ref] > 0 else float("nan"),
            "consolidation_days": cons,
            "range_pct": round(range_pct, 2),
            "drop_from_peak_pct": round(drop_pct, 2),
            "consolidation_high": round(hi, 2),
            "consolidation_low": round(lo, 2),
            "close": round(c[last], 2),
            "to_breakout_pct": round((hi / c[last] - 1) * 100, 2) if c[last] > 0 else float("nan"),
        }
    return None


def build_anticipation_screen(price_data: dict, **params) -> pd.DataFrame:
    """Scans the universe; `params` are passed straight to detect_anticipation()."""
    rows = []
    for ticker, df in price_data.items():
        r = detect_anticipation(df, **params)
        if r is None:
            continue
        peak = r["peak_date"]
        rows.append({
            "Ticker": ticker,
            "Run Days": r["run_days"],
            "Run Gain %": r["run_gain_pct"],
            "Peak Date": peak.strftime("%Y-%m-%d") if hasattr(peak, "strftime") else str(peak),
            "Consolidation Days": r["consolidation_days"],
            "Consolidation Range %": r["range_pct"],
            "Drop From Peak %": r["drop_from_peak_pct"],
            "Consolidation High": r["consolidation_high"],
            "Consolidation Low": r["consolidation_low"],
            "Last Close": r["close"],
            "To Breakout %": r["to_breakout_pct"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Consolidation Range %", "Run Gain %"], ascending=[True, False]).reset_index(drop=True)


def anticipation_chart_data(df: pd.DataFrame, consolidation_days: int, consolidation_high: float,
                            consolidation_low: float, bars: int = 40) -> pd.DataFrame:
    """Close for the last `bars` bars, plus the consolidation box's high/low drawn over the base only."""
    out = pd.DataFrame({"Close": df["Close"]}).tail(bars)
    out["Consolidation High"] = np.nan
    out["Consolidation Low"] = np.nan
    k = min(consolidation_days, len(out))
    out.iloc[-k:, out.columns.get_loc("Consolidation High")] = consolidation_high
    out.iloc[-k:, out.columns.get_loc("Consolidation Low")] = consolidation_low
    return out
