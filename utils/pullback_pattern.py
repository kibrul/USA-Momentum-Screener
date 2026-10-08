"""
"Pullback" pattern (3-9 day straight up, then pullback to the 9 or 18 MA).

Two conditions, matching the reference chart:

  1. RUN: bullish candles going almost straight up for 3-9 days.
     Strict mode (default): every run candle is green (Close > Open) AND
     makes a higher high and a higher low than the candle before it.
     Relaxed mode: every run candle just closes above the prior close
     (allows an occasional red-bodied candle, i.e. "almost" straight up).
     The run is the unbroken streak of qualifying candles that ends at the
     "peak bar" (the very next candle must NOT continue the run), and its
     length must be within [min_run_days, max_run_days].

  2. PULLBACK: after the peak bar, price comes down (the pullback's lowest
     low drops below the peak bar's low) and the candle on the "touch bar"
     reaches the 9-period or 18-period simple moving average of Close:
         touch  <=>  Low <= MA * (1 + tol)
                     and (Close >= MA * (1 - tol)   if require_close_holds_ma
                          else High >= MA * (1 - tol))
     i.e. the candle came down to the MA and (by default) held it, rather
     than closing decisively through it. The touch bar must be among the
     most recent `touch_window` bars (default: the latest bar only), and
     sit 1..max_pullback_days bars after the peak bar.

Optional extra: require MA9 > MA18 at the touch bar (uptrend context, as in
the reference chart). Off by default so the scan follows the two conditions
exactly as specified.

Pure price geometry (Open/High/Low/Close), so it is currency-agnostic and the
same file is used by both the USA and DSE apps.
"""
import numpy as np
import pandas as pd

FAST_MA = 9
SLOW_MA = 18

DEFAULT_MIN_RUN_DAYS = 3
DEFAULT_MAX_RUN_DAYS = 9
DEFAULT_MAX_PULLBACK_DAYS = 5          # max bars between the peak bar and the touch bar
DEFAULT_TOUCH_WINDOW = 1               # how many of the latest bars may be the touch bar
DEFAULT_TOUCH_TOLERANCE_PCT = 0.5      # how close counts as "touching" the MA
DEFAULT_STRICT_RUN = True
DEFAULT_REQUIRE_CLOSE_HOLDS_MA = True
DEFAULT_REQUIRE_MA_UPTREND = False

MA_CHOICES = ("either", "9ma", "18ma")


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


def _touches(low, high, close, ma, tol, require_close_holds):
    """Did this candle come down to the MA (and, optionally, hold it)?"""
    if low > ma * (1 + tol):
        return False  # candle never came down to the MA
    if require_close_holds:
        return close >= ma * (1 - tol)
    return high >= ma * (1 - tol)  # range at least reaches the MA; excludes candles entirely below it


def detect_pullback(df: pd.DataFrame,
                    min_run_days: int = DEFAULT_MIN_RUN_DAYS,
                    max_run_days: int = DEFAULT_MAX_RUN_DAYS,
                    max_pullback_days: int = DEFAULT_MAX_PULLBACK_DAYS,
                    touch_window: int = DEFAULT_TOUCH_WINDOW,
                    touch_tolerance_pct: float = DEFAULT_TOUCH_TOLERANCE_PCT,
                    ma_choice: str = "either",
                    strict_run: bool = DEFAULT_STRICT_RUN,
                    require_close_holds_ma: bool = DEFAULT_REQUIRE_CLOSE_HOLDS_MA,
                    require_ma_uptrend: bool = DEFAULT_REQUIRE_MA_UPTREND):
    """Returns a details dict for the most recent matching setup, or None."""
    if any(col not in df.columns for col in ("Open", "High", "Low", "Close")):
        return None

    # Only the most recent bars can matter (touch bars + pullback + run + the 18-bar MA lookback), so trim the
    # work for long histories. MA values on these bars are identical to those from the full history.
    df = df.tail(SLOW_MA + max_run_days + max_pullback_days + touch_window + 3)
    if len(df) < SLOW_MA + 2:
        return None

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    ma9 = df["Close"].rolling(FAST_MA).mean().to_numpy(dtype=float)
    ma18 = df["Close"].rolling(SLOW_MA).mean().to_numpy(dtype=float)
    n = len(df)
    tol = touch_tolerance_pct / 100.0

    for t in range(n - 1, max(n - 1 - touch_window, 0), -1):  # most recent touch bar first
        if l[t] <= 0 or h[t] <= 0:
            continue  # bad data row (zero price) — never treat as a touch
        m9, m18 = ma9[t], ma18[t]
        if np.isnan(m9) or np.isnan(m18):
            continue
        if require_ma_uptrend and not (m9 > m18):
            continue

        touched = []
        if ma_choice in ("either", "9ma") and _touches(l[t], h[t], c[t], m9, tol, require_close_holds_ma):
            touched.append("9MA")
        if ma_choice in ("either", "18ma") and _touches(l[t], h[t], c[t], m18, tol, require_close_holds_ma):
            touched.append("18MA")
        if not touched:
            continue

        for d in range(1, max_pullback_days + 1):
            p = t - d  # candidate peak bar (last bar of the run)
            if p < 1:
                break

            run_len = _run_length_ending_at(o, h, l, c, p, strict_run, max_run_days)
            if run_len < min_run_days or run_len > max_run_days:
                continue

            # The run must END at the peak bar: if the very next bar still continues the run, this isn't the
            # peak (it would let an 11-day run masquerade as a 9-day one with its last days counted as "pullback").
            if _run_bar_qualifies(o, h, l, c, p + 1, strict_run):
                continue

            # The pullback must actually come down: lowest low after the peak bar drops below the peak bar's low.
            pullback_lows = l[p + 1:t + 1]
            pullback_lows = pullback_lows[pullback_lows > 0]
            if pullback_lows.size == 0 or pullback_lows.min() >= l[p]:
                continue

            run_start_ref = p - run_len  # bar just before the run began
            return {
                "touch_date": df.index[t],
                "days_ago": n - 1 - t,
                "touched_ma": " & ".join(touched),
                "peak_date": df.index[p],
                "run_days": run_len,
                "run_gain_pct": round((c[p] / c[run_start_ref] - 1) * 100, 2) if c[run_start_ref] > 0 else float("nan"),
                "pullback_days": d,
                "pullback_depth_pct": round((h[p] - pullback_lows.min()) / h[p] * 100, 2) if h[p] > 0 else float("nan"),
                "ma9": round(m9, 2),
                "ma18": round(m18, 2),
                "close": round(c[t], 2),
            }
    return None


def build_pullback_screen(price_data: dict, **params) -> pd.DataFrame:
    """Scans the universe; `params` are passed straight to detect_pullback()."""
    rows = []
    for ticker, df in price_data.items():
        r = detect_pullback(df, **params)
        if r is None:
            continue
        fmt = lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        rows.append({
            "Ticker": ticker,
            "Touch Date": fmt(r["touch_date"]),
            "Days Ago": r["days_ago"],
            "MA Touched": r["touched_ma"],
            "Run Days": r["run_days"],
            "Run Gain %": r["run_gain_pct"],
            "Peak Date": fmt(r["peak_date"]),
            "Pullback Days": r["pullback_days"],
            "Pullback Depth %": r["pullback_depth_pct"],
            "MA9": r["ma9"],
            "MA18": r["ma18"],
            "Last Close": round(float(df["Close"].iloc[-1]), 2),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Days Ago", "Run Gain %"], ascending=[True, False]).reset_index(drop=True)


def pullback_chart_data(df: pd.DataFrame, bars: int = 45) -> pd.DataFrame:
    """Close + MA9 + MA18 for the last `bars` bars, for a quick visual check of a match."""
    out = pd.DataFrame({
        "Close": df["Close"],
        "MA9": df["Close"].rolling(FAST_MA).mean(),
        "MA18": df["Close"].rolling(SLOW_MA).mean(),
    })
    return out.tail(bars)
