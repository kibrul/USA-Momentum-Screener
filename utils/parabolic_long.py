"""
Parabolic long scan: stocks that have crashed hard, are still falling several days in a row, and sit far below
their 9 and 18 day moving averages, i.e. an exhausted, over-extended sell-off that is a candidate for a sharp
bounce. This is a scanner for the setup, not a buy signal.

Conditions (every threshold is adjustable)
  1. Big fall        : the last close is at least `min_fall_pct` below the highest close of the last
                       `fall_window` trading days ("down 50-100% in a few days or weeks").
  2. Down streak     : at least `min_down_days` consecutive down days ending on the latest bar
                       ("down 3-5+ days in a row"; "minimum last 3+ consecutive down days" is the same rule).
  3. Below the MAs   : the last close is below BOTH the 9-day and the 18-day simple moving average of Close, and
                       at least `min_below_ma9_pct` below the 9-day ("minimum 3% to 9% distance from the 9 MA").
  4. Extension       : the last close is at least `min_extension_pct` below the highest High of the last
                       `high_window` trading days ("30%+ extension from the recent high").

Corporate actions: DSE prices are not adjusted for bonus shares or dividends, so a 1:1 bonus halves the raw close
overnight and would look like a 50% crash. When the data has DSE's YCP column (the exchange-adjusted previous
close) every price is first re-based so that each day's change is Close / YCP, which removes those jumps; the
adjusted series is anchored to the latest close, so the last price is the real one. Without YCP the raw closes are
used. A "down day" is a close below the previous (adjusted) close; a flat or no-trade day ends the streak.

Windows longer than the available history use all the history (at least 20 bars are needed).
Pure price geometry, so nothing here is DSE-specific except the optional YCP column.
"""
import numpy as np
import pandas as pd

DEFAULT_FALL_WINDOW = 20
DEFAULT_MIN_FALL_PCT = 50.0
DEFAULT_MIN_DOWN_DAYS = 3
DEFAULT_MIN_BELOW_MA9_PCT = 3.0
DEFAULT_HIGH_WINDOW = 60
DEFAULT_MIN_EXTENSION_PCT = 30.0
FAST_MA = 9
SLOW_MA = 18
MIN_BARS = 20

COLUMNS = ["Symbol", "Last Close", "Fall %", "Below High %", "Down Days", "Streak Fall %",
           "Below 9MA %", "Below 18MA %", "Recent High"]


def _adjusted_arrays(df: pd.DataFrame):
    """Corporate-action-adjusted (close, high) arrays, anchored to the latest bar; invalid prices become NaN."""
    c = df["Close"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    n = len(c)
    factor = np.ones(n)
    if "YCP" in df.columns and n > 1:
        ycp = df["YCP"].to_numpy(dtype=float)
        prev = np.concatenate([[np.nan], c[:-1]])
        ok = np.isfinite(ycp) & np.isfinite(prev) & (ycp > 0) & (prev > 0)
        ratio = np.ones(n)
        ratio[ok] = ycp[ok] / prev[ok]                       # ~1.0 on a normal day, 0.5 on a 1:1 bonus record date
        factor = np.append(np.cumprod(ratio[:0:-1])[::-1], 1.0)   # factor[k] = product of ratio[k+1 .. n-1]
    ac = np.where(np.isfinite(c) & (c > 0), c * factor, np.nan)
    ah = np.where(np.isfinite(h) & (h > 0), h * factor, np.nan)
    return ac, ah


def _trailing_mean(values: np.ndarray, length: int) -> float:
    tail = values[-length:]
    return float(tail.mean()) if len(tail) == length and np.isfinite(tail).all() else float("nan")


def parabolic_metrics(df: pd.DataFrame, fall_window: int = DEFAULT_FALL_WINDOW, high_window: int = DEFAULT_HIGH_WINDOW):
    """All the measurements for one stock on its latest bar, or None if there isn't enough usable data."""
    if len(df) < MIN_BARS or any(col not in df.columns for col in ("High", "Close")):
        return None
    ac, ah = _adjusted_arrays(df)
    last = ac[-1]
    if not np.isfinite(last):
        return None

    closes = ac[-fall_window:]
    closes = closes[np.isfinite(closes)]
    highs = ah[-high_window:]
    highs = highs[np.isfinite(highs)]
    if closes.size == 0 or highs.size == 0:
        return None
    window_high_close, recent_high = closes.max(), highs.max()

    streak, t = 0, len(ac) - 1
    while t >= 1 and np.isfinite(ac[t]) and np.isfinite(ac[t - 1]) and ac[t] < ac[t - 1]:
        streak += 1
        t -= 1
    streak_fall = (ac[len(ac) - 1 - streak] - last) / ac[len(ac) - 1 - streak] * 100 if streak else 0.0

    ma9, ma18 = _trailing_mean(ac, FAST_MA), _trailing_mean(ac, SLOW_MA)
    return {
        "last_close": float(df["Close"].iloc[-1]),
        "fall": (window_high_close - last) / window_high_close * 100,
        "extension": (recent_high - last) / recent_high * 100,
        "down_days": streak,
        "streak_fall": streak_fall,
        "below9": (1 - last / ma9) * 100 if np.isfinite(ma9) else float("nan"),
        "below18": (1 - last / ma18) * 100 if np.isfinite(ma18) else float("nan"),
        "recent_high": float(recent_high),
    }


def build_parabolic_long_screen(price_data: dict,
                                fall_window: int = DEFAULT_FALL_WINDOW,
                                min_fall_pct: float = DEFAULT_MIN_FALL_PCT,
                                min_down_days: int = DEFAULT_MIN_DOWN_DAYS,
                                min_below_ma9_pct: float = DEFAULT_MIN_BELOW_MA9_PCT,
                                high_window: int = DEFAULT_HIGH_WINDOW,
                                min_extension_pct: float = DEFAULT_MIN_EXTENSION_PCT):
    """
    Returns (table, funnel). `table` (COLUMNS) lists the stocks that pass every condition, biggest fall first.
    `funnel` says how many stocks pass each condition on its own, which is what tells you which threshold to relax
    when the table is empty; it also carries the biggest fall and longest down streak found (with their symbols).
    """
    rows = {}
    for symbol, df in price_data.items():
        m = parabolic_metrics(df, fall_window, high_window)
        if m is not None:
            rows[symbol] = m
    funnel = {"stocks": len(rows), "fall": 0, "down_days": 0, "below_ma": 0, "extension": 0, "all": 0,
              "largest_fall": None, "longest_streak": None}
    if not rows:
        return pd.DataFrame(columns=COLUMNS), funnel

    m = pd.DataFrame.from_dict(rows, orient="index")
    c_fall = m["fall"] >= min_fall_pct
    c_streak = m["down_days"] >= min_down_days
    c_ma = (m["below9"] >= min_below_ma9_pct) & (m["below18"] > 0)      # NaN compares False
    c_ext = m["extension"] >= min_extension_pct
    passed = c_fall & c_streak & c_ma & c_ext

    funnel.update(fall=int(c_fall.sum()), down_days=int(c_streak.sum()), below_ma=int(c_ma.sum()),
                  extension=int(c_ext.sum()), all=int(passed.sum()),
                  largest_fall=(m["fall"].idxmax(), round(float(m["fall"].max()), 2)),
                  longest_streak=(m["down_days"].idxmax(), int(m["down_days"].max())))

    hits = m[passed].sort_values(["fall", "down_days"], ascending=False, kind="mergesort")
    table = pd.DataFrame({
        "Symbol": hits.index,
        "Last Close": hits["last_close"].round(2).to_numpy(),
        "Fall %": hits["fall"].round(2).to_numpy(),
        "Below High %": hits["extension"].round(2).to_numpy(),
        "Down Days": hits["down_days"].to_numpy(),
        "Streak Fall %": hits["streak_fall"].round(2).to_numpy(),
        "Below 9MA %": hits["below9"].round(2).to_numpy(),
        "Below 18MA %": hits["below18"].round(2).to_numpy(),
        "Recent High": hits["recent_high"].round(2).to_numpy(),
    })[COLUMNS]
    return table, funnel


def parabolic_chart_data(df: pd.DataFrame, bars: int = 40) -> pd.DataFrame:
    """Adjusted Close with its 9 and 18 day moving averages for the last `bars` bars (the series the scan used)."""
    ac, _ = _adjusted_arrays(df)
    s = pd.Series(ac, index=df.index)
    out = pd.DataFrame({"Close": s, "MA9": s.rolling(FAST_MA).mean(), "MA18": s.rolling(SLOW_MA).mean()})
    return out.tail(bars)
