"""
Stockbee-style Market Monitor: breadth indicators used to gauge whether the
overall market favors aggressive momentum trading or defense.

Core numbers Bonde tracks:
  - % of stocks up 4%+ / down 4%+ TODAY (daily momentum ratio)
  - % of stocks up 25%+ in the last 1 / 4 / 10 trading days ("momentum burst")
  - % of stocks up 25%+ / 50%+ over a quarter (longer-term participation)
"""
import pandas as pd


def _pct_change_over(df: pd.DataFrame, days: int) -> float:
    """% change of Close from `days` bars ago to the latest bar."""
    if len(df) <= days:
        return float("nan")
    latest = df["Close"].iloc[-1]
    past = df["Close"].iloc[-1 - days]
    if past == 0 or pd.isna(past):
        return float("nan")
    return (latest / past - 1) * 100


def compute_breadth_stats(price_data: dict[str, pd.DataFrame]) -> dict:
    """
    price_data: {ticker: OHLCV DataFrame}
    Returns a dict of breadth ratios (percentages) across the universe.
    """
    daily_moves = []
    burst_1d, burst_4d, burst_10d = [], [], []
    quarter_25, quarter_50 = [], []

    for ticker, df in price_data.items():
        if len(df) < 2:
            continue

        # Today's daily % move
        today_pct = _pct_change_over(df, 1)
        if not pd.isna(today_pct):
            daily_moves.append(today_pct)

        # Momentum burst windows: % up 25%+ within N trading days
        burst_1d.append(_pct_change_over(df, 1))
        burst_4d.append(_pct_change_over(df, 4))
        burst_10d.append(_pct_change_over(df, 10))

        # Quarter-length participation windows (~63 trading days)
        q_pct = _pct_change_over(df, min(63, len(df) - 1))
        if not pd.isna(q_pct):
            quarter_25.append(q_pct)
            quarter_50.append(q_pct)

    n = len(daily_moves) or 1

    def pct_above(values, threshold):
        vals = [v for v in values if not pd.isna(v)]
        if not vals:
            return 0.0
        return round(100 * sum(1 for v in vals if v >= threshold) / len(vals), 1)

    def pct_below(values, threshold):
        vals = [v for v in values if not pd.isna(v)]
        if not vals:
            return 0.0
        return round(100 * sum(1 for v in vals if v <= threshold) / len(vals), 1)

    stats = {
        "universe_size": len(price_data),
        "pct_up_4pct_today": pct_above(daily_moves, 4),
        "pct_down_4pct_today": pct_below(daily_moves, -4),
        "pct_up_25pct_1d": pct_above(burst_1d, 25),
        "pct_up_25pct_4d": pct_above(burst_4d, 25),
        "pct_up_25pct_10d": pct_above(burst_10d, 25),
        "pct_up_25pct_quarter": pct_above(quarter_25, 25),
        "pct_up_50pct_quarter": pct_above(quarter_50, 50),
    }

    # Simple derived read: ratio of up-4% to down-4% names (Bonde's "momentum ratio")
    up, down = stats["pct_up_4pct_today"], stats["pct_down_4pct_today"]
    stats["momentum_ratio"] = round(up / down, 2) if down > 0 else float("inf") if up > 0 else 0.0

    return stats


def breadth_regime_label(stats: dict) -> str:
    """Rough plain-language read of the breadth numbers, for a quick banner in the UI."""
    up4 = stats["pct_up_4pct_today"]
    down4 = stats["pct_down_4pct_today"]
    burst = stats["pct_up_25pct_10d"]

    if down4 >= 15 and down4 > up4 * 1.5:
        return "Risk-off — high % of stocks down 4%+, favor caution / cash"
    if up4 >= 15 and burst >= 5:
        return "Momentum expansion — favorable for aggressive breakout/EP entries"
    if up4 < 3 and down4 < 3:
        return "Low volatility / quiet tape — fewer high-quality setups likely"
    return "Mixed / neutral breadth"
