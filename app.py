"""
Momentum & Breadth Screener
Stockbee-style market breadth monitor + Qullamaggie-style momentum/breakout/EP screener.
Free data via yfinance — no API key required.
"""
import os
import sys

# Ensure the app's own directory is on sys.path so `utils` resolves regardless
# of the working directory Streamlit was launched from (fixes
# "ModuleNotFoundError: No module named 'utils'" when run via a shortcut,
# a different cwd, or some deployment setups).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils.data import (
    get_sp500_tickers, fetch_price_history, FALLBACK_UNIVERSE,
    fetch_full_listed_universe, filter_equities_only, fetch_last_close_prices,
)
from utils.breadth import compute_breadth_stats, breadth_regime_label
from utils.momentum import build_momentum_screen
from utils.live import is_market_open, fetch_live_snapshot, compute_live_breadth
from utils.volume_spike import build_volume_spike_screen
from utils.narrow_range_spike import build_narrow_range_spike_screen, DEFAULT_MAX_ABS_PCT
from utils.tendon_pattern import build_tendon_screen, compute_ma9, DEFAULT_WINDOW as TENDON_DEFAULT_WINDOW
from utils.bullish_pin_bar import build_bullish_pin_bar_screen, DEFAULT_WINDOW as PIN_BAR_DEFAULT_WINDOW
from utils.pullback_pattern import (
    build_pullback_screen, pullback_chart_data,
    DEFAULT_MIN_RUN_DAYS as PB_DEFAULT_MIN_RUN, DEFAULT_MAX_RUN_DAYS as PB_DEFAULT_MAX_RUN,
    DEFAULT_MAX_PULLBACK_DAYS as PB_DEFAULT_MAX_PULLBACK, DEFAULT_TOUCH_WINDOW as PB_DEFAULT_TOUCH_WINDOW,
    DEFAULT_TOUCH_TOLERANCE_PCT as PB_DEFAULT_TOLERANCE,
)
from utils.anticipation_pattern import (
    build_anticipation_screen, anticipation_chart_data,
    DEFAULT_MIN_RUN_DAYS as AN_MIN_RUN, DEFAULT_MAX_RUN_DAYS as AN_MAX_RUN,
    DEFAULT_MIN_CONSOLIDATION_DAYS as AN_MIN_CONS, DEFAULT_MAX_CONSOLIDATION_DAYS as AN_MAX_CONS,
    DEFAULT_MAX_RANGE_PCT as AN_MAX_RANGE, DEFAULT_MAX_DROP_FROM_PEAK_PCT as AN_MAX_DROP,
)

st.set_page_config(page_title="Momentum & Breadth Screener", layout="wide")

st.title("📈 Momentum & Breadth Screener")
st.caption("Stockbee-style breadth monitor + Qullamaggie-style momentum/breakout/EP screener — free data via yfinance.")

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Universe")
    universe_choice = st.radio(
        "Ticker universe",
        [
            "Full NASDAQ + NYSE (equities only, price-filtered)",
            "S&P 500 (scraped)",
            "Small fallback list (fast/offline-safe)",
            "Custom list",
        ],
        index=2,
    )

    max_price = None
    if universe_choice == "Full NASDAQ + NYSE (equities only, price-filtered)":
        st.caption(
            "Pulls Nasdaq's public symbol directories (~6,000-8,000 symbols total), "
            "filters out ETFs/warrants/units/preferreds/funds, then filters by price. "
            "This is a real full-market scan — expect it to take several minutes."
        )
        max_price = st.number_input("Max last price (USD)", min_value=0.0, value=60.0, step=1.0)
        min_price = st.number_input("Min last price (USD)", min_value=0.0, value=1.0, step=0.5)
    elif universe_choice == "Custom list":
        custom = st.text_area("Comma-separated tickers", "AAPL, NVDA, MSFT, AMD, TSLA")
        tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
    elif universe_choice == "S&P 500 (scraped)":
        tickers = get_sp500_tickers()
    else:
        tickers = FALLBACK_UNIVERSE

    if universe_choice != "Full NASDAQ + NYSE (equities only, price-filtered)":
        st.caption(f"{len(tickers)} tickers loaded")

    st.header("Momentum Screen Settings")
    min_rs_rank = st.slider("Minimum RS Rank (percentile)", 50, 99, 80)
    period = st.selectbox("Price history window", ["3mo", "6mo", "1y"], index=1)

    run_button = st.button("Run Screen", type="primary", use_container_width=True)

# ---------------- Data fetch ----------------
if run_button:
    if universe_choice == "Full NASDAQ + NYSE (equities only, price-filtered)":
        status = st.empty()
        status.info("Step 1/3 — Downloading full NASDAQ + NYSE symbol directory...")
        raw_universe = fetch_full_listed_universe()

        if raw_universe.empty:
            st.error(
                "Could not download the symbol directory from any Nasdaq mirror — this is usually "
                "a network/firewall issue reaching nasdaqtrader.com. Try the S&P 500 or fallback list "
                "instead, or check your network settings / VPN / corporate firewall."
            )
            st.stop()

        equities = filter_equities_only(raw_universe)
        status.info(
            f"Step 1/3 done — {len(raw_universe)} total listed symbols, "
            f"{len(equities)} after filtering to plain equities (no ETFs/warrants/units/preferreds/funds)."
        )

        status.info("Step 2/3 — Fetching last prices to apply the price filter (this is the slow step)...")
        progress_bar = st.progress(0.0)
        last_prices = fetch_last_close_prices(
            equities["YF_Symbol"].tolist(),
            chunk_size=200,
            pause_sec=1.0,
            progress_callback=progress_bar.progress,
        )
        progress_bar.empty()

        price_series = pd.Series(last_prices)
        survivors = price_series[(price_series >= min_price) & (price_series <= max_price)]
        status.info(
            f"Step 2/3 done — got prices for {len(price_series)}/{len(equities)} symbols; "
            f"{len(survivors)} are within ${min_price:.2f}-${max_price:.2f}."
        )

        if survivors.empty:
            st.warning("No symbols matched the price range. Try widening it.")
            st.stop()

        status.info(f"Step 3/3 — Downloading full price history for {len(survivors)} matching symbols...")
        hist_progress = st.progress(0.0)
        price_data = fetch_price_history(
            tuple(survivors.index.tolist()),
            period=period,
            chunk_size=150,
            pause_sec=1.0,
            progress_callback=hist_progress.progress,
        )
        hist_progress.empty()
        status.success(f"Done — full history loaded for {len(price_data)} symbols priced ${min_price:.2f}-${max_price:.2f}.")

        st.session_state["price_data"] = price_data
        st.session_state["universe_attempted"] = len(survivors)
        st.session_state["fetched"] = True
    else:
        with st.spinner(f"Downloading price history for {len(tickers)} tickers..."):
            price_data = fetch_price_history(tuple(tickers), period=period)
        st.session_state["price_data"] = price_data
        st.session_state["universe_attempted"] = len(tickers)
        st.session_state["fetched"] = True

if st.session_state.get("fetched"):
    price_data = st.session_state["price_data"]
    universe_attempted = st.session_state.get("universe_attempted", len(price_data))

    if not price_data:
        st.error("No price data returned. Try a smaller universe or check your network connection.")
        st.stop()

    st.success(f"Loaded data for {len(price_data)} / {universe_attempted} tickers.")

    tab_breadth, tab_momentum, tab_vol_spike, tab_narrow_range, tab_tendon, tab_pin_bar, tab_pullback, tab_anticipation, tab_live = st.tabs(
        ["📊 Market Breadth (Stockbee)", "🚀 Momentum Screener (Qullamaggie)",
         "📈 Volume Spike Scan", "🔍 Narrow Range Volume Spike", "🪢 Tendon Pattern",
         "🔨 Bullish Pin Bar", "🎯 Pullback Pattern", "🔮 Anticipation", "🔴 Live (intraday)"]
    )

    # ---------------- Breadth tab ----------------
    with tab_breadth:
        stats = compute_breadth_stats(price_data)
        regime = breadth_regime_label(stats)

        st.subheader("Market Monitor")
        st.info(f"**Read:** {regime}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("% Up 4%+ Today", f"{stats['pct_up_4pct_today']}%")
        c2.metric("% Down 4%+ Today", f"{stats['pct_down_4pct_today']}%")
        c3.metric("Momentum Ratio (Up4/Down4)", stats["momentum_ratio"])
        c4.metric("Universe Size", stats["universe_size"])

        c5, c6, c7 = st.columns(3)
        c5.metric("% Up 25%+ (1 day)", f"{stats['pct_up_25pct_1d']}%")
        c6.metric("% Up 25%+ (4 days)", f"{stats['pct_up_25pct_4d']}%")
        c7.metric("% Up 25%+ (10 days)", f"{stats['pct_up_25pct_10d']}%")

        c8, c9 = st.columns(2)
        c8.metric("% Up 25%+ (Quarter)", f"{stats['pct_up_25pct_quarter']}%")
        c9.metric("% Up 50%+ (Quarter)", f"{stats['pct_up_50pct_quarter']}%")

        st.caption(
            "These mirror Stockbee's Market Monitor ratios: daily 4% up/down counts gauge short-term "
            "volatility and risk appetite; the 25%/50% multi-day windows gauge how much real momentum "
            "('momentum bursts') is present in the market right now."
        )

    # ---------------- Momentum tab ----------------
    with tab_momentum:
        st.subheader("Momentum Candidates")
        screen_df = build_momentum_screen(price_data, min_rs_rank=min_rs_rank)
        st.session_state["last_screen_df"] = screen_df  # so the Live tab can offer it as a watchlist source

        if screen_df.empty:
            st.warning("No tickers met the RS Rank threshold. Try lowering it in the sidebar.")
        else:
            st.caption(
                f"{len(screen_df)} tickers with RS Rank ≥ {min_rs_rank}. "
                "Tight Base / Breakout Today / Episodic Pivot flag Qullamaggie-style setups."
            )

            def highlight_flags(row):
                styles = [""] * len(row)
                if row.get("Breakout Today"):
                    styles = ["background-color: #d4f4dd"] * len(row)
                elif row.get("Episodic Pivot"):
                    styles = ["background-color: #fde9c8"] * len(row)
                return styles

            st.dataframe(
                screen_df.style.apply(highlight_flags, axis=1),
                use_container_width=True,
                height=500,
            )

            st.markdown("**Flag legend:** 🟩 Breakout today · 🟧 Episodic pivot (gap + volume surge)")

            st.divider()
            st.subheader("Setup filters")
            f1, f2 = st.columns(2)
            with f1:
                if st.checkbox("Show only Breakout Today"):
                    st.dataframe(screen_df[screen_df["Breakout Today"]], use_container_width=True)
            with f2:
                if st.checkbox("Show only Episodic Pivots"):
                    st.dataframe(screen_df[screen_df["Episodic Pivot"]], use_container_width=True)

            st.caption(
                "ADR% (Average Daily Range) is Qullamaggie's standard volatility measure — commonly used "
                "to size stops as a fraction of ADR% rather than a fixed percentage."
            )

    # ---------------- Volume Spike tab ----------------
    with tab_vol_spike:
        st.subheader("Single-Day Volume Spike Scan")
        st.caption(
            "Flags stocks where at least ONE individual day within the lookback window hit the volume "
            "threshold — checked day by day, not as an average. A stock with one huge day and otherwise "
            "quiet volume still qualifies, even though its average volume over the same window is low."
        )

        vc1, vc2 = st.columns(2)
        with vc1:
            spike_window = st.number_input("Lookback window (trading days)", min_value=2, max_value=60, value=9)
        with vc2:
            spike_threshold = st.number_input(
                "Volume threshold (single day)", min_value=0, value=9_000_000, step=500_000, format="%d"
            )

        vol_spike_df = build_volume_spike_screen(price_data, window=int(spike_window), threshold=spike_threshold)

        if vol_spike_df.empty:
            st.warning(
                f"No tickers had a single day with volume ≥ {spike_threshold:,} within the last "
                f"{spike_window} trading days."
            )
        else:
            st.success(
                f"{len(vol_spike_df)} tickers had at least one day with volume ≥ {spike_threshold:,} "
                f"within the last {spike_window} trading days."
            )
            st.dataframe(vol_spike_df, use_container_width=True, height=500)
            st.caption(
                "'Days Ago' counts back from the most recent bar in the window (0 = most recent day). "
                "'Spike Count' is how many separate days in the window individually cleared the threshold."
            )

    # ---------------- Narrow Range Volume Spike tab ----------------
    with tab_narrow_range:
        st.subheader("Narrow Range Volume Spike Scan")
        st.caption(
            "Flags a high-volume day where price barely moved — a possible quiet accumulation/"
            "distribution signal (size traded without pushing price around), as opposed to a spike "
            "that comes with a big directional move."
        )

        nr_mode_label = st.radio(
            "Pattern to look for",
            [
                "Spike day itself was narrow-range (high volume, that day didn't move much)",
                "Spike happened recently, and the most recent day is narrow-range now (quiet after the spike)",
            ],
            index=0,
        )
        nr_mode = "spike_day" if nr_mode_label.startswith("Spike day itself") else "most_recent_day"

        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            nr_window = st.number_input("Lookback window (trading days)", min_value=2, max_value=60, value=9, key="nr_window")
        with nc2:
            nr_vol_threshold = st.number_input(
                "Volume threshold (single day)", min_value=0, value=9_000_000, step=500_000,
                format="%d", key="nr_vol_threshold",
            )
        with nc3:
            nr_max_pct = st.number_input(
                "Narrow-range band (± %, Open→Close)", min_value=0.1, max_value=20.0,
                value=DEFAULT_MAX_ABS_PCT, step=0.1, key="nr_max_pct",
            )

        st.caption(
            f"\"Narrow range\" means the day's Open→Close % change stayed within ±{nr_max_pct}% — "
            f"this is the day's own move, not the High-Low range (that's covered by the Momentum "
            f"tab's Tight Base / ADR% instead)."
        )

        nr_df = build_narrow_range_spike_screen(
            price_data, mode=nr_mode, window=int(nr_window),
            volume_threshold=nr_vol_threshold, max_abs_pct=nr_max_pct,
        )

        if nr_df.empty:
            st.warning(
                f"No tickers matched this pattern within the last {nr_window} trading days. "
                f"Try widening the narrow-range band or lowering the volume threshold."
            )
        else:
            st.success(f"{len(nr_df)} tickers matched this pattern.")
            st.dataframe(nr_df, use_container_width=True, height=500)
            if nr_mode == "spike_day":
                st.caption(
                    "Shows the most recent day (within the window) that had BOTH a volume spike AND "
                    "a narrow Open→Close range on that same day."
                )
            else:
                st.caption(
                    "Shows tickers where a volume spike occurred at some point in the window, and the "
                    "MOST RECENT day is now sitting in a narrow range — the spike and the quiet day can "
                    "be different days."
                )

    # ---------------- Tendon Pattern tab ----------------
    with tab_tendon:
        st.subheader("Tendon Pattern Scan")
        st.caption(
            "Looks for a V/U-shaped decline-then-recovery in the 9-day SMA of Close within a rolling "
            "window, followed AFTER the recovery by a flat, sideways consolidation — the shape: "
            "rise → peak → rounded trough → recovery to a new high → flat tail."
        )

        tc1, tc2 = st.columns(2)
        with tc1:
            tendon_window = st.number_input(
                "Rolling window (trading days, ~3 months = 63)", min_value=20, max_value=252,
                value=TENDON_DEFAULT_WINDOW, key="tendon_window",
            )
            tendon_min_decline = st.number_input(
                "Minimum decline into trough (%)", min_value=1.0, max_value=80.0, value=8.0, step=1.0,
                key="tendon_min_decline",
            )
            tendon_min_recovery = st.number_input(
                "Minimum recovery out of trough (%)", min_value=1.0, max_value=200.0, value=8.0, step=1.0,
                key="tendon_min_recovery",
            )
        with tc2:
            tendon_consolidation_window = st.number_input(
                "Consolidation tail length (trading days)", min_value=3, max_value=60, value=12,
                key="tendon_consolidation_window",
            )
            tendon_max_range = st.number_input(
                "Max consolidation range (%, tighter = flatter)", min_value=0.5, max_value=30.0,
                value=5.0, step=0.5, key="tendon_max_range",
            )

        tendon_df, tendon_matches = build_tendon_screen(
            price_data, window=int(tendon_window), min_decline_pct=tendon_min_decline,
            min_recovery_pct=tendon_min_recovery, consolidation_window=int(tendon_consolidation_window),
            max_consolidation_range_pct=tendon_max_range,
        )

        if tendon_df.empty:
            st.warning(
                "No tickers matched this pattern. Try loosening the decline/recovery minimums or "
                "widening the consolidation range."
            )
        else:
            st.success(f"{len(tendon_df)} tickers matched the Tendon pattern.")
            st.dataframe(tendon_df, use_container_width=True, height=400)
            st.caption(
                "'Consolidation Range %' is how tight the flat tail is (lower = flatter). "
                "'Days Since Peak' is how many trading days ago the recovery peak occurred."
            )

            st.divider()
            st.subheader("Visual confirmation")
            chosen_ticker = st.selectbox("Preview MA9 for a matched ticker", tendon_df["Ticker"].tolist())
            if chosen_ticker:
                match = tendon_matches[chosen_ticker]
                ma_series = match["ma9_series"]
                chart_df = pd.DataFrame({"MA9": ma_series})
                st.line_chart(chart_df, height=300)
                st.caption(
                    f"Trough: {match['trough_date'].strftime('%Y-%m-%d') if hasattr(match['trough_date'], 'strftime') else match['trough_date']} · "
                    f"Recovery peak: {match['post_peak_date'].strftime('%Y-%m-%d') if hasattr(match['post_peak_date'], 'strftime') else match['post_peak_date']} · "
                    f"Decline {match['decline_pct']}% · Recovery {match['recovery_pct']}% · "
                    f"Consolidation range {match['consolidation_range_pct']}%"
                )

    # ---------------- Bullish Pin Bar tab ----------------
    with tab_pin_bar:
        st.subheader("Bullish Pin Bar Scan")
        st.caption(
            "Looks for a bullish pin bar / hammer candle on any single day within the last few "
            "trading days: a small real body sitting near the top of the day's range, a long lower "
            "wick (rejection of the low), and volume confirming (that day's volume ≥ the prior day's)."
        )

        pb1, pb2 = st.columns(2)
        with pb1:
            pin_window = st.number_input(
                "Lookback window (trading days)", min_value=1, max_value=20,
                value=PIN_BAR_DEFAULT_WINDOW, key="pin_window",
            )
            pin_min_lower_wick = st.number_input(
                "Minimum lower wick (% of day's range)", min_value=20.0, max_value=95.0,
                value=66.67, step=1.0, key="pin_min_lower_wick",
            )
        with pb2:
            pin_max_upper_wick = st.number_input(
                "Maximum upper wick (% of day's range, keeps body near the top)", min_value=1.0,
                max_value=40.0, value=10.0, step=1.0, key="pin_max_upper_wick",
            )
            pin_max_body = st.number_input(
                "Maximum body size (% of day's range)", min_value=5.0, max_value=50.0,
                value=33.0, step=1.0, key="pin_max_body",
            )

        pb3, pb4 = st.columns(2)
        with pb3:
            pin_require_green = st.checkbox(
                "Require green body (Close > Open)", value=True, key="pin_require_green",
                help="Uncheck to also allow a red body, as long as the wick/body shape still qualifies.",
            )
        with pb4:
            pin_require_volume = st.checkbox(
                "Require volume ≥ prior day", value=True, key="pin_require_volume",
            )

        pin_df = build_bullish_pin_bar_screen(
            price_data, window=int(pin_window), min_lower_wick_pct=pin_min_lower_wick,
            max_upper_wick_pct=pin_max_upper_wick, max_body_pct=pin_max_body,
            require_green_body=pin_require_green, require_volume_confirmation=pin_require_volume,
        )

        if pin_df.empty:
            st.warning(
                "No tickers matched this pattern within the lookback window. Try loosening the "
                "wick/body thresholds or unchecking the volume/green-body requirements."
            )
        else:
            st.success(f"{len(pin_df)} tickers had a qualifying bullish pin bar within the last {pin_window} trading days.")
            st.dataframe(pin_df, use_container_width=True, height=500)
            st.caption(
                "'Days Ago' counts back from the most recent bar (0 = most recent day). "
                "'Lower Wick %' / 'Upper Wick %' / 'Body %' are each as a share of that day's total "
                "High-Low range, and sum to 100%."
            )

    # ---------------- Pullback Pattern tab ----------------
    with tab_pullback:
        st.subheader("Pullback Pattern Scan")
        st.caption(
            "Condition 1: a strong, almost straight-up run of bullish candles (3-9 days by default). "
            "Condition 2: a pullback that then comes down and touches the 9-day or 18-day simple moving "
            "average of Close."
        )

        pb_ma_options = {"Either 9MA or 18MA": "either", "9MA only": "9ma", "18MA only": "18ma"}

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            pb_min_run = st.number_input("Min run length (days)", min_value=2, max_value=30,
                                         value=PB_DEFAULT_MIN_RUN, key="pb_min_run")
            pb_touch_window = st.number_input(
                "Touch must be within the last N bars", min_value=1, max_value=10,
                value=PB_DEFAULT_TOUCH_WINDOW, key="pb_touch_window",
                help="1 = the most recent bar only. Raise it to also catch touches from the last few days.",
            )
        with pc2:
            pb_max_run = st.number_input("Max run length (days)", min_value=2, max_value=30,
                                         value=PB_DEFAULT_MAX_RUN, key="pb_max_run")
            pb_tolerance = st.number_input(
                "Touch tolerance (% from the MA)", min_value=0.0, max_value=5.0,
                value=PB_DEFAULT_TOLERANCE, step=0.1, key="pb_tolerance",
                help="How close the candle's low must get to the MA to count as a touch.",
            )
        with pc3:
            pb_max_pullback = st.number_input(
                "Max pullback length (days after the peak)", min_value=1, max_value=15,
                value=PB_DEFAULT_MAX_PULLBACK, key="pb_max_pullback",
            )
            pb_ma_label = st.radio("Pullback must touch", list(pb_ma_options), key="pb_ma_choice")

        pk1, pk2, pk3 = st.columns(3)
        with pk1:
            pb_strict = st.checkbox(
                "Strict run: green candles, higher highs & higher lows", value=True, key="pb_strict",
                help="Uncheck to only require each candle to close above the prior close "
                     "(allows an occasional red-bodied candle in the run).",
            )
        with pk2:
            pb_hold = st.checkbox(
                "Touch candle must close at/above the MA", value=True, key="pb_hold",
                help="Uncheck to also allow a candle that wicks to the MA but closes below it.",
            )
        with pk3:
            pb_uptrend = st.checkbox(
                "Require 9MA above 18MA", value=False, key="pb_uptrend",
                help="Uptrend context, as in the reference chart. Off by default so the scan follows "
                     "the two conditions exactly.",
            )

        if pb_min_run > pb_max_run:
            st.error("Min run length can't be greater than max run length.")
        else:
            pb_df = build_pullback_screen(
                price_data, min_run_days=int(pb_min_run), max_run_days=int(pb_max_run),
                max_pullback_days=int(pb_max_pullback), touch_window=int(pb_touch_window),
                touch_tolerance_pct=pb_tolerance, ma_choice=pb_ma_options[pb_ma_label],
                strict_run=pb_strict, require_close_holds_ma=pb_hold, require_ma_uptrend=pb_uptrend,
            )

            if pb_df.empty:
                st.warning(
                    "No tickers matched. Try raising 'Touch must be within the last N bars', widening the "
                    "touch tolerance, or unchecking the strict-run / close-holds-the-MA options."
                )
            else:
                st.success(f"{len(pb_df)} tickers matched the pullback pattern.")
                st.dataframe(pb_df, use_container_width=True, height=450)
                st.caption(
                    "'Days Ago' = how many bars ago the MA touch happened (0 = latest bar). 'Run Gain %' is the "
                    "close-to-close gain over the straight-up run; 'Pullback Depth %' is the drop from the peak "
                    "bar's high to the pullback's lowest low."
                )

                st.divider()
                st.subheader("Visual confirmation")
                pb_choice = st.selectbox("Preview Close with 9MA / 18MA for a matched ticker",
                                         pb_df["Ticker"].tolist(), key="pb_preview_ticker")
                if pb_choice:
                    st.line_chart(pullback_chart_data(price_data[pb_choice], bars=45), height=300)

    # ---------------- Anticipation tab ----------------
    with tab_anticipation:
        st.subheader("Anticipation Pattern Scan")
        st.caption(
            "Condition 1: price goes almost straight up for 3-9 days. Condition 2: it then moves sideways in a "
            "very narrow range for 3-9 days, still in progress on the latest bar. The tight base near the highs "
            "is the coil that anticipates the next move. Bases containing zero-volume (no-trade) days are "
            "skipped."
        )

        an1, an2, an3 = st.columns(3)
        with an1:
            an_min_run = st.number_input("Min run length (days)", min_value=2, max_value=30,
                                         value=AN_MIN_RUN, key="an_min_run")
            an_max_run = st.number_input("Max run length (days)", min_value=2, max_value=30,
                                         value=AN_MAX_RUN, key="an_max_run")
        with an2:
            an_min_cons = st.number_input("Min consolidation (days)", min_value=2, max_value=30,
                                          value=AN_MIN_CONS, key="an_min_cons")
            an_max_cons = st.number_input("Max consolidation (days)", min_value=2, max_value=30,
                                          value=AN_MAX_CONS, key="an_max_cons")
        with an3:
            an_max_range = st.number_input(
                "Max box range (%, tighter = narrower)", min_value=0.2, max_value=15.0,
                value=AN_MAX_RANGE, step=0.1, key="an_max_range",
                help="Whole consolidation box: highest High vs lowest Low.",
            )
            an_max_drop = st.number_input(
                "Max give-back from peak close (%)", min_value=0.0, max_value=15.0,
                value=AN_MAX_DROP, step=0.5, key="an_max_drop",
                help="Keeps the base up near the highs, so 'run, crash, then flat' doesn't match.",
            )

        an_strict = st.checkbox(
            "Strict run: green candles, higher highs & higher lows", value=True, key="an_strict",
            help="Uncheck to only require each candle to close above the prior close "
                 "(allows an occasional red-bodied candle in the run).",
        )

        if an_min_run > an_max_run or an_min_cons > an_max_cons:
            st.error("A minimum can't be greater than its maximum.")
        else:
            an_df = build_anticipation_screen(
                price_data, min_run_days=int(an_min_run), max_run_days=int(an_max_run),
                min_consolidation_days=int(an_min_cons), max_consolidation_days=int(an_max_cons),
                max_range_pct=an_max_range, max_drop_from_peak_pct=an_max_drop, strict_run=an_strict,
            )

            if an_df.empty:
                st.warning(
                    "No tickers matched. Try widening the box range, allowing a bigger give-back from the "
                    "peak, or unchecking the strict-run option."
                )
            else:
                st.success(f"{len(an_df)} tickers matched the anticipation pattern.")
                st.dataframe(an_df, use_container_width=True, height=450)
                st.caption(
                    "Sorted tightest base first. 'Consolidation Range %' is the whole box (highest High vs lowest "
                    "Low); 'Drop From Peak %' is how far the base's low sits below the peak close; 'To Breakout %' "
                    "is the distance from the last close up to the box high (the trigger level)."
                )

                st.divider()
                st.subheader("Visual confirmation")
                an_choice = st.selectbox("Preview Close with the consolidation box for a matched ticker",
                                         an_df["Ticker"].tolist(), key="an_preview_ticker")
                if an_choice:
                    row = an_df[an_df["Ticker"] == an_choice].iloc[0]
                    st.line_chart(
                        anticipation_chart_data(price_data[an_choice], int(row["Consolidation Days"]),
                                                float(row["Consolidation High"]), float(row["Consolidation Low"]),
                                                bars=40),
                        height=300,
                    )

    # ---------------- Live tab ----------------
    with tab_live:
        st.subheader("Live Watchlist Monitor")
        st.caption(
            "Free intraday data only works well for a small watchlist, not the whole market — polling "
            "thousands of tickers every minute would get you rate-limited fast. Pick a watchlist below."
        )

        market_open = is_market_open()
        if market_open:
            st.success("🟢 US market is open (Mon-Fri, 9:30 AM-4:00 PM ET).")
        else:
            st.warning(
                "🔴 US market appears closed right now (or it's a market holiday — this check doesn't "
                "know about holidays). Data below will be stale/last-session."
            )

        last_screen = st.session_state.get("last_screen_df")

        watchlist_source = st.radio(
            "Watchlist source",
            ["Top results from last Momentum Screener run", "Custom list"],
            index=0 if (last_screen is not None and not last_screen.empty) else 1,
            key="live_watchlist_source",
        )

        if watchlist_source == "Top results from last Momentum Screener run":
            if last_screen is None or last_screen.empty:
                st.info("Run the Momentum Screener tab first to populate this option.")
                live_tickers = []
            else:
                top_n = st.slider("How many top-RS tickers to watch live", 5, 100, 25, key="live_top_n")
                live_tickers = last_screen["Ticker"].head(top_n).tolist()
                st.caption(f"Watching: {', '.join(live_tickers)}")
        else:
            live_custom = st.text_area(
                "Comma-separated tickers (keep it under ~100 for reliable polling)",
                "AAPL, NVDA, TSLA, AMD, MSFT",
                key="live_custom_tickers",
            )
            live_tickers = [t.strip().upper() for t in live_custom.split(",") if t.strip()]

        c1, c2 = st.columns([2, 1])
        with c1:
            refresh_minutes = st.select_slider(
                "Auto-refresh interval", options=[1, 2, 3, 5], value=1, key="live_refresh_minutes"
            )
        with c2:
            auto_refresh_on = st.checkbox("Auto-refresh", value=True, key="live_auto_refresh")

        if auto_refresh_on and market_open and live_tickers:
            st_autorefresh(interval=refresh_minutes * 60 * 1000, key="live_autorefresh_timer")
        elif auto_refresh_on and not market_open:
            st.caption("Auto-refresh paused — market is closed, so live data won't change.")

        if live_tickers:
            with st.spinner(f"Fetching live snapshot for {len(live_tickers)} tickers..."):
                snapshot = fetch_live_snapshot(live_tickers)

            if snapshot.empty:
                st.error("No live data returned. Check tickers are valid, or try again.")
            else:
                live_stats = compute_live_breadth(snapshot)
                lc1, lc2, lc3, lc4 = st.columns(4)
                lc1.metric("Watchlist size", live_stats["count"])
                lc2.metric("Up 4%+ now", live_stats["up_4pct"])
                lc3.metric("Down 4%+ now", live_stats["down_4pct"])
                lc4.metric("Avg RVOL", live_stats["avg_rvol"] if live_stats["avg_rvol"] is not None else "—")

                snapshot_sorted = snapshot.sort_values("% Change", ascending=False).reset_index(drop=True)

                def highlight_live(row):
                    if row.get("Breakout Now"):
                        return ["background-color: #d4f4dd"] * len(row)
                    if row.get("Intraday EP"):
                        return ["background-color: #fde9c8"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    snapshot_sorted.style.apply(highlight_live, axis=1),
                    use_container_width=True,
                    height=450,
                )
                st.caption(
                    "🟩 Breakout Now: at/near the day's high with RVOL ≥ 1.5x its 10-day average. "
                    "🟧 Intraday EP: up 10%+ today with RVOL ≥ 2x — an intraday echo of the daily episodic-pivot flag. "
                    "RVOL compares today's cumulative volume to the 10-day average total daily volume (not "
                    "time-of-day adjusted), so it reads highest later in the trading day."
                )
                st.caption(f"Last updated: {pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            st.info("Add tickers to the watchlist above to start monitoring.")
else:
    st.info("Set your universe and settings in the sidebar, then click **Run Screen**.")
