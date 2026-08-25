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
        index=0,
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
    min_rs_rank = st.slider("Minimum RS Rank (percentile)", 50, 99, 60)
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

    tab_breadth, tab_momentum, tab_live = st.tabs(
        ["📊 Market Breadth (Stockbee)", "🚀 Momentum Screener (Qullamaggie)", "🔴 Live (intraday)"]
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
