# Momentum & Breadth Screener

Stockbee-style market breadth monitor + Qullamaggie-style momentum/breakout/episodic-pivot screener, built on free `yfinance` data (no API key needed).

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Full-market scan (NASDAQ + NYSE, equities only, price-filtered)

Selecting **"Full NASDAQ + NYSE (equities only, price-filtered)"** in the sidebar runs a real full-market screen:

1. Downloads Nasdaq's free public symbol directories (`nasdaqlisted.txt` covers Nasdaq; `otherlisted.txt` covers NYSE, NYSE American, NYSE Arca, Cboe BZX, IEX) — roughly 6,000-8,000 symbols combined.
2. Filters out ETFs/ETNs (via the file's own ETF flag) and, by security-name keyword matching, warrants, rights, units, preferred shares, notes/bonds, and funds/trusts — leaving plain common/ordinary equity shares. **This keyword filter is a heuristic, not perfect** — a handful of unusual names may slip through or get excluded incorrectly. Spot-check the results if precision matters.
3. Fetches a cheap 5-day price snapshot for every remaining symbol to apply your min/max price filter (default ≤ $60), so full 6-month history is only downloaded for the stocks that actually pass.
4. Downloads full price history for the survivors and runs the same breadth + momentum screens.

**This takes several minutes** — it's chunked (150-200 tickers per batch) with a pause between batches to avoid Yahoo Finance rate-limiting, and progress bars show each stage. If you see many failed/missing tickers, increase `pause_sec` in `utils/data.py`'s `fetch_last_close_prices` / `fetch_price_history` calls, or narrow the scan (raise the min price, or run NASDAQ and NYSE as separate passes by editing `fetch_full_listed_universe(include_nasdaq=..., include_nyse_other=...)`).

## Live tab (intraday watchlist monitor)

The **🔴 Live (intraday)** tab auto-refreshes a *watchlist* (not the full market) every 1-5 minutes during US market hours. It's intentionally scoped to a small list because polling thousands of tickers every minute on free data will get you rate-limited.

- **Watchlist source**: either the top-N tickers (by RS Rank) from your last Momentum Screener run, or a custom comma-separated list. Keep it under ~100 names for reliable polling.
- **Data**: uses yfinance's lightweight `fast_info` per ticker (cheaper than downloading full bars) — last price, previous close, day high/low, day volume, 10-day average volume.
- **Metrics**: % change from previous close, RVOL (today's volume vs. 10-day average — not time-of-day adjusted, so it naturally reads higher later in the session), % from day's high/low.
- **Flags**: 🟩 *Breakout Now* (at/near the day's high with RVOL ≥ 1.5x), 🟧 *Intraday EP* (up 10%+ today with RVOL ≥ 2x — an intraday echo of the daily episodic-pivot flag).
- **Auto-refresh**: uses the `streamlit-autorefresh` package to re-run the live fetch on a timer without a full page reload (session state is preserved). Auto-refresh pauses automatically when the market-hours check reports the market closed.

**Important caveats:**
- yfinance quotes are near-real-time during market hours but **not guaranteed exchange-direct real-time** — treat this as "good enough for monitoring," not for order execution.
- The market-hours check is a simple Mon-Fri 9:30-4:00 ET window and **does not know about market holidays** — on holidays it will incorrectly say the market is open.
- This is not designed to scan the full market live — it's a focused watchlist monitor. If you want true full-market live scanning, that generally requires a paid real-time data feed (e.g., Polygon.io's websocket, IEX Cloud, or a broker API like IBKR/Alpaca) since free sources rate-limit hard on high-frequency, high-volume polling.

## What it does

**Market Breadth tab** — Stockbee's Market Monitor ratios:
- % of universe up 4%+ / down 4%+ today (short-term risk appetite)
- % up 25%+ over 1 / 4 / 10 trading days ("momentum burst" windows)
- % up 25%+ / 50%+ over a quarter (broader participation)
- A plain-language regime read (risk-off / momentum expansion / quiet tape / mixed)

**Momentum Screener tab** — Qullamaggie-style candidate table:
- RS Rank: percentile rank of 3-month return vs. the rest of the universe
- ADR%: Average Daily Range, his standard volatility metric for sizing stops
- Returns over 1D / 5D / 1M / 3M, and % from 52-week high
- Flags: **Tight Base** (consolidation), **Breakout Today** (break of prior 10-day high on volume), **Episodic Pivot** (gap ≥10% + 2x+ volume — approximates his EP setup without a news feed)

## Known limitations (free-tier tradeoffs)

- **Universe**: S&P 500 scrape depends on Wikipedia's table staying in its current format; a small static fallback list is included so the app never breaks entirely. The full NASDAQ+NYSE scan uses Nasdaq's free symbol directories (see above) — no paid provider needed, but the equity-only filter is keyword-based, not a guaranteed security-type classification.
- **Episodic Pivot detection is approximate**: real EP screening keys off *why* a stock gapped (earnings, guidance, FDA news, etc.). This version only sees the gap % + volume surge, so it will also catch gaps unrelated to a real catalyst. A next step would be pulling an earnings-calendar or news API and cross-referencing.
- **yfinance rate limits**: large universes (full S&P 500+) can be slow or occasionally throttled. The batch downloader and caching help, but a paid data feed (Polygon.io, Tiingo, EOD Historical Data) would be more reliable for daily production use.
- **No intraday data**: everything here is end-of-day. Qullamaggie's own breakout entries are often intraday; this screener is best used as an end-of-day watchlist builder, not a live intraday scanner.

## Extending

- `utils/momentum.py` — add new setup detectors here (e.g. a "backside" parabolic-fade flag, or a 10/20/50 MA scale-out signal)
- `utils/breadth.py` — add more Stockbee ratios (e.g. new-high/new-low ratio, advance/decline line)
- `utils/live.py` — the live tab's logic; swap `fast_info` for a paid real-time feed here if you outgrow free data
- Swap the universe source in `utils/data.py` if you want NASDAQ-wide or a custom watchlist from a broker export
