"""
Data layer: ticker universe + price history fetching, all via yfinance (free).
Cached with Streamlit so repeated screener runs don't re-download everything.
"""
import time

import pandas as pd
import requests
import yfinance as yf
import streamlit as st

# Nasdaq's own public symbol-directory files. Free, no API key, updated daily.
# nasdaqlisted.txt  -> stocks listed ON Nasdaq
# otherlisted.txt   -> stocks listed on NYSE, NYSE American, NYSE Arca, Cboe BZX, IEX
#                      (the "Exchange" column tells you which)
# Multiple mirrors are tried in order since nasdaqtrader.com has changed hosts before.
NASDAQ_LISTED_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://old.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
]
OTHER_LISTED_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    "https://old.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]

_REQUEST_HEADERS = {
    # A plain default User-Agent gets blocked by some Nasdaq endpoints.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def _fetch_first_working(urls: list[str]) -> str | None:
    """Tries each mirror URL in order, returns the first successful response text."""
    last_error = None
    for url in urls:
        try:
            resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=20)
            resp.raise_for_status()
            if resp.text.strip():
                return resp.text
        except Exception as e:
            last_error = e
            continue
    if last_error:
        raise last_error
    return None

OTHER_EXCHANGE_NAMES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

# Security-name keywords that indicate a non-common-equity instrument.
# Used to filter the raw listing down to plain common/ordinary shares.
# Note: plain "depositary shares" (ADRs like BABA, TSM) are NOT excluded —
# those trade as ordinary equity. Only *preferred* depositary shares are
# caught, via the "preferred"/"pfd" keywords below.
_NON_EQUITY_KEYWORDS = [
    "warrant", "warrants", " right", "rights", " unit", "units,",
    "preferred", " pfd", "notes", "debenture",
    "bond", "etf", "etn", "exchange traded", "index fund", " fund",
    "trust preferred", "convertible", "subordinated",
]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_full_listed_universe(include_nasdaq: bool = True, include_nyse_other: bool = True) -> pd.DataFrame:
    """
    Downloads Nasdaq's public symbol directory files and returns a combined
    DataFrame of every listed symbol with columns:
    Symbol, Name, Exchange, ETF (Y/N), TestIssue (Y/N)
    """
    frames = []

    if include_nasdaq:
        try:
            text = _fetch_first_working(NASDAQ_LISTED_URLS)
            lines = [l for l in text.splitlines() if l and not l.startswith("File Creation Time")]
            df = pd.DataFrame([l.split("|") for l in lines[1:]], columns=lines[0].split("|"))
            df = df.rename(columns={"Security Name": "Name", "Test Issue": "TestIssue"})
            df["Exchange"] = "NASDAQ"
            frames.append(df[["Symbol", "Name", "Exchange", "ETF", "TestIssue"]])
        except Exception as e:
            st.warning(f"Could not fetch NASDAQ listing from any mirror: {e}")

    if include_nyse_other:
        try:
            text = _fetch_first_working(OTHER_LISTED_URLS)
            lines = [l for l in text.splitlines() if l and not l.startswith("File Creation Time")]
            df = pd.DataFrame([l.split("|") for l in lines[1:]], columns=lines[0].split("|"))
            df = df.rename(columns={"ACT Symbol": "Symbol", "Security Name": "Name", "Test Issue": "TestIssue"})
            df["Exchange"] = df["Exchange"].map(OTHER_EXCHANGE_NAMES).fillna(df["Exchange"])
            frames.append(df[["Symbol", "Name", "Exchange", "ETF", "TestIssue"]])
        except Exception as e:
            st.warning(f"Could not fetch NYSE/other listing from any mirror: {e}")

    if not frames:
        return pd.DataFrame(columns=["Symbol", "Name", "Exchange", "ETF", "TestIssue"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Symbol"])
    combined = combined[combined["Symbol"].str.len() > 0]
    combined = combined.drop_duplicates(subset="Symbol")
    return combined.reset_index(drop=True)


def filter_equities_only(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a raw listing down to plain common/ordinary equity shares:
    drops ETFs, ETNs, test issues, warrants, rights, units, preferreds,
    trusts/funds, and notes/bonds, based on the ETF flag and security name.
    """
    df = universe.copy()
    df = df[df["ETF"].str.upper() != "Y"]
    df = df[df["TestIssue"].str.upper() != "Y"]

    name_lower = df["Name"].str.lower()
    mask_excluded = pd.Series(False, index=df.index)
    for kw in _NON_EQUITY_KEYWORDS:
        mask_excluded |= name_lower.str.contains(kw, regex=False)
    df = df[~mask_excluded]

    # Keep only reasonably "normal" ticker symbols (letters, digits, . - ^)
    df = df[df["Symbol"].str.match(r"^[A-Za-z0-9.\-^]+$")]

    # yfinance wants '-' instead of '.' for share classes (e.g. BRK.B -> BRK-B)
    df["YF_Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    return df.reset_index(drop=True)


def fetch_last_close_prices(tickers: list[str], chunk_size: int = 200, pause_sec: float = 1.0,
                             progress_callback=None) -> dict[str, float]:
    """
    Cheap first pass: pulls only the last 5 days of daily bars per ticker
    (fast) to get a current-ish last close, so we can apply the price filter
    BEFORE downloading full history for the whole universe.
    progress_callback(fraction_done) is called after each chunk if provided.
    """
    last_prices: dict[str, float] = {}
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(
                chunk, period="5d", interval="1d", group_by="ticker",
                auto_adjust=True, threads=True, progress=False,
            )
        except Exception:
            data = None

        if data is not None and not data.empty:
            if len(chunk) == 1:
                t = chunk[0]
                try:
                    last_prices[t] = float(data["Close"].dropna().iloc[-1])
                except Exception:
                    pass
            else:
                for t in chunk:
                    try:
                        close = data[t]["Close"].dropna()
                        if not close.empty:
                            last_prices[t] = float(close.iloc[-1])
                    except (KeyError, TypeError, IndexError):
                        continue

        if progress_callback:
            progress_callback((i + 1) / len(chunks))
        time.sleep(pause_sec)

    return last_prices

# Small static fallback universe in case the Wikipedia scrape fails (offline, rate-limited, etc.)
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "CRM", "ADBE", "COST", "PEP", "LIN", "TMO", "ABBV", "MRK", "JPM", "V",
    "MA", "UNH", "HD", "WMT", "PG", "XOM", "CVX", "CAT", "DE", "LMT",
]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia. Falls back to a static list on failure."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        # yfinance uses '-' instead of '.' for share classes (e.g. BRK.B -> BRK-B)
        return [t.replace(".", "-") for t in tickers]
    except Exception:
        return FALLBACK_UNIVERSE


def fetch_price_history(tickers: tuple[str, ...], period: str = "6mo", interval: str = "1d",
                         chunk_size: int = 150, pause_sec: float = 1.0,
                         progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Batch-download OHLCV history for a list of tickers, chunked to avoid
    Yahoo rate limits on large universes (hundreds to thousands of tickers).
    Returns a dict of {ticker: DataFrame} with columns Open, High, Low, Close, Volume.
    Tickers with no/insufficient data are dropped.
    """
    tickers = list(tickers)
    result: dict[str, pd.DataFrame] = {}
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(
                chunk, period=period, interval=interval, group_by="ticker",
                auto_adjust=True, threads=True, progress=False,
            )
        except Exception:
            data = None

        if data is not None and not data.empty:
            if len(chunk) == 1:
                t = chunk[0]
                df = data.dropna()
                if len(df) >= 30:
                    result[t] = df
            else:
                for t in chunk:
                    try:
                        df = data[t].dropna()
                        if len(df) >= 30:  # need enough history for MAs / breadth windows
                            result[t] = df
                    except (KeyError, TypeError):
                        continue

        if progress_callback:
            progress_callback((i + 1) / len(chunks))
        time.sleep(pause_sec)

    return result


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_benchmark(ticker: str = "SPY", period: str = "6mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return df.dropna()
