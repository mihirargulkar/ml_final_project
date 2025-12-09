"""Compute 1/2/3-month (30/60/90 calendar day) close-to-close returns for insider trades.

The script reads `insider_trades.csv`, fetches historical Close prices from
Yahoo Finance via `yfinance`, and appends `return_30d_close`, `return_60d_close`,
and `return_90d_close`. For each row, it uses the last available trading Close
on or before `trade_date` and on or before `trade_date + N days` (N in
{30, 60, 90}). Missing prices fall back to the nearest previous trading day; if
none exist the return remains NaN.
"""

from __future__ import annotations

import sys
import time
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "insider_trades.csv"
WINDOWS = [30, 60, 90]  # calendar days
RETURN_COLS = {w: f"return_{w}d_close" for w in WINDOWS}
THROTTLE_SEC = 1.1  # sleep between batch downloads to stay under 60 calls/min
RATE_LIMIT_BACKOFF_SEC = 60  # base backoff when Yahoo rate limits
RATE_LIMIT_RETRIES = 3
BATCH_SIZE = 40  # tickers per multi-download call (60 calls/min budget)
CACHE_DIR = ROOT / "data_cache"
BUFFER_DAYS_BEFORE = 10
BUFFER_DAYS_AFTER = 5

VALID_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip()


def is_valid_ticker(ticker: str) -> bool:
    if not ticker:
        return False
    if ticker.startswith("$"):  # many delisted/OTC symbols in this dataset
        return False
    if ticker in {".", "--"}:
        return False
    if not VALID_TICKER_RE.match(ticker):
        return False
    if not any(ch.isalnum() for ch in ticker):
        return False
    return True


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path_for(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def load_cached_prices(ticker: str) -> Optional[pd.DataFrame]:
    path = cache_path_for(ticker)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            return df
        except Exception:  # noqa: BLE001
            return None
    return None


def save_cached_prices(ticker: str, df: pd.DataFrame) -> None:
    ensure_cache_dir()
    path = cache_path_for(ticker)
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001
        pass


def download_prices(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Download Close prices for a ticker between start and end (inclusive-ish).

    Returns (price_df, used_symbol). price_df has columns [price_date, close].
    Tries a dash-version fallback when the ticker contains dots.
    """
    candidates = [ticker]
    dash_ticker = ticker.replace(".", "-")
    if dash_ticker != ticker:
        candidates.append(dash_ticker)

    for symbol in candidates:
        backoff = RATE_LIMIT_BACKOFF_SEC
        for attempt in range(1, RATE_LIMIT_RETRIES + 1):
            try:
                hist = (
                    yf.Ticker(symbol)
                    .history(
                        start=start.date(),
                        end=(end + timedelta(days=1)).date(),  # history end is exclusive
                        auto_adjust=False,
                        actions=False,
                    )
                    .reset_index()
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "Too Many Requests" in msg or "rate limit" in msg.lower():
                    wait = backoff * attempt
                    print(f"[ratelimit] {symbol} attempt {attempt}/{RATE_LIMIT_RETRIES}; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                print(f"[warn] failed to download {symbol}: {exc}")
                hist = None

            if hist is None or hist.empty or "Close" not in hist:
                # If empty due to rate limit, retry; otherwise break
                if attempt < RATE_LIMIT_RETRIES:
                    wait = backoff * attempt
                    print(f"[retry] {symbol} empty response; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                hist = None
            # Successful or exhausted retries
            break

        if hist is None or hist.empty or "Close" not in hist:
            continue

        price_df = hist.rename(columns={"Date": "price_date", "Close": "close"})[
            ["price_date", "close"]
        ].dropna()
        price_df["price_date"] = (
            pd.to_datetime(price_df["price_date"]).dt.tz_localize(None).dt.normalize()
        )
        return price_df.sort_values("price_date"), symbol

    return None, None


def extract_close_frame(hist: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Extract price_date/close frame for a given ticker from a download result."""
    if hist is None or hist.empty:
        return None

    # Single-ticker case
    if not isinstance(hist.columns, pd.MultiIndex):
        if "Close" not in hist:
            return None
        df = (
            hist[["Close"]]
            .reset_index()
            .rename(columns={"Date": "price_date", "Close": "close"})
            .dropna(subset=["close"])
        )
    else:
        # Multi-ticker case; yfinance uses (ticker, field)
        cols = hist.columns
        if (ticker, "Close") in cols:
            close_series = hist[(ticker, "Close")]
        elif ("Close", ticker) in cols:
            close_series = hist[("Close", ticker)]
        else:
            return None
        df = (
            close_series.reset_index()
            .rename(columns={"Date": "price_date", close_series.name: "close"})
            .dropna(subset=["close"])
        )

    df["price_date"] = pd.to_datetime(df["price_date"]).dt.tz_localize(None).dt.normalize()
    return df[["price_date", "close"]].sort_values("price_date")


def download_batch(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> Dict[str, Optional[pd.DataFrame]]:
    """Download Close prices for a batch of tickers in one call."""
    results: Dict[str, Optional[pd.DataFrame]] = {t: None for t in tickers}
    backoff = RATE_LIMIT_BACKOFF_SEC
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            hist = yf.download(
                tickers=tickers,
                start=start.date(),
                end=(end + timedelta(days=1)).date(),  # end is exclusive
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Too Many Requests" in msg or "rate limit" in msg.lower():
                wait = backoff * attempt
                print(f"[ratelimit] batch {attempt}/{RATE_LIMIT_RETRIES}; sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"[warn] batch download failed: {exc}")
            hist = None

        if hist is None or hist.empty:
            if attempt < RATE_LIMIT_RETRIES:
                wait = backoff * attempt
                print(f"[retry] empty batch response; sleeping {wait}s")
                time.sleep(wait)
                continue
        # Success or exhausted retries; break to parse whatever we have
        break

    if hist is None or hist.empty:
        return results

    for t in tickers:
        df_t = extract_close_frame(hist, t)
        results[t] = df_t
    return results


def compute_returns(df: pd.DataFrame, windows: Optional[list[int]] = None) -> pd.DataFrame:
    """Compute forward returns per row for the requested windows (in days)."""
    if "trade_date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Expected columns 'trade_date' and 'ticker' in input CSV")

    if windows is None:
        windows = WINDOWS

    windows = sorted(set(int(w) for w in windows))
    if not windows:
        raise ValueError("No windows specified")

    # Normalize ticker/text fields and dates
    tickers_clean = df["ticker"].apply(normalize_ticker)
    trade_dt = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    target_dt_by_window = {w: trade_dt + pd.Timedelta(days=w) for w in windows}

    returns_df = pd.DataFrame(index=df.index, dtype="float64")
    tickers = [t for t in tickers_clean.dropna().unique() if is_valid_ticker(t)]

    if not tickers:
        print("[info] no valid tickers to process")
        return returns_df

    # Compute per-ticker download windows
    window_bounds: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for ticker in tickers:
        mask = tickers_clean == ticker
        trade_dates = trade_dt.loc[mask]
        if trade_dates.isna().all():
            continue
        start = trade_dates.min() - pd.Timedelta(days=BUFFER_DAYS_BEFORE)
        end_candidates = [target_dt_by_window[w].loc[mask].max() for w in windows]
        end_max = pd.Series(end_candidates).max()
        end = end_max + pd.Timedelta(days=BUFFER_DAYS_AFTER)
        window_bounds[ticker] = (start, end)

    # Load from cache and prepare batches for missing tickers
    price_cache: Dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for t in tickers:
        cached = load_cached_prices(t)
        if cached is not None and not cached.empty:
            price_cache[t] = cached
        else:
            missing.append(t)

    batches = [missing[i : i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    for b_idx, batch in enumerate(batches, 1):
        if not batch:
            continue
        batch_start = min(window_bounds[t][0] for t in batch)
        batch_end = max(window_bounds[t][1] for t in batch)
        print(f"[info] downloading batch {b_idx}/{len(batches)} ({len(batch)} tickers)")
        batch_data = download_batch(batch, batch_start, batch_end)
        for t, df_prices in batch_data.items():
            if df_prices is not None and not df_prices.empty:
                price_cache[t] = df_prices
                save_cached_prices(t, df_prices)
        if THROTTLE_SEC > 0:
            time.sleep(THROTTLE_SEC)

    print(f"[info] processing {len(tickers)} tickers across {len(df)} rows for windows {windows}")

    for i, ticker in enumerate(sorted(tickers)):
        price_df = price_cache.get(ticker)
        if price_df is None or price_df.empty:
            print(f"[warn] no price data for {ticker}")
            continue

        mask = tickers_clean == ticker
        trade_dates = trade_dt.loc[mask]
        price_series = price_df.set_index("price_date")["close"].sort_index()

        # Entry prices: last close on/before trade_date
        trade_sorted = trade_dates.sort_values()
        entry_vals = price_series.reindex(trade_sorted, method="ffill").to_numpy()
        entry = pd.Series(entry_vals, index=trade_sorted.index).reindex(trade_dates.index)

        for window in windows:
            target_dates = target_dt_by_window[window].loc[mask]
            target_sorted = target_dates.sort_values()
            target_vals = price_series.reindex(target_sorted, method="ffill").to_numpy()
            target = pd.Series(target_vals, index=target_sorted.index).reindex(target_dates.index)

            ticker_returns = (target - entry) / entry
            returns_df.loc[ticker_returns.index, RETURN_COLS[window]] = ticker_returns

        if (i + 1) % 50 == 0 or i == len(tickers) - 1:
            print(f"[info] processed {i + 1}/{len(tickers)} tickers (last: {ticker})")

    return returns_df


def main(csv_path: Path = CSV_PATH) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    returns_df = compute_returns(df, windows=WINDOWS)
    for w, col in RETURN_COLS.items():
        df[col] = returns_df.get(col, pd.NA)

    filled = {col: df[col].notna().sum() for col in RETURN_COLS.values()}
    missing = {col: df[col].isna().sum() for col in RETURN_COLS.values()}

    df.to_csv(csv_path, index=False)

    print(f"[done] wrote {csv_path}")
    print(f"[stats] rows={len(df)}")
    for w, col in RETURN_COLS.items():
        print(f"  {col}: filled={filled[col]}, missing={missing[col]}")
    preview_cols = ["ticker", "trade_date"] + list(RETURN_COLS.values())
    print(df[preview_cols].head(10))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)

