"""Compute 30/60/90 calendar-day close-to-close returns for insider trades.

Sequential, per-ticker downloads (no batching) with throttling to ~50 calls/min.
Outputs a new CSV `insider_trades_with_returns.csv` alongside the source file.
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
SRC_CSV = ROOT / "insider_trades.csv"
OUT_CSV = ROOT / "insider_trades_with_returns.csv"
WINDOWS = [30, 60, 90]  # calendar days
RETURN_COLS = {w: f"return_{w}d_close" for w in WINDOWS}
THROTTLE_SEC = 1.2  # ~50 calls/min safety
BUFFER_DAYS_BEFORE = 10
BUFFER_DAYS_AFTER = 5
VALID_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
FLUSH_EVERY = 50  # write partial progress every N tickers


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip()


def is_valid_ticker(ticker: str) -> bool:
    if not ticker:
        return False
    if ticker.startswith("$"):  # skip OTC-like
        return False
    if ticker in {".", "--"}:
        return False
    if not VALID_TICKER_RE.match(ticker):
        return False
    if not any(ch.isalnum() for ch in ticker):
        return False
    return True


def download_prices(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Download Close prices for a ticker between start and end (inclusive-ish)."""
    symbols = [ticker]
    if "." in ticker:
        symbols.append(ticker.replace(".", "-"))

    for sym in symbols:
        try:
            hist = (
                yf.Ticker(sym)
                .history(
                    start=start.date(),
                    end=(end + timedelta(days=1)).date(),  # end is exclusive
                    auto_adjust=False,
                    actions=False,
                )
                .reset_index()
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to download {sym}: {exc}")
            continue

        if hist is None or hist.empty or "Close" not in hist:
            continue

        price_df = hist.rename(columns={"Date": "price_date", "Close": "close"})[
            ["price_date", "close"]
        ].dropna()
        price_df["price_date"] = (
            pd.to_datetime(price_df["price_date"]).dt.tz_localize(None).dt.normalize()
        )
        return price_df.sort_values("price_date"), sym

    return None, None


def ensure_return_columns(df: pd.DataFrame, windows: list[int]) -> None:
    for w in windows:
        col = RETURN_COLS[w]
        if col not in df.columns:
            df[col] = pd.NA


def compute_returns(df: pd.DataFrame, windows: Optional[list[int]] = None, flush_path: Optional[Path] = None) -> pd.DataFrame:
    """Compute forward returns per row for requested windows (in days), resuming from existing values."""
    if "trade_date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Expected columns 'trade_date' and 'ticker' in input CSV")

    if windows is None:
        windows = WINDOWS
    windows = sorted(set(int(w) for w in windows))

    ensure_return_columns(df, windows)

    tickers_clean = df["ticker"].apply(normalize_ticker)
    trade_dt = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    target_dt_by_window = {w: trade_dt + pd.Timedelta(days=w) for w in windows}

    # Determine which tickers still need processing (any NA in return cols for that ticker)
    return_cols = [RETURN_COLS[w] for w in windows]
    tickers = []
    skipped = 0
    for t in tickers_clean.dropna().unique():
        if not is_valid_ticker(t):
            continue
        mask = tickers_clean == t
        needs = df.loc[mask, return_cols].isna().any(axis=1).any()
        if needs:
            tickers.append(t)
        else:
            skipped += 1

    print(f"[info] processing {len(tickers)} tickers across {len(df)} rows (skipped already-complete: {skipped})")

    for i, ticker in enumerate(sorted(tickers)):
        mask = tickers_clean == ticker
        trade_dates = trade_dt.loc[mask]
        if trade_dates.isna().all():
            continue

        start = trade_dates.min() - pd.Timedelta(days=BUFFER_DAYS_BEFORE)
        end_candidates = [target_dt_by_window[w].loc[mask].max() for w in windows]
        end_max = pd.Series(end_candidates).max()
        end = end_max + pd.Timedelta(days=BUFFER_DAYS_AFTER)

        price_df, used_symbol = download_prices(ticker, start, end)
        time.sleep(THROTTLE_SEC)

        if price_df is None or price_df.empty:
            print(f"[warn] no price data for {ticker}")
            continue

        # Entry prices via merge_asof (handles gaps cleanly)
        entry = (
            pd.merge_asof(
                trade_dates.reset_index().sort_values("trade_date"),
                price_df,
                left_on="trade_date",
                right_on="price_date",
                direction="backward",
            )
            .set_index("index")["close"]
            .reindex(trade_dates.index)
        )

        for w in windows:
            target_dates = target_dt_by_window[w].loc[mask]
            target = (
                pd.merge_asof(
                    target_dates.reset_index().sort_values(target_dates.name),
                    price_df,
                    left_on=target_dates.name,
                    right_on="price_date",
                    direction="backward",
                )
                .set_index("index")["close"]
                .reindex(target_dates.index)
            )

            ticker_returns = (target - entry) / entry
            df.loc[mask, RETURN_COLS[w]] = ticker_returns.values

        if (i + 1) % FLUSH_EVERY == 0 and flush_path:
            df.to_csv(flush_path, index=False)
            print(f"[info] flushed progress at {i + 1}/{len(tickers)} tickers to {flush_path}")

        if (i + 1) % 50 == 0 or i == len(tickers) - 1:
            print(f"[info] processed {i + 1}/{len(tickers)} tickers (last: {used_symbol or ticker})")

    return df


def main(src: Path = SRC_CSV, out: Path = OUT_CSV) -> None:
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}")

    df = pd.read_csv(src)
    print(f"[info] loaded source {src}")

    # If an output file exists, pre-fill returns from it using (transaction_date, trade_date, ticker)
    if out.exists():
        print(f"[info] found existing {out}; pre-filling returns")
        prev = pd.read_csv(out)
        ensure_return_columns(df, WINDOWS)
        ensure_return_columns(prev, WINDOWS)
        key_cols = ["transaction_date", "trade_date", "ticker"]
        df["_key"] = df[key_cols].astype(str).agg("|".join, axis=1)
        prev["_key"] = prev[key_cols].astype(str).agg("|".join, axis=1)
        prev_returns = (
            prev.drop_duplicates("_key")
            .set_index("_key")[[RETURN_COLS[w] for w in WINDOWS]]
        )
        for w, col in RETURN_COLS.items():
            if col in prev_returns:
                returns_map = prev_returns[col].to_dict()
                df[col] = df[col].combine_first(df["_key"].map(returns_map))
        df.drop(columns=["_key"], inplace=True)

    df = compute_returns(df, windows=WINDOWS, flush_path=out)

    df.to_csv(out, index=False)
    print(f"[done] wrote {out}")
    for w, col in RETURN_COLS.items():
        filled = df[col].notna().sum()
        missing = df[col].isna().sum()
        print(f"[stats] {col}: rows={len(df)}, filled={filled}, missing={missing}")
    preview_cols = ["ticker", "trade_date"] + list(RETURN_COLS.values())
    print(df[preview_cols].head(10))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)

