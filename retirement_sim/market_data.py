"""
Historical market data fetching for retirement simulations.

Fetches annual price returns for ticker symbols via yfinance and annual
CPI inflation from the FRED public data endpoint (no API key required).

Proxy tickers
-------------
When a holding has limited history (e.g. a newer ETF), you can specify a
proxy ticker whose returns are used to fill any missing years.  The proxy
is only consulted for years the primary ticker lacks; primary data always
takes precedence.  If the primary ticker fails entirely, all years are
filled from the proxy.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@dataclass
class HistoricalData:
    """Aligned historical annual returns for portfolio tickers and CPI inflation."""
    # columns = ticker symbols, index = year (int), values = annual total return
    returns: pd.DataFrame
    # index = year (int), values = annual CPI inflation rate
    inflation: pd.Series
    # sorted list of years present in both datasets
    years: list[int]
    # (ticker, dollar_value) pairs as provided by the user
    ticker_values: list[tuple[str, float]]

    def weights(self) -> np.ndarray:
        """Portfolio weights (fraction of total) in ticker order."""
        values = np.array([v for _, v in self.ticker_values], dtype=float)
        return values / values.sum()

    def total_value(self) -> float:
        return sum(v for _, v in self.ticker_values)

    def summary_lines(self) -> list[str]:
        """Human-readable lines describing the dataset."""
        lines = [
            f"  Historical data: {self.years[0]}–{self.years[-1]} "
            f"({len(self.years)} years)",
        ]
        total = self.total_value()
        for ticker, value in self.ticker_values:
            w = value / total * 100
            lines.append(f"    {ticker:<8} ${value:>12,.0f}  ({w:.1f}%)")
        inf_mean = self.inflation.mean() * 100
        inf_std = self.inflation.std() * 100
        lines.append(
            f"  CPI inflation (historical): mean {inf_mean:.1f}%, "
            f"std {inf_std:.1f}%"
        )
        return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_ticker_annual_returns(
    ticker: str,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """
    Download adjusted closing prices for `ticker` and compute annual returns.

    Fetches one extra year before `start_year` so the first return is a
    full-year price change.  Raises ValueError if no data is available.
    """
    raw = yf.download(
        ticker,
        start=f"{start_year - 1}-01-01",
        end=f"{end_year + 1}-01-01",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if raw.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")

    close = raw["Close"].squeeze()
    annual_price = close.resample("YE").last()
    returns = annual_price.pct_change().dropna()
    returns.index = returns.index.year.astype(int)
    filtered = returns[(returns.index >= start_year) & (returns.index <= end_year)]
    if filtered.empty:
        raise ValueError(
            f"Ticker '{ticker}' has no data in the range {start_year}–{end_year}."
        )
    return filtered.rename(ticker)


def _fetch_ticker_with_proxy(
    ticker: str,
    proxy: str | None,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """
    Fetch annual returns for `ticker`, using `proxy` to fill any missing years.

    Behaviour:
    - If `ticker` has full coverage, proxy is never used.
    - If `ticker` is missing some years and proxy is set, proxy data fills
      those gaps.  Primary data is never overwritten.
    - If `ticker` fetch fails entirely and proxy is set, proxy provides all
      years (with a warning).
    - If no proxy is set and the ticker fails or has gaps, the gaps remain
      (years without data will be dropped during alignment).
    """
    primary: pd.Series | None = None
    try:
        primary = _fetch_ticker_annual_returns(ticker, start_year, end_year)
    except ValueError as exc:
        if proxy is None:
            raise
        print(f"    Warning: {exc}  Falling back to proxy {proxy} for all years.")

    if proxy is None:
        return primary  # type: ignore[return-value]

    # Determine which years are missing from the primary series
    all_years = set(range(start_year, end_year + 1))
    covered = set(primary.index) if primary is not None else set()
    missing = all_years - covered

    if not missing:
        return primary  # type: ignore[return-value]

    # Fetch proxy
    try:
        proxy_series = _fetch_ticker_annual_returns(proxy, start_year, end_year)
    except ValueError as exc:
        if primary is not None:
            print(f"    Warning: proxy {proxy} unavailable ({exc}). Using primary data only.")
            return primary
        raise ValueError(
            f"Neither '{ticker}' nor proxy '{proxy}' returned data "
            f"for {start_year}–{end_year}."
        ) from exc

    # Fill only the genuinely missing years from proxy
    fill_years = sorted(missing & set(proxy_series.index))
    if fill_years:
        n = len(fill_years)
        yr_range = f"{fill_years[0]}–{fill_years[-1]}" if n > 1 else str(fill_years[0])
        print(f"    {ticker}: filled {n} year(s) from proxy {proxy} ({yr_range})")

    fill = proxy_series.loc[fill_years].rename(ticker)
    combined = (
        pd.concat([primary, fill]).sort_index()
        if primary is not None
        else fill.copy()
    )
    return combined


def _fetch_cpi_annual(start_year: int, end_year: int) -> pd.Series:
    """
    Fetch monthly CPIAUCSL from FRED's public CSV endpoint (no API key needed),
    then compute annual average-over-average inflation rates.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not fetch CPI data from FRED: {exc}\n"
            "Check your internet connection or use --inflation-mode fixed."
        ) from exc

    df = pd.read_csv(
        io.StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.columns = ["CPI"]
    annual_avg = df["CPI"].resample("YE").mean()
    inflation = annual_avg.pct_change().dropna()
    inflation.index = inflation.index.year.astype(int)
    return inflation[(inflation.index >= start_year) & (inflation.index <= end_year)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_historical_dataset(
    ticker_values: list[tuple[str, float]],
    start_year: int,
    end_year: int,
    ticker_proxies: dict[str, str] | None = None,
) -> HistoricalData:
    """
    Download and align annual returns for each ticker plus CPI inflation.

    Only years where *all* tickers and CPI have data are kept.

    Parameters
    ----------
    ticker_values:
        List of (ticker_symbol, current_dollar_value) pairs.
    start_year, end_year:
        Requested date range. Actual range may be narrower due to data gaps.
    ticker_proxies:
        Optional mapping of ticker → proxy symbol.  For any year where a
        ticker has no data, the proxy's return is substituted.  A ticker
        not present in the dict (or where the dict is None) gets no proxy.
    """
    tickers = [t for t, _ in ticker_values]
    print(f"\nFetching price history for: {', '.join(tickers)}")

    proxies = ticker_proxies or {}
    returns_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        proxy = proxies.get(ticker)
        proxy_note = f" (proxy: {proxy})" if proxy else ""
        print(f"  {ticker}{proxy_note}...", end=" ", flush=True)
        series = _fetch_ticker_with_proxy(ticker, proxy, start_year, end_year)
        returns_dict[ticker] = series
        print(f"{len(series)} years ({series.index[0]}–{series.index[-1]})")

    returns_df = pd.DataFrame(returns_dict)
    # Drop years where any ticker still has no data after proxy filling
    returns_df = returns_df.dropna()

    print("  Fetching CPI inflation (FRED CPIAUCSL)...", end=" ", flush=True)
    inflation = _fetch_cpi_annual(start_year, end_year)
    print(f"{len(inflation)} years ({inflation.index[0]}–{inflation.index[-1]})")

    common_years = sorted(set(returns_df.index) & set(inflation.index))
    if not common_years:
        raise ValueError(
            "No overlapping years found between ticker data and CPI data. "
            "Try widening --data-start / --data-end, or add a proxy for "
            "tickers with limited history."
        )

    returns_df = returns_df.loc[common_years]
    inflation = inflation.loc[common_years]

    print(
        f"  Aligned dataset: {len(common_years)} years "
        f"({common_years[0]}–{common_years[-1]})\n"
    )

    return HistoricalData(
        returns=returns_df,
        inflation=inflation,
        years=common_years,
        ticker_values=ticker_values,
    )
