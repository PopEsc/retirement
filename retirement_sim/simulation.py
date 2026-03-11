"""
Monte Carlo simulation engine for retirement portfolio.

Two simulation modes:

Parametric (default, no tickers):
  Returns each year are drawn independently from normal distributions using
  historical long-run averages.  Fast and requires no internet connection.

Historical block-bootstrap (when HistoricalData is supplied):
  Returns are drawn in consecutive chunks from actual historical years,
  preserving multi-year autocorrelation (bear/bull runs, recessions, etc.).
  Inflation can be taken from the same historical years ('actual') or held
  fixed at a user-specified rate ('fixed').

Historical parameter defaults (nominal):
  Stocks:    mean ~10%, std ~17%  (US large-cap, ~1928–present)
  Bonds:     mean  ~4%, std  ~7%  (intermediate-term US treasuries)
  Inflation: mean  ~3%, std ~1.5%
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .market_data import HistoricalData


@dataclass
class MarketParams:
    """Annual nominal return assumptions used in parametric mode."""
    stock_mean: float = 0.10
    stock_std: float = 0.17
    bond_mean: float = 0.04
    bond_std: float = 0.07
    inflation_mean: float = 0.03
    inflation_std: float = 0.015


@dataclass
class PortfolioParams:
    """
    Retirement portfolio configuration.

    Exactly one of `stock_fraction` (parametric mode) or `tickers`
    (historical mode) must be provided.
    """
    initial_balance: float
    annual_withdrawal: float       # Gross, in today's dollars
    stock_fraction: float | None = None  # 0.0–1.0; remainder is bonds
    tickers: list[tuple[str, float]] | None = None  # (symbol, current_value)
    social_security: float = 0.0
    years: int = 30
    num_simulations: int = 10_000
    market: MarketParams = field(default_factory=MarketParams)
    cash_fraction: float = 0.0   # fraction of initial_balance held in cash/fixed-rate
    cash_rate: float = 0.0       # annual return on the cash portion (e.g. 0.045)

    def __post_init__(self) -> None:
        if self.stock_fraction is None and self.tickers is None:
            raise ValueError("Provide either stock_fraction or tickers.")
        if self.stock_fraction is not None and self.tickers is not None:
            raise ValueError("Provide stock_fraction OR tickers, not both.")
        if self.stock_fraction is not None and not (0.0 <= self.stock_fraction <= 1.0):
            raise ValueError("stock_fraction must be between 0.0 and 1.0.")

    @property
    def bond_fraction(self) -> float | None:
        return None if self.stock_fraction is None else 1.0 - self.stock_fraction

    @property
    def net_withdrawal(self) -> float:
        """Portfolio withdrawal needed after SS/pension offsets."""
        return max(0.0, self.annual_withdrawal - self.social_security)

    @property
    def ticker_weights(self) -> np.ndarray | None:
        """Normalised allocation weights in ticker order (sums to 1.0)."""
        if self.tickers is None:
            return None
        values = np.array([v for _, v in self.tickers], dtype=float)
        return values / values.sum()

    def allocation_label(self) -> str:
        """Short human-readable allocation description for chart titles."""
        cash_label = f" + {self.cash_fraction * 100:.0f}% cash" if self.cash_fraction > 0 else ""
        if self.tickers:
            names = "/".join(t for t, _ in self.tickers)
            return names + cash_label
        return f"{self.stock_fraction * 100:.0f}% stocks" + cash_label


@dataclass
class SimulationResults:
    """Raw results from a Monte Carlo simulation run."""
    params: PortfolioParams
    # balances[sim_i, year_j]: portfolio balance at end of year j (col 0 = initial)
    balances: np.ndarray
    # depletion_year[sim_i]: year portfolio hit $0, or NaN if never depleted
    depletion_year: np.ndarray


# ---------------------------------------------------------------------------
# Parametric simulation (no historical data required)
# ---------------------------------------------------------------------------

def run_simulation(
    params: PortfolioParams,
    seed: int | None = None,
) -> SimulationResults:
    """
    Parametric Monte Carlo: each year's returns are drawn independently
    from normal distributions.

    Year sequence for one simulation run:
      1. Apply random portfolio return (weighted stock + bond draw)
      2. Subtract inflation-adjusted net withdrawal
      3. Floor balance at 0
    """
    rng = np.random.default_rng(seed)
    n_sims, n_years = params.num_simulations, params.years
    mkt = params.market

    stock_returns = rng.normal(mkt.stock_mean, mkt.stock_std, (n_sims, n_years))
    bond_returns  = rng.normal(mkt.bond_mean,  mkt.bond_std,  (n_sims, n_years))
    inflation_rates = rng.normal(mkt.inflation_mean, mkt.inflation_std, (n_sims, n_years))
    inflation_rates = np.clip(inflation_rates, -0.05, 0.15)

    market_frac = 1.0 - params.cash_fraction
    portfolio_returns = (
        params.stock_fraction * market_frac * stock_returns
        + params.bond_fraction * market_frac * bond_returns
        + params.cash_fraction * params.cash_rate
    )

    return _simulate_balances(params, portfolio_returns, inflation_rates)


# ---------------------------------------------------------------------------
# Historical block-bootstrap simulation
# ---------------------------------------------------------------------------

def run_simulation_historical(
    params: PortfolioParams,
    historical: HistoricalData,
    chunk_size: int = 5,
    inflation_mode: str = "actual",
    seed: int | None = None,
) -> SimulationResults:
    """
    Block-bootstrap Monte Carlo using actual historical annual returns.

    For each simulation the required years are filled by repeatedly drawing
    a random consecutive chunk of `chunk_size` years from the historical
    dataset and appending them end-to-end.  This preserves the multi-year
    autocorrelation present in real market data (e.g. extended bear/bull runs).

    Parameters
    ----------
    chunk_size:
        Number of consecutive historical years per block.  Larger values
        keep more serial correlation; 1 is equivalent to IID resampling.
    inflation_mode:
        'actual' — use CPI inflation from the same historical years as the
                   return data.
        'fixed'  — use params.market.inflation_mean (no year-to-year variance).
    """
    rng = np.random.default_rng(seed)
    n_sims, n_years = params.num_simulations, params.years

    returns_array = historical.returns.values.astype(float)   # (n_hist, n_tickers)
    inflation_array = historical.inflation.values.astype(float)  # (n_hist,)
    n_hist = len(historical.years)
    weights = historical.weights()

    # Clamp chunk_size so we always have at least one valid starting position
    chunk_size = min(chunk_size, n_hist)
    max_start = n_hist - chunk_size  # inclusive upper bound for starting index

    # ---- Vectorised chunk-index construction --------------------------------
    # For each sim and each chunk position, draw a random start index into the
    # historical array.  Then expand into per-year indices and truncate to
    # exactly n_years.
    n_chunks = math.ceil(n_years / chunk_size)
    # chunk_starts[sim_i, chunk_j] = starting historical index for that block
    chunk_starts = rng.integers(0, max_start + 1, (n_sims, n_chunks))

    # offsets within a chunk: 0, 1, ..., chunk_size-1
    offsets = np.arange(chunk_size, dtype=int)

    # hist_indices[sim_i, yr] = index into historical arrays for that year
    # shape after reshape: (n_sims, n_chunks * chunk_size), then slice to n_years
    hist_indices = (
        chunk_starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    ).reshape(n_sims, -1)[:, :n_years]   # (n_sims, n_years)

    # ---- Returns and inflation -----------------------------------------------
    # Fancy-index into historical arrays: returns_array[hist_indices] has shape
    # (n_sims, n_years, n_tickers); dot with weights gives (n_sims, n_years).
    # Market-only weighted returns (weights sum to 1 over non-cash tickers)
    portfolio_returns = returns_array[hist_indices] @ weights   # (n_sims, n_years)
    # Blend with fixed cash return: scale market portion down, add cash portion
    if params.cash_fraction > 0:
        portfolio_returns = (
            portfolio_returns * (1.0 - params.cash_fraction)
            + params.cash_fraction * params.cash_rate
        )

    if inflation_mode == "actual":
        simulation_inflation = inflation_array[hist_indices]    # (n_sims, n_years)
    else:
        # Fixed rate — no year-to-year variance
        simulation_inflation = np.full(
            (n_sims, n_years), params.market.inflation_mean
        )

    return _simulate_balances(params, portfolio_returns, simulation_inflation)


# ---------------------------------------------------------------------------
# Shared balance-evolution kernel
# ---------------------------------------------------------------------------

def _simulate_balances(
    params: PortfolioParams,
    portfolio_returns: np.ndarray,   # (n_sims, n_years)
    inflation_rates: np.ndarray,     # (n_sims, n_years)
) -> SimulationResults:
    """
    Given pre-computed portfolio returns and inflation rates, evolve balances
    year by year and record depletion.

    Each year:
      balance = balance × (1 + r_portfolio) − withdrawal
      withdrawal grows by that year's inflation rate
      balance is floored at 0 once depleted
    """
    n_sims, n_years = portfolio_returns.shape
    base_withdrawal = params.net_withdrawal

    # Cumulative inflation multipliers: cum[sim, yr] = ∏(1+inf) for years 0..yr
    cum_inflation = np.cumprod(1.0 + inflation_rates, axis=1)

    # Nominal withdrawal for each year
    withdrawals = np.empty((n_sims, n_years))
    withdrawals[:, 0] = base_withdrawal
    if n_years > 1:
        withdrawals[:, 1:] = base_withdrawal * cum_inflation[:, :-1]

    balances = np.empty((n_sims, n_years + 1))
    balances[:, 0] = params.initial_balance
    already_depleted = np.zeros(n_sims, dtype=bool)

    for yr in range(n_years):
        after_return = balances[:, yr] * (1.0 + portfolio_returns[:, yr])
        after_withdrawal = after_return - withdrawals[:, yr]
        after_withdrawal = np.where(already_depleted, 0.0, after_withdrawal)
        after_withdrawal = np.maximum(after_withdrawal, 0.0)
        balances[:, yr + 1] = after_withdrawal
        already_depleted |= after_withdrawal == 0.0

    # Record first year of depletion
    depletion_year = np.full(n_sims, np.nan)
    for yr in range(1, n_years + 1):
        depleted_now = (balances[:, yr] == 0.0) & np.isnan(depletion_year)
        depletion_year = np.where(depleted_now, float(yr), depletion_year)

    return SimulationResults(
        params=params,
        balances=balances,
        depletion_year=depletion_year,
    )
