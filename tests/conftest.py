"""
Shared fixtures and test helpers.

Also sets the matplotlib backend to Agg (non-interactive) before any chart
module is imported, so tests can run headlessly.
"""

import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

import numpy as np
import pandas as pd
import pytest

from retirement_sim.market_data import HistoricalData
from retirement_sim.simulation import MarketParams, PortfolioParams, run_simulation


# ---------------------------------------------------------------------------
# Parameter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_params():
    return MarketParams()


@pytest.fixture
def parametric_params():
    """Small parametric PortfolioParams suitable for fast unit tests."""
    return PortfolioParams(
        initial_balance=1_000_000,
        annual_withdrawal=50_000,
        stock_fraction=0.60,
        social_security=12_000,
        years=20,
        num_simulations=200,
    )


@pytest.fixture
def ticker_params():
    """PortfolioParams in historical (ticker) mode."""
    return PortfolioParams(
        initial_balance=1_000_000,
        annual_withdrawal=50_000,
        tickers=[("SPY", 600_000), ("BND", 400_000)],
        social_security=12_000,
        years=20,
        num_simulations=200,
    )


# ---------------------------------------------------------------------------
# Historical data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_historical():
    """
    Ten years of synthetic annual returns for SPY and BND, plus inflation.
    Uses a fixed seed so tests are deterministic.
    """
    rng = np.random.default_rng(0)
    years = list(range(2000, 2010))
    returns = pd.DataFrame(
        {
            "SPY": rng.normal(0.10, 0.17, len(years)),
            "BND": rng.normal(0.04, 0.07, len(years)),
        },
        index=years,
    )
    inflation = pd.Series(
        np.clip(rng.normal(0.03, 0.015, len(years)), -0.05, 0.15),
        index=years,
    )
    return HistoricalData(
        returns=returns,
        inflation=inflation,
        years=years,
        ticker_values=[("SPY", 600_000), ("BND", 400_000)],
    )


# ---------------------------------------------------------------------------
# Pre-run simulation results fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def parametric_results(parametric_params):
    return run_simulation(parametric_params, seed=0)


# ---------------------------------------------------------------------------
# Helpers for mocking external data sources
# ---------------------------------------------------------------------------

def make_price_dataframe(start_year: int, end_year: int, annual_return: float = 0.08) -> pd.DataFrame:
    """
    Build a fake yfinance-style Close-price DataFrame with monthly frequency.
    Prices grow at a constant compound rate so annual returns are predictable.
    """
    # Extra year before so pct_change produces a return for start_year
    dates = pd.date_range(
        f"{start_year - 1}-01-31",
        periods=(end_year - start_year + 3) * 12,
        freq="ME",
    )
    monthly_factor = (1 + annual_return) ** (1 / 12)
    prices = 100.0 * (monthly_factor ** np.arange(len(dates)))
    return pd.DataFrame({"Close": prices}, index=dates)


def make_cpi_csv(start_year: int = 1990, end_year: int = 2024, annual_rate: float = 0.03) -> str:
    """
    Build a fake FRED CPIAUCSL CSV string with monthly observations.
    CPI grows at a constant annual rate for predictable inflation figures.
    """
    dates = pd.date_range(f"{start_year - 1}-01-01", f"{end_year}-12-01", freq="MS")
    monthly_factor = (1 + annual_rate) ** (1 / 12)
    cpis = 200.0 * (monthly_factor ** np.arange(len(dates)))
    lines = ["observation_date,CPIAUCSL"]
    for d, v in zip(dates, cpis):
        lines.append(f"{d.strftime('%Y-%m-%d')},{v:.3f}")
    return "\n".join(lines)
