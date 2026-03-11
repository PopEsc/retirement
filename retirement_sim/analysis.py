"""
Analysis functions: success rates, safe withdrawal rate, summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .simulation import PortfolioParams, SimulationResults, run_simulation

if TYPE_CHECKING:
    from .market_data import HistoricalData


@dataclass
class SimConfig:
    """
    Extra configuration for the historical block-bootstrap simulation path.
    Pass this alongside PortfolioParams to route through the correct engine.
    """
    historical: HistoricalData | None = None
    chunk_size: int = 5
    inflation_mode: str = "fixed"   # 'actual' or 'fixed'


def _run(
    params: PortfolioParams,
    cfg: SimConfig | None = None,
    seed: int | None = None,
) -> SimulationResults:
    """Dispatch to the correct simulation engine based on SimConfig."""
    if cfg is not None and cfg.historical is not None:
        from .simulation import run_simulation_historical
        return run_simulation_historical(
            params, cfg.historical,
            chunk_size=cfg.chunk_size,
            inflation_mode=cfg.inflation_mode,
            seed=seed,
        )
    return run_simulation(params, seed=seed)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def success_rate(results: SimulationResults) -> float:
    """Fraction of simulations where the portfolio was never depleted."""
    return float(np.mean(np.isnan(results.depletion_year)))


def find_safe_withdrawal_rate(
    params: PortfolioParams,
    target_success: float = 0.95,
    seed: int | None = None,
    tolerance: float = 500.0,
    cfg: SimConfig | None = None,
) -> float:
    """
    Binary-search for the maximum annual gross withdrawal (today's dollars)
    that achieves at least `target_success` probability of not depleting the
    portfolio over `params.years` years.

    Returns the safe annual withdrawal amount (gross, before SS offset).
    """
    lo = params.social_security   # withdrawals ≤ SS are always "safe"
    hi = params.initial_balance * 2.0

    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        p = replace(params, annual_withdrawal=mid)
        rate = success_rate(_run(p, cfg, seed=seed))
        if rate >= target_success:
            lo = mid
        else:
            hi = mid

    return lo


def sweep_withdrawal_rates(
    params: PortfolioParams,
    withdrawals: list[float] | None = None,
    seed: int | None = None,
    cfg: SimConfig | None = None,
) -> list[tuple[float, float]]:
    """
    Compute success rates for a range of withdrawal amounts.

    Returns a list of (withdrawal, success_rate) tuples.
    """
    if withdrawals is None:
        max_w = params.initial_balance * 0.12
        withdrawals = list(np.linspace(0, max_w, 25)[1:])  # skip $0

    return [
        (w, success_rate(_run(replace(params, annual_withdrawal=w), cfg, seed=seed)))
        for w in withdrawals
    ]


# ---------------------------------------------------------------------------
# Balance and depletion statistics
# ---------------------------------------------------------------------------

def balance_percentiles(
    results: SimulationResults,
    percentiles: list[float] | None = None,
) -> dict[float, np.ndarray]:
    """
    Compute portfolio balance percentiles across simulations at each year.

    Returns dict mapping percentile → array of length (years + 1).
    """
    if percentiles is None:
        percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]
    return {
        p: np.percentile(results.balances, p, axis=0)
        for p in percentiles
    }


def depletion_summary(results: SimulationResults) -> dict:
    """Return summary statistics about portfolio depletion."""
    depletion = results.depletion_year
    depleted_mask = ~np.isnan(depletion)
    n_total = len(depletion)
    n_depleted = int(np.sum(depleted_mask))

    summary = {
        "num_simulations": n_total,
        "num_depleted": n_depleted,
        "success_rate": 1.0 - n_depleted / n_total,
    }
    if n_depleted > 0:
        depleted_years = depletion[depleted_mask]
        summary["depletion_year_mean"]   = float(np.mean(depleted_years))
        summary["depletion_year_median"] = float(np.median(depleted_years))
        summary["depletion_year_p10"]    = float(np.percentile(depleted_years, 10))
        summary["depletion_year_p90"]    = float(np.percentile(depleted_years, 90))
    return summary


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(
    results: SimulationResults,
    safe_withdrawal: float | None = None,
    target_success: float | None = None,
    cfg: SimConfig | None = None,
) -> None:
    """Print a formatted summary table to stdout."""
    p = results.params
    summary = depletion_summary(results)

    print("\n" + "=" * 60)
    print("  RETIREMENT PORTFOLIO MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"  Starting balance:      ${p.initial_balance:>15,.0f}")

    if p.tickers:
        total = sum(v for _, v in p.tickers)
        print("  Portfolio holdings:")
        for ticker, value in p.tickers:
            print(f"    {ticker:<8} ${value:>12,.0f}  ({value/total*100:.1f}%)")
        if cfg is not None and cfg.historical is not None:
            hist = cfg.historical
            print(
                f"  Historical data:       {hist.years[0]}–{hist.years[-1]} "
                f"({len(hist.years)} years)"
            )
            print(f"  Chunk size:            {cfg.chunk_size} years")
            inf_label = "actual CPI" if cfg.inflation_mode == "actual" else "fixed"
            print(f"  Inflation mode:        {inf_label}")
    else:
        print(f"  Stock / Bond split:    {p.stock_fraction*100:.0f}% / {p.bond_fraction*100:.0f}%")

    print(f"  Gross withdrawal:      ${p.annual_withdrawal:>15,.0f} / yr")
    print(f"  Social Security:       ${p.social_security:>15,.0f} / yr")
    print(f"  Net portfolio draw:    ${p.net_withdrawal:>15,.0f} / yr")
    print(f"  Withdrawal rate:       {p.annual_withdrawal/p.initial_balance*100:.2f}% of portfolio")
    print(f"  Simulation years:      {p.years}")
    print(f"  Simulations run:       {p.num_simulations:,}")
    print("-" * 60)
    print(f"  Success rate:          {summary['success_rate']*100:.1f}%")
    print(f"  Portfolios depleted:   {summary['num_depleted']:,} / {summary['num_simulations']:,}")
    if summary["num_depleted"] > 0:
        print(f"  Avg depletion year:    {summary['depletion_year_mean']:.1f}")
        print(f"  Median depletion year: {summary['depletion_year_median']:.1f}")
    if safe_withdrawal is not None and target_success is not None:
        print("-" * 60)
        print(f"  Safe withdrawal ({target_success*100:.0f}%): ${safe_withdrawal:>13,.0f} / yr")
        swr_pct = safe_withdrawal / p.initial_balance * 100
        print(f"  Safe withdrawal rate:  {swr_pct:.2f}% of portfolio")
    print("=" * 60 + "\n")
