"""Tests for retirement_sim/analysis.py."""

import io

import numpy as np
import pytest

from retirement_sim.analysis import (
    SimConfig,
    _run,
    balance_percentiles,
    depletion_summary,
    find_safe_withdrawal_rate,
    print_summary,
    success_rate,
    sweep_withdrawal_rates,
)
from retirement_sim.simulation import PortfolioParams, SimulationResults, run_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(depleted_fraction: float, n_sims: int = 1000, years: int = 20) -> SimulationResults:
    """
    Build a SimulationResults where exactly `depleted_fraction` of simulations
    are marked as depleted (at year 10), the rest as survivors.
    """
    params = PortfolioParams(
        initial_balance=1_000_000,
        annual_withdrawal=50_000,
        stock_fraction=0.6,
        years=years,
        num_simulations=n_sims,
    )
    balances = np.ones((n_sims, years + 1)) * 500_000
    balances[:, 0] = 1_000_000
    depletion_year = np.full(n_sims, np.nan)

    n_depleted = int(n_sims * depleted_fraction)
    depletion_year[:n_depleted] = 10.0
    balances[:n_depleted, 10:] = 0.0

    return SimulationResults(params=params, balances=balances, depletion_year=depletion_year)


def _make_params(**kwargs) -> PortfolioParams:
    defaults = dict(
        initial_balance=1_000_000,
        annual_withdrawal=50_000,
        stock_fraction=0.60,
        years=20,
        num_simulations=300,
    )
    defaults.update(kwargs)
    return PortfolioParams(**defaults)


# ---------------------------------------------------------------------------
# SimConfig
# ---------------------------------------------------------------------------

class TestSimConfig:
    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.historical is None
        assert cfg.chunk_size == 5
        assert cfg.inflation_mode == "fixed"

    def test_custom_values(self):
        cfg = SimConfig(chunk_size=3, inflation_mode="actual")
        assert cfg.chunk_size == 3
        assert cfg.inflation_mode == "actual"


# ---------------------------------------------------------------------------
# _run dispatcher
# ---------------------------------------------------------------------------

class TestRunDispatcher:
    def test_dispatches_to_parametric_without_cfg(self):
        params = _make_params(num_simulations=50, years=5)
        results = _run(params, cfg=None, seed=0)
        assert results.balances.shape == (50, 6)

    def test_dispatches_to_parametric_with_empty_cfg(self):
        params = _make_params(num_simulations=50, years=5)
        results = _run(params, cfg=SimConfig(), seed=0)
        assert results.balances.shape == (50, 6)

    def test_dispatches_to_historical_with_historical_cfg(self, small_historical):
        params = PortfolioParams(
            initial_balance=1_000_000,
            annual_withdrawal=50_000,
            tickers=[("SPY", 600_000), ("BND", 400_000)],
            years=5,
            num_simulations=50,
        )
        cfg = SimConfig(historical=small_historical)
        results = _run(params, cfg=cfg, seed=0)
        assert results.balances.shape == (50, 6)


# ---------------------------------------------------------------------------
# success_rate
# ---------------------------------------------------------------------------

class TestSuccessRate:
    def test_all_survived(self):
        results = _make_results(depleted_fraction=0.0)
        assert success_rate(results) == pytest.approx(1.0)

    def test_all_depleted(self):
        results = _make_results(depleted_fraction=1.0)
        assert success_rate(results) == pytest.approx(0.0)

    def test_partial(self):
        results = _make_results(depleted_fraction=0.20, n_sims=1000)
        assert success_rate(results) == pytest.approx(0.80)

    def test_return_type_is_float(self):
        results = _make_results(depleted_fraction=0.5)
        assert isinstance(success_rate(results), float)


# ---------------------------------------------------------------------------
# balance_percentiles
# ---------------------------------------------------------------------------

class TestBalancePercentiles:
    def test_shape(self, parametric_results):
        pcts = balance_percentiles(parametric_results)
        for arr in pcts.values():
            assert arr.shape == (parametric_results.params.years + 1,)

    def test_default_percentiles_keys(self, parametric_results):
        pcts = balance_percentiles(parametric_results)
        assert set(pcts.keys()) == {10.0, 25.0, 50.0, 75.0, 90.0}

    def test_ordering(self, parametric_results):
        # At every year, lower percentile ≤ higher percentile
        pcts = balance_percentiles(parametric_results)
        assert np.all(pcts[10.0] <= pcts[50.0])
        assert np.all(pcts[50.0] <= pcts[90.0])

    def test_custom_percentiles(self, parametric_results):
        pcts = balance_percentiles(parametric_results, percentiles=[5.0, 95.0])
        assert set(pcts.keys()) == {5.0, 95.0}

    def test_year_zero_equals_initial_balance(self, parametric_params):
        results = run_simulation(parametric_params, seed=0)
        pcts = balance_percentiles(results)
        for arr in pcts.values():
            assert arr[0] == pytest.approx(parametric_params.initial_balance)


# ---------------------------------------------------------------------------
# depletion_summary
# ---------------------------------------------------------------------------

class TestDepletionSummary:
    def test_no_depletions(self):
        results = _make_results(depleted_fraction=0.0, n_sims=500)
        summary = depletion_summary(results)
        assert summary["success_rate"] == pytest.approx(1.0)
        assert summary["num_depleted"] == 0
        assert "depletion_year_mean" not in summary

    def test_all_depleted(self):
        results = _make_results(depleted_fraction=1.0, n_sims=500)
        summary = depletion_summary(results)
        assert summary["success_rate"] == pytest.approx(0.0)
        assert summary["num_depleted"] == 500
        assert "depletion_year_mean" in summary
        assert summary["depletion_year_mean"] == pytest.approx(10.0)

    def test_partial_depletion(self):
        results = _make_results(depleted_fraction=0.30, n_sims=1000)
        summary = depletion_summary(results)
        assert summary["num_depleted"] == 300
        assert summary["success_rate"] == pytest.approx(0.70)

    def test_num_simulations_matches(self):
        results = _make_results(depleted_fraction=0.5, n_sims=400)
        summary = depletion_summary(results)
        assert summary["num_simulations"] == 400

    def test_depletion_stats_present_when_depleted(self):
        results = _make_results(depleted_fraction=0.5, n_sims=200)
        summary = depletion_summary(results)
        for key in ("depletion_year_mean", "depletion_year_median",
                    "depletion_year_p10", "depletion_year_p90"):
            assert key in summary


# ---------------------------------------------------------------------------
# find_safe_withdrawal_rate
# ---------------------------------------------------------------------------

class TestFindSafeWithdrawalRate:
    def test_high_withdrawal_lower_success(self):
        params = _make_params(num_simulations=500, years=30)
        swr_90 = find_safe_withdrawal_rate(params, target_success=0.90, seed=42)
        swr_95 = find_safe_withdrawal_rate(params, target_success=0.95, seed=42)
        # 90% target should allow higher withdrawal than 95%
        assert swr_90 >= swr_95

    def test_result_achieves_target(self):
        # Run a simulation AT the found SWR and verify success rate ≥ target
        params = _make_params(num_simulations=500, years=20)
        target = 0.90
        swr = find_safe_withdrawal_rate(params, target_success=target, seed=0, tolerance=200)
        from dataclasses import replace
        check_params = replace(params, annual_withdrawal=swr)
        results = run_simulation(check_params, seed=0)
        assert success_rate(results) >= target - 0.03  # allow 3% tolerance for binary search

    def test_respects_tolerance(self):
        # With loose tolerance, result should still be a positive number
        params = _make_params(num_simulations=200, years=10)
        swr = find_safe_withdrawal_rate(
            params, target_success=0.90, seed=0, tolerance=5_000
        )
        assert swr > 0

    def test_returns_float(self):
        params = _make_params(num_simulations=100, years=10)
        swr = find_safe_withdrawal_rate(params, target_success=0.90, seed=0)
        assert isinstance(swr, float)


# ---------------------------------------------------------------------------
# sweep_withdrawal_rates
# ---------------------------------------------------------------------------

class TestSweepWithdrawalRates:
    def test_returns_correct_count(self):
        params = _make_params(num_simulations=100, years=10)
        sweep = sweep_withdrawal_rates(params, seed=0)
        assert len(sweep) == 24  # default: linspace(0, 12%, 25)[1:]

    def test_custom_withdrawals(self):
        params = _make_params(num_simulations=100, years=10)
        amounts = [10_000, 50_000, 100_000]
        sweep = sweep_withdrawal_rates(params, withdrawals=amounts, seed=0)
        assert len(sweep) == 3
        assert [w for w, _ in sweep] == amounts

    def test_rates_between_zero_and_one(self):
        params = _make_params(num_simulations=100, years=10)
        sweep = sweep_withdrawal_rates(params, seed=0)
        for _, rate in sweep:
            assert 0.0 <= rate <= 1.0

    def test_monotonically_decreasing_with_fixed_seed(self):
        # With same seed per call, higher withdrawals → lower/equal success
        params = _make_params(num_simulations=500, years=20)
        amounts = [20_000, 40_000, 60_000, 80_000, 100_000]
        sweep = sweep_withdrawal_rates(params, withdrawals=amounts, seed=42)
        rates = [r for _, r in sweep]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1] - 0.02  # allow tiny noise


# ---------------------------------------------------------------------------
# print_summary — stdout content checks
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_parametric_output_contains_key_fields(self, parametric_results, capsys):
        print_summary(parametric_results)
        out = capsys.readouterr().out
        assert "1,000,000" in out
        assert "Stock / Bond split" in out
        assert "Success rate" in out

    def test_safe_withdrawal_shown_when_provided(self, parametric_results, capsys):
        print_summary(parametric_results, safe_withdrawal=55_000, target_success=0.95)
        out = capsys.readouterr().out
        assert "55,000" in out
        assert "95%" in out

    def test_ticker_mode_output(self, small_historical, capsys):
        params = PortfolioParams(
            initial_balance=1_000_000,
            annual_withdrawal=50_000,
            tickers=[("SPY", 600_000), ("BND", 400_000)],
            years=10,
            num_simulations=100,
        )
        from retirement_sim.simulation import run_simulation_historical
        results = run_simulation_historical(params, small_historical, seed=0)
        cfg = SimConfig(historical=small_historical)
        print_summary(results, cfg=cfg)
        out = capsys.readouterr().out
        assert "SPY" in out
        assert "BND" in out
        assert "Historical data" in out

    def test_ticker_mode_shows_inflation_label(self, small_historical, capsys):
        params = PortfolioParams(
            initial_balance=1_000_000,
            annual_withdrawal=50_000,
            tickers=[("SPY", 600_000), ("BND", 400_000)],
            years=10,
            num_simulations=50,
        )
        from retirement_sim.simulation import run_simulation_historical
        results = run_simulation_historical(params, small_historical, seed=0)
        cfg = SimConfig(historical=small_historical, inflation_mode="actual")
        print_summary(results, cfg=cfg)
        out = capsys.readouterr().out
        assert "actual CPI" in out

    def test_no_depletion_skips_depletion_year_line(self, capsys):
        results = _make_results(depleted_fraction=0.0)
        print_summary(results)
        out = capsys.readouterr().out
        assert "depletion year" not in out.lower()
