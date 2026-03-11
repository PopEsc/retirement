"""Tests for retirement_sim/simulation.py."""

import numpy as np
import pytest

from retirement_sim.simulation import (
    MarketParams,
    PortfolioParams,
    SimulationResults,
    _simulate_balances,
    run_simulation,
    run_simulation_historical,
)


# ---------------------------------------------------------------------------
# MarketParams
# ---------------------------------------------------------------------------

class TestMarketParams:
    def test_defaults(self):
        m = MarketParams()
        assert m.stock_mean == pytest.approx(0.10)
        assert m.stock_std  == pytest.approx(0.17)
        assert m.bond_mean  == pytest.approx(0.04)
        assert m.bond_std   == pytest.approx(0.07)
        assert m.inflation_mean == pytest.approx(0.03)
        assert m.inflation_std  == pytest.approx(0.015)

    def test_custom_values(self):
        m = MarketParams(stock_mean=0.07, bond_mean=0.02)
        assert m.stock_mean == pytest.approx(0.07)
        assert m.bond_mean  == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# PortfolioParams — validation
# ---------------------------------------------------------------------------

class TestPortfolioParamsValidation:
    def test_requires_stock_fraction_or_tickers(self):
        with pytest.raises(ValueError, match="stock_fraction or tickers"):
            PortfolioParams(initial_balance=1e6, annual_withdrawal=50_000)

    def test_rejects_both_stock_fraction_and_tickers(self):
        with pytest.raises(ValueError, match="not both"):
            PortfolioParams(
                initial_balance=1e6,
                annual_withdrawal=50_000,
                stock_fraction=0.6,
                tickers=[("SPY", 600_000)],
            )

    def test_stock_fraction_above_one_rejected(self):
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            PortfolioParams(
                initial_balance=1e6, annual_withdrawal=50_000, stock_fraction=1.5
            )

    def test_stock_fraction_below_zero_rejected(self):
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            PortfolioParams(
                initial_balance=1e6, annual_withdrawal=50_000, stock_fraction=-0.1
            )

    def test_stock_fraction_boundary_values_accepted(self):
        # 0.0 and 1.0 are valid (100% bonds or 100% stocks)
        PortfolioParams(initial_balance=1e6, annual_withdrawal=0, stock_fraction=0.0)
        PortfolioParams(initial_balance=1e6, annual_withdrawal=0, stock_fraction=1.0)


# ---------------------------------------------------------------------------
# PortfolioParams — properties
# ---------------------------------------------------------------------------

class TestPortfolioParamsProperties:
    def test_bond_fraction(self):
        p = PortfolioParams(
            initial_balance=1e6, annual_withdrawal=50_000, stock_fraction=0.6
        )
        assert p.bond_fraction == pytest.approx(0.4)

    def test_bond_fraction_is_none_for_tickers(self):
        p = PortfolioParams(
            initial_balance=1e6,
            annual_withdrawal=50_000,
            tickers=[("SPY", 1_000_000)],
        )
        assert p.bond_fraction is None

    def test_net_withdrawal_subtracts_ss(self):
        p = PortfolioParams(
            initial_balance=1e6,
            annual_withdrawal=60_000,
            stock_fraction=0.6,
            social_security=15_000,
        )
        assert p.net_withdrawal == pytest.approx(45_000)

    def test_net_withdrawal_floored_at_zero(self):
        p = PortfolioParams(
            initial_balance=1e6,
            annual_withdrawal=10_000,
            stock_fraction=0.6,
            social_security=20_000,
        )
        assert p.net_withdrawal == 0.0

    def test_ticker_weights_sum_to_one(self):
        p = PortfolioParams(
            initial_balance=1e6,
            annual_withdrawal=50_000,
            tickers=[("SPY", 600_000), ("BND", 400_000)],
        )
        weights = p.ticker_weights
        assert weights is not None
        assert weights.sum() == pytest.approx(1.0)
        assert weights[0] == pytest.approx(0.6)
        assert weights[1] == pytest.approx(0.4)

    def test_ticker_weights_none_for_parametric(self):
        p = PortfolioParams(
            initial_balance=1e6, annual_withdrawal=50_000, stock_fraction=0.6
        )
        assert p.ticker_weights is None

    def test_allocation_label_parametric(self):
        p = PortfolioParams(
            initial_balance=1e6, annual_withdrawal=50_000, stock_fraction=0.6
        )
        label = p.allocation_label()
        assert "60%" in label
        assert "stocks" in label

    def test_allocation_label_tickers(self):
        p = PortfolioParams(
            initial_balance=1e6,
            annual_withdrawal=50_000,
            tickers=[("SPY", 600_000), ("BND", 400_000)],
        )
        label = p.allocation_label()
        assert "SPY" in label
        assert "BND" in label


# ---------------------------------------------------------------------------
# run_simulation — output shape and basic correctness
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def _make_params(self, **kwargs):
        defaults = dict(
            initial_balance=1_000_000,
            annual_withdrawal=40_000,
            stock_fraction=0.6,
            years=10,
            num_simulations=100,
        )
        defaults.update(kwargs)
        return PortfolioParams(**defaults)

    def test_output_shapes(self):
        params = self._make_params(years=10, num_simulations=100)
        results = run_simulation(params, seed=0)
        assert results.balances.shape == (100, 11)
        assert results.depletion_year.shape == (100,)

    def test_initial_balance_column_unchanged(self):
        params = self._make_params()
        results = run_simulation(params, seed=0)
        np.testing.assert_array_equal(results.balances[:, 0], 1_000_000)

    def test_balances_nonnegative(self):
        params = self._make_params(annual_withdrawal=80_000)
        results = run_simulation(params, seed=0)
        assert np.all(results.balances >= 0)

    def test_reproducible_with_same_seed(self):
        params = self._make_params()
        r1 = run_simulation(params, seed=42)
        r2 = run_simulation(params, seed=42)
        np.testing.assert_array_equal(r1.balances, r2.balances)
        np.testing.assert_array_equal(r1.depletion_year, r2.depletion_year)

    def test_different_seeds_produce_different_results(self):
        params = self._make_params(num_simulations=200)
        r1 = run_simulation(params, seed=1)
        r2 = run_simulation(params, seed=2)
        assert not np.array_equal(r1.balances, r2.balances)

    def test_zero_net_withdrawal_never_depletes(self):
        # Zero withdrawal → balance only changes via returns; can never reach 0
        # unless returns are catastrophically negative, which normal dist makes
        # extremely unlikely over 200 sims.  Use very small withdrawal and high
        # SS to guarantee net=0.
        params = self._make_params(
            annual_withdrawal=10_000, social_security=10_000, num_simulations=500
        )
        results = run_simulation(params, seed=0)
        assert np.all(np.isnan(results.depletion_year))

    def test_massive_withdrawal_depletes_all_year_one(self):
        # Withdrawal so large the portfolio cannot survive year 1 regardless of return
        params = self._make_params(
            initial_balance=1_000,
            annual_withdrawal=10_000_000,
            num_simulations=100,
        )
        results = run_simulation(params, seed=0)
        np.testing.assert_array_equal(results.depletion_year, 1.0)

    def test_depletion_year_recorded_correctly(self):
        # Once depleted, balance stays at 0
        params = self._make_params(annual_withdrawal=200_000, num_simulations=200)
        results = run_simulation(params, seed=0)
        for sim_i in range(results.balances.shape[0]):
            if not np.isnan(results.depletion_year[sim_i]):
                yr = int(results.depletion_year[sim_i])
                # Balance at depletion year should be 0
                assert results.balances[sim_i, yr] == 0.0
                # All subsequent years should also be 0
                assert np.all(results.balances[sim_i, yr:] == 0.0)

    def test_params_stored_on_results(self):
        params = self._make_params()
        results = run_simulation(params, seed=0)
        assert results.params is params


# ---------------------------------------------------------------------------
# run_simulation_historical
# ---------------------------------------------------------------------------

class TestRunSimulationHistorical:
    def _make_params(self, small_historical, **kwargs):
        defaults = dict(
            initial_balance=1_000_000,
            annual_withdrawal=40_000,
            tickers=[("SPY", 600_000), ("BND", 400_000)],
            years=10,
            num_simulations=100,
        )
        defaults.update(kwargs)
        return PortfolioParams(**defaults)

    def test_output_shapes(self, small_historical):
        params = self._make_params(small_historical, years=10, num_simulations=100)
        results = run_simulation_historical(params, small_historical, seed=0)
        assert results.balances.shape == (100, 11)
        assert results.depletion_year.shape == (100,)

    def test_initial_balance_preserved(self, small_historical):
        params = self._make_params(small_historical)
        results = run_simulation_historical(params, small_historical, seed=0)
        np.testing.assert_array_equal(results.balances[:, 0], 1_000_000)

    def test_balances_nonnegative(self, small_historical):
        params = self._make_params(small_historical, annual_withdrawal=200_000)
        results = run_simulation_historical(params, small_historical, seed=0)
        assert np.all(results.balances >= 0)

    def test_reproducible_with_same_seed(self, small_historical):
        params = self._make_params(small_historical)
        r1 = run_simulation_historical(params, small_historical, seed=99)
        r2 = run_simulation_historical(params, small_historical, seed=99)
        np.testing.assert_array_equal(r1.balances, r2.balances)

    def test_different_seeds_differ(self, small_historical):
        params = self._make_params(small_historical, num_simulations=200)
        r1 = run_simulation_historical(params, small_historical, chunk_size=3, seed=1)
        r2 = run_simulation_historical(params, small_historical, chunk_size=3, seed=2)
        assert not np.array_equal(r1.balances, r2.balances)

    def test_fixed_inflation_mode(self, small_historical):
        params = self._make_params(small_historical)
        results = run_simulation_historical(
            params, small_historical, inflation_mode="fixed", seed=0
        )
        assert results.balances.shape == (100, 11)

    def test_chunk_size_one(self, small_historical):
        params = self._make_params(small_historical, num_simulations=50)
        results = run_simulation_historical(
            params, small_historical, chunk_size=1, seed=0
        )
        assert results.balances.shape == (50, 11)

    def test_chunk_size_larger_than_history(self, small_historical):
        # chunk_size is clamped to n_hist, should not raise
        params = self._make_params(small_historical, num_simulations=50)
        results = run_simulation_historical(
            params, small_historical, chunk_size=100, seed=0
        )
        assert results.balances.shape == (50, 11)

    def test_chunk_size_affects_sequence(self, small_historical):
        # chunk_size=1 and chunk_size=5 should produce different year-index sequences
        params = self._make_params(small_historical, num_simulations=50)
        r1 = run_simulation_historical(
            params, small_historical, chunk_size=1, seed=7
        )
        r5 = run_simulation_historical(
            params, small_historical, chunk_size=5, seed=7
        )
        assert not np.array_equal(r1.balances, r5.balances)


# ---------------------------------------------------------------------------
# _simulate_balances — shared kernel
# ---------------------------------------------------------------------------

class TestSimulateBalances:
    def _run(self, n_sims=50, n_years=10, withdrawal=40_000, balance=1_000_000,
             ret=0.07, inf=0.03):
        params = PortfolioParams(
            initial_balance=balance,
            annual_withdrawal=withdrawal,
            stock_fraction=0.6,
            years=n_years,
            num_simulations=n_sims,
        )
        portfolio_returns = np.full((n_sims, n_years), ret)
        inflation_rates   = np.full((n_sims, n_years), inf)
        return _simulate_balances(params, portfolio_returns, inflation_rates)

    def test_deterministic_single_path(self):
        # With constant 7% return, 3% inflation, $40k withdrawal on $1M portfolio
        # year 1: 1_000_000 * 1.07 - 40_000 = 1_030_000
        results = self._run(n_sims=1, ret=0.07, inf=0.03, withdrawal=40_000)
        assert results.balances[0, 1] == pytest.approx(1_030_000, rel=1e-6)

    def test_withdrawal_grows_with_inflation(self):
        # Year 2 withdrawal = 40_000 * 1.03 = 41_200
        # Balance after year 2 = year_1_balance * 1.07 - 41_200
        results = self._run(n_sims=1, ret=0.07, inf=0.03, withdrawal=40_000)
        expected_yr1 = 1_000_000 * 1.07 - 40_000
        expected_yr2 = expected_yr1 * 1.07 - 40_000 * 1.03
        assert results.balances[0, 2] == pytest.approx(expected_yr2, rel=1e-6)

    def test_depletion_stays_at_zero(self):
        # Extreme withdrawal → depletion in year 1 → all subsequent years = 0
        results = self._run(n_sims=1, withdrawal=5_000_000, balance=100)
        assert results.balances[0, 1] == 0.0
        assert np.all(results.balances[0, 1:] == 0.0)
        assert results.depletion_year[0] == 1.0
