"""
Tests for retirement_sim/charts.py.

All tests use the Agg (non-interactive) backend set in conftest.py so no
display is required.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from retirement_sim.analysis import balance_percentiles, sweep_withdrawal_rates
from retirement_sim.charts import (
    plot_balance_percentiles,
    plot_depletion_histogram,
    plot_success_rates,
    show_all_charts,
)
from retirement_sim.simulation import PortfolioParams, SimulationResults, run_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(n_sims=100, years=20, depleted_fraction=0.3):
    params = PortfolioParams(
        initial_balance=1_000_000,
        annual_withdrawal=60_000,
        stock_fraction=0.6,
        years=years,
        num_simulations=n_sims,
    )
    balances = np.ones((n_sims, years + 1)) * 500_000
    balances[:, 0] = 1_000_000
    depletion_year = np.full(n_sims, np.nan)

    n_depleted = int(n_sims * depleted_fraction)
    # Space depletion years across the horizon so the histogram has spread
    dep_years = np.linspace(5, years, n_depleted, dtype=int)
    for i, yr in enumerate(dep_years):
        depletion_year[i] = float(yr)
        balances[i, yr:] = 0.0

    return SimulationResults(params=params, balances=balances, depletion_year=depletion_year)


def _make_sweep(params):
    return sweep_withdrawal_rates(
        params,
        withdrawals=[20_000, 40_000, 60_000, 80_000, 100_000],
        seed=0,
    )


# ---------------------------------------------------------------------------
# plot_balance_percentiles
# ---------------------------------------------------------------------------

class TestPlotBalancePercentiles:
    def test_returns_figure(self, parametric_results):
        fig = plot_balance_percentiles(parametric_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_x_axis_length_matches_years(self, parametric_results):
        fig = plot_balance_percentiles(parametric_results)
        ax = fig.axes[0]
        years = parametric_results.params.years
        assert ax.get_xlim()[1] == pytest.approx(years)
        plt.close(fig)

    def test_title_contains_withdrawal_amount(self, parametric_results):
        fig = plot_balance_percentiles(parametric_results)
        title = fig.axes[0].get_title()
        withdrawal = parametric_results.params.annual_withdrawal
        assert f"{withdrawal:,.0f}" in title
        plt.close(fig)

    def test_saves_png(self, parametric_results, tmp_path):
        out = tmp_path / "balance.png"
        fig = plot_balance_percentiles(parametric_results, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_success_rates
# ---------------------------------------------------------------------------

class TestPlotSuccessRates:
    def test_returns_figure(self, parametric_results):
        sweep = _make_sweep(parametric_results.params)
        fig = plot_success_rates(
            parametric_results,
            sweep=sweep,
            current_withdrawal=60_000,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_y_axis_upper_limit(self, parametric_results):
        sweep = _make_sweep(parametric_results.params)
        fig = plot_success_rates(
            parametric_results, sweep=sweep, current_withdrawal=60_000
        )
        assert fig.axes[0].get_ylim()[1] == pytest.approx(105)
        plt.close(fig)

    def test_with_safe_withdrawal_marker(self, parametric_results):
        sweep = _make_sweep(parametric_results.params)
        fig = plot_success_rates(
            parametric_results,
            sweep=sweep,
            current_withdrawal=60_000,
            safe_withdrawal=50_000,
            target_success=0.95,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png(self, parametric_results, tmp_path):
        sweep = _make_sweep(parametric_results.params)
        out = tmp_path / "success.png"
        fig = plot_success_rates(
            parametric_results, sweep=sweep,
            current_withdrawal=60_000, output_path=out
        )
        assert out.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_depletion_histogram
# ---------------------------------------------------------------------------

class TestPlotDepletionHistogram:
    def test_returns_figure(self):
        results = _make_results(depleted_fraction=0.3)
        fig = plot_depletion_histogram(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_depletions(self):
        results = _make_results(depleted_fraction=0.0)
        fig = plot_depletion_histogram(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_depleted(self):
        results = _make_results(depleted_fraction=1.0)
        fig = plot_depletion_histogram(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png(self, tmp_path):
        results = _make_results()
        out = tmp_path / "depletion.png"
        fig = plot_depletion_histogram(results, output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_title_contains_allocation_label(self):
        results = _make_results()
        fig = plot_depletion_histogram(results)
        title = fig.axes[0].get_title()
        # allocation_label() for 60% stocks should appear somewhere
        assert "60%" in title
        plt.close(fig)


# ---------------------------------------------------------------------------
# show_all_charts
# ---------------------------------------------------------------------------

class TestShowAllCharts:
    def test_saves_three_pngs(self, parametric_results, tmp_path):
        sweep = _make_sweep(parametric_results.params)
        show_all_charts(
            parametric_results,
            sweep=sweep,
            safe_withdrawal=55_000,
            target_success=0.95,
            save_dir=tmp_path,
        )
        plt.close("all")
        assert (tmp_path / "balance_percentiles.png").exists()
        assert (tmp_path / "success_rates.png").exists()
        assert (tmp_path / "depletion_histogram.png").exists()

    def test_runs_without_save_dir(self, parametric_results):
        sweep = _make_sweep(parametric_results.params)
        # Should not raise even without a save_dir (Agg backend suppresses display)
        show_all_charts(parametric_results, sweep=sweep)
        plt.close("all")

    def test_runs_without_safe_withdrawal(self, parametric_results, tmp_path):
        sweep = _make_sweep(parametric_results.params)
        show_all_charts(
            parametric_results,
            sweep=sweep,
            save_dir=tmp_path,
        )
        plt.close("all")
        assert (tmp_path / "balance_percentiles.png").exists()
