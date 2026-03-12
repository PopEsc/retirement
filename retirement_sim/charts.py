"""
Matplotlib charts for retirement simulation results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .simulation import SimulationResults
from .analysis import balance_percentiles, sweep_withdrawal_rates


_ACCOUNT_COLORS = {
    "Cash":             "#7f7f7f",   # gray   — spent first
    "Brokerage":        "#ff7f0e",   # orange
    "Traditional 401k": "#1f77b4",   # blue
    "Traditional IRA":  "#aec7e8",   # light blue
    "After-Tax 401k":   "#17becf",   # cyan
    "Roth 401k":        "#2ca02c",   # green
    "Roth IRA":         "#98df8a",   # light green — spent last
}

# Draw order: typically spend Cash/Brokerage before tax-deferred, Roth last
_ACCOUNT_ORDER = [
    "Cash", "Brokerage",
    "Traditional 401k", "Traditional IRA", "After-Tax 401k",
    "Roth 401k", "Roth IRA",
]


_PERCENTILE_COLORS = {
    10.0: "#d62728",   # red
    25.0: "#ff7f0e",   # orange
    50.0: "#2ca02c",   # green (median)
    75.0: "#1f77b4",   # blue
    90.0: "#9467bd",   # purple
}


def _money(x: float, pos=None) -> str:  # noqa: ANN001
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"


def plot_balance_percentiles(
    results: SimulationResults,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Fan chart showing portfolio balance percentiles over time.
    Shaded bands between 10–90 and 25–75, median line highlighted.
    """
    pcts = balance_percentiles(results, percentiles=[10.0, 25.0, 50.0, 75.0, 90.0])
    years = np.arange(results.params.years + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded bands
    ax.fill_between(years, pcts[10.0], pcts[90.0],
                    alpha=0.15, color="#1f77b4", label="10th–90th percentile")
    ax.fill_between(years, pcts[25.0], pcts[75.0],
                    alpha=0.25, color="#1f77b4", label="25th–75th percentile")

    # Percentile lines
    for p, color in _PERCENTILE_COLORS.items():
        lw = 2.5 if p == 50.0 else 1.2
        ls = "-" if p == 50.0 else "--"
        ax.plot(years, pcts[p], color=color, linewidth=lw, linestyle=ls,
                label=f"{p:.0f}th percentile")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Portfolio Balance", fontsize=12)
    ax.set_title(
        f"Portfolio Balance Percentiles Over {results.params.years} Years\n"
        f"(${results.params.annual_withdrawal:,.0f}/yr gross withdrawal, "
        f"{results.params.allocation_label()})",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, results.params.years)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Chart saved: {output_path}")
    return fig


def plot_success_rates(
    results: SimulationResults,
    sweep: list[tuple[float, float]],
    current_withdrawal: float,
    safe_withdrawal: float | None = None,
    target_success: float | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Line chart of success rate vs. annual withdrawal amount.
    Marks the current withdrawal and (optionally) the safe withdrawal level.
    """
    withdrawals = [w for w, _ in sweep]
    rates = [r * 100 for _, r in sweep]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(withdrawals, rates, color="#1f77b4", linewidth=2, marker="o",
            markersize=4, label="Success rate")

    # Mark current withdrawal
    ax.axvline(current_withdrawal, color="#d62728", linewidth=1.5, linestyle="--",
               label=f"Current withdrawal (${current_withdrawal:,.0f})")

    # Mark safe withdrawal
    if safe_withdrawal is not None and target_success is not None:
        ax.axvline(safe_withdrawal, color="#2ca02c", linewidth=1.5, linestyle="--",
                   label=f"Safe withdrawal @ {target_success*100:.0f}% (${safe_withdrawal:,.0f})")
        ax.axhline(target_success * 100, color="#2ca02c", linewidth=0.8,
                   linestyle=":", alpha=0.6)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel("Annual Gross Withdrawal", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(
        f"Success Rate vs. Annual Withdrawal\n"
        f"(${results.params.initial_balance:,.0f} portfolio, "
        f"{results.params.years} years, {results.params.num_simulations:,} simulations)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Chart saved: {output_path}")
    return fig


def plot_depletion_histogram(
    results: SimulationResults,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Histogram of the year portfolios were depleted, plus a bar for survivors.
    """
    depletion = results.depletion_year
    depleted_mask = ~np.isnan(depletion)
    n_total = len(depletion)
    n_depleted = int(np.sum(depleted_mask))
    n_survived = n_total - n_depleted

    fig, ax = plt.subplots(figsize=(10, 6))

    if n_depleted > 0:
        depleted_years = depletion[depleted_mask].astype(int)
        bins = np.arange(0.5, results.params.years + 1.5, 1)
        ax.hist(depleted_years, bins=bins, color="#d62728", alpha=0.75,
                label=f"Depleted ({n_depleted:,} runs, {n_depleted/n_total*100:.1f}%)")

    # Represent survivors as a bar past the end of the simulation window
    survivor_x = results.params.years + 1
    ax.bar(survivor_x, n_survived, color="#2ca02c", alpha=0.75, width=0.8,
           label=f"Survived {results.params.years} years ({n_survived:,} runs, "
                 f"{n_survived/n_total*100:.1f}%)")

    # X-axis labels
    tick_positions = list(range(1, results.params.years + 1, max(1, results.params.years // 10)))
    tick_labels = [str(t) for t in tick_positions]
    tick_positions.append(survivor_x)
    tick_labels.append(f"Survived\n{results.params.years}yr")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)

    ax.set_xlabel("Year of Portfolio Depletion", fontsize=12)
    ax.set_ylabel("Number of Simulations", fontsize=12)
    ax.set_title(
        f"Portfolio Depletion Distribution\n"
        f"(${results.params.annual_withdrawal:,.0f}/yr, "
        f"{results.params.allocation_label()}, "
        f"{results.params.num_simulations:,} simulations)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Chart saved: {output_path}")
    return fig


def plot_account_balances(
    results: SimulationResults,
    account_groups: dict[str, float],
) -> plt.Figure:
    """
    Stacked area chart showing each account type's projected balance over the
    retirement horizon, using the median total-portfolio path.

    Assumes proportional depletion across accounts (each account shrinks at the
    same rate as the total portfolio). Stacked in typical spending order:
    Cash → Brokerage → Traditional → Roth.
    """
    years = np.arange(results.params.years + 1)
    median_balance = np.median(results.balances, axis=0)
    total_initial = sum(account_groups.values())

    # Build ordered list, preserving _ACCOUNT_ORDER, unknown types at the top
    ordered: list[tuple[str, float]] = []
    for acct in _ACCOUNT_ORDER:
        if acct in account_groups:
            ordered.append((acct, account_groups[acct]))
    for acct, val in account_groups.items():
        if acct not in _ACCOUNT_ORDER:
            ordered.append((acct, val))

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(years))
    for acct, initial_val in ordered:
        fraction = initial_val / total_initial if total_initial > 0 else 0.0
        acct_balance = median_balance * fraction
        color = _ACCOUNT_COLORS.get(acct, "#bcbd22")
        ax.fill_between(
            years, bottom, bottom + acct_balance,
            alpha=0.80, color=color,
            label=f"{acct}  (${initial_val:,.0f})",
        )
        bottom += acct_balance

    ax.plot(years, median_balance, color="black", linewidth=1.5,
            linestyle="--", label="Total (median)", zorder=5)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.set_xlabel("Retirement Year", fontsize=12)
    ax.set_ylabel("Portfolio Balance", fontsize=12)
    ax.set_title(
        "Account Balance by Type — Median Scenario\n"
        "(proportional withdrawals assumed across account types)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, results.params.years)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def show_all_charts(
    results: SimulationResults,
    sweep: list[tuple[float, float]],
    safe_withdrawal: float | None = None,
    target_success: float | None = None,
    save_dir: str | Path | None = None,
) -> None:
    """
    Render (and optionally save) all three charts.
    Calls plt.show() at the end to display interactively.
    """
    save = Path(save_dir) if save_dir else None

    plot_balance_percentiles(
        results,
        output_path=save / "balance_percentiles.png" if save else None,
    )
    plot_success_rates(
        results,
        sweep=sweep,
        current_withdrawal=results.params.annual_withdrawal,
        safe_withdrawal=safe_withdrawal,
        target_success=target_success,
        output_path=save / "success_rates.png" if save else None,
    )
    plot_depletion_histogram(
        results,
        output_path=save / "depletion_histogram.png" if save else None,
    )

    plt.show()
