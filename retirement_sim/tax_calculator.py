"""
Bracket-based tax calculator for retirement simulation.

Uses 2026 federal brackets and 2025/2026 state brackets from tax_data.py.
Provides binary-search gross-up: given a net (after-tax) spending goal,
find the gross portfolio withdrawal needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tax_data import (
    EXTRA_DEDUCTION_65_PLUS,
    FEDERAL_LTCG_BRACKETS,
    FEDERAL_ORDINARY_BRACKETS,
    NIIT_THRESHOLD,
    SS_TAXABILITY,
    STANDARD_DEDUCTION,
    STATE_TAX,
    STATES_NO_SS_TAX,
)


@dataclass
class TaxProfile:
    """Personal tax profile for bracket-based gross-up calculations."""
    filing_status: str        # "single", "married_joint", "head_of_household", "married_separate"
    age: int                  # retiree's age (for 65+ extra deduction)
    state: str | None         # two-letter state code, or None/"None" for no state tax
    spouse_also_65_plus: bool = False   # applies extra deduction for MFJ when spouse is also 65+


# ---------------------------------------------------------------------------
# Low-level bracket helpers
# ---------------------------------------------------------------------------

def _bracket_tax(income: float, brackets: list[tuple[float, float]]) -> float:
    """Compute tax using a progressive bracket schedule.

    brackets: [(income_floor, marginal_rate), ...] sorted ascending.
    """
    if income <= 0:
        return 0.0
    tax = 0.0
    for i, (floor, rate) in enumerate(brackets):
        next_floor = brackets[i + 1][0] if i + 1 < len(brackets) else float("inf")
        if income <= floor:
            break
        taxable_in_bracket = min(income, next_floor) - floor
        tax += taxable_in_bracket * rate
    return tax


def _ltcg_tax(
    ordinary_taxable: float,
    ltcg: float,
    brackets: list[tuple[float, float]],
) -> float:
    """Compute LTCG tax using the income-stacking method.

    LTCG rate depends on where total taxable income (ordinary + LTCG) falls.
    Ordinary income fills lower brackets first; LTCG is taxed at the rate of
    the band it occupies when stacked on top.
    """
    if ltcg <= 0:
        return 0.0
    floors = [b[0] for b in brackets]
    rates  = [b[1] for b in brackets]
    ceilings = floors[1:] + [float("inf")]
    tax = 0.0
    for rate, floor, ceiling in zip(rates, floors, ceilings):
        overlap_start = max(ordinary_taxable, floor)
        overlap_end   = min(ordinary_taxable + ltcg, ceiling)
        if overlap_end > overlap_start:
            tax += (overlap_end - overlap_start) * rate
    return tax


# ---------------------------------------------------------------------------
# Social Security taxability (IRC §86)
# ---------------------------------------------------------------------------

def _taxable_ss(ss_annual: float, agi_before_ss: float, filing_status: str) -> float:
    """Compute taxable portion of Social Security / pension (IRC §86).

    Provisional income = AGI (before SS) + 50% of SS benefits.
    Up to 50% of SS taxable above first threshold; up to 85% above second.
    """
    if ss_annual <= 0:
        return 0.0
    thresholds = SS_TAXABILITY.get(filing_status, SS_TAXABILITY["single"])
    t1 = thresholds["threshold_50pct"]
    t2 = thresholds["threshold_85pct"]
    provisional = agi_before_ss + 0.5 * ss_annual
    if provisional <= t1:
        return 0.0
    if provisional <= t2:
        return min(0.5 * (provisional - t1), 0.5 * ss_annual)
    # Above t2: 85% cap
    t1_portion = min(0.5 * (t2 - t1), 0.5 * ss_annual)
    t2_portion = 0.85 * (provisional - t2)
    return min(t1_portion + t2_portion, 0.85 * ss_annual)


# ---------------------------------------------------------------------------
# Federal tax
# ---------------------------------------------------------------------------

def _federal_std_deduction(profile: TaxProfile) -> float:
    """Standard deduction including 65+ extra amounts."""
    base  = STANDARD_DEDUCTION[profile.filing_status]
    extra = EXTRA_DEDUCTION_65_PLUS[profile.filing_status]
    bonus = extra if profile.age >= 65 else 0.0
    if profile.filing_status == "married_joint" and profile.spouse_also_65_plus:
        bonus += extra
    return base + bonus


def compute_federal_tax(
    ordinary_income: float,
    ltcg: float,
    ss_annual: float,
    profile: TaxProfile,
) -> float:
    """Estimate federal income tax.

    Parameters
    ----------
    ordinary_income : pre-tax traditional IRA/401k withdrawal (today's dollars)
    ltcg            : long-term capital gains from brokerage gains portion
    ss_annual       : annual Social Security / pension income
    profile         : TaxProfile (filing status, age, state)

    Returns total federal tax (ordinary + LTCG + NIIT).
    """
    fs = profile.filing_status

    # Taxable SS (provisional income = ordinary + ltcg + 0.5 × SS)
    agi_before_ss = ordinary_income + ltcg
    taxable_ss    = _taxable_ss(ss_annual, agi_before_ss, fs)

    # Ordinary taxable income
    std_ded           = _federal_std_deduction(profile)
    ordinary_agi      = ordinary_income + taxable_ss
    ordinary_taxable  = max(0.0, ordinary_agi - std_ded)

    # Ordinary income tax
    ordinary_tax = _bracket_tax(ordinary_taxable, FEDERAL_ORDINARY_BRACKETS[fs])

    # LTCG tax (stacked on ordinary taxable)
    ltcg_tax = _ltcg_tax(ordinary_taxable, ltcg, FEDERAL_LTCG_BRACKETS[fs])

    # Net Investment Income Tax: 3.8% on lesser of NII or excess MAGI over threshold
    magi     = ordinary_agi + ltcg
    niit_base = max(0.0, magi - NIIT_THRESHOLD[fs])
    niit      = 0.038 * min(ltcg, niit_base)

    return ordinary_tax + ltcg_tax + niit


# ---------------------------------------------------------------------------
# State tax
# ---------------------------------------------------------------------------

def compute_state_tax(
    ordinary_income: float,
    ltcg: float,
    ss_annual: float,
    profile: TaxProfile,
) -> float:
    """Estimate state income tax.

    Most states tax LTCG as ordinary income.  Washington (WA) has a
    capital-gains-only tax.  States in STATES_NO_SS_TAX exempt SS income.
    """
    state = profile.state
    if not state or state == "None":
        return 0.0

    state_data = STATE_TAX.get(state)
    if state_data is None:
        return 0.0

    fs     = profile.filing_status
    is_mfj = fs == "married_joint"

    if is_mfj and "married_joint" in state_data:
        brackets   = state_data["married_joint"]
        std_ded    = float(state_data.get("std_ded_mfj", 0))
        tax_credit = float(state_data.get("tax_credit_mfj", 0))
    else:
        brackets   = state_data["single"]
        std_ded    = float(state_data.get("std_ded_single", 0))
        tax_credit = float(state_data.get("tax_credit_single", 0))

    # Washington: LTCG-only tax, no ordinary income tax
    if state == "WA":
        wa_ltcg_brackets = state_data.get("ltcg_brackets", [(0, 0.0)])
        return _bracket_tax(ltcg, wa_ltcg_brackets)

    # SS inclusion: states in STATES_NO_SS_TAX exempt SS entirely
    if state in STATES_NO_SS_TAX:
        state_ss = 0.0
    else:
        # Use federal IRC §86 fraction as approximation
        state_ss = _taxable_ss(ss_annual, ordinary_income + ltcg, fs)

    # Most states tax LTCG as ordinary income
    state_agi      = ordinary_income + ltcg + state_ss
    state_taxable  = max(0.0, state_agi - std_ded)
    state_tax_amt  = _bracket_tax(state_taxable, brackets)

    # Apply flat tax credit (e.g., Utah)
    return max(0.0, state_tax_amt - tax_credit)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_total_tax(
    ordinary_income: float,
    ltcg: float,
    ss_annual: float,
    profile: TaxProfile,
) -> float:
    """Total estimated tax (federal + state)."""
    return (
        compute_federal_tax(ordinary_income, ltcg, ss_annual, profile)
        + compute_state_tax(ordinary_income, ltcg, ss_annual, profile)
    )


# ---------------------------------------------------------------------------
# Binary-search gross-up
# ---------------------------------------------------------------------------

def gross_up_withdrawal(
    net_spending: float,
    pretax_frac: float,
    brokerage_frac: float,
    gains_frac: float,
    ss_annual: float,
    profile: TaxProfile,
    tolerance: float = 10.0,
) -> float:
    """Find the gross portfolio withdrawal that yields the desired net spending.

    Solves for W (gross portfolio draw) such that:
        W + ss_annual − total_tax(W, ss_annual, profile) = net_spending

    Returns W + ss_annual, matching the ``annual_withdrawal`` convention in
    ``PortfolioParams`` (gross total spending before SS offset).

    Parameters
    ----------
    net_spending   : after-tax annual spending goal (today's dollars)
    pretax_frac    : share of portfolio in traditional IRA/401k
    brokerage_frac : share of portfolio in taxable brokerage
    gains_frac     : aggregate unrealised-gains fraction of brokerage value
                     = (total_value − total_basis) / total_value
    ss_annual      : annual Social Security / pension income (today's dollars)
    profile        : TaxProfile
    tolerance      : binary-search convergence tolerance in dollars
    """
    if net_spending <= 0:
        return max(0.0, net_spending)

    # Target: W − tax(W, ss) = net_from_portfolio
    net_from_portfolio = net_spending - ss_annual
    if net_from_portfolio <= 0:
        # SS covers all spending; no portfolio draw needed
        return max(0.0, ss_annual)

    def _net_after_tax(w: float) -> float:
        ordinary = pretax_frac * w
        ltcg     = gains_frac * brokerage_frac * w
        taxes    = compute_total_tax(ordinary, ltcg, ss_annual, profile)
        return w - taxes

    # Binary search bounds
    lo = net_from_portfolio          # lower bound: zero-tax case
    hi = net_from_portfolio * 4.0   # generous upper bound
    # Expand hi if taxes are extreme
    for _ in range(20):
        if _net_after_tax(hi) >= net_from_portfolio:
            break
        hi *= 2.0

    for _ in range(80):
        if hi - lo < tolerance:
            break
        mid = (lo + hi) / 2.0
        if _net_after_tax(mid) < net_from_portfolio:
            lo = mid
        else:
            hi = mid

    # lo = gross portfolio draw (W); return W + ss_annual = annual_withdrawal
    return lo + ss_annual
