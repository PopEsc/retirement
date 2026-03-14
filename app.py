"""
Retirement Portfolio Monte Carlo Simulator — Streamlit GUI

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date as _date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from retirement_sim.analysis import (
    SimConfig,
    _run,
    depletion_summary,
    find_safe_withdrawal_rate,
    sweep_withdrawal_rates,
)
from retirement_sim.charts import (
    plot_account_balances,
    plot_balance_percentiles,
    plot_depletion_histogram,
    plot_success_rates,
)
from retirement_sim.simulation import MarketParams, PortfolioParams
from retirement_sim.tax_calculator import TaxProfile, compute_total_tax, gross_up_withdrawal
from retirement_sim.tax_data import FILING_STATUS_LABELS, STATE_NAMES


# ── Cached data fetching ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_historical(
    ticker_values: tuple[tuple[str, float], ...],
    start_year: int,
    end_year: int,
    ticker_proxies: tuple[tuple[str, str], ...] | None,
):
    """
    Thin wrapper around build_historical_dataset with Streamlit-level caching.
    Avoids re-reading even disk-cached files when the user reruns with the
    same parameters within the same Streamlit session (ttl=1 hour).
    """
    from retirement_sim.market_data import build_historical_dataset
    return build_historical_dataset(
        list(ticker_values),
        start_year=start_year,
        end_year=end_year,
        ticker_proxies=dict(ticker_proxies) if ticker_proxies else None,
    )


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retirement Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PWA support — links the web app manifest so Android Chrome shows
# "Add to Home Screen" and the installed app launches full-screen.
st.markdown(
    """
    <link rel="manifest" href="app/static/manifest.json">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="RetireSim">
    <meta name="theme-color" content="#1f77b4">
    """,
    unsafe_allow_html=True,
)


# ── Account type constants ─────────────────────────────────────────────────────

_ACCOUNT_TYPES = [
    "Traditional IRA",
    "Traditional 401k",
    "Roth IRA",
    "Roth 401k",
    "After-Tax 401k",
    "Brokerage",
    "Cash",
]
_PRETAX_TYPES    = {"Traditional IRA", "Traditional 401k"}
_BROKERAGE_TYPES = {"Brokerage"}
_CASH_TYPES      = {"Cash"}
# Roth IRA, Roth 401k, After-Tax 401k are tax-free on withdrawal


# ── Session state initialisation ──────────────────────────────────────────────

_EMPTY_HOLDINGS = pd.DataFrame({
    "Ticker":         pd.Series([], dtype=str),
    "Value ($)":      pd.Series([], dtype=float),
    "Account Type":   pd.Series([], dtype=str),
    "Cash Rate (%)":  pd.Series([], dtype=float),   # annual rate for Cash accounts
    "Cost Basis ($)": pd.Series([], dtype=float),   # cost basis for Brokerage accounts
    "Proxy":          pd.Series([], dtype=str),
})

def _init_state() -> None:
    defaults: dict = {
        "holdings_df":  _EMPTY_HOLDINGS.copy(),
        "editor_key":   0,          # incremented on JSON import to reset the table
        "import_id":    None,       # tracks which file has been imported
        "ss_override":           None,  # social_security read from the imported file
        "ss_delay_override":     None,  # ss_delay_years read from the imported file
        "ytr_override":          None,  # years_to_retirement from imported file
        "annual_savings_override": None, # annual_savings from imported file
        "account_groups":        None,  # {account_type: initial_value} for chart
        "results":      None,
        "sweep":        None,
        "safe_withdrawal": None,
        "target_success":  None,
        "cfg":          None,
        "error_msg":    None,
        "tax_profile":       None,  # TaxProfile used for this run
        "portfolio_fracs":   None,  # (pretax_frac, brokerage_frac, gains_frac)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _df_to_json(
    df: pd.DataFrame,
    social_security: float,
    ss_delay_years: int = 0,
    years_to_retirement: int = 0,
    annual_savings: float = 0.0,
) -> str:
    """Serialise the holdings DataFrame to our portfolio JSON format."""
    holdings = []
    cash_index = 0
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).upper().strip()
        acct_check = str(row.get("Account Type", "")).strip()
        if not ticker:
            if acct_check in _CASH_TYPES:
                cash_index += 1
                ticker = "CASH" if cash_index == 1 else f"CASH{cash_index}"
            else:
                continue
        try:
            value = float(row["Value ($)"])
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        entry: dict = {"ticker": ticker, "value": value}
        acct = str(row.get("Account Type", "")).strip()
        if acct and acct in _ACCOUNT_TYPES:
            entry["account_type"] = acct
        if acct in _CASH_TYPES:
            try:
                rate = float(row.get("Cash Rate (%)", 0.0) or 0.0)
                if rate > 0:
                    entry["cash_rate"] = rate
            except (TypeError, ValueError):
                pass
        if acct in _BROKERAGE_TYPES:
            try:
                basis = float(row.get("Cost Basis ($)", 0.0) or 0.0)
                if basis > 0:
                    entry["cost_basis"] = basis
            except (TypeError, ValueError):
                pass
        proxy = str(row.get("Proxy", "")).upper().strip()
        if proxy:
            entry["proxy"] = proxy
        holdings.append(entry)

    out: dict = {"holdings": holdings}
    if social_security > 0:
        out["social_security"] = social_security
    if ss_delay_years > 0:
        out["ss_delay_years"] = ss_delay_years
    if years_to_retirement > 0:
        out["years_to_retirement"] = years_to_retirement
    if annual_savings > 0:
        out["annual_savings"] = annual_savings
    return json.dumps(out, indent=2)


def _json_to_df(data: dict) -> tuple[pd.DataFrame, float | None, int | None, int | None, float | None]:
    """Parse a portfolio JSON dict into (DataFrame, social_security)."""
    rows = []
    for h in data.get("holdings", []):
        acct = str(h.get("account_type", "Brokerage")).strip()
        if acct not in _ACCOUNT_TYPES:
            acct = "Brokerage"
        rows.append({
            "Ticker":         str(h.get("ticker", "")).upper().strip(),
            "Value ($)":      float(h.get("value", 0.0)),
            "Account Type":   acct,
            "Cash Rate (%)":  float(h["cash_rate"]) if "cash_rate" in h else None,
            "Cost Basis ($)": float(h["cost_basis"]) if "cost_basis" in h else None,
            "Proxy":          str(h.get("proxy", "")).upper().strip(),
        })

    if rows:
        df = pd.DataFrame(rows).astype({
            "Ticker": str, "Value ($)": float, "Account Type": str, "Proxy": str,
        })
    else:
        df = _EMPTY_HOLDINGS.copy()

    ss = data.get("social_security")
    ss_delay = data.get("ss_delay_years")
    ytr = data.get("years_to_retirement")
    savings = data.get("annual_savings")
    return (
        df,
        float(ss) if ss is not None else None,
        int(ss_delay) if ss_delay is not None else None,
        int(ytr) if ytr is not None else None,
        float(savings) if savings is not None else None,
    )


def _parse_holdings(
    df: pd.DataFrame,
) -> tuple[
    list[tuple[str, float]],  # non-cash (ticker, value) pairs
    dict[str, str],            # proxy mapping
    dict[str, str],            # account types
    float,                     # total cash value
    float,                     # weighted-average cash rate (0–1 decimal)
    dict[str, float],          # cost basis by ticker (Brokerage only)
]:
    """Extract holdings info, separating cash from market assets."""
    tv: list[tuple[str, float]] = []
    tp: dict[str, str] = {}
    ta: dict[str, str] = {}
    basis: dict[str, float] = {}
    cash_weighted_rate = 0.0
    cash_total = 0.0
    cash_index = 0

    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).upper().strip()
        try:
            value = float(row["Value ($)"])
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue

        acct = str(row.get("Account Type", "Brokerage")).strip()
        if acct not in _ACCOUNT_TYPES:
            acct = "Brokerage"

        if acct in _CASH_TYPES:
            # Cash rows don't need a real ticker — generate a label if blank
            if not ticker:
                cash_index += 1
                ticker = "CASH" if cash_index == 1 else f"CASH{cash_index}"
            try:
                rate_pct = float(row.get("Cash Rate (%)", 0.0) or 0.0)
            except (TypeError, ValueError):
                rate_pct = 0.0
            cash_weighted_rate += value * (rate_pct / 100.0)
            cash_total += value
        else:
            if not ticker:
                continue  # non-Cash rows still require a ticker symbol
            tv.append((ticker, value))
            ta[ticker] = acct
            proxy = str(row.get("Proxy", "")).upper().strip()
            if proxy:
                tp[ticker] = proxy
            if acct in _BROKERAGE_TYPES:
                try:
                    b = float(row.get("Cost Basis ($)", 0.0) or 0.0)
                    if b > 0:
                        basis[ticker] = b
                except (TypeError, ValueError):
                    pass

    avg_cash_rate = cash_weighted_rate / cash_total if cash_total > 0 else 0.0
    return tv, tp, ta, cash_total, avg_cash_rate, basis


def _compute_portfolio_fracs(
    ticker_values: list[tuple[str, float]],
    cash_total: float,
    account_types: dict[str, str],
    basis_by_ticker: dict[str, float],
) -> tuple[float, float, float]:
    """Return (pretax_frac, brokerage_frac, gains_frac) for the holdings."""
    total = sum(v for _, v in ticker_values) + cash_total
    if total == 0:
        return 0.0, 0.0, 0.0

    pretax_val    = sum(v for t, v in ticker_values if account_types.get(t, "Brokerage") in _PRETAX_TYPES)
    brokerage_val = sum(v for t, v in ticker_values if account_types.get(t, "Brokerage") in _BROKERAGE_TYPES)

    pretax_frac    = pretax_val    / total
    brokerage_frac = brokerage_val / total

    if brokerage_val > 0:
        brokerage_tickers = [(t, v) for t, v in ticker_values
                             if account_types.get(t, "Brokerage") in _BROKERAGE_TYPES]
        weighted_gains = sum(
            v * max(0.0, (v - basis_by_ticker.get(t, 0.0)) / v)
            for t, v in brokerage_tickers
        )
        gains_frac = weighted_gains / brokerage_val
    else:
        gains_frac = 0.0

    return pretax_frac, brokerage_frac, gains_frac


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Retirement Simulator")
    st.caption("Monte Carlo portfolio analysis")
    st.divider()

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = st.radio(
        "Simulation mode",
        ["🗃️  Historical (ticker data)", "📐  Parametric (normal dist.)"],
        help=(
            "**Historical**: downloads actual annual returns from Yahoo Finance "
            "and resamples them in consecutive blocks.\n\n"
            "**Parametric**: samples returns from normal distributions using "
            "long-run historical averages — fast, no internet needed."
        ),
    )
    historical_mode = mode.startswith("🗃️")
    st.divider()

    # ── Parametric-only: portfolio balance + allocation ────────────────────
    if not historical_mode:
        st.subheader("Portfolio")
        portfolio_balance_param = st.number_input(
            "Portfolio balance ($)",
            min_value=1.0, value=1_000_000.0, step=10_000.0, format="%f",
        )
        stocks_pct = st.slider(
            "Stock allocation", min_value=0, max_value=100,
            value=60, step=5, format="%d%%",
        )
        st.divider()

    # ── Income ────────────────────────────────────────────────────────────────
    st.subheader("Income")
    # When a JSON file is imported the SS value is stored in session_state.
    # Changing editor_key (on import) forces the widget to reset to the new value.
    ss_default = float(st.session_state.ss_override or 0.0)
    ss_widget_key = f"ss_{st.session_state.editor_key}_{int(ss_default)}"
    social_security = st.number_input(
        "Social Security / Pension ($/yr)",
        min_value=0.0, value=ss_default, step=1_000.0, format="%f",
        key=ss_widget_key,
        help="Annual SS/pension income in today's dollars — subtracted from the "
             "portfolio withdrawal each year.",
    )
    ss_delay_default = int(st.session_state.ss_delay_override or 0)
    ss_delay_widget_key = f"ss_delay_{st.session_state.editor_key}_{ss_delay_default}"
    ss_delay_years = st.number_input(
        "Years before collecting SS",
        min_value=0, max_value=40, value=ss_delay_default, step=1,
        key=ss_delay_widget_key,
        help="Number of years from retirement start until SS/pension income begins. "
             "During this period the full withdrawal comes from the portfolio.",
    )
    st.divider()

    # ── Pre-retirement accumulation ───────────────────────────────────────────
    with st.expander("Pre-retirement accumulation"):
        st.caption(
            "Model the growth of your portfolio before retirement. "
            "Savings are added each year and grow at the portfolio's blended return. "
            "No withdrawals occur until retirement begins."
        )
        ytr_default = int(st.session_state.ytr_override or 0)
        ytr_widget_key = f"ytr_{st.session_state.editor_key}_{ytr_default}"
        years_to_retirement = st.number_input(
            "Years to retirement",
            min_value=0, max_value=50, value=ytr_default, step=1,
            key=ytr_widget_key,
            help="Years until you stop working. During this period no withdrawals "
                 "are made and annual savings are added to the portfolio.",
        )
        savings_default = float(st.session_state.annual_savings_override or 0.0)
        savings_widget_key = f"savings_{st.session_state.editor_key}_{int(savings_default)}"
        annual_savings = st.number_input(
            "Annual savings ($/yr)",
            min_value=0.0, value=savings_default, step=1_000.0, format="%f",
            key=savings_widget_key,
            help="Annual contribution in today's dollars (inflation-adjusted each year).",
        )
    st.divider()

    # ── Withdrawal ────────────────────────────────────────────────────────────
    st.subheader("Withdrawal")
    withdrawal_mode = st.radio(
        "withdrawal_radio",
        ["Fixed amount", "Find safe withdrawal rate"],
        label_visibility="collapsed",
    )
    if withdrawal_mode == "Fixed amount":
        annual_withdrawal: float | None = st.number_input(
            "Annual spending goal ($/yr, after tax)",
            min_value=0.0, value=60_000.0, step=1_000.0, format="%f",
            help="Total yearly spending need in today's dollars (after tax, before SS offset). "
                 "The simulator grosses this up using your tax profile and account mix.",
        )
        target_success: float | None = None
    else:
        annual_withdrawal = None
        target_success = st.slider(
            "Target success rate", min_value=80, max_value=99,
            value=95, step=1, format="%d%%",
        ) / 100.0
    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────────
    st.subheader("Simulation")
    col_age1, col_age2 = st.columns(2)
    with col_age1:
        current_age = st.number_input(
            "Current age", min_value=0, max_value=100, value=65, step=1,
        )
    with col_age2:
        expected_lifespan = st.number_input(
            "Life expectancy", min_value=50, max_value=120, value=90, step=1,
        )
    sim_years = max(1, int(expected_lifespan) - int(current_age))
    st.caption(f"Simulation horizon: **{sim_years} years**")
    n_sims = st.select_slider(
        "Monte Carlo runs",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    st.divider()

    # ── Historical-specific settings ──────────────────────────────────────────
    if historical_mode:
        st.subheader("Historical Data")
        col_yr1, col_yr2 = st.columns(2)
        with col_yr1:
            data_start = st.number_input(
                "From year", min_value=1950, max_value=2020, value=1993, step=1,
            )
        with col_yr2:
            data_end = st.number_input(
                "To year", min_value=1960, max_value=_date.today().year,
                value=_date.today().year - 1, step=1,
            )
        chunk_size = st.slider(
            "Chunk size (years per block)", min_value=1, max_value=10, value=5,
            help="Number of consecutive historical years drawn per bootstrap block. "
                 "Larger values preserve more serial correlation across simulation years.",
        )
        inflation_opt = st.radio(
            "Inflation source",
            ["Actual CPI (FRED)", "Fixed rate"],
            help=(
                "**Actual CPI**: uses real CPI inflation from the same historical "
                "years as the return data.\n\n"
                "**Fixed rate**: applies a constant annual rate with no variance."
            ),
        )
        if inflation_opt == "Fixed rate":
            fixed_inflation_pct = st.number_input(
                "Annual inflation rate (%)",
                min_value=0.0, max_value=20.0, value=3.0, step=0.1, format="%.1f",
                help="Fixed annual inflation rate applied each year of the simulation.",
            )
            fixed_inflation_rate = fixed_inflation_pct / 100.0
        else:
            fixed_inflation_rate = 0.03  # unused in actual-CPI mode
        st.divider()

    # ── Parametric: market parameter overrides ────────────────────────────────
    if not historical_mode:
        with st.expander("Market parameters (advanced)"):
            stock_ret = st.slider(
                "Stock mean return", 0.02, 0.15, 0.10, 0.005, format="%.1f%%",
            )
            stock_std_val = st.slider(
                "Stock return std dev", 0.05, 0.35, 0.17, 0.005, format="%.1f%%",
            )
            bond_ret = st.slider(
                "Bond mean return", 0.00, 0.10, 0.04, 0.005, format="%.1f%%",
            )
            bond_std_val = st.slider(
                "Bond return std dev", 0.01, 0.20, 0.07, 0.005, format="%.1f%%",
            )
            inf_mean = st.slider(
                "Inflation mean", 0.00, 0.10, 0.03, 0.005, format="%.1f%%",
            )
            inf_std_val = st.slider(
                "Inflation std dev", 0.00, 0.05, 0.015, 0.005, format="%.1f%%",
            )
        st.divider()

    # ── Tax settings ──────────────────────────────────────────────────────────
    with st.expander("Tax settings"):
        st.caption(
            "Withdrawal amounts are entered as **after-tax spending goals**. "
            "The simulator uses 2026 federal and state tax brackets to gross up "
            "the portfolio withdrawal based on your account mix and tax profile."
        )
        filing_status = st.selectbox(
            "Filing status",
            options=list(FILING_STATUS_LABELS.keys()),
            format_func=lambda k: FILING_STATUS_LABELS[k],
            index=0,
        )
        state_options = list(STATE_NAMES.keys())
        tax_state = st.selectbox(
            "State",
            options=state_options,
            format_func=lambda k: f"{k} — {STATE_NAMES[k]}" if k != "None" else STATE_NAMES[k],
            index=state_options.index("None"),
        )
        spouse_65_plus = False
        if filing_status == "married_joint" and int(current_age) >= 65:
            spouse_65_plus = st.checkbox(
                "Spouse is also 65 or older",
                help="Adds an extra standard deduction for a qualifying spouse.",
            )
    tax_profile = TaxProfile(
        filing_status=filing_status,
        age=int(current_age),
        state=tax_state if tax_state != "None" else None,
        spouse_also_65_plus=spouse_65_plus,
    )
    st.divider()

    # ── Advanced ──────────────────────────────────────────────────────────────
    with st.expander("Advanced"):
        use_seed = st.checkbox("Fix random seed (for reproducibility)")
        rng_seed: int | None = (
            int(st.number_input("Seed value", value=42, step=1)) if use_seed else None
        )

    # ── Run button ────────────────────────────────────────────────────────────
    run_clicked = st.button(
        "🚀  Run Simulation", width="stretch", type="primary",
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📈 Retirement Portfolio Monte Carlo Simulator")

# ── Portfolio holdings section (historical mode only) ─────────────────────────
if historical_mode:
    with st.expander("📁  Portfolio Holdings", expanded=True):

        # Import controls
        imp_col, _, dl_col = st.columns([3, 0.3, 1.5])
        with imp_col:
            st.caption("Import from JSON")
            uploaded_file = st.file_uploader(
                "Upload portfolio JSON",
                type="json",
                label_visibility="collapsed",
                help="Select a portfolio JSON file to pre-fill the table below.",
            )
            if uploaded_file is not None:
                import_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if import_id != st.session_state.import_id:
                    try:
                        data = json.load(uploaded_file)
                        df_imported, ss_imported, ss_delay_imported, ytr_imported, savings_imported = _json_to_df(data)
                        st.session_state.holdings_df = df_imported.reset_index(drop=True)
                        st.session_state.editor_key += 1
                        st.session_state.import_id = import_id
                        if ss_imported is not None:
                            st.session_state.ss_override = ss_imported
                        if ss_delay_imported is not None:
                            st.session_state.ss_delay_override = ss_delay_imported
                        if ytr_imported is not None:
                            st.session_state.ytr_override = ytr_imported
                        if savings_imported is not None:
                            st.session_state.annual_savings_override = savings_imported
                        n = len(df_imported)
                        st.success(
                            f"Imported {n} holding{'s' if n != 1 else ''}."
                            + (f" SS ${ss_imported:,.0f}/yr." if ss_imported else "")
                            + (f" SS delay {ss_delay_imported} yr." if ss_delay_imported else "")
                            + (f" {ytr_imported} yr to retirement." if ytr_imported else "")
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not read portfolio file: {exc}")

        # Editable holdings table (full width)
        st.caption("Click any cell to edit · ＋ to add a row · checkbox + 🗑 to delete")
        edited_df = st.data_editor(
            st.session_state.holdings_df,
            key=f"holdings_{st.session_state.editor_key}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Ticker": st.column_config.TextColumn(
                    "Ticker",
                    help="Yahoo Finance symbol (e.g. SPY, VTI, BND). Optional for Cash accounts — leave blank or enter any label.",
                    max_chars=10,
                    required=False,
                ),
                "Value ($)": st.column_config.NumberColumn(
                    "Current Value ($)",
                    help="Dollar amount currently invested in this holding",
                    min_value=0,
                    format="$%,.0f",
                    required=True,
                ),
                "Account Type": st.column_config.SelectboxColumn(
                    "Account Type",
                    help=(
                        "Tax treatment of this holding.\n\n"
                        "• **Traditional IRA / 401k** — withdrawals taxed as ordinary income\n"
                        "• **Roth IRA / 401k / After-Tax 401k** — withdrawals tax-free\n"
                        "• **Brokerage** — capital gains tax on the gains portion"
                    ),
                    options=_ACCOUNT_TYPES,
                    required=True,
                ),
                "Cash Rate (%)": st.column_config.NumberColumn(
                    "Cash Rate (%)",
                    help="Annual interest rate for Cash accounts (e.g. 4.5 for 4.5% APY). Ignored for other account types.",
                    min_value=0.0,
                    max_value=25.0,
                    step=0.1,
                    format="%.2f%%",
                ),
                "Cost Basis ($)": st.column_config.NumberColumn(
                    "Cost Basis ($)",
                    help="Original purchase cost for Brokerage accounts — used to compute the taxable gains fraction. Ignored for other account types.",
                    min_value=0.0,
                    format="$%,.0f",
                ),
                "Proxy": st.column_config.TextColumn(
                    "Proxy Ticker",
                    help=(
                        "Optional: ticker whose returns are used to fill years "
                        "where this holding has no data.  Example: use AGG as "
                        "proxy for BND before 2003 when BND didn't exist."
                    ),
                    max_chars=10,
                ),
            },
        )
        # Persist edits to session state immediately (normalize index to avoid
        # row-tracking issues on next render with hide_index=True)
        st.session_state.holdings_df = edited_df.reset_index(drop=True)

        # Footer: totals + export
        tv_preview, _, ta_preview, cash_total_preview, _, _ = _parse_holdings(edited_df)
        total_val = sum(v for _, v in tv_preview)

        foot_l, foot_r = st.columns([3, 1.5])
        with foot_l:
            if tv_preview or cash_total_preview > 0:
                grand_total = total_val + cash_total_preview
                pct_strs = "  ·  ".join(
                    f"**{t}** {v/grand_total*100:.0f}%" for t, v in tv_preview
                )
                if cash_total_preview > 0:
                    cash_pct = cash_total_preview / grand_total * 100
                    pct_strs += f"  ·  **Cash** {cash_pct:.0f}%"
                st.caption(
                    f"{len(tv_preview)} holding(s)  ·  Total: **${grand_total:,.0f}**\n\n"
                    + pct_strs
                )
            else:
                st.caption("No valid holdings yet — add rows above.")
        with foot_r:
            json_export = _df_to_json(
                edited_df, social_security, int(ss_delay_years),
                int(years_to_retirement), float(annual_savings),
            )
            st.download_button(
                "💾  Save as JSON",
                data=json_export,
                file_name="portfolio.json",
                mime="application/json",
                width="stretch",
                disabled=len(tv_preview) == 0,
                help="Download the current holdings table as a portfolio JSON file.",
            )

        # Optional: override computed portfolio balance
        with st.expander("Override total portfolio balance"):
            use_balance_override = st.checkbox(
                f"Override (table sum is ${total_val:,.0f})",
                key="balance_override_cb",
            )
            balance_override: float | None = (
                st.number_input(
                    "Custom portfolio balance ($)",
                    min_value=1.0,
                    value=float(total_val) if total_val > 0 else 1_000_000.0,
                    step=10_000.0, format="%f",
                )
                if use_balance_override else None
            )
else:
    # Keep these defined so the run handler can reference them safely
    use_balance_override = False
    balance_override = None
    tv_preview = []
    ta_preview = {}
    cash_total_preview = 0.0

st.divider()


# ── Run simulation ────────────────────────────────────────────────────────────

if run_clicked:
    st.session_state.error_msg = None
    st.session_state.results = None

    with st.status("Running simulation…", expanded=True) as status:
        try:
            # ── Build parameters ──────────────────────────────────────────────
            if historical_mode:
                holdings_df = st.session_state.holdings_df
                ticker_values, ticker_proxies, account_types, cash_total, avg_cash_rate, basis_by_ticker = _parse_holdings(holdings_df)
                if not ticker_values:
                    raise ValueError(
                        "Add at least one holding (ticker + value) before running."
                    )

                st.write("⬇️ Loading historical market data…")
                proxies_tuple = (
                    tuple(sorted(ticker_proxies.items())) if ticker_proxies else None
                )
                historical = _fetch_historical(
                    tuple(ticker_values),
                    int(data_start),
                    int(data_end),
                    proxies_tuple,
                )

                inf_mode = "actual" if inflation_opt == "Actual CPI (FRED)" else "fixed"
                market = MarketParams(
                    inflation_mean=fixed_inflation_rate,
                    inflation_std=0.0 if inf_mode == "fixed" else 0.015,
                )
                cfg: SimConfig | None = SimConfig(
                    historical=historical,
                    chunk_size=int(chunk_size),
                    inflation_mode=inf_mode,
                )
                portfolio_balance_val = (
                    balance_override
                    if (use_balance_override and balance_override)
                    else historical.total_value() + cash_total
                )
                stock_fraction_val: float | None = None

            else:
                ticker_values = None
                account_types = {}
                basis_by_ticker = {}
                cash_total = 0.0
                avg_cash_rate = 0.0
                cfg = None
                portfolio_balance_val = portfolio_balance_param
                stock_fraction_val = stocks_pct / 100.0
                market = MarketParams(
                    stock_mean=stock_ret,   stock_std=stock_std_val,
                    bond_mean=bond_ret,     bond_std=bond_std_val,
                    inflation_mean=inf_mean, inflation_std=inf_std_val,
                )

            # ── Account groups for the per-account balance chart ──────────────
            if ticker_values or cash_total > 0:
                acct_groups: dict[str, float] = {}
                for t, v in (ticker_values or []):
                    key = account_types.get(t, "Brokerage")
                    acct_groups[key] = acct_groups.get(key, 0.0) + v
                if cash_total > 0:
                    acct_groups["Cash"] = acct_groups.get("Cash", 0.0) + cash_total
                st.session_state.account_groups = acct_groups
            else:
                st.session_state.account_groups = None

            # ── Tax gross-up ──────────────────────────────────────────────────
            # Convert after-tax spending goal → pre-tax portfolio withdrawal
            net_spend = annual_withdrawal  # may be None if finding SWR
            if net_spend is not None and (ticker_values or cash_total > 0):
                pretax_frac, brokerage_frac, gains_frac = _compute_portfolio_fracs(
                    ticker_values or [], cash_total, account_types, basis_by_ticker,
                )
                gross_annual_withdrawal = gross_up_withdrawal(
                    net_spending=net_spend,
                    pretax_frac=pretax_frac,
                    brokerage_frac=brokerage_frac,
                    gains_frac=gains_frac,
                    ss_annual=float(social_security),
                    profile=tax_profile,
                )
            else:
                gross_annual_withdrawal = net_spend

            # Compute cash fraction relative to total portfolio (market + cash)
            total_portfolio = portfolio_balance_val
            cash_frac_param = cash_total / total_portfolio if total_portfolio > 0 and cash_total > 0 else 0.0

            base_params = PortfolioParams(
                initial_balance=portfolio_balance_val,
                annual_withdrawal=gross_annual_withdrawal or 0.0,
                stock_fraction=stock_fraction_val,
                tickers=ticker_values,
                social_security=social_security,
                years=int(sim_years),
                num_simulations=int(n_sims),
                market=market,
                cash_fraction=cash_frac_param,
                cash_rate=avg_cash_rate,
                ss_delay_years=int(ss_delay_years),
                years_to_retirement=int(years_to_retirement),
                annual_savings=float(annual_savings),
            )

            # ── Safe withdrawal rate search (if requested) ────────────────────
            resolved_target = target_success
            safe_w: float | None = None

            if target_success is not None:
                st.write(
                    f"🔍 Searching for safe withdrawal at "
                    f"{target_success * 100:.0f}% success…"
                )
                safe_w = find_safe_withdrawal_rate(
                    base_params, target_success=target_success,
                    seed=rng_seed, cfg=cfg,
                )
                withdrawal_val = safe_w
            else:
                withdrawal_val = gross_annual_withdrawal

            params = replace(base_params, annual_withdrawal=withdrawal_val)

            # ── Main simulation ───────────────────────────────────────────────
            st.write(f"⚙️ Running {n_sims:,} Monte Carlo simulations…")
            results = _run(params, cfg, seed=rng_seed)

            # If a fixed withdrawal was given, also find the 95% SWR for context
            if gross_annual_withdrawal is not None and target_success is None:
                resolved_target = 0.95
                st.write("🔍 Computing safe withdrawal rate at 95% success…")
                safe_w = find_safe_withdrawal_rate(
                    params, target_success=resolved_target,
                    seed=rng_seed, cfg=cfg,
                )

            # ── Withdrawal sweep for the success-rate chart ───────────────────
            st.write("📊 Computing success-rate sweep…")
            sweep = sweep_withdrawal_rates(params, seed=rng_seed, cfg=cfg)

            # ── Persist to session state ──────────────────────────────────────
            st.session_state.results       = results
            st.session_state.sweep         = sweep
            st.session_state.safe_withdrawal = safe_w
            st.session_state.target_success  = resolved_target
            st.session_state.cfg           = cfg
            st.session_state.tax_profile   = tax_profile
            st.session_state.portfolio_fracs = (
                (pretax_frac, brokerage_frac, gains_frac)
                if (ticker_values or cash_total > 0) else None
            )

            status.update(label="✅ Simulation complete", state="complete")

        except Exception as exc:
            st.session_state.error_msg = str(exc)
            status.update(label="❌ Simulation failed", state="error")

    st.rerun()


# ── Error banner ──────────────────────────────────────────────────────────────

if st.session_state.error_msg:
    st.error(f"**Error:** {st.session_state.error_msg}")


# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.results is not None:
    results  = st.session_state.results
    sweep    = st.session_state.sweep
    safe_w   = st.session_state.safe_withdrawal
    t_succ   = st.session_state.target_success
    summary  = depletion_summary(results)
    p        = results.params

    # ── After-tax SWR estimate ─────────────────────────────────────────────────
    # Compute the after-tax spendable equivalent of the gross safe withdrawal.
    _stored_profile = st.session_state.tax_profile
    _stored_fracs   = st.session_state.portfolio_fracs

    def _net_of_tax(gross_w: float) -> float:
        """Estimate after-tax spendable from a gross portfolio withdrawal."""
        if _stored_profile is None or _stored_fracs is None:
            return gross_w
        pf, bf, gf = _stored_fracs
        net_draw  = max(0.0, gross_w - float(p.social_security))
        ordinary  = pf * net_draw
        ltcg      = gf * bf * net_draw
        taxes     = compute_total_tax(ordinary, ltcg, float(p.social_security), _stored_profile)
        return gross_w - taxes

    net_safe_w = _net_of_tax(safe_w) if safe_w is not None else None

    # ── Summary metric cards ──────────────────────────────────────────────────
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if p.years_to_retirement > 0:
            ret_start = float(np.median(results.balances[:, 0]))
            st.metric(
                "Balance at Retirement (median)",
                f"${ret_start:,.0f}",
                delta=f"from ${p.initial_balance:,.0f} over {p.years_to_retirement} yr",
                delta_color="off",
                help=f"Median portfolio at start of retirement after {p.years_to_retirement} "
                     f"year(s) of accumulation with ${p.annual_savings:,.0f}/yr savings.",
            )
        else:
            st.metric(
                "Starting Balance",
                f"${p.initial_balance:,.0f}",
                help=f"Net portfolio draw: ${p.net_withdrawal:,.0f}/yr after "
                     f"${p.social_security:,.0f}/yr SS/pension offset.",
            )
    with m2:
        rate_pct = summary["success_rate"] * 100
        survivors = summary["num_simulations"] - summary["num_depleted"]
        st.metric(
            "Success Rate",
            f"{rate_pct:.1f}%",
            delta=f"{survivors:,} / {summary['num_simulations']:,} survived {p.years} yrs",
            delta_color="off",
        )
    with m3:
        if safe_w is not None:
            label = (
                f"Safe Withdrawal ({t_succ * 100:.0f}%)"
                if t_succ is not None else "Safe Withdrawal"
            )
            swr_pct = safe_w / p.initial_balance * 100
            if net_safe_w is not None and _stored_profile is not None:
                st.metric(
                    label,
                    f"${net_safe_w:,.0f}/yr after tax",
                    delta=f"${safe_w:,.0f}/yr gross  ·  {swr_pct:.2f}% SWR",
                    delta_color="off",
                    help="After-tax spendable amount estimated using your tax profile. "
                         "Gross = pre-tax portfolio withdrawal (+ SS) before income taxes.",
                )
            else:
                st.metric(
                    label,
                    f"${safe_w:,.0f}/yr",
                    delta=f"{swr_pct:.2f}% SWR",
                    delta_color="off",
                )
        else:
            st.metric("Annual Withdrawal", f"${p.annual_withdrawal:,.0f}/yr")
    with m4:
        dep_pct = summary["num_depleted"] / summary["num_simulations"] * 100
        st.metric(
            "Portfolios Depleted",
            f"{summary['num_depleted']:,}",
            delta=f"{dep_pct:.1f}% of runs",
            delta_color="inverse",
        )

    if summary["num_depleted"] > 0:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric(
                "Avg Depletion Year",
                f"Year {summary['depletion_year_mean']:.1f}",
            )
        with d2:
            st.metric(
                "Median Depletion Year",
                f"Year {summary['depletion_year_median']:.1f}",
            )
        with d3:
            st.metric(
                "10th–90th Pct Depletion",
                f"Yr {summary['depletion_year_p10']:.0f} – "
                f"Yr {summary['depletion_year_p90']:.0f}",
            )

    st.divider()

    # ── Charts in tabs ────────────────────────────────────────────────────────
    account_groups = st.session_state.account_groups
    tab_labels = [
        "📉  Balance Percentiles",
        "✅  Success Rate vs. Withdrawal",
        "📊  Depletion Year Distribution",
    ]
    if account_groups:
        tab_labels.insert(1, "🗂️  Account Balances")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    with tabs[tab_idx]:
        fig = plot_balance_percentiles(results)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    tab_idx += 1

    if account_groups:
        with tabs[tab_idx]:
            fig = plot_account_balances(results, account_groups)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        tab_idx += 1

    with tabs[tab_idx]:
        fig = plot_success_rates(
            results,
            sweep=sweep,
            current_withdrawal=p.annual_withdrawal,
            safe_withdrawal=safe_w,
            target_success=t_succ,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    tab_idx += 1

    with tabs[tab_idx]:
        fig = plot_depletion_histogram(results)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

else:
    # ── Welcome screen ────────────────────────────────────────────────────────
    if not st.session_state.error_msg:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🗃️  Historical Mode")
            st.markdown("""
**Uses actual market returns** downloaded from Yahoo Finance.

- Add ticker symbols and dollar values to the holdings table above
- Set an optional **proxy ticker** for funds with limited history
  *(e.g. use `AGG` as proxy for `BND` before 2003)*
- **Import** an existing portfolio JSON or **export** after editing
- Social Security or pension income from the JSON file is auto-filled
- Chooses between real CPI inflation or a fixed rate
- Draws years in consecutive blocks to preserve real market cycles
            """)
        with col_b:
            st.subheader("📐  Parametric Mode")
            st.markdown("""
**No internet connection needed.** Uses long-run historical averages.

- Set total portfolio value and stock / bond split
- Adjust expected returns, volatility, and inflation in the sidebar
- Returns and inflation are sampled independently each year from
  normal distributions — fast and simple
- Good for quick sensitivity analysis with custom assumptions
            """)

        st.info(
            "👈  Configure your portfolio and settings in the sidebar, "
            "then click **🚀  Run Simulation**.",
        )

        # Example JSON format card
        with st.expander("📄  Portfolio JSON file format"):
            st.markdown(
                "The portfolio JSON file is the same format used by `main.py --portfolio-file`."
            )
            st.code(
                """{
  "holdings": [
    { "ticker": "VTI",  "value": 500000, "account_type": "Traditional IRA",  "proxy": "VTSMX" },
    { "ticker": "VXUS", "value": 200000, "account_type": "Roth IRA" },
    { "ticker": "BND",  "value": 300000, "account_type": "Brokerage",        "proxy": "AGG"   }
  ],
  "social_security": 24000
}""",
                language="json",
            )
            st.markdown("""
| Field | Required | Description |
|---|---|---|
| `holdings[].ticker` | ✅ | Yahoo Finance symbol |
| `holdings[].value`  | ✅ | Current dollar value of this holding |
| `holdings[].account_type` | ❌ | `Traditional IRA`, `Traditional 401k`, `Roth IRA`, `Roth 401k`, `After-Tax 401k`, `Brokerage` (default), or `Cash` |
| `holdings[].cash_rate` | ❌ | Annual interest rate for `Cash` accounts (e.g. `4.5` for 4.5%) |
| `holdings[].cost_basis` | ❌ | Original purchase cost in dollars for `Brokerage` accounts |
| `holdings[].proxy`  | ❌ | Ticker used to fill years with no data (not applicable to `Cash`) |
| `social_security`   | ❌ | Annual SS/pension income (pre-fills the sidebar) |
""")
