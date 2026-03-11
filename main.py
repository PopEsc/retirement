#!/usr/bin/env python3
"""
Retirement Portfolio Monte Carlo Simulator
==========================================

PARAMETRIC MODE (default)
--------------------------
Uses long-run historical normal distributions for returns.  No internet
connection required.

    python main.py --portfolio 1000000 --stocks 0.60 --withdrawal 60000 \\
                   --social-security 24000

    python main.py --portfolio 1000000 --stocks 0.60 --social-security 24000 \\
                   --target-success 0.95


HISTORICAL MODE (--tickers or --portfolio-file)
------------------------------------------------
Downloads actual annual returns from Yahoo Finance and draws simulation
years in consecutive blocks to preserve real market cycles.

Inline tickers:

    python main.py --tickers SPY:600000 BND:400000 \\
                   --withdrawal 60000 --social-security 24000

From a JSON portfolio file:

    python main.py --portfolio-file portfolio.json --target-success 0.95

With a global proxy for tickers that lack early history:

    python main.py --tickers FSKAX:700000 FTBFX:300000 \\
                   --proxy SPY --data-start 1993 --withdrawal 55000


COMMON OPTIONS
--------------
    --save-charts ./output    Save PNGs instead of interactive display
    --no-charts               Skip chart generation
    --seed 42                 Fix random seed for reproducibility
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from retirement_sim.simulation import MarketParams, PortfolioParams
from retirement_sim.analysis import (
    SimConfig,
    find_safe_withdrawal_rate,
    print_summary,
    sweep_withdrawal_rates,
    _run,
)
from retirement_sim.charts import show_all_charts

_CURRENT_YEAR = date.today().year


# ---------------------------------------------------------------------------
# Portfolio JSON loading
# ---------------------------------------------------------------------------

def load_portfolio_file(path: str) -> tuple[
    list[tuple[str, float]],  # ticker_values
    dict[str, str],           # ticker_proxies  (may be empty)
    float | None,             # social_security (if present in file)
]:
    """
    Load portfolio holdings from a JSON file.

    Expected format::

        {
          "holdings": [
            {"ticker": "VTI",  "value": 500000, "proxy": "VTSMX"},
            {"ticker": "VXUS", "value": 200000},
            {"ticker": "BND",  "value": 300000, "proxy": "AGG"}
          ],
          "social_security": 24000
        }

    The ``proxy`` field on each holding is optional.  ``social_security``
    at the top level is also optional and can be overridden by --social-security.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading portfolio file '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON in '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if "holdings" not in data or not isinstance(data["holdings"], list):
        print(
            f"Error: '{path}' must contain a top-level \"holdings\" array.",
            file=sys.stderr,
        )
        sys.exit(1)

    ticker_values: list[tuple[str, float]] = []
    ticker_proxies: dict[str, str] = {}

    for i, entry in enumerate(data["holdings"]):
        # ticker
        if "ticker" not in entry:
            print(f"Error: holdings[{i}] is missing \"ticker\".", file=sys.stderr)
            sys.exit(1)
        ticker = str(entry["ticker"]).upper().strip()

        # value
        if "value" not in entry:
            print(f"Error: holdings[{i}] ({ticker}) is missing \"value\".", file=sys.stderr)
            sys.exit(1)
        try:
            value = float(entry["value"])
        except (TypeError, ValueError):
            print(
                f"Error: holdings[{i}] ({ticker}) has non-numeric value '{entry['value']}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        if value <= 0:
            print(f"Error: value for {ticker} must be positive.", file=sys.stderr)
            sys.exit(1)

        ticker_values.append((ticker, value))

        # optional per-ticker proxy
        if "proxy" in entry and entry["proxy"]:
            ticker_proxies[ticker] = str(entry["proxy"]).upper().strip()

    if not ticker_values:
        print(f"Error: '{path}' contains no holdings.", file=sys.stderr)
        sys.exit(1)

    social_security: float | None = None
    if "social_security" in data:
        try:
            social_security = float(data["social_security"])
        except (TypeError, ValueError):
            print(
                "Warning: 'social_security' in portfolio file is not a number; ignoring.",
                file=sys.stderr,
            )

    return ticker_values, ticker_proxies, social_security


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo retirement portfolio simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Portfolio inputs ---------------------------------------------------
    port_group = parser.add_argument_group("Portfolio")
    port_group.add_argument(
        "--portfolio", "-p", type=float, default=None, metavar="DOLLARS",
        help=(
            "Starting portfolio balance in dollars. "
            "Optional in historical mode (defaults to sum of holding values)."
        ),
    )
    port_group.add_argument(
        "--stocks", "-s", type=float, default=0.60, metavar="FRACTION",
        help="Fraction in stocks, 0.0–1.0 (parametric mode only, default: 0.60)",
    )

    holdings_group = port_group.add_mutually_exclusive_group()
    holdings_group.add_argument(
        "--tickers", nargs="+", metavar="TICKER:VALUE",
        help=(
            "Holdings as TICKER:VALUE pairs, e.g. SPY:600000 BND:400000. "
            "Enables historical block-bootstrap simulation."
        ),
    )
    holdings_group.add_argument(
        "--portfolio-file", metavar="PATH",
        help=(
            "Path to a JSON file describing your holdings. "
            "See README.md for the file format. "
            "Enables historical block-bootstrap simulation."
        ),
    )

    port_group.add_argument(
        "--social-security", "--ss", type=float, default=None, metavar="DOLLARS",
        help=(
            "Annual Social Security / pension in today's dollars. "
            "Overrides the value in a portfolio JSON file."
        ),
    )
    port_group.add_argument(
        "--proxy", metavar="TICKER",
        help=(
            "Global fallback proxy ticker for any holding that lacks data in "
            "the requested date range (e.g. ^GSPC, SPY). "
            "Per-holding proxies in the JSON file take precedence."
        ),
    )

    # ---- Withdrawal --------------------------------------------------------
    w_group = parser.add_mutually_exclusive_group(required=True)
    w_group.add_argument(
        "--withdrawal", "-w", type=float, metavar="DOLLARS",
        help="Annual gross withdrawal to evaluate (today's dollars)",
    )
    w_group.add_argument(
        "--target-success", type=float, metavar="FRACTION",
        help="Find the max withdrawal achieving this success rate (e.g. 0.95)",
    )

    # ---- Historical data options -------------------------------------------
    hist_group = parser.add_argument_group("Historical data (historical mode)")
    hist_group.add_argument(
        "--data-start", type=int, default=1993, metavar="YEAR",
        help="First year of historical data to download (default: 1993)",
    )
    hist_group.add_argument(
        "--data-end", type=int, default=_CURRENT_YEAR - 1, metavar="YEAR",
        help=f"Last year of historical data (default: {_CURRENT_YEAR - 1})",
    )
    hist_group.add_argument(
        "--chunk-size", type=int, default=5, metavar="N",
        help=(
            "Years per consecutive block in the bootstrap (default: 5). "
            "Larger values preserve more serial correlation."
        ),
    )
    hist_group.add_argument(
        "--inflation-mode", choices=["actual", "fixed"], default=None,
        help=(
            "actual: use CPI from the same sampled historical years. "
            "fixed: use --inflation rate with no year-to-year variance. "
            "Default: actual in historical mode, fixed in parametric mode."
        ),
    )

    # ---- Simulation parameters ---------------------------------------------
    sim_group = parser.add_argument_group("Simulation")
    sim_group.add_argument(
        "--years", "-y", type=int, default=30, metavar="N",
        help="Years to simulate (default: 30)",
    )
    sim_group.add_argument(
        "--simulations", "-n", type=int, default=10_000, metavar="N",
        help="Number of Monte Carlo runs (default: 10000)",
    )

    # ---- Market parameter overrides (parametric mode) ----------------------
    mkt_group = parser.add_argument_group("Market parameters (parametric mode)")
    mkt_group.add_argument("--stock-return", type=float, default=0.10,
                           help="Expected nominal stock return (default: 0.10)")
    mkt_group.add_argument("--stock-std",    type=float, default=0.17,
                           help="Stock return std dev (default: 0.17)")
    mkt_group.add_argument("--bond-return",  type=float, default=0.04,
                           help="Expected nominal bond return (default: 0.04)")
    mkt_group.add_argument("--bond-std",     type=float, default=0.07,
                           help="Bond return std dev (default: 0.07)")
    mkt_group.add_argument("--inflation",    type=float, default=0.03,
                           help="Expected annual inflation (default: 0.03)")
    mkt_group.add_argument("--inflation-std", type=float, default=0.015,
                           help="Inflation std dev (default: 0.015)")

    # ---- Output ------------------------------------------------------------
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--save-charts", metavar="DIR",
        help="Save chart PNGs to DIR instead of displaying interactively",
    )
    out_group.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation entirely",
    )
    out_group.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inline ticker parsing
# ---------------------------------------------------------------------------

def parse_tickers(raw: list[str]) -> list[tuple[str, float]]:
    """Parse ['SPY:600000', 'BND:400000'] → [('SPY', 600000.0), ('BND', 400000.0)]."""
    result = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != 2:
            print(f"Error: ticker '{item}' must be in TICKER:VALUE format.", file=sys.stderr)
            sys.exit(1)
        ticker, raw_value = parts[0].upper().strip(), parts[1].strip()
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            print(f"Error: value '{raw_value}' for {ticker} is not a number.", file=sys.stderr)
            sys.exit(1)
        if value <= 0:
            print(f"Error: value for {ticker} must be positive.", file=sys.stderr)
            sys.exit(1)
        result.append((ticker, value))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Build MarketParams ------------------------------------------------
    market = MarketParams(
        stock_mean=args.stock_return,
        stock_std=args.stock_std,
        bond_mean=args.bond_return,
        bond_std=args.bond_std,
        inflation_mean=args.inflation,
        inflation_std=args.inflation_std,
    )

    # ---- Resolve holdings --------------------------------------------------
    historical = None
    ticker_values = None
    cfg: SimConfig | None = None
    ss_from_file: float | None = None

    using_historical = args.tickers or args.portfolio_file

    if using_historical:
        if args.tickers:
            ticker_values = parse_tickers(args.tickers)
            ticker_proxies: dict[str, str] = {}
        else:
            ticker_values, ticker_proxies, ss_from_file = load_portfolio_file(
                args.portfolio_file
            )

        # Apply global --proxy as fallback for any ticker without a per-ticker proxy
        if args.proxy:
            global_proxy = args.proxy.upper().strip()
            for ticker, _ in ticker_values:
                if ticker not in ticker_proxies:
                    ticker_proxies[ticker] = global_proxy

        inflation_mode = args.inflation_mode or "actual"

        from retirement_sim.market_data import build_historical_dataset
        historical = build_historical_dataset(
            ticker_values,
            start_year=args.data_start,
            end_year=args.data_end,
            ticker_proxies=ticker_proxies or None,
        )
        cfg = SimConfig(
            historical=historical,
            chunk_size=args.chunk_size,
            inflation_mode=inflation_mode,
        )
        portfolio_balance = args.portfolio or historical.total_value()

    else:
        # Parametric mode
        if args.portfolio is None:
            print("Error: --portfolio is required when --tickers/--portfolio-file is not used.",
                  file=sys.stderr)
            sys.exit(1)
        if not (0.0 <= args.stocks <= 1.0):
            print("Error: --stocks must be between 0.0 and 1.0", file=sys.stderr)
            sys.exit(1)
        portfolio_balance = args.portfolio
        if args.inflation_mode == "actual":
            print(
                "Warning: --inflation-mode actual requires historical mode. "
                "Falling back to fixed inflation.",
                file=sys.stderr,
            )

    # --social-security on the CLI overrides any value from the JSON file
    social_security = (
        args.social_security
        if args.social_security is not None
        else (ss_from_file or 0.0)
    )

    if args.target_success is not None and not (0.0 < args.target_success < 1.0):
        print("Error: --target-success must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # ---- Build base PortfolioParams ----------------------------------------
    base_params = PortfolioParams(
        initial_balance=portfolio_balance,
        annual_withdrawal=args.withdrawal or 0.0,
        stock_fraction=args.stocks if not using_historical else None,
        tickers=ticker_values,
        social_security=social_security,
        years=args.years,
        num_simulations=args.simulations,
        market=market,
    )

    # ---- Find safe withdrawal rate (if requested) --------------------------
    safe_withdrawal = None
    target_success = args.target_success

    if target_success is not None:
        print(f"Searching for safe withdrawal at {target_success*100:.0f}% success...")
        safe_withdrawal = find_safe_withdrawal_rate(
            base_params,
            target_success=target_success,
            seed=args.seed,
            cfg=cfg,
        )
        withdrawal = safe_withdrawal
    else:
        withdrawal = args.withdrawal

    # ---- Run main simulation -----------------------------------------------
    params = replace(base_params, annual_withdrawal=withdrawal)

    print(f"Running {args.simulations:,} simulations over {args.years} years...")
    results = _run(params, cfg, seed=args.seed)

    # ---- If evaluating a specific withdrawal, also compute SWR at 95% ------
    if args.withdrawal is not None and target_success is None:
        target_success = 0.95
        print(f"Computing safe withdrawal at {target_success*100:.0f}% success...")
        safe_withdrawal = find_safe_withdrawal_rate(
            params, target_success=target_success, seed=args.seed, cfg=cfg
        )

    print_summary(results, safe_withdrawal=safe_withdrawal,
                  target_success=target_success, cfg=cfg)

    if args.no_charts:
        return

    print("Computing success rates across withdrawal range...")
    sweep = sweep_withdrawal_rates(params, seed=args.seed, cfg=cfg)

    show_all_charts(
        results,
        sweep=sweep,
        safe_withdrawal=safe_withdrawal,
        target_success=target_success,
        save_dir=args.save_charts,
    )


if __name__ == "__main__":
    main()
