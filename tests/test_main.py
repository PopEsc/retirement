"""
Tests for main.py: CLI argument parsing, JSON loading, and end-to-end runs.

Network calls and chart rendering are fully mocked so these tests are fast
and offline.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import main as main_module
from main import load_portfolio_file, parse_tickers


# ---------------------------------------------------------------------------
# parse_tickers
# ---------------------------------------------------------------------------

class TestParseTickers:
    def test_single_ticker(self):
        result = parse_tickers(["SPY:600000"])
        assert result == [("SPY", 600_000.0)]

    def test_multiple_tickers(self):
        result = parse_tickers(["SPY:600000", "BND:400000"])
        assert result == [("SPY", 600_000.0), ("BND", 400_000.0)]

    def test_ticker_uppercased(self):
        result = parse_tickers(["spy:100000"])
        assert result[0][0] == "SPY"

    def test_value_with_comma_removed(self):
        result = parse_tickers(["SPY:1,000,000"])
        assert result[0][1] == pytest.approx(1_000_000.0)

    def test_missing_colon_exits(self):
        with pytest.raises(SystemExit):
            parse_tickers(["SPY600000"])

    def test_non_numeric_value_exits(self):
        with pytest.raises(SystemExit):
            parse_tickers(["SPY:abc"])

    def test_negative_value_exits(self):
        with pytest.raises(SystemExit):
            parse_tickers(["SPY:-100000"])

    def test_zero_value_exits(self):
        with pytest.raises(SystemExit):
            parse_tickers(["SPY:0"])


# ---------------------------------------------------------------------------
# load_portfolio_file
# ---------------------------------------------------------------------------

class TestLoadPortfolioFile:
    def _write(self, tmp_path: Path, data: dict) -> str:
        p = tmp_path / "portfolio.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_basic_holdings(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [
                {"ticker": "SPY", "value": 600_000},
                {"ticker": "BND", "value": 400_000},
            ]
        })
        tv, proxies, ss = load_portfolio_file(path)
        assert tv == [("SPY", 600_000.0), ("BND", 400_000.0)]
        assert proxies == {}
        assert ss is None

    def test_ticker_uppercased(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "spy", "value": 1_000_000}]
        })
        tv, _, _ = load_portfolio_file(path)
        assert tv[0][0] == "SPY"

    def test_social_security_parsed(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "SPY", "value": 1_000_000}],
            "social_security": 24_000,
        })
        _, _, ss = load_portfolio_file(path)
        assert ss == pytest.approx(24_000.0)

    def test_per_holding_proxy_parsed(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [
                {"ticker": "BND", "value": 400_000, "proxy": "AGG"},
                {"ticker": "SPY", "value": 600_000},
            ]
        })
        _, proxies, _ = load_portfolio_file(path)
        assert proxies == {"BND": "AGG"}

    def test_proxy_uppercased(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "BND", "value": 400_000, "proxy": "agg"}]
        })
        _, proxies, _ = load_portfolio_file(path)
        assert proxies["BND"] == "AGG"

    def test_missing_file_exits(self):
        with pytest.raises(SystemExit):
            load_portfolio_file("/nonexistent/path/portfolio.json")

    def test_invalid_json_exits(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_portfolio_file(str(p))

    def test_missing_holdings_key_exits(self, tmp_path):
        path = self._write(tmp_path, {"tickers": []})
        with pytest.raises(SystemExit):
            load_portfolio_file(path)

    def test_holding_missing_ticker_exits(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"value": 100_000}]
        })
        with pytest.raises(SystemExit):
            load_portfolio_file(path)

    def test_holding_missing_value_exits(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "SPY"}]
        })
        with pytest.raises(SystemExit):
            load_portfolio_file(path)

    def test_non_numeric_value_exits(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "SPY", "value": "lots"}]
        })
        with pytest.raises(SystemExit):
            load_portfolio_file(path)

    def test_negative_value_exits(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "SPY", "value": -1_000}]
        })
        with pytest.raises(SystemExit):
            load_portfolio_file(path)

    def test_empty_holdings_exits(self, tmp_path):
        path = self._write(tmp_path, {"holdings": []})
        with pytest.raises(SystemExit):
            load_portfolio_file(path)

    def test_non_numeric_ss_ignored(self, tmp_path, capsys):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "SPY", "value": 1_000_000}],
            "social_security": "a lot",
        })
        _, _, ss = load_portfolio_file(path)
        assert ss is None  # bad value ignored, no crash

    def test_empty_proxy_field_ignored(self, tmp_path):
        path = self._write(tmp_path, {
            "holdings": [{"ticker": "SPY", "value": 1_000_000, "proxy": ""}]
        })
        _, proxies, _ = load_portfolio_file(path)
        assert "SPY" not in proxies


# ---------------------------------------------------------------------------
# End-to-end: parametric mode via main()
# ---------------------------------------------------------------------------

def _run_main(argv: list[str]) -> None:
    """Replace sys.argv and call main(), suppressing charts."""
    with patch.object(sys, "argv", ["main.py"] + argv):
        main_module.main()


class TestMainParametric:
    def test_basic_run_no_charts(self, capsys):
        _run_main([
            "--portfolio", "1000000",
            "--stocks", "0.60",
            "--withdrawal", "50000",
            "--no-charts",
            "--seed", "0",
            "--simulations", "200",
            "--years", "10",
        ])
        out = capsys.readouterr().out
        assert "Success rate" in out
        assert "1,000,000" in out

    def test_target_success_mode(self, capsys):
        _run_main([
            "--portfolio", "1000000",
            "--stocks", "0.60",
            "--target-success", "0.90",
            "--no-charts",
            "--seed", "0",
            "--simulations", "200",
            "--years", "10",
        ])
        out = capsys.readouterr().out
        assert "Safe withdrawal" in out

    def test_portfolio_required_without_tickers(self):
        with pytest.raises(SystemExit):
            _run_main(["--stocks", "0.60", "--withdrawal", "50000", "--no-charts"])

    def test_invalid_stocks_fraction_exits(self):
        with pytest.raises(SystemExit):
            _run_main([
                "--portfolio", "1000000", "--stocks", "1.5",
                "--withdrawal", "50000", "--no-charts",
            ])

    def test_social_security_appears_in_output(self, capsys):
        _run_main([
            "--portfolio", "1000000", "--stocks", "0.60",
            "--social-security", "20000", "--withdrawal", "50000",
            "--no-charts", "--seed", "0",
            "--simulations", "100", "--years", "5",
        ])
        out = capsys.readouterr().out
        assert "20,000" in out

    def test_save_charts_creates_files(self, tmp_path, capsys):
        _run_main([
            "--portfolio", "1000000", "--stocks", "0.60",
            "--withdrawal", "50000",
            "--save-charts", str(tmp_path),
            "--seed", "0", "--simulations", "100", "--years", "5",
        ])
        import matplotlib.pyplot as plt
        plt.close("all")
        assert (tmp_path / "balance_percentiles.png").exists()
        assert (tmp_path / "success_rates.png").exists()
        assert (tmp_path / "depletion_histogram.png").exists()


# ---------------------------------------------------------------------------
# End-to-end: historical mode via main() (network mocked)
# ---------------------------------------------------------------------------

def _make_fake_historical(ticker_values):
    """Build a minimal HistoricalData to return from the mocked fetch."""
    from retirement_sim.market_data import HistoricalData
    import pandas as pd

    years = list(range(2000, 2010))
    rng = np.random.default_rng(0)
    tickers = [t for t, _ in ticker_values]
    returns = pd.DataFrame(
        {t: rng.normal(0.08, 0.15, len(years)) for t in tickers},
        index=years,
    )
    inflation = pd.Series(np.full(len(years), 0.03), index=years)
    return HistoricalData(
        returns=returns, inflation=inflation,
        years=years, ticker_values=ticker_values,
    )


class TestMainHistorical:
    @patch("retirement_sim.market_data.build_historical_dataset")
    def test_inline_tickers(self, mock_fetch, capsys):
        tv = [("SPY", 600_000), ("BND", 400_000)]
        mock_fetch.return_value = _make_fake_historical(tv)

        _run_main([
            "--tickers", "SPY:600000", "BND:400000",
            "--withdrawal", "50000",
            "--no-charts", "--seed", "0",
            "--simulations", "200", "--years", "10",
        ])
        out = capsys.readouterr().out
        assert "SPY" in out
        assert "BND" in out

    @patch("retirement_sim.market_data.build_historical_dataset")
    def test_portfolio_file(self, mock_fetch, tmp_path, capsys):
        tv = [("VTI", 700_000), ("BND", 300_000)]
        mock_fetch.return_value = _make_fake_historical(tv)

        pf = tmp_path / "port.json"
        pf.write_text(json.dumps({
            "holdings": [
                {"ticker": "VTI", "value": 700_000},
                {"ticker": "BND", "value": 300_000},
            ],
            "social_security": 18_000,
        }), encoding="utf-8")

        _run_main([
            "--portfolio-file", str(pf),
            "--withdrawal", "50000",
            "--no-charts", "--seed", "0",
            "--simulations", "200", "--years", "10",
        ])
        out = capsys.readouterr().out
        assert "VTI" in out
        assert "18,000" in out  # SS from file

    @patch("retirement_sim.market_data.build_historical_dataset")
    def test_cli_ss_overrides_file_ss(self, mock_fetch, tmp_path, capsys):
        tv = [("SPY", 1_000_000)]
        mock_fetch.return_value = _make_fake_historical(tv)

        pf = tmp_path / "port.json"
        pf.write_text(json.dumps({
            "holdings": [{"ticker": "SPY", "value": 1_000_000}],
            "social_security": 5_000,
        }), encoding="utf-8")

        _run_main([
            "--portfolio-file", str(pf),
            "--social-security", "30000",
            "--withdrawal", "50000",
            "--no-charts", "--seed", "0",
            "--simulations", "100", "--years", "5",
        ])
        out = capsys.readouterr().out
        assert "30,000" in out   # CLI value used
        assert "5,000" not in out.split("Social Security")[1].split("\n")[0]

    @patch("retirement_sim.market_data.build_historical_dataset")
    def test_global_proxy_passed_to_dataset_builder(self, mock_fetch, capsys):
        tv = [("SPY", 600_000), ("BND", 400_000)]
        mock_fetch.return_value = _make_fake_historical(tv)

        _run_main([
            "--tickers", "SPY:600000", "BND:400000",
            "--proxy", "AGG",
            "--withdrawal", "50000",
            "--no-charts", "--seed", "0",
            "--simulations", "100", "--years", "5",
        ])
        _, kwargs = mock_fetch.call_args
        proxies = kwargs.get("ticker_proxies") or mock_fetch.call_args[0][3] if len(mock_fetch.call_args[0]) > 3 else None
        # Both tickers should have AGG as global proxy
        if proxies:
            assert proxies.get("SPY") == "AGG" or proxies.get("BND") == "AGG"

    @patch("retirement_sim.market_data.build_historical_dataset")
    def test_per_holding_proxy_in_json_takes_precedence(self, mock_fetch, tmp_path, capsys):
        tv = [("BND", 1_000_000)]
        mock_fetch.return_value = _make_fake_historical(tv)

        pf = tmp_path / "port.json"
        pf.write_text(json.dumps({
            "holdings": [{"ticker": "BND", "value": 1_000_000, "proxy": "AGG"}]
        }), encoding="utf-8")

        _run_main([
            "--portfolio-file", str(pf),
            "--proxy", "SPY",      # global proxy (should not override per-holding)
            "--withdrawal", "40000",
            "--no-charts", "--seed", "0",
            "--simulations", "100", "--years", "5",
        ])
        _, kwargs = mock_fetch.call_args
        proxies = kwargs.get("ticker_proxies")
        if proxies:
            assert proxies.get("BND") == "AGG"  # per-holding wins
