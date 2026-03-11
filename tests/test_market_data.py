"""Tests for retirement_sim/market_data.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from retirement_sim.market_data import (
    HistoricalData,
    _fetch_cpi_annual,
    _fetch_ticker_annual_returns,
    _fetch_ticker_with_proxy,
    build_historical_dataset,
)
from tests.conftest import make_cpi_csv, make_price_dataframe


# ---------------------------------------------------------------------------
# HistoricalData
# ---------------------------------------------------------------------------

class TestHistoricalData:
    def _make(self, ticker_values=None):
        years = list(range(2000, 2010))
        rng = np.random.default_rng(0)
        tv = ticker_values or [("SPY", 600_000), ("BND", 400_000)]
        tickers = [t for t, _ in tv]
        returns = pd.DataFrame(
            {t: rng.normal(0.08, 0.15, len(years)) for t in tickers},
            index=years,
        )
        inflation = pd.Series(rng.normal(0.03, 0.01, len(years)), index=years)
        return HistoricalData(
            returns=returns, inflation=inflation,
            years=years, ticker_values=tv,
        )

    def test_weights_sum_to_one(self):
        hd = self._make()
        weights = hd.weights()
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_proportional_to_values(self):
        hd = self._make([("SPY", 300_000), ("BND", 700_000)])
        weights = hd.weights()
        assert weights[0] == pytest.approx(0.3)
        assert weights[1] == pytest.approx(0.7)

    def test_total_value(self):
        hd = self._make([("SPY", 600_000), ("BND", 400_000)])
        assert hd.total_value() == pytest.approx(1_000_000)

    def test_total_value_single_ticker(self):
        hd = self._make([("SPY", 750_000)])
        assert hd.total_value() == pytest.approx(750_000)

    def test_summary_lines_contains_year_range(self):
        hd = self._make()
        lines = "\n".join(hd.summary_lines())
        assert "2000" in lines
        assert "2009" in lines

    def test_summary_lines_contains_tickers(self):
        hd = self._make()
        lines = "\n".join(hd.summary_lines())
        assert "SPY" in lines
        assert "BND" in lines

    def test_summary_lines_contains_inflation_stats(self):
        hd = self._make()
        lines = "\n".join(hd.summary_lines())
        assert "inflation" in lines.lower()


# ---------------------------------------------------------------------------
# _fetch_ticker_annual_returns
# ---------------------------------------------------------------------------

class TestFetchTickerAnnualReturns:
    def _mock_yf_download(self, start_year=2000, end_year=2010, annual_return=0.08):
        return make_price_dataframe(start_year, end_year, annual_return)

    @patch("retirement_sim.market_data.yf.download")
    def test_returns_series_in_requested_range(self, mock_dl):
        mock_dl.return_value = self._mock_yf_download(2000, 2010)
        result = _fetch_ticker_annual_returns("SPY", 2000, 2010)
        assert all(2000 <= yr <= 2010 for yr in result.index)

    @patch("retirement_sim.market_data.yf.download")
    def test_series_name_is_ticker(self, mock_dl):
        mock_dl.return_value = self._mock_yf_download(2000, 2005)
        result = _fetch_ticker_annual_returns("SPY", 2000, 2005)
        assert result.name == "SPY"

    @patch("retirement_sim.market_data.yf.download")
    def test_returns_are_floats(self, mock_dl):
        mock_dl.return_value = self._mock_yf_download(2000, 2005)
        result = _fetch_ticker_annual_returns("SPY", 2000, 2005)
        assert result.dtype == float

    @patch("retirement_sim.market_data.yf.download")
    def test_empty_dataframe_raises_value_error(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No price data"):
            _fetch_ticker_annual_returns("FAKE", 2000, 2005)

    @patch("retirement_sim.market_data.yf.download")
    def test_no_data_in_range_raises_value_error(self, mock_dl):
        # Return data only for 2020; requesting 2000–2005 should raise
        mock_dl.return_value = self._mock_yf_download(2020, 2022)
        with pytest.raises(ValueError, match="no data in the range"):
            _fetch_ticker_annual_returns("SPY", 2000, 2005)

    @patch("retirement_sim.market_data.yf.download")
    def test_positive_constant_return_value(self, mock_dl):
        # 8% annual return → each year's pct_change ≈ 0.08
        mock_dl.return_value = self._mock_yf_download(2000, 2005, annual_return=0.08)
        result = _fetch_ticker_annual_returns("SPY", 2000, 2005)
        assert all(result > 0)


# ---------------------------------------------------------------------------
# _fetch_ticker_with_proxy
# ---------------------------------------------------------------------------

class TestFetchTickerWithProxy:
    def _price_df(self, start_year, end_year, annual_return=0.08):
        return make_price_dataframe(start_year, end_year, annual_return)

    @patch("retirement_sim.market_data.yf.download")
    def test_no_proxy_returns_primary(self, mock_dl):
        mock_dl.return_value = self._price_df(2000, 2010)
        result = _fetch_ticker_with_proxy("SPY", None, 2000, 2010)
        assert result.name == "SPY"
        assert all(2000 <= yr <= 2010 for yr in result.index)

    @patch("retirement_sim.market_data.yf.download")
    def test_no_proxy_failure_raises(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError):
            _fetch_ticker_with_proxy("FAKE", None, 2000, 2005)

    @patch("retirement_sim.market_data.yf.download")
    def test_primary_full_coverage_proxy_unused(self, mock_dl):
        # Primary has all years; proxy should never be called
        mock_dl.return_value = self._price_df(2000, 2010)
        result = _fetch_ticker_with_proxy("SPY", "AGG", 2000, 2010)
        # yf.download should only have been called once (for the primary)
        assert mock_dl.call_count == 1
        assert result.name == "SPY"

    @patch("retirement_sim.market_data.yf.download")
    def test_proxy_fills_missing_years(self, mock_dl):
        # Primary covers 2005–2010; proxy covers 2000–2010
        # Requesting 2000–2010: years 2000–2004 should come from proxy
        def side_effect(ticker, **kwargs):
            if ticker == "BND":
                return self._price_df(2005, 2010)  # limited history
            return self._price_df(2000, 2010)       # proxy has full history
        mock_dl.side_effect = side_effect

        result = _fetch_ticker_with_proxy("BND", "AGG", 2000, 2010)
        # Result should have years from 2000 onward (proxy filled the gaps)
        assert min(result.index) <= 2005
        assert max(result.index) >= 2010
        assert result.name == "BND"

    @patch("retirement_sim.market_data.yf.download")
    def test_primary_total_failure_uses_proxy(self, mock_dl):
        def side_effect(ticker, **kwargs):
            if ticker == "NEWFUND":
                return pd.DataFrame()   # primary fails entirely
            return self._price_df(2000, 2010)  # proxy succeeds
        mock_dl.side_effect = side_effect

        result = _fetch_ticker_with_proxy("NEWFUND", "SPY", 2000, 2010)
        assert len(result) > 0

    @patch("retirement_sim.market_data.yf.download")
    def test_both_fail_raises(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="proxy"):
            _fetch_ticker_with_proxy("FAKE1", "FAKE2", 2000, 2005)

    @patch("retirement_sim.market_data.yf.download")
    def test_proxy_unavailable_returns_primary(self, mock_dl):
        # Primary has partial data; proxy also fails → return what primary has
        def side_effect(ticker, **kwargs):
            if ticker == "PROXY":
                return pd.DataFrame()
            return self._price_df(2005, 2010)
        mock_dl.side_effect = side_effect

        result = _fetch_ticker_with_proxy("BND", "PROXY", 2000, 2010)
        # Should get primary data despite proxy failure
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _fetch_cpi_annual
# ---------------------------------------------------------------------------

class TestFetchCpiAnnual:
    def _mock_response(self, csv_text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = csv_text
        resp.raise_for_status = MagicMock()
        return resp

    @patch("retirement_sim.market_data.requests.get")
    def test_returns_series_in_requested_range(self, mock_get):
        mock_get.return_value = self._mock_response(make_cpi_csv(1990, 2024))
        result = _fetch_cpi_annual(2000, 2010)
        assert all(2000 <= yr <= 2010 for yr in result.index)

    @patch("retirement_sim.market_data.requests.get")
    def test_inflation_values_are_positive_for_growing_cpi(self, mock_get):
        # Our synthetic CPI grows at 3% per year → all inflation values > 0
        mock_get.return_value = self._mock_response(make_cpi_csv(1990, 2024))
        result = _fetch_cpi_annual(2000, 2010)
        assert all(result > 0)

    @patch("retirement_sim.market_data.requests.get")
    def test_inflation_approximately_correct(self, mock_get):
        # Synthetic CPI grows at 3% → annual inflation ≈ 0.03
        mock_get.return_value = self._mock_response(make_cpi_csv(1990, 2024, annual_rate=0.03))
        result = _fetch_cpi_annual(2000, 2010)
        assert all(abs(v - 0.03) < 0.005 for v in result)

    @patch("retirement_sim.market_data.requests.get")
    def test_http_error_raises_runtime_error(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.RequestException("timeout")
        with pytest.raises(RuntimeError, match="FRED"):
            _fetch_cpi_annual(2000, 2010)


# ---------------------------------------------------------------------------
# build_historical_dataset
# ---------------------------------------------------------------------------

class TestBuildHistoricalDataset:
    """
    These tests mock _fetch_ticker_with_proxy and _fetch_cpi_annual to avoid
    any network calls while still exercising the alignment and proxy-dispatch
    logic inside build_historical_dataset.
    """

    def _make_ticker_series(self, ticker, years, ret=0.08):
        return pd.Series(
            np.full(len(years), ret), index=years, name=ticker
        )

    def _make_cpi_series(self, years, rate=0.03):
        return pd.Series(np.full(len(years), rate), index=years)

    @patch("retirement_sim.market_data._fetch_cpi_annual")
    @patch("retirement_sim.market_data._fetch_ticker_with_proxy")
    def test_basic_dataset_aligned(self, mock_ticker, mock_cpi):
        years = list(range(2000, 2010))
        mock_ticker.side_effect = [
            self._make_ticker_series("SPY", years),
            self._make_ticker_series("BND", years),
        ]
        mock_cpi.return_value = self._make_cpi_series(years)

        hd = build_historical_dataset(
            [("SPY", 600_000), ("BND", 400_000)],
            start_year=2000, end_year=2009,
        )
        assert hd.years == years
        assert list(hd.returns.columns) == ["SPY", "BND"]
        assert len(hd.inflation) == len(years)

    @patch("retirement_sim.market_data._fetch_cpi_annual")
    @patch("retirement_sim.market_data._fetch_ticker_with_proxy")
    def test_overlapping_years_only(self, mock_ticker, mock_cpi):
        # SPY has 2000-2010; BND only 2005-2010; CPI 2000-2010
        # aligned result should be 2005-2010 (common years)
        spy_years = list(range(2000, 2011))
        bnd_years = list(range(2005, 2011))
        cpi_years = list(range(2000, 2011))

        mock_ticker.side_effect = [
            self._make_ticker_series("SPY", spy_years),
            self._make_ticker_series("BND", bnd_years),
        ]
        mock_cpi.return_value = self._make_cpi_series(cpi_years)

        hd = build_historical_dataset(
            [("SPY", 600_000), ("BND", 400_000)],
            start_year=2000, end_year=2010,
        )
        assert all(yr >= 2005 for yr in hd.years)

    @patch("retirement_sim.market_data._fetch_cpi_annual")
    @patch("retirement_sim.market_data._fetch_ticker_with_proxy")
    def test_no_overlap_raises(self, mock_ticker, mock_cpi):
        mock_ticker.side_effect = [
            self._make_ticker_series("SPY", list(range(2000, 2005))),
            self._make_ticker_series("BND", list(range(2000, 2005))),
        ]
        mock_cpi.return_value = self._make_cpi_series(list(range(2010, 2015)))

        with pytest.raises(ValueError, match="No overlapping years"):
            build_historical_dataset(
                [("SPY", 600_000), ("BND", 400_000)],
                start_year=2000, end_year=2014,
            )

    @patch("retirement_sim.market_data._fetch_cpi_annual")
    @patch("retirement_sim.market_data._fetch_ticker_with_proxy")
    def test_proxy_passed_to_ticker_fetch(self, mock_ticker, mock_cpi):
        years = list(range(2000, 2010))
        mock_ticker.side_effect = [
            self._make_ticker_series("SPY", years),
            self._make_ticker_series("BND", years),
        ]
        mock_cpi.return_value = self._make_cpi_series(years)

        build_historical_dataset(
            [("SPY", 600_000), ("BND", 400_000)],
            start_year=2000, end_year=2009,
            ticker_proxies={"BND": "AGG"},
        )

        # Second call should pass proxy="AGG" for BND
        calls = mock_ticker.call_args_list
        bnd_call_kwargs = {k: v for k, v in zip(
            ["ticker", "proxy", "start_year", "end_year"],
            calls[1][0]
        )}
        assert bnd_call_kwargs.get("proxy") == "AGG" or calls[1][0][1] == "AGG"

    @patch("retirement_sim.market_data._fetch_cpi_annual")
    @patch("retirement_sim.market_data._fetch_ticker_with_proxy")
    def test_ticker_values_stored_on_result(self, mock_ticker, mock_cpi):
        years = list(range(2000, 2010))
        tv = [("SPY", 600_000), ("BND", 400_000)]
        mock_ticker.side_effect = [
            self._make_ticker_series("SPY", years),
            self._make_ticker_series("BND", years),
        ]
        mock_cpi.return_value = self._make_cpi_series(years)

        hd = build_historical_dataset(tv, start_year=2000, end_year=2009)
        assert hd.ticker_values == tv
