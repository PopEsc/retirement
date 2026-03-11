# Retirement Portfolio Monte Carlo Simulator

Runs Monte Carlo simulations on a retirement portfolio to estimate the
annual income that can be withdrawn without depleting the portfolio.

## Requirements

- Python 3.11 or later
- Internet connection (historical mode only — fetches data from Yahoo Finance and FRED)

## Installation

```bash
pip install -r requirements.txt
```

## Two simulation modes

### Parametric mode

Returns are drawn from normal distributions using long-run historical averages
(stocks ~10%/17% std, bonds ~4%/7% std, inflation ~3%).  No internet connection
needed.

```bash
# Evaluate a specific withdrawal amount
python main.py --portfolio 1000000 --stocks 0.60 \
               --withdrawal 60000 --social-security 24000

# Find the safe withdrawal rate at 95% success
python main.py --portfolio 1000000 --stocks 0.60 \
               --social-security 24000 --target-success 0.95
```

### Historical mode

Downloads actual annual price returns for your holdings from Yahoo Finance.
Simulation years are drawn in consecutive blocks from real history, preserving
multi-year market cycles (recessions, bull runs).

**Via inline tickers:**

```bash
python main.py --tickers SPY:600000 BND:400000 \
               --withdrawal 60000 --social-security 24000
```

**Via a JSON portfolio file** (recommended — supports per-holding proxies):

```bash
python main.py --portfolio-file portfolio.json --target-success 0.95
```

**Using actual historical CPI inflation** (default in historical mode):

```bash
python main.py --tickers VTI:500000 VXUS:200000 BND:300000 \
               --inflation-mode actual --withdrawal 55000
```

## Portfolio JSON file

The JSON file lets you define your holdings once and reuse them across runs.
It also supports per-holding proxy tickers (see below).

```json
{
  "holdings": [
    { "ticker": "VTI",  "value": 500000, "proxy": "VTSMX" },
    { "ticker": "VXUS", "value": 200000 },
    { "ticker": "BND",  "value": 300000, "proxy": "AGG"   }
  ],
  "social_security": 24000
}
```

| Field | Required | Description |
|---|---|---|
| `holdings` | Yes | Array of holding objects |
| `holdings[].ticker` | Yes | Yahoo Finance ticker symbol |
| `holdings[].value` | Yes | Current dollar value of this holding |
| `holdings[].proxy` | No | Ticker to use when this holding lacks data |
| `social_security` | No | Annual SS/pension income (today's dollars). Overridden by `--social-security` on the CLI. |

The portfolio balance defaults to the sum of all holding values.  Override
with `--portfolio` on the CLI.

## Proxy tickers

Many mutual funds and newer ETFs have limited history.  A proxy ticker
substitutes its returns for any year the primary holding lacks data.

**How it works:**
- Primary data always takes precedence.  The proxy is only used for missing years.
- If the primary ticker fails entirely, all years fall back to the proxy.
- If neither has data for a year, that year is dropped from the dataset.

**Per-holding proxy** (in the JSON file):
```json
{ "ticker": "FSKAX", "value": 700000, "proxy": "SPY" }
```

**Global fallback proxy** (via CLI, applies to all holdings without a per-holding proxy):
```bash
python main.py --tickers FSKAX:700000 FTBFX:300000 \
               --proxy SPY --data-start 1993
```

Per-holding proxies in the JSON file take precedence over `--proxy`.

**Choosing a proxy:**
| Asset class | Suggested proxy |
|---|---|
| US total market | `SPY` or `^GSPC` |
| US large-cap growth | `QQQ` or `^NDX` |
| International stocks | `EFA` |
| US bonds | `AGG` or `BND` |

## Options reference

### Portfolio

| Option | Default | Description |
|---|---|---|
| `--portfolio`, `-p` | — | Starting balance in dollars. Optional in historical mode. |
| `--stocks`, `-s` | `0.60` | Stock fraction 0–1 (parametric mode only) |
| `--tickers TICKER:VALUE …` | — | Holdings inline. Enables historical mode. |
| `--portfolio-file PATH` | — | Load holdings from JSON file. Enables historical mode. |
| `--social-security`, `--ss` | `0` | Annual SS/pension income (today's dollars) |
| `--proxy TICKER` | — | Global proxy for tickers missing early data |

### Withdrawal (one required)

| Option | Description |
|---|---|
| `--withdrawal`, `-w` | Evaluate this annual gross withdrawal amount |
| `--target-success` | Find the max withdrawal achieving this success rate (e.g. `0.95`) |

### Historical data

| Option | Default | Description |
|---|---|---|
| `--data-start YEAR` | `1993` | First year of historical data |
| `--data-end YEAR` | last year | Last year of historical data |
| `--chunk-size N` | `5` | Consecutive years per bootstrap block |
| `--inflation-mode` | `actual` (hist) / `fixed` (param) | `actual` = real CPI; `fixed` = constant `--inflation` rate |

### Market parameters (parametric mode)

| Option | Default | Description |
|---|---|---|
| `--stock-return` | `0.10` | Expected nominal annual stock return |
| `--stock-std` | `0.17` | Stock return standard deviation |
| `--bond-return` | `0.04` | Expected nominal annual bond return |
| `--bond-std` | `0.07` | Bond return standard deviation |
| `--inflation` | `0.03` | Annual inflation rate |
| `--inflation-std` | `0.015` | Inflation standard deviation |

### Simulation

| Option | Default | Description |
|---|---|---|
| `--years`, `-y` | `30` | Simulation horizon in years |
| `--simulations`, `-n` | `10000` | Number of Monte Carlo runs |
| `--seed` | — | Random seed for reproducibility |

### Output

| Option | Description |
|---|---|
| `--save-charts DIR` | Save chart PNGs to DIR instead of showing interactively |
| `--no-charts` | Skip all chart generation |

## Output

The simulator prints a summary table and (unless `--no-charts`) opens three charts:

1. **Balance percentile fan chart** — 10th/25th/50th/75th/90th percentile portfolio
   balance over time.
2. **Success rate vs. withdrawal** — probability of not depleting the portfolio
   across a range of annual withdrawal amounts, with markers for the current
   withdrawal and the safe withdrawal rate.
3. **Depletion year histogram** — distribution of the year the portfolio ran out,
   plus a bar for simulations that survived the full horizon.

## Examples

```bash
# 1. Parametric: 30-year run, find safe withdrawal at 90% confidence
python main.py --portfolio 1200000 --stocks 0.70 \
               --social-security 30000 --target-success 0.90

# 2. Historical: use actual SPY + BND data from 1993–2024
python main.py --tickers SPY:800000 BND:200000 \
               --withdrawal 50000 --inflation-mode actual

# 3. Portfolio file with proxies, save charts to disk
python main.py --portfolio-file portfolio.json \
               --target-success 0.95 --years 35 \
               --save-charts ./charts

# 4. Mutual funds with limited history, global SPY proxy
python main.py --tickers FSKAX:600000 FSRNX:100000 FTBFX:300000 \
               --proxy SPY --data-start 1993 --withdrawal 55000

# 5. Parametric with conservative assumptions, 40-year horizon
python main.py --portfolio 900000 --stocks 0.50 \
               --stock-return 0.07 --inflation 0.035 \
               --years 40 --simulations 50000 --target-success 0.95
```
