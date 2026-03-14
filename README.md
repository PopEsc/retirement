# Retirement Portfolio Monte Carlo Simulator

Runs Monte Carlo simulations on a retirement portfolio to estimate the
annual income that can be withdrawn without depleting the portfolio.

## Requirements

- Python 3.11 or later
- Internet connection (historical mode only — fetches data from Yahoo Finance and FRED)

## Running the GUI

The interactive web UI is the easiest way to use the simulator.

### 1. Create a virtual environment (first time only)

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows (Git Bash / MINGW):**
```bash
source .venv/Scripts/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies (first time only)

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
python -m streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### Quick start

1. Select **Historical** or **Parametric** mode in the sidebar
2. **Historical mode:** add your holdings (ticker, value, account type) in the Portfolio Holdings table, or import a `portfolio.json` file
3. **Parametric mode:** enter your portfolio balance and stock/bond allocation in the sidebar
4. Set your **annual spending goal** (after-tax dollars) and Social Security income
5. Enter your **current age** and **life expectancy** — the simulation horizon is computed automatically
6. Open **Tax settings** and select your filing status and state
7. Click **🚀 Run Simulation**

The simulator grosses up your spending goal to a pre-tax withdrawal using 2026 federal and state
tax brackets, then shows both the gross safe-withdrawal rate and the estimated **after-tax
spendable amount** in the results.

Historical market data is cached locally in `.market_cache/` after the first download, so subsequent runs are fast.

---

## Tax engine

The GUI uses actual 2026 tax brackets to compute how much you need to withdraw from your
portfolio to meet your after-tax spending goal.

**Federal taxes modelled:**
- Ordinary income (TCJA brackets: 10/12/22/24/32/35/37%) for Traditional IRA/401k withdrawals
- Long-term capital gains (0/15/20%) stacked on ordinary income per IRS Publication 550 thresholds
- Net Investment Income Tax (3.8% NIIT) on investment income above the MAGI threshold
- Social Security taxability (IRC §86) — up to 85% included in income based on provisional income
- Additional standard deduction for taxpayers aged 65+ (and spouse if married filing jointly)

**State taxes modelled:**
- Brackets for all 50 states + D.C. (Tax Foundation 2026 data)
- States with no income tax (AK, FL, NV, NH, SD, TN, TX, WY) produce zero state tax
- Washington state capital-gains-only tax (7% above $278K, 9% above $1M) on brokerage gains
- Utah flat-rate credit ($966 single / $1,932 MFJ) subtracted from computed state tax
- Social Security exemption for states that do not tax SS benefits

**Account-type mix:**
- Traditional IRA / 401k withdrawals → ordinary income
- Brokerage → capital gains on the (value − cost basis) / value fraction
- Roth IRA / Roth 401k / After-Tax 401k → tax-free
- Cash → return of principal, no gross-up

Tax data is stored in `retirement_sim/tax_data.py` and updated annually.

---

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
    { "ticker": "VTI",  "value": 500000, "account_type": "Traditional IRA",  "proxy": "VTSMX" },
    { "ticker": "VXUS", "value": 200000, "account_type": "Roth IRA" },
    { "ticker": "BND",  "value": 300000, "account_type": "Brokerage",        "cost_basis": 180000, "proxy": "AGG" },
    { "value": 50000,   "account_type": "Cash", "cash_rate": 4.5 }
  ],
  "social_security": 24000,
  "ss_delay_years": 10,
  "years_to_retirement": 5,
  "annual_savings": 30000
}
```

| Field | Required | Description |
|---|---|---|
| `holdings` | Yes | Array of holding objects |
| `holdings[].ticker` | Yes* | Yahoo Finance ticker symbol. *Optional for Cash accounts. |
| `holdings[].value` | Yes | Current dollar value of this holding |
| `holdings[].account_type` | No | `Traditional IRA`, `Traditional 401k`, `Roth IRA`, `Roth 401k`, `After-Tax 401k`, `Brokerage` (default), or `Cash` |
| `holdings[].cash_rate` | No | Annual interest rate for `Cash` accounts (e.g. `4.5` for 4.5%) |
| `holdings[].cost_basis` | No | Original purchase cost for `Brokerage` accounts — used to compute taxable gains |
| `holdings[].proxy` | No | Ticker to use when this holding lacks data |
| `social_security` | No | Annual SS/pension income (today's dollars) |
| `ss_delay_years` | No | Years from retirement start until SS/pension begins |
| `years_to_retirement` | No | Years of pre-retirement accumulation before withdrawals begin |
| `annual_savings` | No | Annual savings contribution during accumulation (today's dollars) |

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
| `--years`, `-y` | `30` | Simulation horizon in years (CLI only; GUI derives this from current age and life expectancy) |
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
