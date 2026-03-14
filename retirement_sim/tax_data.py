"""
US Federal and State income tax data for 2026.

Federal ordinary income brackets: 2026 data (TCJA rates in effect).
  Source: taxbrackets_2026.csv provided by user.

Federal standard deductions: Derived from conforming-state data showing $16,100/$32,200
  for states that track the federal amount (CO, IA, ID, MO, MT, NM, ND, DC).

Federal LTCG brackets: Estimated ~4% CPI adjustment from 2025 IRS amounts.

State brackets: 2026 data from Tax Foundation "2026 State Income Tax Rates and Brackets".
  Standard deductions listed are state-level amounts (or personal exemption equivalents
  where noted). States with both a standard deduction and personal exemption use the sum.

Note: Update this file annually as IRS releases Rev. Proc. with adjusted amounts.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Filing status keys and labels
# ─────────────────────────────────────────────────────────────────────────────

FILING_STATUSES: list[str] = [
    "single",
    "married_joint",
    "head_of_household",
    "married_separate",
]

FILING_STATUS_LABELS: dict[str, str] = {
    "single":           "Single",
    "married_joint":    "Married Filing Jointly",
    "head_of_household": "Head of Household",
    "married_separate": "Married Filing Separately",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2026 Federal Ordinary Income Tax Brackets
# Source: User-provided 2026 tax bracket data (TCJA rates)
# Format: list of (income_floor, marginal_rate) sorted ascending.
#   Income above floor and below next bracket's floor is taxed at that rate.
# ─────────────────────────────────────────────────────────────────────────────

FEDERAL_ORDINARY_BRACKETS: dict[str, list[tuple[float, float]]] = {
    "single": [
        (0,        0.10),
        (12_400,   0.12),
        (50_400,   0.22),
        (105_700,  0.24),
        (201_775,  0.32),
        (256_225,  0.35),
        (640_600,  0.37),
    ],
    "married_joint": [
        (0,        0.10),
        (24_800,   0.12),
        (100_800,  0.22),
        (211_400,  0.24),
        (403_550,  0.32),
        (512_450,  0.35),
        (768_700,  0.37),
    ],
    "head_of_household": [
        (0,        0.10),
        (17_700,   0.12),
        (67_450,   0.22),
        (105_700,  0.24),
        (201_750,  0.32),
        (256_200,  0.35),
        (640_600,  0.37),
    ],
    "married_separate": [
        (0,        0.10),
        (12_400,   0.12),
        (50_400,   0.22),
        (105_700,  0.24),
        (201_775,  0.32),
        (256_225,  0.35),
        (384_350,  0.37),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2026 Federal Standard Deductions
# Derived from conforming states (CO, IA, ID, MO, MT, NM, ND, DC all = $16,100/$32,200)
# HOH estimated at 1.5× single; MFS = same as single.
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_DEDUCTION: dict[str, float] = {
    "single":            16_100,
    "married_joint":     32_200,
    "head_of_household": 24_150,   # estimated
    "married_separate":  16_100,
}

# Additional standard deduction per qualifying person aged 65+ or blind
# (estimated ~2% adj from 2025 IRS amounts of $1,950 single / $1,550 MFJ per spouse)
EXTRA_DEDUCTION_65_PLUS: dict[str, float] = {
    "single":            2_000,    # per person
    "married_joint":     1_600,    # per qualifying spouse (doubles if both 65+)
    "head_of_household": 2_000,
    "married_separate":  1_600,
}

# ─────────────────────────────────────────────────────────────────────────────
# Federal Long-Term Capital Gains Brackets
# Source: IRS Publication 550, taxable years beginning in 2025.
# (2026 amounts not yet released; use these until Rev. Proc. update.)
# LTCG rate is determined by total taxable income (ordinary + LTCG stacked).
# ─────────────────────────────────────────────────────────────────────────────

FEDERAL_LTCG_BRACKETS: dict[str, list[tuple[float, float]]] = {
    # Format: (income_floor_for_this_rate, rate)
    # Income at or below the first threshold: 0%; above second: 20%.
    "single": [
        (0,        0.000),
        (48_350,   0.150),
        (533_400,  0.200),
    ],
    "married_joint": [
        (0,        0.000),
        (96_700,   0.150),
        (600_050,  0.200),
    ],
    "head_of_household": [
        (0,        0.000),
        (64_750,   0.150),
        (566_700,  0.200),
    ],
    "married_separate": [
        (0,        0.000),
        (48_350,   0.150),
        (300_000,  0.200),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Net Investment Income Tax (3.8% surtax) — NOT inflation-adjusted
# ─────────────────────────────────────────────────────────────────────────────

NIIT_THRESHOLD: dict[str, float] = {
    "single":            200_000,
    "married_joint":     250_000,
    "head_of_household": 200_000,
    "married_separate":  125_000,
}

# ─────────────────────────────────────────────────────────────────────────────
# Social Security Benefit Taxability (IRC §86)
# Thresholds have NOT been inflation-adjusted since enacted (1983/1993).
# Provisional income = AGI + non-taxable interest + 50% of SS benefits.
# ─────────────────────────────────────────────────────────────────────────────

SS_TAXABILITY: dict[str, dict[str, float]] = {
    "single": {
        "threshold_50pct": 25_000,
        "threshold_85pct": 34_000,
    },
    "married_joint": {
        "threshold_50pct": 32_000,
        "threshold_85pct": 44_000,
    },
    "head_of_household": {
        "threshold_50pct": 25_000,
        "threshold_85pct": 34_000,
    },
    "married_separate": {
        # MFS who lived together all year: 85% always taxable
        "threshold_50pct": 0,
        "threshold_85pct": 0,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# State Income Tax Data — 2026
# Source: Tax Foundation "2026 State Income Tax Rates and Brackets"
#
# Structure per state:
#   "single"           – (floor, rate) bracket list for single filers
#   "married_joint"    – (floor, rate) bracket list for MFJ (omitted if same as single)
#   "std_ded_single"   – standard deduction for single (or personal exemption if no SD)
#   "std_ded_mfj"      – standard deduction for MFJ
#   "tax_credit_single"– dollar credit subtracted from computed tax (optional, e.g. UT)
#   "tax_credit_mfj"   – same for MFJ
#
# Washington (WA) taxes only capital gains, not ordinary income.
# States with no income tax omit bracket entries (0% flat).
# ─────────────────────────────────────────────────────────────────────────────

STATE_TAX: dict[str, dict] = {

    # ── No income tax ─────────────────────────────────────────────────────────
    "AK": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "FL": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "NV": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "NH": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "SD": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "TN": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "TX": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},
    "WY": {"single": [(0, 0.000)], "std_ded_single": 0, "std_ded_mfj": 0},

    # WA: no ordinary income tax; capital gains only (7% above $278K, 9% above $1M)
    # Ordinary income brackets set to 0%; WA LTCG handled separately below.
    "WA": {
        "single":          [(0, 0.000)],
        "std_ded_single":  0,
        "std_ded_mfj":     0,
        "ltcg_brackets":   [(0, 0.000), (278_000, 0.070), (1_000_000, 0.090)],
        "ltcg_threshold":  278_000,
    },

    # ── Alabama ──────────────────────────────────────────────────────────────
    "AL": {
        "single":         [(0, 0.020), (500, 0.040), (3_000, 0.050)],
        "married_joint":  [(0, 0.020), (1_000, 0.040), (6_000, 0.050)],
        "std_ded_single": 3_000,     # SD $3,000 + PE $1,500 = $4,500
        "std_ded_mfj":    8_500,     # SD $8,500 + PE $3,000 = $11,500
        # Using SD only per Tax Foundation; PE is separate credit in AL
    },

    # ── Arizona ──────────────────────────────────────────────────────────────
    "AZ": {
        "single":         [(0, 0.025)],
        "std_ded_single": 8_350,
        "std_ded_mfj":    16_700,
    },

    # ── Arkansas ─────────────────────────────────────────────────────────────
    "AR": {
        "single":         [(0, 0.020), (4_600, 0.039)],
        "std_ded_single": 2_470,
        "std_ded_mfj":    4_940,
    },

    # ── California ───────────────────────────────────────────────────────────
    "CA": {
        "single": [
            (0,           0.010),
            (11_079,      0.020),
            (26_264,      0.040),
            (41_452,      0.060),
            (57_542,      0.080),
            (72_724,      0.093),
            (371_479,     0.103),
            (445_771,     0.113),
            (742_953,     0.123),
            (1_000_000,   0.133),
        ],
        "married_joint": [
            (0,           0.010),
            (22_158,      0.020),
            (52_528,      0.040),
            (82_904,      0.060),
            (115_084,     0.080),
            (145_448,     0.093),
            (742_958,     0.103),
            (891_542,     0.113),
            (1_000_000,   0.123),
            (1_485_906,   0.133),
        ],
        "std_ded_single": 5_540,
        "std_ded_mfj":    11_080,
    },

    # ── Colorado ─────────────────────────────────────────────────────────────
    "CO": {
        "single":         [(0, 0.044)],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── Connecticut ──────────────────────────────────────────────────────────
    "CT": {
        "single": [
            (0,        0.020),
            (10_000,   0.045),
            (50_000,   0.055),
            (100_000,  0.060),
            (200_000,  0.065),
            (250_000,  0.069),
            (500_000,  0.0699),
        ],
        "married_joint": [
            (0,        0.020),
            (20_000,   0.045),
            (100_000,  0.055),
            (200_000,  0.060),
            (400_000,  0.065),
            (500_000,  0.069),
            (1_000_000, 0.0699),
        ],
        "std_ded_single": 15_000,   # personal exemption (no standard deduction)
        "std_ded_mfj":    24_000,
    },

    # ── Delaware ─────────────────────────────────────────────────────────────
    "DE": {
        "single": [
            (0,       0.000),
            (2_000,   0.022),
            (5_000,   0.039),
            (10_000,  0.048),
            (20_000,  0.052),
            (25_000,  0.0555),
            (60_000,  0.066),
        ],
        "std_ded_single": 3_250,
        "std_ded_mfj":    6_500,
    },

    # ── Georgia ──────────────────────────────────────────────────────────────
    "GA": {
        "single":         [(0, 0.0519)],
        "std_ded_single": 12_000,
        "std_ded_mfj":    24_000,
    },

    # ── Hawaii ───────────────────────────────────────────────────────────────
    "HI": {
        "single": [
            (0,        0.014),
            (9_600,    0.032),
            (14_400,   0.055),
            (19_200,   0.064),
            (24_000,   0.068),
            (36_000,   0.072),
            (48_000,   0.076),
            (125_000,  0.079),
            (175_000,  0.0825),
            (225_000,  0.090),
            (275_000,  0.100),
            (325_000,  0.110),
        ],
        "married_joint": [
            (0,        0.014),
            (19_200,   0.032),
            (28_800,   0.055),
            (38_400,   0.064),
            (48_000,   0.068),
            (72_000,   0.072),
            (96_000,   0.076),
            (250_000,  0.079),
            (350_000,  0.0825),
            (450_000,  0.090),
            (550_000,  0.100),
            (650_000,  0.110),
        ],
        "std_ded_single": 4_400,
        "std_ded_mfj":    8_800,
    },

    # ── Idaho (0% below $4,811 single / $9,622 MFJ) ──────────────────────────
    "ID": {
        "single":         [(0, 0.000), (4_811, 0.053)],
        "married_joint":  [(0, 0.000), (9_622, 0.053)],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── Illinois (personal exemption $2,925/$5,850; no standard deduction) ──
    "IL": {
        "single":         [(0, 0.0495)],
        "std_ded_single": 2_925,     # personal exemption
        "std_ded_mfj":    5_850,
    },

    # ── Indiana (personal exemption $1,000/$2,000) ────────────────────────────
    "IN": {
        "single":         [(0, 0.0295)],
        "std_ded_single": 1_000,     # personal exemption
        "std_ded_mfj":    2_000,
    },

    # ── Iowa (flat 3.8% since 2024) ──────────────────────────────────────────
    "IA": {
        "single":         [(0, 0.038)],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── Kansas ───────────────────────────────────────────────────────────────
    "KS": {
        "single":         [(0, 0.052), (23_000, 0.0558)],
        "married_joint":  [(0, 0.052), (46_000, 0.0558)],
        "std_ded_single": 3_605 + 9_160,   # SD $3,605 + PE $9,160 = $12,765
        "std_ded_mfj":    8_240 + 18_320,  # SD $8,240 + PE $18,320 = $26,560
    },

    # ── Kentucky ─────────────────────────────────────────────────────────────
    "KY": {
        "single":         [(0, 0.035)],
        "std_ded_single": 3_360,
        "std_ded_mfj":    3_360,     # same for both per Tax Foundation
    },

    # ── Louisiana ────────────────────────────────────────────────────────────
    "LA": {
        "single":         [(0, 0.030)],
        "std_ded_single": 12_875,
        "std_ded_mfj":    25_750,
    },

    # ── Maine ────────────────────────────────────────────────────────────────
    "ME": {
        "single": [
            (0,        0.058),
            (27_399,   0.0675),
            (64_849,   0.0715),
        ],
        "married_joint": [
            (0,        0.058),
            (54_849,   0.0675),
            (129_749,  0.0715),
        ],
        "std_ded_single": 8_350,
        "std_ded_mfj":    16_700,
    },

    # ── Maryland ─────────────────────────────────────────────────────────────
    "MD": {
        "single": [
            (0,          0.020),
            (1_000,      0.030),
            (2_000,      0.040),
            (3_000,      0.0475),
            (100_000,    0.050),
            (125_000,    0.0525),
            (150_000,    0.0550),
            (250_000,    0.0575),
            (500_000,    0.0625),
            (1_000_000,  0.0650),
        ],
        "married_joint": [
            (0,          0.020),
            (1_000,      0.030),
            (2_000,      0.040),
            (3_000,      0.0475),
            (150_000,    0.050),
            (175_000,    0.0525),
            (225_000,    0.0550),
            (300_000,    0.0575),
            (600_000,    0.0625),
            (1_200_000,  0.0650),
        ],
        "std_ded_single": 3_350,
        "std_ded_mfj":    6_700,
    },

    # ── Massachusetts (personal exemption $4,400/$8,800) ────────────────────
    "MA": {
        "single": [
            (0,           0.050),
            (1_083_150,   0.090),
        ],
        "std_ded_single": 4_400,     # personal exemption
        "std_ded_mfj":    8_800,
    },

    # ── Michigan (personal exemption $5,900/$11,800) ─────────────────────────
    "MI": {
        "single":         [(0, 0.0425)],
        "std_ded_single": 5_900,     # personal exemption
        "std_ded_mfj":    11_800,
    },

    # ── Minnesota ────────────────────────────────────────────────────────────
    "MN": {
        "single": [
            (0,         0.0535),
            (33_310,    0.0680),
            (109_430,   0.0785),
            (203_150,   0.0985),
        ],
        "married_joint": [
            (0,         0.0535),
            (48_700,    0.0680),
            (193_480,   0.0785),
            (337_930,   0.0985),
        ],
        "std_ded_single": 15_300,
        "std_ded_mfj":    30_600,
    },

    # ── Mississippi (0% on first $10,000) ────────────────────────────────────
    "MS": {
        "single":         [(0, 0.000), (10_000, 0.040)],
        "std_ded_single": 2_300 + 6_000,   # SD $2,300 + PE $6,000 = $8,300
        "std_ded_mfj":    4_600 + 12_000,  # SD $4,600 + PE $12,000 = $16,600
    },

    # ── Missouri ─────────────────────────────────────────────────────────────
    "MO": {
        "single": [
            (0,       0.000),
            (1_348,   0.020),
            (2_696,   0.025),
            (4_044,   0.030),
            (5_392,   0.035),
            (6_740,   0.040),
            (8_088,   0.045),
            (9_436,   0.047),
        ],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── Montana ──────────────────────────────────────────────────────────────
    "MT": {
        "single":         [(0, 0.047), (47_500, 0.0565)],
        "married_joint":  [(0, 0.047), (95_000, 0.0565)],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── Nebraska ─────────────────────────────────────────────────────────────
    "NE": {
        "single": [
            (0,        0.0246),
            (4_130,    0.0351),
            (24_760,   0.0455),
        ],
        "married_joint": [
            (0,        0.0246),
            (8_250,    0.0351),
            (49_530,   0.0455),
        ],
        "std_ded_single": 8_850,
        "std_ded_mfj":    17_700,
    },

    # ── New Jersey ───────────────────────────────────────────────────────────
    "NJ": {
        "single": [
            (0,          0.0140),
            (20_000,     0.0175),
            (35_000,     0.0350),
            (40_000,     0.0553),
            (75_000,     0.0637),
            (500_000,    0.0897),
            (1_000_000,  0.1075),
        ],
        "married_joint": [
            (0,          0.0140),
            (20_000,     0.0175),
            (50_000,     0.0245),
            (70_000,     0.0350),
            (80_000,     0.0553),
            (150_000,    0.0637),
            (500_000,    0.0897),
            (1_000_000,  0.1075),
        ],
        "std_ded_single": 1_000,     # personal exemption
        "std_ded_mfj":    2_000,
    },

    # ── New Mexico ───────────────────────────────────────────────────────────
    "NM": {
        "single": [
            (0,         0.015),
            (5_500,     0.032),
            (16_500,    0.043),
            (33_500,    0.047),
            (66_500,    0.049),
            (210_000,   0.059),
        ],
        "married_joint": [
            (0,         0.015),
            (8_000,     0.032),
            (25_000,    0.043),
            (50_000,    0.047),
            (100_000,   0.049),
            (315_000,   0.059),
        ],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── New York ─────────────────────────────────────────────────────────────
    "NY": {
        "single": [
            (0,           0.039),
            (8_500,       0.044),
            (11_700,      0.0515),
            (13_900,      0.054),
            (80_650,      0.059),
            (215_400,     0.0685),
            (1_077_550,   0.0965),
            (5_000_000,   0.1030),
            (25_000_000,  0.1090),
        ],
        "married_joint": [
            (0,           0.039),
            (17_150,      0.044),
            (23_600,      0.0515),
            (27_900,      0.054),
            (161_550,     0.059),
            (323_200,     0.0685),
            (2_155_350,   0.0965),
            (5_000_000,   0.1030),
            (25_000_000,  0.1090),
        ],
        "std_ded_single": 8_000,
        "std_ded_mfj":    16_050,
    },

    # ── North Carolina ───────────────────────────────────────────────────────
    "NC": {
        "single":         [(0, 0.0399)],
        "std_ded_single": 12_750,
        "std_ded_mfj":    25_500,
    },

    # ── North Dakota (0% below $48,475 single / $80,975 MFJ) ────────────────
    "ND": {
        "single":         [(0, 0.000), (48_475, 0.0195), (244_825, 0.025)],
        "married_joint":  [(0, 0.000), (80_975, 0.0195), (298_075, 0.025)],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },

    # ── Ohio (0% below $26,050; personal exemption $2,400/$4,800) ────────────
    "OH": {
        "single":         [(0, 0.000), (26_050, 0.0275)],
        "std_ded_single": 2_400,     # personal exemption
        "std_ded_mfj":    4_800,
    },

    # ── Oklahoma (0% below $3,750 single / $7,500 MFJ) ───────────────────────
    "OK": {
        "single":         [(0, 0.000), (3_750, 0.025), (4_900, 0.035), (7_200, 0.045)],
        "married_joint":  [(0, 0.000), (7_500, 0.025), (9_800, 0.035), (14_400, 0.045)],
        "std_ded_single": 6_350 + 1_000,   # SD $6,350 + PE $1,000 = $7,350
        "std_ded_mfj":    12_700 + 2_000,  # SD $12,700 + PE $2,000 = $14,700
    },

    # ── Oregon ───────────────────────────────────────────────────────────────
    "OR": {
        "single": [
            (0,         0.0475),
            (4_550,     0.0675),
            (11_400,    0.0875),
            (125_000,   0.0990),
        ],
        "married_joint": [
            (0,         0.0475),
            (9_100,     0.0675),
            (22_800,    0.0875),
            (250_000,   0.0990),
        ],
        "std_ded_single": 2_910,
        "std_ded_mfj":    5_820,
    },

    # ── Pennsylvania ────────────────────────────────────────────────────────
    "PA": {
        "single":         [(0, 0.0307)],
        "std_ded_single": 0,         # no standard deduction or personal exemption
        "std_ded_mfj":    0,
    },

    # ── Rhode Island ─────────────────────────────────────────────────────────
    "RI": {
        "single": [
            (0,         0.0375),
            (82_050,    0.0475),
            (186_450,   0.0599),
        ],
        "std_ded_single": 11_200,
        "std_ded_mfj":    22_400,
    },

    # ── South Carolina ───────────────────────────────────────────────────────
    "SC": {
        "single":         [(0, 0.000), (3_640, 0.030), (18_230, 0.060)],
        "std_ded_single": 8_350,
        "std_ded_mfj":    16_700,
    },

    # ── Utah (4.5% flat; tax credit $966 single / $1,932 MFJ reduces computed tax)
    "UT": {
        "single":             [(0, 0.045)],
        "std_ded_single":     0,
        "std_ded_mfj":        0,
        "tax_credit_single":  966,
        "tax_credit_mfj":     1_932,
    },

    # ── Vermont ──────────────────────────────────────────────────────────────
    "VT": {
        "single": [
            (0,         0.0335),
            (49_400,    0.066),
            (119_700,   0.076),
            (249_700,   0.0875),
        ],
        "married_joint": [
            (0,         0.0335),
            (82_500,    0.066),
            (199_450,   0.076),
            (304_000,   0.0875),
        ],
        "std_ded_single": 7_650,
        "std_ded_mfj":    15_300,
    },

    # ── Virginia ─────────────────────────────────────────────────────────────
    "VA": {
        "single": [
            (0,        0.020),
            (3_000,    0.030),
            (5_000,    0.050),
            (17_000,   0.0575),
        ],
        "std_ded_single": 8_750,
        "std_ded_mfj":    17_500,
    },

    # ── West Virginia ────────────────────────────────────────────────────────
    "WV": {
        "single": [
            (0,        0.0222),
            (10_000,   0.0296),
            (25_000,   0.0333),
            (40_000,   0.0444),
            (60_000,   0.0482),
        ],
        "std_ded_single": 2_000,     # personal exemption
        "std_ded_mfj":    4_000,
    },

    # ── Wisconsin ────────────────────────────────────────────────────────────
    "WI": {
        "single": [
            (0,          0.0350),
            (15_110,     0.0440),
            (51_950,     0.0530),
            (332_720,    0.0765),
        ],
        "married_joint": [
            (0,          0.0350),
            (20_150,     0.0440),
            (69_260,     0.0530),
            (443_630,    0.0765),
        ],
        "std_ded_single": 13_960,
        "std_ded_mfj":    25_840,
    },

    # ── Washington D.C. ──────────────────────────────────────────────────────
    "DC": {
        "single": [
            (0,           0.040),
            (10_000,      0.060),
            (40_000,      0.065),
            (60_000,      0.085),
            (250_000,     0.0925),
            (500_000,     0.0975),
            (1_000_000,   0.1075),
        ],
        "std_ded_single": 16_100,    # conforms to federal
        "std_ded_mfj":    32_200,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# State display information for UI
# ─────────────────────────────────────────────────────────────────────────────

STATE_NAMES: dict[str, str] = {
    "None": "No state income tax / unknown",
    "AL": "Alabama",        "AK": "Alaska",          "AZ": "Arizona",
    "AR": "Arkansas",       "CA": "California",      "CO": "Colorado",
    "CT": "Connecticut",    "DC": "Washington D.C.", "DE": "Delaware",
    "FL": "Florida",        "GA": "Georgia",         "HI": "Hawaii",
    "ID": "Idaho",          "IL": "Illinois",        "IN": "Indiana",
    "IA": "Iowa",           "KS": "Kansas",          "KY": "Kentucky",
    "LA": "Louisiana",      "ME": "Maine",           "MD": "Maryland",
    "MA": "Massachusetts",  "MI": "Michigan",        "MN": "Minnesota",
    "MS": "Mississippi",    "MO": "Missouri",        "MT": "Montana",
    "NE": "Nebraska",       "NV": "Nevada",          "NH": "New Hampshire",
    "NJ": "New Jersey",     "NM": "New Mexico",      "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota",    "OH": "Ohio",
    "OK": "Oklahoma",       "OR": "Oregon",          "PA": "Pennsylvania",
    "RI": "Rhode Island",   "SC": "South Carolina",  "SD": "South Dakota",
    "TN": "Tennessee",      "TX": "Texas",           "UT": "Utah",
    "VT": "Vermont",        "VA": "Virginia",        "WA": "Washington",
    "WV": "West Virginia",  "WI": "Wisconsin",       "WY": "Wyoming",
}

# States that do NOT tax Social Security income (for reference; not modeled
# in brackets, but the calculator notes this for informational purposes).
STATES_NO_SS_TAX: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KY", "LA", "ME", "MD", "MA", "MI", "MS",
    "NH", "NJ", "NY", "NC", "OH", "OK", "OR", "PA", "SC", "SD",
    "TN", "TX", "VA", "WA", "WI", "WY",
})
