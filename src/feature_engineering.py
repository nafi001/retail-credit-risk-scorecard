"""Deterministic, stateless feature engineering (Step 3.1).
No fitted parameters here, so applying it to any sample cannot leak information.
Sentinel codes: -1 = not applicable / not computable, -2 = implausible value."""
from __future__ import annotations
import numpy as np
import pandas as pd

ID_COL, TARGET = "row_id", "SeriousDlqin2yrs"
NOT_APPLICABLE, IMPLAUSIBLE = -1.0, -2.0
UTIL_IMPLAUSIBLE_ABOVE = 10.0          # balance > 10x the credit limit is treated as implausible

RENAME = {
    "RevolvingUtilizationOfUnsecuredLines": "utilisation",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "n_dependents",
}
RAW_PAST_DUE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
PD_COLS = ["past_due_30_59", "past_due_60_89", "past_due_90plus"]
FEATURE_ORDER = ["age", "monthly_income", "n_dependents", "open_credit_lines", "real_estate_loans",
                 "past_due_30_59", "past_due_60_89", "past_due_90plus", "utilisation", "debt_ratio",
                 "total_past_due", "worst_delinquency", "open_lines_non_re"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in RENAME if c not in df.columns]
    if miss:
        raise KeyError(f"Missing raw columns: {miss}")
    raw = df[list(RENAME)].rename(columns=RENAME)
    if (raw[PD_COLS] >= 90).any().any():
        raise ValueError("Special-coded past-due values (>=90): route to manual review; do not score.")
    if (raw["age"] < 18).any():
        raise ValueError("age < 18: outside the modelling population.")

    out = pd.DataFrame(index=df.index)
    for c in (ID_COL, TARGET):
        if c in df.columns:
            out[c] = df[c]
    # pass-through variables (kept as-is; missing stays NaN; binning will treat it explicitly)
    for c in ["age", "n_dependents", "open_credit_lines", "real_estate_loans"] + PD_COLS:
        out[c] = raw[c]
    # income: zero and missing both mean "not available" (indistinguishable default rates; DebtRatio not computable for both)
    out["monthly_income"] = raw["monthly_income"].where(raw["monthly_income"] > 0)
    # utilisation: not meaningful without open lines; implausible above the cap
    util = raw["utilisation"].astype(float)
    out["utilisation"] = np.select([raw["open_credit_lines"] == 0, util > UTIL_IMPLAUSIBLE_ABOVE],
                                   [NOT_APPLICABLE, IMPLAUSIBLE], default=util)
    # debt ratio: only a ratio when income is positive
    out["debt_ratio"] = np.where(raw["monthly_income"] > 0, raw["debt_ratio"], NOT_APPLICABLE)
    # delinquency summaries
    out["total_past_due"] = raw[PD_COLS].sum(axis=1)
    out["worst_delinquency"] = np.select([raw["past_due_90plus"] > 0, raw["past_due_60_89"] > 0, raw["past_due_30_59"] > 0],
                                         [3, 2, 1], default=0)
    # credit lines that are not real-estate loans (real estate never exceeded open lines in development)
    out["open_lines_non_re"] = np.where(raw["open_credit_lines"] >= raw["real_estate_loans"],
                                        raw["open_credit_lines"] - raw["real_estate_loans"], IMPLAUSIBLE)
    front = [c for c in (ID_COL, TARGET) if c in out.columns]
    return out[front + FEATURE_ORDER]

FEATURE_DICTIONARY = pd.DataFrame([
    ("age", "raw", "age", "Age in years", "none", "older -> lower PD", "Policy-sensitive; run with/without"),
    ("monthly_income", "raw", "MonthlyIncome", "Monthly income", "NaN = missing or zero (not available)", "higher -> lower PD", "Very low values may be unreliable"),
    ("n_dependents", "raw", "NumberOfDependents", "Number of dependents", "NaN = missing", "more -> higher PD (weak)", "Household characteristic; fairness review"),
    ("open_credit_lines", "raw", "NumberOfOpenCreditLinesAndLoans", "Open loans and credit lines", "none", "unclear / U-shaped", ""),
    ("real_estate_loans", "raw", "NumberRealEstateLoansOrLines", "Real-estate loans and lines", "none", "unclear / U-shaped", ""),
    ("past_due_30_59", "raw", "NumberOfTime30-59DaysPastDueNotWorse", "Count of 30-59 day delinquencies", "none", "more -> higher PD", ""),
    ("past_due_60_89", "raw", "NumberOfTime60-89DaysPastDueNotWorse", "Count of 60-89 day delinquencies", "none", "more -> higher PD", ""),
    ("past_due_90plus", "raw", "NumberOfTimes90DaysLate", "Count of 90+ day delinquencies", "none", "more -> higher PD", "Timing relative to outcome undocumented"),
    ("utilisation", "cleaned", "RevolvingUtilizationOfUnsecuredLines; open lines", "Revolving utilisation", "-1: no open lines; -2: > 10", "higher -> higher PD", "Zero is its own bin (higher risk than low)"),
    ("debt_ratio", "cleaned", "DebtRatio; MonthlyIncome", "Debt ratio where income > 0", "-1: income missing or zero", "higher -> higher PD", "Includes living costs per dictionary"),
    ("total_past_due", "derived", "three past-due counts", "Sum of the three delinquency counts", "none", "more -> higher PD", "Inherits 90+ timing caveat"),
    ("worst_delinquency", "derived", "three past-due counts", "Worst level: 0 none, 1 30-59, 2 60-89, 3 90+", "none", "higher -> higher PD", "Inherits 90+ timing caveat"),
    ("open_lines_non_re", "derived", "open lines; real-estate loans", "Open lines excluding real-estate loans", "-2: real estate > open lines", "unclear", "Correlated with open_credit_lines"),
], columns=["feature", "kind", "source_fields", "definition", "special_codes", "hypothesised_direction", "note"])
