"""Data-quality diagnostics (Step 2.1). Call these on the DEVELOPMENT sample only."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
ID_COL = "row_id"
P30, P60, P90 = ("NumberOfTime30-59DaysPastDueNotWorse",
                 "NumberOfTime60-89DaysPastDueNotWorse",
                 "NumberOfTimes90DaysLate")
PAST_DUE = [P30, P60, P90]
FEATURES = ["RevolvingUtilizationOfUnsecuredLines", "age", P30, "DebtRatio", "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans", P90, "NumberRealEstateLoansOrLines", P60,
            "NumberOfDependents"]

PREDICTOR_TYPES = pd.DataFrame([
    ("RevolvingUtilizationOfUnsecuredLines", "ratio (continuous)", "credit utilisation", "higher -> higher PD"),
    ("age", "integer", "demographic (policy-sensitive)", "older -> lower PD"),
    (P30, "count", "delinquency history", "more -> higher PD"),
    ("DebtRatio", "ratio (continuous)", "debt burden", "higher -> higher PD"),
    ("MonthlyIncome", "amount (continuous)", "repayment capacity", "higher -> lower PD"),
    ("NumberOfOpenCreditLinesAndLoans", "count", "credit depth", "unclear / possibly non-monotonic"),
    (P90, "count", "delinquency history (leakage check)", "more -> higher PD"),
    ("NumberRealEstateLoansOrLines", "count", "credit mix", "unclear"),
    (P60, "count", "delinquency history", "more -> higher PD"),
    ("NumberOfDependents", "count", "household burden (demographic)", "weak: more -> higher PD"),
], columns=["variable", "type", "business_group", "hypothesised_direction_of_risk"])

def target_summary(df):
    n, d = len(df), int(df[TARGET].sum())
    p = d / n
    se = np.sqrt(p * (1 - p) / n)
    return pd.Series({"n": n, "defaults": d, "non_defaults": n - d, "default_rate": p,
                      "ci95_low": p - 1.96 * se, "ci95_high": p + 1.96 * se,
                      "nondefault_per_default": (n - d) / d})

def duplicate_summary(df):
    cols = FEATURES + [TARGET]
    return pd.Series({
        "extra_copies_of_identical_rows": int(df.duplicated(subset=cols).sum()),
        "rows_in_any_duplicate_group": int(df.duplicated(subset=cols, keep=False).sum()),
        "feature_patterns_with_conflicting_targets": int((df.groupby(FEATURES, dropna=False)[TARGET].nunique() > 1).sum()),
    })

def top_duplicate_patterns(df, k=10):
    cols = FEATURES + [TARGET]
    d = df[df.duplicated(subset=cols, keep=False)]
    if d.empty:
        return d
    return d.groupby(cols, dropna=False).size().rename("n_rows").sort_values(ascending=False).head(k).reset_index()

def missing_summary(df):
    rows = []
    for c in FEATURES:
        m = df[c].isna()
        if m.sum() == 0:
            continue
        rows.append({"variable": c, "n_missing": int(m.sum()), "pct_missing": 100 * m.mean(),
                     "default_rate_if_missing": df.loc[m, TARGET].mean(),
                     "default_rate_if_present": df.loc[~m, TARGET].mean()})
    return pd.DataFrame(rows)

def missing_overlap(df):
    return pd.crosstab(df["MonthlyIncome"].isna().rename("income_missing"),
                       df["NumberOfDependents"].isna().rename("dependents_missing"), margins=True)

def rule_checks(df):
    """Review flags, NOT deletion rules."""
    inc_na = df["MonthlyIncome"].isna()
    big = df[PAST_DUE] >= 90
    rules = {
        "age == 0": df["age"] == 0,
        "age < 18": df["age"] < 18,
        "age > 100": df["age"] > 100,
        "utilisation > 1 (over limit)": df["RevolvingUtilizationOfUnsecuredLines"] > 1,
        "utilisation > 10 (implausible?)": df["RevolvingUtilizationOfUnsecuredLines"] > 10,
        "DebtRatio > 1 (payments exceed income)": df["DebtRatio"] > 1,
        "DebtRatio > 100 (artefact?)": df["DebtRatio"] > 100,
        "DebtRatio > 1 AND income missing": (df["DebtRatio"] > 1) & inc_na,
        "DebtRatio > 1 AND income present": (df["DebtRatio"] > 1) & ~inc_na,
        "MonthlyIncome == 0": df["MonthlyIncome"] == 0,
        "any past-due count >= 90 (special code?)": big.any(axis=1),
        "all three past-due counts >= 90": big.all(axis=1),
        "real-estate loans > open credit lines": df["NumberRealEstateLoansOrLines"] > df["NumberOfOpenCreditLinesAndLoans"],
    }
    rows = []
    for name, mask in rules.items():
        n = int(mask.sum())
        rows.append({"rule": name, "n_flagged": n, "pct_flagged": 100 * n / len(df),
                     "default_rate_flagged": df.loc[mask, TARGET].mean() if n else np.nan,
                     "default_rate_not_flagged": df.loc[~mask, TARGET].mean()})
    return pd.DataFrame(rows)

def value_counts_top(df, col, k=8):
    return df[col].value_counts(dropna=False).head(k).rename("n").to_frame()

def quantile_table(df, cols=None):
    q = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
    t = df[cols or FEATURES].quantile(q).T
    t.columns = [f"q{x:g}" for x in q]
    return t

def iqr_outliers(df, cols=None):
    rows = []
    for c in cols or FEATURES:
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        r = {"variable": c, "iqr": iqr}
        for k in (1.5, 3.0):
            r[f"n_outside_{k}xIQR"] = int(((s < q1 - k * iqr) | (s > q3 + k * iqr)).sum())
        rows.append(r)
    return pd.DataFrame(rows)

def past_due_profile(df):
    bins = [-1, 0, 1, 2, 3, 9, 89, np.inf]
    labels = ["0", "1", "2", "3", "4-9", "10-89", "90+ (special code?)"]
    parts = []
    for c in PAST_DUE:
        b = pd.cut(df[c], bins=bins, labels=labels).rename("bucket")
        g = df.groupby(b, observed=False)[TARGET].agg(n="size", default_rate="mean").reset_index()
        g.insert(0, "variable", c)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)

def past_due_overlap(df):
    gt0 = df[PAST_DUE] > 0
    flags = pd.DataFrame({
        "90+late_gt0": gt0[P90],
        "any_pastdue_gt0": gt0.any(axis=1),
        "90+late_gt0_but_30_59_and_60_89_zero": gt0[P90] & ~gt0[P30] & ~gt0[P60],
    })
    out = flags.groupby(df[TARGET].map({0: "non-default", 1: "default"})).mean()
    out.insert(0, "n", df.groupby(df[TARGET].map({0: "non-default", 1: "default"})).size())
    return out
