"""Data acquisition and validation utilities (Step 1.3)."""
from __future__ import annotations
import hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

SEED = 42
TARGET = "SeriousDlqin2yrs"
ID_COL = "row_id"
EXPECTED_COLUMNS = [
    ID_COL, TARGET, "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
EXPECTED_TRAIN_ROWS = 150_000   # to be CHECKED, not assumed

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def load_raw_training(path: Path) -> pd.DataFrame:
    """Read cs-training.csv without altering values; name the unnamed index column."""
    df = pd.read_csv(path)
    if str(df.columns[0]).startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: ID_COL})
    return df

def validate_training(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(name, ok, detail): checks.append({"check": name, "passed": bool(ok), "detail": detail})
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    add("expected columns present", not missing_cols, f"missing={missing_cols}")
    add("no unexpected columns", not extra_cols, f"extra={extra_cols}")
    add("row count equals expected", len(df) == EXPECTED_TRAIN_ROWS,
        f"observed={len(df):,}, expected={EXPECTED_TRAIN_ROWS:,}")
    if TARGET in df.columns:
        add("target has no nulls", df[TARGET].isna().sum() == 0,
            f"nulls={int(df[TARGET].isna().sum())}")
        add("target values subset of {0,1}", set(df[TARGET].dropna().unique()) <= {0, 1},
            f"values={sorted(df[TARGET].dropna().unique().tolist())}")
    if ID_COL in df.columns:
        add("row_id unique", df[ID_COL].is_unique, f"duplicated ids={int(df[ID_COL].duplicated().sum())}")
    feats = [c for c in df.columns if c != ID_COL]
    add("exact duplicate rows (excluding row_id)", True,
        f"count={int(df.duplicated(subset=feats).sum())} (informational; handled in Step 2.1)")
    return pd.DataFrame(checks)

# Descriptions recalled from the public competition dictionary: VERIFY against the official file.
_DICT = [
    (ID_COL, "identifier", "Row index in the raw file (unnamed first column).", "Not a predictor"),
    (TARGET, "target", "Person experienced 90 days past due delinquency or worse.", "1 = default, 0 = non-default"),
    ("RevolvingUtilizationOfUnsecuredLines", "predictor", "Total balance on credit cards and personal lines of credit (excluding real estate and installment debt) divided by the sum of credit limits.", "Ratio"),
    ("age", "predictor", "Age of borrower in years.", "Years. Sensitive attribute: policy review needed"),
    ("NumberOfTime30-59DaysPastDueNotWorse", "predictor", "Number of times borrower has been 30-59 days past due but no worse.", "Count. CHECK time window vs outcome window"),
    ("DebtRatio", "predictor", "Monthly debt payments, alimony and living costs divided by monthly gross income.", "Ratio"),
    ("MonthlyIncome", "predictor", "Monthly income.", "Currency unit undocumented"),
    ("NumberOfOpenCreditLinesAndLoans", "predictor", "Number of open loans and lines of credit.", "Count"),
    ("NumberOfTimes90DaysLate", "predictor", "Number of times borrower has been 90 days or more past due.", "Count. CHECK time window vs outcome window (leakage risk)"),
    ("NumberRealEstateLoansOrLines", "predictor", "Number of mortgage and real estate loans including home equity lines of credit.", "Count"),
    ("NumberOfTime60-89DaysPastDueNotWorse", "predictor", "Number of times borrower has been 60-89 days past due but no worse.", "Count. CHECK time window vs outcome window"),
    ("NumberOfDependents", "predictor", "Number of dependents in family excluding themselves.", "Count"),
]
def build_data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(_DICT, columns=["variable", "role", "description", "notes"]).assign(
        description_source="recalled from public dictionary; verify against official Data Dictionary file")

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append({"variable": c, "dtype": str(s.dtype), "n_non_null": int(s.notna().sum()),
                     "n_missing": int(s.isna().sum()), "pct_missing": round(100 * s.isna().mean(), 3),
                     "n_unique": int(s.nunique(dropna=True)),
                     "min": s.min() if num else None, "max": s.max() if num else None})
    return pd.DataFrame(rows)

def write_manifest(raw_dir: Path, out_path: Path, source_url: str) -> dict:
    files = sorted(p for p in raw_dir.iterdir() if p.is_file())
    info = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": source_url,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
    }
    out_path.write_text(json.dumps(info, indent=2))
    return info


# ---------------- Step 2.1: development / validation / test split ----------------
from sklearn.model_selection import train_test_split

def make_split(df: pd.DataFrame, seed: int = SEED,
               dev: float = 0.60, val: float = 0.20, test: float = 0.20) -> pd.DataFrame:
    """Stratified (on default) three-way split. NOT out-of-time: the data has no dates."""
    assert abs(dev + val + test - 1.0) < 1e-9
    ids, y = df[ID_COL], df[TARGET]
    ids_dev, ids_tmp, _, y_tmp = train_test_split(ids, y, train_size=dev, stratify=y, random_state=seed)
    ids_val, ids_test = train_test_split(ids_tmp, train_size=val / (val + test),
                                         stratify=y_tmp, random_state=seed)
    out = pd.concat([
        pd.DataFrame({ID_COL: ids_dev.values, "sample": "development"}),
        pd.DataFrame({ID_COL: ids_val.values, "sample": "validation"}),
        pd.DataFrame({ID_COL: ids_test.values, "sample": "test"}),
    ])
    return out.sort_values(ID_COL).reset_index(drop=True)

def load_or_create_split(df: pd.DataFrame, path) -> pd.DataFrame:
    """The split is created once and then frozen on disk."""
    path = Path(path)
    if path.exists():
        split = pd.read_csv(path)
        assert len(split) == len(df) and set(split[ID_COL]) == set(df[ID_COL]), "Saved split does not match data"
        return split
    split = make_split(df)
    split.to_csv(path, index=False)
    return split

def summarise_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(split, on=ID_COL, how="left", validate="one_to_one")
    s = m.groupby("sample")[TARGET].agg(n="size", defaults="sum", default_rate="mean")
    s["share_of_rows"] = s["n"] / len(m)
    return s.loc[["development", "validation", "test"]]

def cross_split_duplicates(df: pd.DataFrame, split: pd.DataFrame) -> pd.Series:
    """Integrity count only: identical feature rows landing in more than one sample."""
    m = df.merge(split, on=ID_COL, validate="one_to_one")
    feats = [c for c in df.columns if c != ID_COL]
    m["n_samples"] = m.groupby(feats, dropna=False)["sample"].transform("nunique")
    return pd.Series({"rows_whose_feature_pattern_appears_in_>1_sample": int((m["n_samples"] > 1).sum())})

def save_samples(df: pd.DataFrame, split: pd.DataFrame, out_dir) -> None:
    out_dir = Path(out_dir)
    m = df.merge(split, on=ID_COL, validate="one_to_one")
    for name in ["development", "validation", "test"]:
        m[m["sample"] == name].drop(columns="sample").to_csv(out_dir / f"{name}.csv", index=False)


# ---------------- Step 2.2: population rules (feature-based; never uses the target) ----------------
PAST_DUE_COLS = ["NumberOfTime30-59DaysPastDueNotWorse",
                 "NumberOfTime60-89DaysPastDueNotWorse",
                 "NumberOfTimes90DaysLate"]

def apply_population_rules(df: pd.DataFrame):
    """Return (kept, excluded). Rules: age >= 18; no special-code (>=90) past-due values."""
    reason = pd.Series("", index=df.index, dtype=object)
    reason[df["age"] < 18] = "age < 18"
    special = (df[PAST_DUE_COLS] >= 90).any(axis=1)
    reason[special & (reason == "")] = "special code (>=90) in past-due counts"
    keep = reason == ""
    excluded = df.loc[~keep].assign(exclusion_reason=reason[~keep])
    return df.loc[keep].copy(), excluded
