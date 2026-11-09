"""Single-applicant and batch scoring (Phase 8). Uses ONLY the frozen bundle: nothing is fitted here.
Invalid, ineligible or special-coded applications never receive a silent score.
EDUCATIONAL PROTOTYPE built on public data (not Bangladeshi bank data). Not for live lending decisions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import feature_engineering as fe
import scorecard as sc
import cutoff as co

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models" / "scorecard_m1_bundle_v2.json"
DISCLAIMER = ("Educational prototype built on a public dataset of unknown origin; it is not Bangladeshi bank data. "
              "Scores and zones are illustrative, financial assumptions are not observed values, and the model has not been "
              "independently validated or approved by any regulator. Not for live lending decisions.")

# key, source column, label, kind, low, high, required
FIELDS = [
    ("age", "age", "Age (years)", "int", 0, 120, True),
    ("monthly_income", "MonthlyIncome", "Monthly income", "float", 0, 1e9, False),
    ("debt_ratio", "DebtRatio", "Debt ratio", "float", 0, 1e7, False),
    ("utilisation", "RevolvingUtilizationOfUnsecuredLines", "Revolving utilisation", "float", 0, 1e6, False),
    ("open_credit_lines", "NumberOfOpenCreditLinesAndLoans", "Open credit lines and loans", "int", 0, 100, True),
    ("real_estate_loans", "NumberRealEstateLoansOrLines", "Real-estate loans and lines", "int", 0, 100, True),
    ("past_due_30_59", "NumberOfTime30-59DaysPastDueNotWorse", "Times 30-59 days past due", "int", 0, 98, True),
    ("past_due_60_89", "NumberOfTime60-89DaysPastDueNotWorse", "Times 60-89 days past due", "int", 0, 98, True),
    ("past_due_90plus", "NumberOfTimes90DaysLate", "Times 90+ days past due", "int", 0, 98, True),
    ("n_dependents", "NumberOfDependents", "Dependents (not used by the model)", "int", 0, 30, False),
]
RAW_COLS = {k: raw for k, raw, *_ in FIELDS}
VAR_LABELS = {"utilisation": "Revolving utilisation", "past_due_30_59": "30-59 days past due", "past_due_60_89": "60-89 days past due",
              "past_due_90plus": "90+ days past due", "debt_ratio": "Debt ratio", "real_estate_loans": "Real-estate loans"}
PD_KEYS = ("past_due_30_59", "past_due_60_89", "past_due_90plus")

def load_bundle(path=None):
    p_ = Path(path) if path else DEFAULT_BUNDLE
    if p_.suffix == ".json":
        import json
        b = json.loads(p_.read_text())
        b["card"]["points"] = {k: np.asarray(v, dtype=int) for k, v in b["card"]["points"].items()}
        b["card"]["points_exact"] = {k: np.asarray(v, dtype=float) for k, v in b["card"]["points_exact"].items()}
        for spec in b["specs"].values():
            spec["woe_by_code"] = np.asarray(spec["woe_by_code"], dtype=float)
    else:
        import joblib
        b = joblib.load(p_)
    missing = {"variables", "specs", "card", "master_scale", "risk_band_score_thresholds", "risk_band_labels", "status"} - set(b)
    if missing:
        raise KeyError(f"Bundle is missing keys: {sorted(missing)}")
    return b

def _parse(v, kind):
    """Return the value, or None if missing. Raise ValueError with a short message if invalid."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("not a number")
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if v == "":
            return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError("not a number")
    if np.isnan(x):
        return None
    if np.isinf(x):
        raise ValueError("not a finite number")
    if kind == "int" and x != int(x):
        raise ValueError("must be a whole number")
    return int(x) if kind == "int" else x

def validate_inputs(raw):
    clean, issues, warnings = {}, [], []
    for key, _, label, kind, lo, hi, req in FIELDS:
        try:
            v = _parse(raw.get(key), kind)
        except ValueError as e:
            issues.append(f"{label}: {e}"); clean[key] = None; continue
        if v is None:
            if req:
                issues.append(f"{label}: required")
        elif not (lo <= v <= hi):
            issues.append(f"{label}: must be between {lo:g} and {hi:g}"); v = None
        clean[key] = v
    named = lambda lab: any(i.startswith(lab) for i in issues)
    if clean.get("open_credit_lines") and clean.get("utilisation") is None and not named("Revolving utilisation"):
        issues.append("Revolving utilisation: required when the applicant has open credit lines")
    inc = clean.get("monthly_income")
    if inc is not None and inc > 0 and clean.get("debt_ratio") is None and not named("Debt ratio"):
        issues.append("Debt ratio: required when monthly income is above zero")
    re_, ol_ = clean.get("real_estate_loans"), clean.get("open_credit_lines")
    if re_ is not None and ol_ is not None and re_ > ol_:
        issues.append("Real-estate loans cannot exceed open credit lines and loans (open lines include real-estate loans)")
    if clean.get("age") is not None and clean["age"] > 100:
        warnings.append("Age above 100 is unusual: please check the entry.")
    return clean, issues, warnings

def _result(status, **kw):
    base = dict(status=status, zone=None, score=None, calibrated_pd=None, risk_band=None, base_points=None, breakdown=None,
                reasons=[], flags=[], issues=[], warnings=[], guidance="", override_applied=False, model_status=None, disclaimer=DISCLAIMER)
    base.update(kw)
    return base

def score_applicant(raw, bundle=None):
    bundle = bundle or load_bundle()
    clean, issues, warnings = validate_inputs(raw)
    if issues:
        return _result("INVALID_INPUT", issues=issues, warnings=warnings, guidance="No score: correct the listed fields.")
    if clean["age"] < 18:
        return _result("INELIGIBLE", warnings=warnings, guidance="Applicants under 18 are outside the modelling population: no score is produced.")
    if max(clean[k] for k in PD_KEYS) >= 90:
        return _result("REFERRED_SPECIAL_CODE", warnings=warnings,
                       guidance="Past-due counts of 90 or more are special codes in the source data, and such applicants defaulted at over 50% "
                                "in the development sample. The application is referred for manual review and is not scored.")
    inc = clean["monthly_income"]
    row = {RAW_COLS[k]: (np.nan if clean[k] is None else clean[k]) for k in RAW_COLS}
    if inc is None or inc <= 0:
        row[RAW_COLS["debt_ratio"]] = 0.0                      # not computable without income; the feature rule marks it so
    if clean["open_credit_lines"] == 0 and clean["utilisation"] is None:
        row[RAW_COLS["utilisation"]] = 0.0                     # not applicable without open lines
    try:
        feat = fe.engineer_features(pd.DataFrame([row]))
    except (KeyError, ValueError) as e:
        return _result("INVALID_INPUT", issues=[str(e)], warnings=warnings, guidance="No score: the application could not be processed.")

    card, specs, ms = bundle["card"], bundle["specs"], bundle["master_scale"]
    S = sc.score_features(feat, card, specs); score = int(S["score"].iloc[0])
    pd_cal = float(sc.master_scale_pd(np.array([score]), ms)[0])
    zone = str(co.final_zone(feat, np.array([score]))[0][0])
    pre = str(co.zone_of(np.array([score]), co.FINAL_THRESHOLDS["t_decline"], co.FINAL_THRESHOLDS["t_accept"])[0])
    thr = sorted(bundle["risk_band_score_thresholds"]); band = bundle["risk_band_labels"][int(np.digitize(score, thr))]

    rows = []
    for v in bundle["variables"]:
        code = int(sc.assign_codes(feat[v], specs[v]["splits"], specs[v]["specials"])[0])
        pts, best = int(card["points"][v][code]), int(card["points"][v].max())
        rows.append(dict(variable=VAR_LABELS.get(v, v), applicant_bin=specs[v]["labels"][code], points=pts, best_possible=best, points_lost=best - pts))
    bd = pd.DataFrame(rows)
    assert int(card["base_points"]) + int(bd["points"].sum()) == score, "Points breakdown does not add up to the score"
    reasons = [f"{r.variable}: bin '{r.applicant_bin}' scores {r.points} of {r.best_possible} possible points ({r.points_lost} below the best bin)"
               for r in bd[bd["points_lost"] > 0].nlargest(3, "points_lost", keep="all").itertuples()]

    flags = []
    if clean["open_credit_lines"] == 0:
        flags.append("No open credit lines: utilisation is not applicable and the model ranks this group weakly; the outcome is capped at Review.")
    if float(feat["utilisation"].iloc[0]) == fe.IMPLAUSIBLE:
        flags.append("Utilisation above 10 is implausible for a ratio: the outcome is capped at Review for a data check.")
    if inc is None or inc <= 0:
        flags.append("Income not available: the debt ratio could not be used and was treated as neutral (no reward or penalty).")
    t = co.FINAL_THRESHOLDS
    guidance = {"Accept": f"Illustrative outcome: approve at standard terms (score at or above {t['t_accept']}).",
                "Review": f"Illustrative outcome: refer to an underwriter or price for risk (score {t['t_decline']} to {t['t_accept'] - 1}, or capped by a data flag).",
                "Decline": f"Illustrative outcome: decline (score below {t['t_decline']})."}[zone]
    return _result("SCORED", zone=zone, score=score, calibrated_pd=pd_cal, risk_band=band, base_points=int(card["base_points"]), breakdown=bd,
                   reasons=reasons, flags=flags, warnings=warnings, guidance=guidance, override_applied=(pre == "Accept" and zone == "Review"),
                   model_status=bundle["status"])

def score_dataframe(raw_df, bundle=None):
    """Batch scoring of source-named rows that already satisfy the population rules."""
    bundle = bundle or load_bundle()
    feat = fe.engineer_features(raw_df)
    score = sc.score_features(feat, bundle["card"], bundle["specs"])["score"].to_numpy()
    zones, _ = co.final_zone(feat, score)
    thr = sorted(bundle["risk_band_score_thresholds"]); labels = np.array(bundle["risk_band_labels"], dtype=object)
    return pd.DataFrame({"score": score, "calibrated_pd": sc.master_scale_pd(score, bundle["master_scale"]),
                         "zone": zones, "risk_band": labels[np.digitize(score, thr)]}, index=feat.index)
