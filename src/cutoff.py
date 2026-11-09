"""Cut-off and business decision analysis (Phase 7). Accept if score >= threshold.
ALL financial parameters are ILLUSTRATIVE assumptions, normalised to EAD = 1 per applicant."""
from __future__ import annotations
import numpy as np, pandas as pd

SCENARIOS = {
    "Tight margin / high loss": dict(R=0.06, LGD=0.75),
    "Base": dict(R=0.10, LGD=0.65),
    "Wide margin / lower loss": dict(R=0.16, LGD=0.55),
}
BB_RATES = {"20% (substandard)": 0.20, "50% (doubtful)": 0.50, "100% (bad/loss)": 1.00}

def breakeven_pd(R, lgd):
    return R / (R + lgd)

def decision_table(score, y, pd_cal, thresholds, n_excl=0, d_excl=0):
    score = np.asarray(score, float); y = np.asarray(y, int); p = np.asarray(pd_cal, float)
    N, D = len(y), int(y.sum()); G = N - D; N_all, D_all = N + n_excl, D + d_excl
    rows = []
    for t in thresholds:
        acc = score >= t; n_a = int(acc.sum()); d_a = int(y[acc].sum()); g_a = n_a - d_a
        sens = (D - d_a) / D; spec = g_a / G
        prec = (D - d_a) / (N - n_a) if n_a < N else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (prec == prec and prec + sens > 0) else np.nan
        row = dict(threshold=t, accept_rate_scored=n_a / N, reject_rate_scored=1 - n_a / N, accept_rate_all_applications=n_a / N_all,
                   accepted=n_a, defaults_among_accepted=d_a, default_rate_accepted=d_a / n_a if n_a else np.nan,
                   calibrated_pd_accepted=p[acc].mean() if n_a else np.nan, defaults_caught=sens,
                   defaults_caught_incl_policy_referrals=(D_all - d_a) / D_all, good_customers_lost=1 - spec,
                   reject_precision=prec, f1=f1, balanced_accuracy=(sens + spec) / 2)
        for k, r in BB_RATES.items():
            row[f"provision_per_100_applied|{k}"] = 100 * d_a * r / N_all
        rows.append(row)
    return pd.DataFrame(rows)

def profit_table(score, y, pd_cal, thresholds, n_excl=0, scenarios=SCENARIOS):
    score = np.asarray(score, float); y = np.asarray(y, int); p = np.asarray(pd_cal, float); N_all = len(y) + n_excl
    rows = []
    for name, s in scenarios.items():
        R, L = s["R"], s["LGD"]
        for t in thresholds:
            acc = score >= t; n_a = int(acc.sum()); d_a = int(y[acc].sum()); g_a = n_a - d_a; pe = p[acc]
            rows.append(dict(scenario=name, R=R, LGD=L, breakeven_pd=breakeven_pd(R, L), threshold=t,
                             profit_obs_per_100=100 * (g_a * R - d_a * L) / N_all,
                             profit_pd_based_per_100=100 * ((1 - pe).sum() * R - pe.sum() * L) / N_all,
                             exp_loss_obs_per_100=100 * d_a * L / N_all, exp_loss_pd_based_per_100=100 * pe.sum() * L / N_all))
    return pd.DataFrame(rows)

def bootstrap_optimal_threshold(score, y, grid, scenarios=SCENARIOS, n_boot=300, seed=42, smin=300, size=401):
    """Profit-maximising threshold in each bootstrap resample of the applicants (integer scores 300..700)."""
    rng = np.random.default_rng(seed); score = np.asarray(score).astype(int); y = np.asarray(y, int); n = len(y)
    idx = np.asarray(grid).astype(int) - smin; out = {k: [] for k in scenarios}
    for _ in range(n_boot):
        i = rng.integers(0, n, n); s = score[i] - smin; yy = y[i]
        bad = np.bincount(s[yy == 1], minlength=size); good = np.bincount(s[yy == 0], minlength=size)
        cg = np.cumsum(good[::-1])[::-1]; cb = np.cumsum(bad[::-1])[::-1]       # accepted if score >= t
        for k, sc_ in scenarios.items():
            out[k].append(int(grid[np.argmax(cg[idx] * sc_["R"] - cb[idx] * sc_["LGD"])]))
    return pd.DataFrame(out)

def flat_region(profit_by_threshold, tol=0.02):
    m = profit_by_threshold.max(); ok = profit_by_threshold[profit_by_threshold >= (1 - tol) * m]
    return int(ok.index.min()), int(ok.index.max())

def risk_appetite_cutoffs(table, targets):
    rows = []
    for tg in targets:
        ok = table[table["default_rate_accepted"] <= tg]
        if ok.empty: rows.append(dict(max_default_rate_accepted=tg, threshold=np.nan)); continue
        r = ok.sort_values("threshold").iloc[0]
        rows.append(dict(max_default_rate_accepted=tg, threshold=int(r["threshold"]), accept_rate_scored=r["accept_rate_scored"],
                         accept_rate_all_applications=r["accept_rate_all_applications"], default_rate_accepted=r["default_rate_accepted"],
                         defaults_caught=r["defaults_caught"]))
    return pd.DataFrame(rows)


# ---------------- Phase 7b: three-zone policy evaluation ----------------
def zone_of(score, t_decline, t_accept):
    s = np.asarray(score, float)
    return np.where(s >= t_accept, "Accept", np.where(s >= t_decline, "Review", "Decline"))

def zone_table(score, y, pd_cal, t_decline, t_accept, n_excl=0, d_excl=0, scenarios=SCENARIOS):
    score = np.asarray(score, float); y = np.asarray(y, int); p = np.asarray(pd_cal, float)
    z = zone_of(score, t_decline, t_accept); N_all = len(y) + n_excl; rows = []
    for name in ("Accept", "Review", "Decline"):
        m = z == name; n = int(m.sum()); d = int(y[m].sum()); g = n - d
        row = dict(zone=name, n=n, share_of_scored=n / len(y), share_of_all_applications=n / N_all, defaults=d,
                   default_rate=d / n if n else np.nan, mean_calibrated_pd=p[m].mean() if n else np.nan,
                   min_score=score[m].min() if n else np.nan, max_score=score[m].max() if n else np.nan)
        for k, s in scenarios.items():
            row[f"profit_if_approved_per_100_applied|{k}"] = 100 * (g * s["R"] - d * s["LGD"]) / N_all
        rows.append(row)
    rows.append(dict(zone="Policy referral (special codes)", n=n_excl, share_of_all_applications=n_excl / N_all,
                     defaults=d_excl, default_rate=d_excl / n_excl if n_excl else np.nan))
    return pd.DataFrame(rows)

def segment_check(score, y, mask, t_decline, t_accept, scenarios=SCENARIOS):
    score = np.asarray(score, float)[mask]; y = np.asarray(y, int)[mask]; n = len(y); d = int(y.sum())
    z = zone_of(score, t_decline, t_accept); acc = z == "Accept"
    row = dict(n=n, defaults=d, default_rate=d / n, share_in_accept_zone=acc.mean(), share_in_review_zone=(z == "Review").mean(),
               share_in_decline_zone=(z == "Decline").mean(), n_in_accept_zone=int(acc.sum()),
               default_rate_in_accept_zone=y[acc].mean() if acc.any() else np.nan)
    for k, s in scenarios.items():
        row[f"profit_per_100_applicants_if_all_approved|{k}"] = 100 * ((n - d) * s["R"] - d * s["LGD"]) / n
    return row


# ---------------- Phase 7b-B: final policy ----------------
FINAL_THRESHOLDS = dict(t_decline=529, t_accept=563)    # fixed from break-even PDs; never tuned on any sample

def final_zone(feat, score, t_decline=FINAL_THRESHOLDS["t_decline"], t_accept=FINAL_THRESHOLDS["t_accept"]):
    """Zone per scored applicant. Overrides: no open credit lines or implausible utilisation -> at most Review."""
    z = zone_of(score, t_decline, t_accept).astype(object)
    flagged = np.asarray(feat["open_credit_lines"] == 0) | np.asarray(feat["utilisation"] == -2)
    moved = (z == "Accept") & flagged
    return np.where(moved, "Review", z), int(moved.sum())

def zone_summary(zones, y, p, n_excl, d_excl, scenarios=SCENARIOS):
    from statsmodels.stats.proportion import proportion_confint
    y = np.asarray(y, int); p = np.asarray(p, float); zones = np.asarray(zones, object); N_all = len(y) + n_excl
    lgds = sorted({s["LGD"] for s in scenarios.values()}); rows = []
    for name in ("Accept", "Review", "Decline"):
        m = zones == name; n = int(m.sum()); d = int(y[m].sum()); g = n - d
        lo, hi = proportion_confint(d, n, method="wilson") if n else (np.nan, np.nan)
        rate = d / n if n else np.nan; mp = p[m].mean() if n else np.nan
        row = dict(zone=name, n=n, share_of_scored=n / len(y), share_of_all_applications=n / N_all, defaults=d, default_rate=rate,
                   ci_low=lo, ci_high=hi, mean_calibrated_pd=mp, pd_inside_ci=(bool(lo <= mp <= hi) if n else False))
        for k, s in scenarios.items(): row[f"profit_if_approved_per_100|{k}"] = 100 * (g * s["R"] - d * s["LGD"]) / N_all
        for L in lgds: row[f"required_net_margin|LGD={L}"] = rate / (1 - rate) * L if n else np.nan
        rows.append(row)
    rows.append(dict(zone="Referral (special codes / ineligible)", n=n_excl, share_of_all_applications=n_excl / N_all,
                     defaults=d_excl, default_rate=d_excl / n_excl if n_excl else np.nan))
    return pd.DataFrame(rows)

def policy_criteria(tab, scenarios=SCENARIOS):
    be = [breakeven_pd(s["R"], s["LGD"]) for s in scenarios.values()]; lo_be, hi_be = min(be), max(be); t = tab.set_index("zone")
    pk = lambda z, k: t.loc[z, f"profit_if_approved_per_100|{k}"]
    return pd.Series({
        f"C1 Accept-zone default-rate upper bound < {lo_be:.1%}": bool(t.loc["Accept", "ci_high"] < lo_be),
        f"C2 Decline-zone default-rate lower bound > {hi_be:.1%}": bool(t.loc["Decline", "ci_low"] > hi_be),
        "C3 Review zone <= 20% of scored applicants": bool(t.loc["Review", "share_of_scored"] <= 0.20),
        "C4 Accept profitable and Decline loss-making in all scenarios": bool(all(pk("Accept", k) > 0 and pk("Decline", k) < 0 for k in scenarios)),
        "C5 Calibrated PD inside observed 95% CI in all three zones": bool(t.loc[["Accept", "Review", "Decline"], "pd_inside_ci"].all())})
