"""Validation metrics (Phase 6). Positive class = default; LOWER score = riskier."""
from __future__ import annotations
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from statsmodels.stats.proportion import proportion_confint

EPS = 1e-6

def discrimination(y, score):
    y = np.asarray(y, int); risk = -np.asarray(score, float)
    auc = roc_auc_score(y, risk); fpr, tpr, _ = roc_curve(y, risk)
    return dict(auc=float(auc), gini=float(2 * auc - 1), ks=float(np.max(tpr - fpr)))

def bootstrap_discrimination(y, score, n_boot=300, seed=42):
    rng = np.random.default_rng(seed); y = np.asarray(y, int); score = np.asarray(score, float); n = len(y); rows = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n); d = discrimination(y[i], score[i]); rows.append((d["auc"], d["gini"], d["ks"]))
    return pd.DataFrame(rows, columns=["auc", "gini", "ks"])

def with_ci(point, boot):
    return pd.DataFrame({k: dict(estimate=point[k], ci_low=np.percentile(boot[k], 2.5), ci_high=np.percentile(boot[k], 97.5))
                         for k in ("auc", "gini", "ks")}).T

def gini_ci(y, risk, n_boot=300, seed=7):
    rng = np.random.default_rng(seed); y = np.asarray(y, int); risk = np.asarray(risk, float); n = len(y)
    g = [2 * roc_auc_score(y[i], risk[i]) - 1 for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return 2 * roc_auc_score(y, risk) - 1, float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))

def classification_at(y, score, threshold):
    y = np.asarray(y, int); decline = np.asarray(score, float) < threshold
    tp = int((decline & (y == 1)).sum()); fp = int((decline & (y == 0)).sum())
    fn = int((~decline & (y == 1)).sum()); tn = int((~decline & (y == 0)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else np.nan
    return dict(decline_if_score_below=threshold, decline_rate=decline.mean(), tp=tp, fp=fp, fn=fn, tn=tn,
                sensitivity=sens, specificity=spec, precision=prec, f1=f1, balanced_accuracy=(sens + spec) / 2,
                default_rate_accepted=fn / (fn + tn), default_rate_declined=prec)

def score_groups(score, n_groups=10, seed=42):
    """Group 1 = riskiest (lowest scores). Ties broken at random with a fixed seed."""
    rng = np.random.default_rng(seed); score = np.asarray(score, float); n = len(score)
    order = np.lexsort((rng.random(n), score)); g = np.empty(n, int); g[order] = (np.arange(n) * n_groups) // n
    return g + 1

def gains_table(y, score, n_groups=10, seed=42):
    y = np.asarray(y, int); df = pd.DataFrame({"group": score_groups(score, n_groups, seed), "y": y, "score": np.asarray(score, float)})
    t = df.groupby("group").agg(n=("y", "size"), defaults=("y", "sum"), min_score=("score", "min"), max_score=("score", "max"))
    t["default_rate"] = t["defaults"] / t["n"]; t["cum_pct_pop"] = t["n"].cumsum() / t["n"].sum()
    t["cum_pct_defaults"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["lift"] = t["default_rate"] / y.mean(); t["cum_lift"] = t["cum_pct_defaults"] / t["cum_pct_pop"]
    return t.reset_index()

def calibration_stats(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), EPS, 1 - EPS); lp = np.log(p / (1 - p)); prev = y.mean()
    sl = sm.Logit(y, sm.add_constant(pd.DataFrame({"lp": lp}))).fit(disp=0); sci = sl.conf_int()
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(); cc = np.asarray(citl.conf_int())[0]
    br = brier_score_loss(y, p)
    return dict(mean_pd=p.mean(), observed=prev, brier=br, brier_skill=1 - br / (prev * (1 - prev)),
                log_loss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                citl=float(citl.params[0]), citl_low=float(cc[0]), citl_high=float(cc[1]),
                slope=float(sl.params["lp"]), slope_low=float(sci.loc["lp", 0]), slope_high=float(sci.loc["lp", 1]))

def reliability_table(y, p, score, n_groups=10, seed=42):
    df = pd.DataFrame({"group": score_groups(score, n_groups, seed), "y": np.asarray(y, int), "p": np.asarray(p, float)})
    t = df.groupby("group").agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"), observed=("y", "mean"))
    t["ci_low"], t["ci_high"] = proportion_confint(t["defaults"], t["n"], method="wilson")
    t["pd_inside_ci"] = (t["mean_pd"] >= t["ci_low"]) & (t["mean_pd"] <= t["ci_high"])
    return t.reset_index()

def assign_band(score, thr_desc, labels_low_to_high):
    idx = np.digitize(np.asarray(score, float), np.array(sorted(thr_desc)))
    return np.array(labels_low_to_high, dtype=object)[idx]

def band_table(score, y, p, bands, order):
    df = pd.DataFrame({"band": pd.Categorical(bands, categories=order, ordered=True), "score": np.asarray(score, float),
                       "y": np.asarray(y, int), "p": np.asarray(p, float)})
    t = df.groupby("band", observed=False).agg(n=("y", "size"), defaults=("y", "sum"), observed=("y", "mean"),
                                               calibrated_pd=("p", "mean"), min_score=("score", "min"), max_score=("score", "max"))
    t["ci_low"], t["ci_high"] = proportion_confint(t["defaults"], t["n"], method="wilson")
    t["pd_inside_ci"] = (t["calibrated_pd"] >= t["ci_low"]) & (t["calibrated_pd"] <= t["ci_high"])
    t["share_pct"] = 100 * t["n"] / t["n"].sum()
    return t

def psi_counts(a, b, eps=1e-4):
    a = np.asarray(a, float); b = np.asarray(b, float)
    pa = np.clip(a / a.sum(), eps, None); pb = np.clip(b / b.sum(), eps, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())

def psi_score(ref_score, new_score, edges):
    k = len(edges) + 1
    return psi_counts(np.bincount(np.digitize(ref_score, edges), minlength=k), np.bincount(np.digitize(new_score, edges), minlength=k))

def segment_row(name, mask, y, score, p):
    y = np.asarray(y, int)[mask]; s = np.asarray(score, float)[mask]; p = np.asarray(p, float)[mask]
    d = int(y.sum()); lo, hi = proportion_confint(d, len(y), method="wilson") if len(y) else (np.nan, np.nan)
    g = (2 * roc_auc_score(y, -s) - 1) if (d >= 30 and (len(y) - d) >= 30) else np.nan
    return dict(segment=name, n=len(y), defaults=d, observed=d / len(y), obs_ci_low=lo, obs_ci_high=hi, mean_calibrated_pd=p.mean(),
                obs_over_pred=(d / len(y)) / p.mean(), gini=g)
