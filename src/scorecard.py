"""WoE / IV binning utilities (Step 3.2). All bins are FITTED on the development sample only.

WoE convention: WoE_i = ln( %non-default_i / %default_i )  (higher WoE = lower risk).
Smoothing: additive 0.5 on every non-empty bin's good and bad counts:
  %good_i = (g_i + 0.5)/(G + 0.5*k),  %bad_i = (b_i + 0.5)/(B + 0.5*k),  k = number of non-empty bins.
Empty bins get WoE = 0 (neutral).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
SMOOTHING = 0.5

_CNT = dict(specials=[], names={}, integer=True)
BIN_CONFIG = {
    "age":               dict(trend="descending", min_bin_size=0.05,  max_bins=8, **_CNT),
    "monthly_income":    dict(trend="descending", min_bin_size=0.04,  max_bins=8, specials=[0], names={0: "zero income"}, integer=False),
    "n_dependents":      dict(trend="ascending",  min_bin_size=0.02,  max_bins=6, **_CNT),
    "open_credit_lines": dict(trend="valley",     min_bin_size=0.03,  max_bins=8, specials=[0], names={0: "no open lines"}, integer=True),
    "real_estate_loans": dict(trend="valley",     min_bin_size=0.02,  max_bins=6, **_CNT),
    "past_due_30_59":    dict(trend="ascending",  min_bin_size=0.005, max_bins=6, **_CNT),
    "past_due_60_89":    dict(trend="ascending",  min_bin_size=0.005, max_bins=6, **_CNT),
    "past_due_90plus":   dict(trend="ascending",  min_bin_size=0.005, max_bins=6, **_CNT),
    "utilisation":       dict(trend="ascending",  min_bin_size=0.03,  max_bins=8, specials=[-1, -2, 0],
                              names={-1: "not applicable (no open lines)", -2: "implausible (>10)", 0: "zero utilisation"}, integer=False),
    "debt_ratio":        dict(trend="ascending",  min_bin_size=0.03,  max_bins=8, specials=[-1],
                              names={-1: "not computable (income missing/zero)"}, integer=False),
    "total_past_due":    dict(trend="ascending",  min_bin_size=0.005, max_bins=8, **_CNT),
    "worst_delinquency": dict(trend="ascending",  min_bin_size=0.005, max_bins=4, **_CNT),
    "open_lines_non_re": dict(trend="valley",     min_bin_size=0.03,  max_bins=8, **_CNT),
}

def _fmt(v, integer):
    if integer:
        return f"{int(v)}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"

def make_labels(splits, specials, names, integer, vmin):
    s = list(splits); labels = []
    if not s:
        labels.append("all valid values")
    else:
        for i in range(len(s) + 1):
            if integer:
                lo = int(np.ceil(s[i - 1])) if i > 0 else int(vmin)
                hi = int(np.ceil(s[i])) - 1 if i < len(s) else None
                labels.append(f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}"))
            else:
                if i == 0:            labels.append(f"< {_fmt(s[0], False)}")
                elif i == len(s):     labels.append(f">= {_fmt(s[-1], False)}")
                else:                 labels.append(f"[{_fmt(s[i-1], False)}, {_fmt(s[i], False)})")
    labels += [f"Special: {names.get(c, c)}" for c in specials]
    labels.append("Missing")
    return labels

def assign_codes(x, splits, specials):
    """Bin code per row: 0..len(splits) regular (left-closed intervals), then specials, then Missing (last)."""
    v = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    n_reg = len(splits) + 1
    codes = np.searchsorted(np.asarray(splits, dtype=float), v, side="right")
    for j, s in enumerate(specials):
        codes[v == s] = n_reg + j
    codes[np.isnan(v)] = n_reg + len(specials)
    return codes

def bin_table(x, y, variable, splits, specials, names, integer, smoothing=SMOOTHING):
    x = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
    yv = np.asarray(y, dtype=int)
    codes = assign_codes(x, splits, specials)
    vm = x[~x.isna() & ~x.isin(specials)]
    labels = make_labels(splits, specials, names, integer, float(vm.min()) if len(vm) else 0.0)
    K = len(labels)
    n = np.bincount(codes, minlength=K).astype(float)
    d = np.bincount(codes, weights=yv, minlength=K)
    g = n - d
    ne = n > 0
    k = int(ne.sum()); G, B = g.sum(), d.sum()
    pg = (g + smoothing) / (G + smoothing * k)
    pb = (d + smoothing) / (B + smoothing * k)
    woe = np.where(ne, np.log(pg / pb), 0.0)
    ivc = np.where(ne, (pg - pb) * woe, 0.0)
    safe_n = np.where(ne, n, 1.0)
    tab = pd.DataFrame({
        "predictor": variable, "bin_code": np.arange(K), "bin": labels,
        "n": n.astype(int), "defaults": d.astype(int), "non_defaults": g.astype(int),
        "pct_population": n / n.sum(),
        "default_rate": np.where(ne, d / safe_n, np.nan),
        "non_default_rate": np.where(ne, g / safe_n, np.nan),
        "pct_of_defaults": d / B, "pct_of_non_defaults": g / G,
        "woe": woe, "iv_contribution": ivc})
    tab["total_iv"] = ivc.sum()
    return tab[ne].reset_index(drop=True), woe, ne, labels

def fine_splits(x_valid, n_bins=20):
    u = np.sort(np.unique(x_valid))
    if len(u) <= n_bins:
        return [float(s) for s in (u[:-1] + u[1:]) / 2]
    q = np.quantile(x_valid, np.linspace(0, 1, n_bins + 1)[1:-1])
    return [float(s) for s in np.unique(q)]

def check_shape(rates, trend):
    r = np.asarray(rates, dtype=float)
    if len(r) < 2:
        return True
    d = np.diff(r); tol = 1e-12
    if trend == "ascending":  return bool((d >= -tol).all())
    if trend == "descending": return bool((d <= tol).all())
    if trend == "valley":
        i = int(np.argmin(r))
        return bool((np.diff(r[:i + 1]) <= tol).all() and (np.diff(r[i:]) >= -tol).all())
    return True

def fit_variable(x, y, variable, cfg):
    """Fit bins for one variable on the supplied (development) data."""
    x = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
    yy = np.asarray(y, dtype=int)
    specials = cfg["specials"]
    valid = (x.notna() & ~x.isin(specials)).to_numpy()
    assert (x[valid] >= 0).all(), f"{variable}: unhandled negative sentinel values"
    xv, yv = x[valid].to_numpy(float), yy[valid]

    from optbinning import OptimalBinning   # imported lazily: only needed when FITTING bins
    status, splits = None, []
    for solver in ("cp", "mip"):
        try:
            ob = OptimalBinning(name=variable, dtype="numerical", solver=solver,
                                monotonic_trend=cfg["trend"], min_bin_size=cfg["min_bin_size"],
                                max_n_bins=cfg["max_bins"], max_n_prebins=30,
                                min_prebin_size=min(0.05, cfg["min_bin_size"] / 2))
            ob.fit(xv, yv); status = ob.status
            if status in ("OPTIMAL", "FEASIBLE"):
                splits = [float(s) for s in ob.splits]; break
        except Exception as e:
            status = f"error: {e}"

    tab, woe, ne, labels = bin_table(x, yy, variable, splits, specials, cfg["names"], cfg["integer"])
    ftab, _, _, _ = bin_table(x, yy, variable, fine_splits(xv), specials, cfg["names"], cfg["integer"])
    is_reg = ~tab["bin"].str.startswith("Special:") & (tab["bin"] != "Missing")
    spec = dict(variable=variable, splits=splits, specials=list(specials), names=cfg["names"],
                integer=cfg["integer"], labels=labels, woe_by_code=[float(w) for w in woe],
                empty_codes=[i for i, e in enumerate(ne) if not e], trend=cfg["trend"], config=cfg)
    iv_c, iv_f = float(tab["total_iv"].iloc[0]), float(ftab["total_iv"].iloc[0])
    info = dict(variable=variable, solver_status=status, n_bins=len(tab), n_regular_bins=int(is_reg.sum()),
                iv_fine=iv_f, iv_coarse=iv_c, iv_retained_pct=100 * iv_c / iv_f if iv_f > 0 else np.nan,
                trend=cfg["trend"], shape_ok=check_shape(tab.loc[is_reg, "default_rate"].to_numpy(), cfg["trend"]),
                min_bin_share=float(tab["pct_population"].min()), min_bin_defaults=int(tab["defaults"].min()))
    return spec, tab, ftab, info

def apply_woe(df, specs, prefix="woe_"):
    """Transform engineered features to WoE using FITTED specs. Unseen/empty bins map to WoE = 0."""
    out = pd.DataFrame(index=df.index)
    for var, s in specs.items():
        codes = assign_codes(df[var], s["splits"], s["specials"])
        out[prefix + var] = np.asarray(s["woe_by_code"])[codes]
    return out

def iv_identity_check(woe_col, y):
    """Identity: IV ~ E[WoE | non-default] - E[WoE | default]; an independent check of table and transform."""
    woe_col, y = np.asarray(woe_col), np.asarray(y)
    return float(woe_col[y == 0].mean() - woe_col[y == 1].mean())



# ---------------- Step 3.2b overrides (changes made after reviewing development results) ----------------
BIN_CONFIG["utilisation"].update(min_bin_size=0.015, max_bins=9,
                                 names={-1: "no open lines (n/a)", -2: "implausible >10", 0: "zero"})
BIN_CONFIG["debt_ratio"].update(names={-1: "not computable"})

# ---------------- Step 3.3 helpers: IV stability and characteristic stability ----------------
def iv_from_counts(n, d, smoothing=SMOOTHING):
    n = np.asarray(n, float); d = np.asarray(d, float); g = n - d
    ne = n > 0; k = int(ne.sum()); G, B = g.sum(), d.sum()
    pg = (g + smoothing) / (G + smoothing * k); pb = (d + smoothing) / (B + smoothing * k)
    return float(np.where(ne, (pg - pb) * np.log(pg / pb), 0.0).sum())

def iv_bootstrap(codes, y, K, n_boot=300, seed=SEED if "SEED" in globals() else 42):
    rng = np.random.default_rng(seed); y = np.asarray(y, float); n = len(y); out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n); c = codes[idx]
        out[b] = iv_from_counts(np.bincount(c, minlength=K), np.bincount(c, weights=y[idx], minlength=K))
    return out

def psi_from_codes(codes_a, codes_b, K, eps=1e-4):
    a = np.clip(np.bincount(codes_a, minlength=K) / len(codes_a), eps, None)
    b = np.clip(np.bincount(codes_b, minlength=K) / len(codes_b), eps, None)
    return float(((b - a) * np.log(b / a)).sum())
# ---------------- Step 3.3b: refit changes (development evidence only) ----------------
BIN_CONFIG["monthly_income"].update(specials=[], names={})        # zero income now arrives as missing
BIN_CONFIG["utilisation"]["neutral"] = [-2]                        # thin bin (<50 dev defaults): neutral WoE

def apply_neutral(spec, tab, cfg):
    """Force WoE = 0 on listed special bins (thin-bin rule) and recompute that bin's IV columns."""
    neutral = cfg.get("neutral", [])
    if not neutral:
        return spec, tab
    n_reg = len(spec["splits"]) + 1
    for sv in neutral:
        code = n_reg + spec["specials"].index(sv)
        spec["woe_by_code"][code] = 0.0
        tab.loc[tab["bin_code"] == code, ["woe", "iv_contribution"]] = 0.0
    spec["neutral_codes"] = list(neutral)
    tab["total_iv"] = tab["iv_contribution"].sum()
    return spec, tab

BIN_CONFIG["debt_ratio"]["neutral"] = [-1]    # Step 4.1 decision: "not computable" gets neutral WoE


# ---------------- Phase 5: points scaling and scorecard construction ----------------
BASE_SCORE, BASE_ODDS, PDO = 600.0, 50.0, 20.0     # illustrative convention; odds = non-default : default

def scaling(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return dict(base_score=base_score, base_odds=base_odds, pdo=pdo, factor=factor, offset=offset)

def pd_from_score(score, s):
    odds_good = s["base_odds"] * 2.0 ** ((np.asarray(score, float) - s["base_score"]) / s["pdo"])
    return 1.0 / (1.0 + odds_good)

def score_from_pd(pd_, s):
    pd_ = np.asarray(pd_, float)
    return s["offset"] + s["factor"] * np.log((1.0 - pd_) / pd_)

def build_scorecard(params, variables, specs, s):
    """params: statsmodels MLE for log-odds of DEFAULT ('const', 'woe_<var>').
    Score = (offset - factor*b0) + sum_i(-factor*b_i*WoE_ij); per-variable minimum shifted to 0."""
    f = s["factor"]
    base_exact = s["offset"] - f * params["const"]
    pts_exact = {}
    for v in variables:
        raw = -f * params["woe_" + v] * np.asarray(specs[v]["woe_by_code"], float)
        m = raw.min()
        pts_exact[v] = raw - m
        base_exact += m
    return dict(variables=list(variables), scaling=s, base_exact=float(base_exact),
                base_points=int(np.round(base_exact)), points_exact=pts_exact,
                points={v: np.round(a).astype(int) for v, a in pts_exact.items()})

def score_features(feat, card, specs):
    out = pd.DataFrame(index=feat.index)
    exact = np.full(len(feat), card["base_exact"], float)
    pts = np.full(len(feat), float(card["base_points"]))
    for v in card["variables"]:
        codes = assign_codes(feat[v], specs[v]["splits"], specs[v]["specials"])
        exact += card["points_exact"][v][codes]
        p = card["points"][v][codes]; pts += p; out["pts_" + v] = p
    out["score_exact"] = exact
    out["score"] = pts.astype(int)
    return out

def scorecard_table(card, coarse):
    rows = [dict(predictor="(base points)", bin="", n=np.nan, defaults=np.nan, default_rate=np.nan,
                 woe=np.nan, points=card["base_points"], points_unrounded=card["base_exact"])]
    for v in card["variables"]:
        for _, r in coarse[coarse.predictor == v].iterrows():
            c = int(r["bin_code"])
            rows.append(dict(predictor=v, bin=r["bin"], n=r["n"], defaults=r["defaults"], default_rate=r["default_rate"],
                             woe=r["woe"], points=int(card["points"][v][c]), points_unrounded=float(card["points_exact"][v][c])))
    return pd.DataFrame(rows)


# ---------------- Step 5c: monotone master scale (score -> calibrated PD) ----------------
def fit_master_scale(scores, y, center=580.0, scale=20.0):
    """logit(PD) = a + b*x + c*x^2, x = (score - center)/scale, fitted by ML on DEVELOPMENT scores only."""
    import statsmodels.api as sm
    x = (np.asarray(scores, float) - center) / scale
    X = sm.add_constant(pd.DataFrame({"x": x, "x2": x ** 2}))
    r = sm.Logit(np.asarray(y, int), X).fit(disp=0)
    a, b, c = (float(r.params[k]) for k in ("const", "x", "x2"))
    if c >= 0:
        raise ValueError("Quadratic term is not negative: the turning-point logic does not apply")
    x_star = -b / (2.0 * c)
    return dict(a=a, b=b, c=c, center=center, scale=scale, x_star=float(x_star), score_star=float(center + scale * x_star))

def master_scale_pd(scores, ms):
    """Calibrated PD. Below the turning point the PD is held constant, so PD never rises as the score falls."""
    s_eff = np.maximum(np.asarray(scores, float), ms["score_star"])
    x = (s_eff - ms["center"]) / ms["scale"]
    return 1.0 / (1.0 + np.exp(-(ms["a"] + ms["b"] * x + ms["c"] * x ** 2)))

def score_thresholds(pd_cuts, ms, lo=300, hi=700):
    """Smallest integer score whose calibrated PD is <= each cut (cuts ascending -> thresholds descending)."""
    grid = np.arange(lo, hi + 1); p = master_scale_pd(grid, ms); out = []
    for c in pd_cuts:
        ok = np.where(p <= c)[0]
        if len(ok) == 0:
            raise ValueError(f"No score in [{lo},{hi}] reaches PD <= {c}")
        out.append(int(grid[ok[0]]))
    return out
