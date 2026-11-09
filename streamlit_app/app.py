"""Retail credit risk scorecard: Streamlit dashboard (EDUCATIONAL PROTOTYPE).
Uses the frozen scoring engine (src/predict.py) and aggregate-only files in streamlit_app/data/.
Nothing is trained here, and no applicant-level data is shipped."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import cutoff as co      # noqa: E402
import predict           # noqa: E402

st.set_page_config(page_title="Credit risk scorecard (prototype)", page_icon=":bar_chart:", layout="wide")

C = dict(main="#1F4E79", accent="#C0504D", mid="#5B9BD5", light="#9DC3E6", grey="#7F7F7F")
BAND_ORDER = ["Very low risk", "Low risk", "Moderate risk", "High risk", "Very high risk"]
PAGES = ["1. Executive overview", "2. Credit risk assessment", "3. Model performance", "4. Portfolio risk analysis", "5. Model monitoring"]
PSI_BINS = np.arange(420, 621, 20)
GRID = np.arange(300, 701)
CAL = "Illustrative assumptions: exposure is 1 for every applicant and margin/loss figures are not observed values. Outcomes come from previously accepted loans only (selection bias)."


# ------------------------------------------------------------------ data and helpers
@st.cache_resource
def get_bundle():
    return predict.load_bundle()


@st.cache_data
def load_data():
    f = lambda n: pd.read_csv(HERE / "data" / n)
    return dict(counts=f("score_counts.csv"), risk_bands=f("risk_bands.csv"), zones=f("policy_zones.csv"), rel=f("reliability_deciles.csv"),
                psi=f("psi.csv"), lookup=f("master_scale_lookup.csv"), summary=json.loads((HERE / "data" / "summary.json").read_text()),
                examples=json.loads((HERE / "data" / "examples.json").read_text()))


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def style(fig, title="", height=360):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=45, b=10), template="plotly_white", legend=dict(orientation="h", y=-0.22))
    return fig


def counts(data, sample):
    c = data["counts"]
    return c[c["sample"] == sample].sort_values("score").reset_index(drop=True)


def roc_gains(c):
    d, g = c["n_default"].to_numpy(float), c["n_nondefault"].to_numpy(float)
    td, tg = d.sum(), g.sum()
    tpr, fpr = np.r_[0.0, np.cumsum(d) / td], np.r_[0.0, np.cumsum(g) / tg]
    pop = np.r_[0.0, np.cumsum(d + g) / (td + tg)]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, pop, auc, float(np.max(np.abs(tpr - fpr)))


def confusion(c, t):
    s, d, g = c["score"].to_numpy(), c["n_default"].to_numpy(float), c["n_nondefault"].to_numpy(float)
    dec = s < t
    tp, fp, fn, tn = d[dec].sum(), g[dec].sum(), d[~dec].sum(), g[~dec].sum()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec == prec and prec + sens > 0) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sens=sens, spec=spec, prec=prec, f1=f1, bacc=(sens + spec) / 2)


def psi_from(a, b, eps=1e-4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, pb = np.clip(a / a.sum(), eps, None), np.clip(b / b.sum(), eps, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def binned(c, shift=0):
    tot = (c["n_default"] + c["n_nondefault"]).to_numpy(float)
    return np.bincount(np.digitize(c["score"].to_numpy() + shift, PSI_BINS), weights=tot, minlength=len(PSI_BINS) + 1)


def psi_status(v):
    return "Stable" if v < 0.10 else ("Investigate" if v < 0.25 else "Major shift")


def tail_counts(c):
    s = c["score"].to_numpy().astype(int)
    d, g = np.zeros(len(GRID)), np.zeros(len(GRID))
    d[s - 300], g[s - 300] = c["n_default"].to_numpy(), c["n_nondefault"].to_numpy()
    return d.sum(), g.sum(), np.cumsum(d[::-1])[::-1], np.cumsum(g[::-1])[::-1]     # tails: accepted if score >= t


# ------------------------------------------------------------------ page 1
def page_overview(data):
    S = data["summary"]
    perf = pd.DataFrame(S["performance"]).T
    st.title("Retail credit risk scorecard")
    st.caption("Development, validation and deployment of an explainable credit risk scorecard for retail lending")
    st.warning(predict.DISCLAIMER)
    st.subheader("Purpose")
    st.write("An application scorecard estimates the probability that a borrower reaches 90+ days past due within two years. "
             "This prototype shows the full workflow: Weight-of-Evidence logistic regression, points-based scorecard, calibrated PDs, "
             "validation, an accept / review / decline policy, and monitoring. It demonstrates a method on a public benchmark dataset.")
    disc, cal = pd.DataFrame(S["discrimination"]), pd.DataFrame(S["calibration"])
    g = disc[(disc["sample"] == "test") & (disc["metric"] == "gini")].iloc[0]
    k = disc[(disc["sample"] == "test") & (disc["metric"] == "ks")].iloc[0]
    ct = cal[cal["sample"] == "test"].iloc[0]
    cols = st.columns(5)
    cols[0].metric("Gini (test)", f"{g['estimate']:.3f}", help=f"95% bootstrap interval {g['ci_low']:.3f} to {g['ci_high']:.3f}")
    cols[1].metric("KS (test)", f"{k['estimate']:.3f}", help=f"95% bootstrap interval {k['ci_low']:.3f} to {k['ci_high']:.3f}")
    cols[2].metric("Calibration slope (test)", f"{ct['slope']:.3f}", help=f"95% interval {ct['slope_low']:.3f} to {ct['slope_high']:.3f}; ideal 1")
    cols[3].metric("Brier skill score (test)", f"{perf.loc['test', 'brier_skill']:.3f}", help="Improvement over predicting the overall default rate for everyone")
    cols[4].metric("Overall default rate", pct(S["samples"]["test"]["default_rate"], 2))
    st.subheader("Dataset")
    st.write(f"**{S['data_source']['name']}** ({S['data_source']['url']}). {S['data_source']['note']} Split: {S['split']}.")
    ds = pd.DataFrame(S["samples"]).T
    ds["default_rate"] = (100 * ds["default_rate"]).round(2).astype(str) + "%"
    ds = ds.rename(columns={"n_scored": "Scored applicants", "n_default": "Defaults", "default_rate": "Default rate",
                            "n_policy_referrals": "Referred by policy", "n_policy_referral_defaults": "Defaults among referred"})
    st.dataframe(ds)
    st.subheader("Decision policy on the test sample")
    z = data["zones"][data["zones"]["sample"] == "test"][["zone", "n", "share_of_scored", "default_rate", "mean_calibrated_pd"]].copy()
    for col in ("share_of_scored", "default_rate", "mean_calibrated_pd"):
        z[col] = z[col].map(lambda v: pct(v, 1))
    st.dataframe(z.rename(columns={"zone": "Zone", "n": "Applicants", "share_of_scored": "Share of scored", "default_rate": "Observed default rate", "mean_calibrated_pd": "Calibrated PD"}), hide_index=True)
    t = S["policy"]["thresholds"]
    st.caption(f"Accept at score {t['t_accept']} or above; review {t['t_decline']} to {t['t_accept'] - 1}; decline below {t['t_decline']}. Overrides: " + "; ".join(S["policy"]["overrides"]) + ".")
    st.subheader("Important limitations")
    st.markdown("- Not Bangladeshi bank data; the source dataset's country, period and lender are undocumented.\n"
                "- All outcomes come from loans an earlier lender accepted (selection bias; no reject inference).\n"
                "- Margin, loss and exposure figures are illustrative assumptions, not observed values.\n"
                "- Calibration holds overall and by risk band but not by group: older applicants are over-predicted and younger ones under-predicted.\n"
                "- The timing of the 90+ day delinquency count relative to the outcome window is undocumented.\n"
                "- Random split only: no out-of-time validation, so near-zero PSI reflects the split, not stability over time.\n"
                "- Not independently validated and not approved by any regulator.")


# ------------------------------------------------------------------ page 2
INPUT_DEFAULTS = dict(f_age=45, f_has_income=True, f_income=5000.0, f_debt=0.30, f_util=0.20, f_open=6, f_re=1, f_pd30=0, f_pd60=0, f_pd90=0, f_dep=2)


def apply_example():
    name = st.session_state.get("example_choice", "(custom)")
    ex = next((e for e in load_data()["examples"] if e["name"] == name), None)
    if ex is None:
        return
    i = ex["inputs"]
    st.session_state.update(
        f_age=int(i["age"]), f_has_income=i.get("monthly_income") is not None,
        f_income=float(i["monthly_income"]) if i.get("monthly_income") is not None else 0.0,
        f_debt=float(i["debt_ratio"]) if i.get("debt_ratio") is not None else 0.0,
        f_util=float(i["utilisation"]) if i.get("utilisation") is not None else 0.0,
        f_open=int(i["open_credit_lines"]), f_re=int(i["real_estate_loans"]), f_pd30=int(i["past_due_30_59"]),
        f_pd60=int(i["past_due_60_89"]), f_pd90=int(i["past_due_90plus"]), f_dep=int(i.get("n_dependents") or 0))


def page_assess(data, bundle):
    st.title("Credit risk assessment")
    st.warning(predict.DISCLAIMER)
    st.caption("Simulated applicants only: never enter real personal data. Enter values or load an example; the frozen engine returns the score, calibrated PD, risk band and an illustrative decision.")
    for k, v in INPUT_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.selectbox("Load a simulated example", ["(custom)"] + [e["name"] for e in data["examples"]], key="example_choice", on_change=apply_example)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Age (years)", min_value=0, max_value=120, step=1, key="f_age")
        has_income = st.checkbox("Monthly income provided", key="f_has_income")
        st.number_input("Monthly income", min_value=0.0, step=100.0, key="f_income", disabled=not has_income)
        st.number_input("Debt ratio (monthly debt payments / monthly income)", min_value=0.0, step=0.01, format="%.3f", key="f_debt", disabled=not has_income)
        st.number_input("Dependents (not used by the model)", min_value=0, max_value=30, step=1, key="f_dep")
    with c2:
        st.number_input("Open credit lines and loans (including real-estate loans)", min_value=0, max_value=100, step=1, key="f_open")
        st.number_input("Revolving utilisation (balance / credit limit)", min_value=0.0, step=0.01, format="%.3f", key="f_util", disabled=(st.session_state["f_open"] == 0))
        st.number_input("Real-estate loans and lines", min_value=0, max_value=100, step=1, key="f_re")
        st.number_input("Times 30-59 days past due", min_value=0, max_value=98, step=1, key="f_pd30")
        st.number_input("Times 60-89 days past due", min_value=0, max_value=98, step=1, key="f_pd60")
        st.number_input("Times 90+ days past due", min_value=0, max_value=98, step=1, key="f_pd90")
    inc = st.session_state["f_income"] if has_income else None
    payload = dict(age=st.session_state["f_age"], monthly_income=inc,
                   debt_ratio=(st.session_state["f_debt"] if (has_income and inc and inc > 0) else None),
                   utilisation=(st.session_state["f_util"] if st.session_state["f_open"] > 0 else None),
                   open_credit_lines=st.session_state["f_open"], real_estate_loans=st.session_state["f_re"], past_due_30_59=st.session_state["f_pd30"],
                   past_due_60_89=st.session_state["f_pd60"], past_due_90plus=st.session_state["f_pd90"], n_dependents=st.session_state["f_dep"])
    r = predict.score_applicant(payload, bundle)
    st.divider()
    for w in r["warnings"]:
        st.warning(w)
    if r["status"] == "INVALID_INPUT":
        st.error("No score. " + "; ".join(r["issues"]))
        return
    if r["status"] != "SCORED":
        st.warning(f"{r['status'].replace('_', ' ').title()}: {r['guidance']}")
        return
    m = st.columns(4)
    m[0].metric("Score", r["score"])
    m[1].metric("Calibrated PD", pct(r["calibrated_pd"], 2), help="Master-scale PD. Treat as reliable at band level; point PDs near the top of the score range are extrapolations.")
    m[2].metric("Risk band", r["risk_band"])
    m[3].metric("Decision zone", r["zone"])
    {"Accept": st.success, "Review": st.warning, "Decline": st.error}[r["zone"]](r["guidance"])
    st.caption("The risk band describes the PD; the zone is the decision. They can differ when a data flag caps the outcome at Review.")
    for f in r["flags"]:
        st.info(f)
    st.markdown("**Main drivers of this score**")
    st.markdown("\n".join(f"- {x}" for x in r["reasons"]) if r["reasons"] else "- No variable is below its best bin.")
    bd = r["breakdown"]
    fig = go.Figure()
    fig.add_bar(y=bd["variable"], x=bd["best_possible"], orientation="h", name="Best possible", marker_color=C["light"])
    fig.add_bar(y=bd["variable"], x=bd["points"], orientation="h", name="This applicant", marker_color=C["main"])
    fig.update_layout(barmode="overlay", yaxis=dict(autorange="reversed"), xaxis_title="Points")
    left, right = st.columns(2)
    left.plotly_chart(style(fig, f"Points by variable (base points {r['base_points']})", 330))
    cv = counts(data, "validation")
    share = (cv["n_default"] + cv["n_nondefault"]) / (cv["n_default"] + cv["n_nondefault"]).sum()
    f2 = go.Figure(go.Bar(x=cv["score"], y=100 * share, marker_color=C["light"], name="Validation applicants"))
    f2.add_vline(x=r["score"], line_color=C["accent"], line_width=2)
    f2.update_layout(xaxis_title="Score", yaxis_title="% of applicants")
    right.plotly_chart(style(f2, "Where this score falls (validation sample)", 330))
    st.dataframe(bd.rename(columns={"variable": "Variable", "applicant_bin": "Applicant's bin", "points": "Points", "best_possible": "Best possible", "points_lost": "Points below best"}), hide_index=True)


# ------------------------------------------------------------------ page 3
def page_performance(data):
    S = data["summary"]
    st.title("Model performance")
    st.warning(predict.DISCLAIMER)
    sample = st.radio("Sample", ["validation", "test", "development"], horizontal=True, key="perf_sample")
    c = counts(data, sample)
    fpr, tpr, pop, auc, ks = roc_gains(c)
    stored = S["performance"][sample]
    m = st.columns(5)
    m[0].metric("AUC", f"{auc:.4f}")
    m[1].metric("Gini", f"{2 * auc - 1:.4f}")
    m[2].metric("KS", f"{ks:.4f}")
    m[3].metric("Brier score", f"{stored['brier']:.4f}")
    m[4].metric("Calibration slope", f"{stored['slope']:.3f}")
    st.caption(f"ROC, Gini and KS are recomputed here from per-score counts; the validation report gave AUC {stored['auc']:.4f}, Gini {stored['gini']:.4f}, KS {stored['ks']:.4f}. "
               "Test results are from a single evaluation of the frozen model. The split is random, not out-of-time.")
    t1, t2, t3, t4, t5 = st.tabs(["Discrimination", "Confusion matrix", "Calibration", "Score distribution", "All samples"])
    with t1:
        a, b = st.columns(2)
        f = go.Figure()
        f.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{sample} (AUC {auc:.3f})", line=dict(color=C["main"], width=2)))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=C["grey"], dash="dash")))
        f.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        a.plotly_chart(style(f, "ROC curve"))
        cd_, cg_ = np.cumsum(c["n_default"]) / c["n_default"].sum(), np.cumsum(c["n_nondefault"]) / c["n_nondefault"].sum()
        k_ = int(np.argmax(np.abs(cd_.to_numpy() - cg_.to_numpy())))
        f = go.Figure()
        f.add_trace(go.Scatter(x=c["score"], y=cd_, name="Defaults", line=dict(color=C["accent"], shape="hv")))
        f.add_trace(go.Scatter(x=c["score"], y=cg_, name="Non-defaults", line=dict(color=C["main"], shape="hv")))
        f.add_shape(type="line", x0=c["score"].iloc[k_], x1=c["score"].iloc[k_], y0=cg_.iloc[k_], y1=cd_.iloc[k_], line=dict(color="black", width=2))
        f.update_layout(xaxis_title="Score", yaxis_title="Cumulative share")
        b.plotly_chart(style(f, f"KS = {ks:.3f} at score {int(c['score'].iloc[k_])}"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=100 * pop, y=100 * tpr, mode="lines", name="Scorecard", line=dict(color=C["main"], width=2)))
        f.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random", line=dict(color=C["grey"], dash="dash")))
        f.update_layout(xaxis_title="% of applicants (riskiest first)", yaxis_title="% of defaults captured")
        st.plotly_chart(style(f, "Cumulative gains"))
    with t2:
        t = st.slider("Decline if score is below", 500, 626, 563, key="conf_t")
        cf = confusion(c, t)
        f = go.Figure(go.Heatmap(z=[[cf["tn"], cf["fp"]], [cf["fn"], cf["tp"]]], x=["Accepted", "Declined"], y=["Did not default", "Defaulted"],
                                 text=[[f"{cf['tn']:,.0f}", f"{cf['fp']:,.0f}"], [f"{cf['fn']:,.0f}", f"{cf['tp']:,.0f}"]], texttemplate="%{text}", colorscale="Blues", showscale=False))
        st.plotly_chart(style(f, f"Confusion matrix at cut-off {t} (positive class = default)", 330))
        m = st.columns(5)
        for col, (lab, key) in zip(m, [("Sensitivity (recall)", "sens"), ("Specificity", "spec"), ("Precision", "prec"), ("F1", "f1"), ("Balanced accuracy", "bacc")]):
            col.metric(lab, f"{cf[key]:.3f}")
        st.caption("Precision is low at loose cut-offs because only about 6.6% of applicants default (base-rate effect). Accuracy is not shown: accepting everyone is about 93% accurate.")
    with t3:
        rel = data["rel"][data["rel"]["sample"] == sample]
        a, b = st.columns(2)
        if rel.empty:
            a.info("Reliability deciles were exported for validation and test only.")
        else:
            lo, hi = (rel["observed"] - rel["ci_low"]).clip(lower=0), (rel["ci_high"] - rel["observed"]).clip(lower=0)
            lim = [float(rel[["mean_pd", "observed"]].min().min()) * 0.7, float(rel[["mean_pd", "observed"]].max().max()) * 1.4]
            f = go.Figure()
            f.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect calibration", line=dict(color=C["grey"], dash="dash")))
            f.add_trace(go.Scatter(x=rel["mean_pd"], y=rel["observed"], mode="markers", name="Deciles", marker=dict(color=C["accent"], size=8),
                                   error_y=dict(type="data", symmetric=False, array=hi.tolist(), arrayminus=lo.tolist())))
            f.update_xaxes(type="log", title="Mean calibrated PD")
            f.update_yaxes(type="log", title="Observed default rate")
            a.plotly_chart(style(f, "Reliability by decile (95% Wilson intervals)"))
        rb = data["risk_bands"][data["risk_bands"]["sample"] == sample].set_index("band").loc[BAND_ORDER].reset_index()
        f = go.Figure()
        f.add_bar(x=rb["band"], y=100 * rb["observed"], name="Observed", marker_color=C["main"])
        f.add_bar(x=rb["band"], y=100 * rb["calibrated_pd"], name="Calibrated PD", marker_color=C["light"])
        f.update_layout(barmode="group", yaxis_title="Default rate (%)")
        b.plotly_chart(style(f, "Risk bands: observed vs calibrated"))
        st.write(f"Brier skill score {stored['brier_skill']:.3f}; calibration-in-the-large {stored['citl']:+.3f} (ideal 0); calibration slope {stored['slope']:.3f} (ideal 1).")
    with t4:
        f = go.Figure()
        f.add_bar(x=c["score"], y=c["n_nondefault"] / c["n_nondefault"].sum(), name="Non-default", marker_color=C["main"], opacity=0.65)
        f.add_bar(x=c["score"], y=c["n_default"] / c["n_default"].sum(), name="Default", marker_color=C["accent"], opacity=0.65)
        f.update_layout(barmode="overlay", xaxis_title="Score", yaxis_title="Share within outcome group")
        st.plotly_chart(style(f, "Score distribution by outcome"))
    with t5:
        tab = pd.DataFrame(S["performance"]).T[["n", "defaults", "observed", "auc", "gini", "ks", "brier", "brier_skill", "log_loss", "citl", "slope"]]
        st.dataframe(tab.round(4))


# ------------------------------------------------------------------ page 4
def page_portfolio(data):
    S = data["summary"]
    st.title("Portfolio risk analysis")
    st.warning(predict.DISCLAIMER)
    sample = st.radio("Sample", ["validation", "test", "development"], horizontal=True, key="pf_sample")
    rb = data["risk_bands"][data["risk_bands"]["sample"] == sample].set_index("band").loc[BAND_ORDER].reset_index()
    a, b = st.columns(2)
    f = go.Figure()
    f.add_bar(x=rb["band"], y=100 * rb["observed"], name="Observed default rate", marker_color=C["main"],
              error_y=dict(type="data", symmetric=False, array=(100 * (rb["ci_high"] - rb["observed"]).clip(lower=0)).tolist(), arrayminus=(100 * (rb["observed"] - rb["ci_low"]).clip(lower=0)).tolist()))
    f.add_trace(go.Scatter(x=rb["band"], y=100 * rb["calibrated_pd"], mode="markers", name="Calibrated PD", marker=dict(color="black", symbol="diamond", size=9)))
    f.update_layout(yaxis_title="Default rate (%)")
    a.plotly_chart(style(f, "Default rate by risk grade"))
    f = go.Figure(go.Bar(x=rb["band"], y=rb["share_pct"], marker_color=C["mid"]))
    f.update_layout(yaxis_title="% of scored applicants")
    b.plotly_chart(style(f, "Portfolio risk distribution"))
    z = data["zones"][data["zones"]["sample"] == sample]
    zz = z[z["zone"].isin(["Accept", "Review", "Decline"])]
    st.subheader("Decision zones (illustrative policy)")
    cols = st.columns(4)
    for col, (_, row) in zip(cols, zz.iterrows()):
        col.metric(f"{row['zone']}: share of all applications", pct(row["share_of_all_applications"], 1), help=f"Observed default rate {pct(row['default_rate'], 1)}")
    ref = z[z["zone"].str.startswith("Referral")].iloc[0]
    cols[3].metric("Referred by policy", pct(ref["share_of_all_applications"], 2), help=f"Default rate among referred applicants: {pct(ref['default_rate'], 1)}")
    st.subheader("Interactive cut-off analysis")
    st.caption(CAL)
    c = counts(data, sample)
    n_def, n_good, cd, cg = tail_counts(c)
    n_all = n_def + n_good + S["samples"][sample]["n_policy_referrals"]
    k = st.columns(3)
    t = k[0].slider("Cut-off score (accept if score is at or above)", 500, 626, 563, key="cut_t")
    scen = k[1].selectbox("Scenario (illustrative)", list(co.SCENARIOS) + ["Custom"], key="scenario")
    if scen == "Custom":
        R = k[2].slider("Net revenue if performing (R)", 0.0, 0.30, 0.10, 0.01, key="cust_R")
        L = k[2].slider("Loss given default (LGD)", 0.20, 1.00, 0.65, 0.05, key="cust_L")
    else:
        R, L = co.SCENARIOS[scen]["R"], co.SCENARIOS[scen]["LGD"]
        k[2].write(f"R = {R:.2f}, LGD = {L:.2f}")
    i = t - 300
    n_a, d_a, g_a = cd[i] + cg[i], cd[i], cg[i]
    profit, loss = 100 * (g_a * R - d_a * L) / n_all, 100 * d_a * L / n_all
    base_profit = 100 * (n_good * R - n_def * L) / n_all
    be = co.breakeven_pd(R, L)
    lk = data["lookup"]
    hit = lk[lk["pd_master_scale"] <= be]
    analytic = int(hit["score"].min()) if len(hit) else None
    m1 = st.columns(4)
    m1[0].metric("Acceptance rate (all applications)", pct(n_a / n_all, 1))
    m1[1].metric("Default rate among accepted", pct(d_a / n_a, 2))
    m1[2].metric("Defaults caught (declined)", pct((n_def - d_a) / n_def, 1))
    m1[3].metric("Good customers declined", pct((n_good - g_a) / n_good, 1))
    m2 = st.columns(4)
    m2[0].metric("Profit per 100 applied", f"{profit:.2f}", help=f"Accept-everyone baseline: {base_profit:.2f}")
    m2[1].metric("Expected loss per 100 applied", f"{loss:.2f}", help="Observed defaults x LGD x exposure of 1")
    m2[2].metric("Break-even PD", pct(be, 1), help="R / (R + LGD): accept applicants whose PD is below this")
    m2[3].metric("Analytic best cut-off", "n/a" if analytic is None else str(analytic), help="Score where the calibrated PD first falls to the break-even PD")
    th = np.arange(500, 627)
    curve = 100 * (cg[th - 300] * R - cd[th - 300] * L) / n_all
    f = go.Figure()
    f.add_trace(go.Scatter(x=th, y=curve, mode="lines", name="Profit per 100 applied", line=dict(color=C["main"], width=2)))
    f.add_hline(y=base_profit, line_dash="dot", line_color=C["grey"], annotation_text="Accept everyone")
    f.add_vline(x=t, line_color=C["accent"], line_width=2)
    for v, lab in [(S["policy"]["thresholds"]["t_decline"], "Decline below"), (S["policy"]["thresholds"]["t_accept"], "Accept at or above")]:
        f.add_vline(x=v, line_dash="dash", line_color=C["grey"], annotation_text=lab, annotation_position="top")
    f.update_layout(xaxis_title="Cut-off score", yaxis_title="Profit per 100 units applied")
    st.plotly_chart(style(f, "Profit by cut-off (red line = selected cut-off)", 380))
    rows = []
    for nm, s_ in co.SCENARIOS.items():
        rows.append({"Scenario": nm, "R": s_["R"], "LGD": s_["LGD"], "Break-even PD": pct(co.breakeven_pd(s_["R"], s_["LGD"]), 1),
                     "Profit per 100 at selected cut-off": round(100 * (g_a * s_["R"] - d_a * s_["LGD"]) / n_all, 2),
                     "Expected loss per 100": round(100 * d_a * s_["LGD"] / n_all, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown("**Provisioning burden at the selected cut-off (per 100 applied), if every default reached the stated class**")
    st.dataframe(pd.DataFrame([{"Class and provision rate": nm, "Provision per 100 applied": round(100 * d_a * r_ / n_all, 3)} for nm, r_ in S["bb_rates"].items()]), hide_index=True)
    st.caption("Provision rates follow Bangladesh Bank's classification stages (substandard 20%, doubtful 50%, bad/loss 100%); applying them to the full exposure is a simplification, and actual rules depend on loan type and security.")


# ------------------------------------------------------------------ page 5
def page_monitoring(data):
    st.title("Model monitoring")
    st.warning(predict.DISCLAIMER)
    st.info("Development, validation and test come from one random split, so PSI values near zero are expected by construction. They demonstrate the monitoring tools, not stability over time.")
    ref = counts(data, "development")
    sample = st.radio("Compare development with", ["validation", "test"], horizontal=True, key="mon_sample")
    cur = counts(data, sample)
    shift = st.slider("Simulate a shift in the current score distribution (points)", -40, 40, 0, 1, key="mon_shift")
    live = psi_from(binned(ref), binned(cur, shift))
    ps = data["psi"]
    stored = float(ps[(ps["item"] == "score distribution (20-point bins)") & (ps["sample"] == sample)]["psi"].iloc[0])
    m = st.columns(3)
    m[0].metric("Score PSI (live)", f"{live:.4f}", help="Population Stability Index on 20-point score bins")
    m[1].metric("Stored PSI (no shift)", f"{stored:.4f}")
    m[2].metric("Status (heuristic)", psi_status(live))
    f = go.Figure()
    for nm, cc, col, sh in [("Development", ref, C["light"], 0), (f"{sample.title()} (shifted by {shift:+d})", cur, C["main"], shift)]:
        tot = cc["n_default"] + cc["n_nondefault"]
        f.add_trace(go.Scatter(x=cc["score"] + sh, y=100 * tot / tot.sum(), mode="lines", name=nm, line=dict(color=col, width=2)))
    f.update_layout(xaxis_title="Score", yaxis_title="% of applicants")
    st.plotly_chart(style(f, "Score distribution shift"))
    st.subheader("Stored stability results")
    pv = ps.pivot(index="item", columns="sample", values="psi")
    pv["Status (worst of samples)"] = pv.max(axis=1).map(psi_status)
    st.dataframe(pv.round(5))
    st.caption("Heuristic thresholds: PSI below 0.10 stable; 0.10 to 0.25 investigate; above 0.25 major shift. They are rules of thumb, not universal limits.")
    st.subheader("Suggested monitoring plan (illustrative thresholds)")
    plan = pd.DataFrame([
        ("Score and risk-band PSI", "Monthly", "< 0.10", "0.10 to 0.25", "> 0.25"),
        ("Characteristic PSI (each model variable)", "Monthly", "< 0.10", "0.10 to 0.25", "> 0.25"),
        ("Gini on matured cohorts", "Quarterly", "Within validation interval (about 0.69 to 0.73)", "Below interval once", "Below interval twice running"),
        ("Observed vs calibrated default rate by band", "Quarterly", "Inside 95% interval", "Outside in one band", "Outside in two or more bands"),
        ("Share referred (special codes) and override rate", "Monthly", "Near development levels (about 0.2% and 0.15%)", "Doubles", "Triples"),
        ("Missing-income share and implausible-utilisation share", "Monthly", "Near development levels", "Doubles", "Triples"),
        ("Calibration by age group and past-due history", "Semi-annual", "Gaps no wider than documented", "Wider gaps", "Reversed direction"),
    ], columns=["Indicator", "Frequency", "Green", "Amber", "Red"])
    st.dataframe(plan, hide_index=True)
    st.subheader("Model limitations and recommendations")
    st.markdown("- Re-validate on out-of-time data before any real use; this prototype only has random-split evidence.\n"
                "- Collect outcomes for a controlled band of applicants below the old cut-off (reject inference) to test extrapolated regions.\n"
                "- Document the age and no-open-lines findings, and review whether age may lawfully be used.\n"
                "- Replace illustrative margin and loss assumptions with the institution's own data and weight by exposure.\n"
                "- Recalibrate the score-to-PD mapping if band-level default rates drift, and keep the model, bins and policy under change control.")


# ------------------------------------------------------------------ main
def main():
    data, bundle = load_data(), get_bundle()
    st.sidebar.title("Scorecard prototype")
    page = st.sidebar.radio("Page", PAGES, key="nav")
    st.sidebar.caption("Educational prototype on a public dataset. Not for live lending decisions.")
    st.sidebar.caption("Scoring engine: " + bundle["status"])
    {PAGES[0]: lambda: page_overview(data), PAGES[1]: lambda: page_assess(data, bundle), PAGES[2]: lambda: page_performance(data),
     PAGES[3]: lambda: page_portfolio(data), PAGES[4]: lambda: page_monitoring(data)}[page]()


main()
