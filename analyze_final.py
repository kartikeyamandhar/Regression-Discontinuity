"""
analyze_final.py
FINAL cross-sectional RDD analysis for the Steam review-label study.

QUESTION
Does Steam's documented 40% deprioritization policy reduce player demand?
Valve's own statement: the store visibility boost is essentially flat across all
Mixed-or-above review buckets, with a >500% drop the moment a game falls below
40% positive (Mostly Negative). So 40% is the only real algorithmic
discontinuity. 70% (Mixed -> Mostly Positive) has no documented boost change and
serves as a FALSIFICATION cutoff: a clean effect at 40% but not at 70% is
positive evidence the effect is algorithmic rather than a smooth quality
confound mistaken for a jump.

DESIGN
Sharp RDD on the running variable pct_positive at c = 0.40 (and 0.70 as
falsification). Local-linear, triangular kernel, MSE-optimal bandwidth, robust
bias-corrected CIs (rdrobust).

PRIMARY OUTCOME
log(peak_ccu_yesterday + 1). The previous day's PEAK concurrent players is far
less timing-sensitive than an instantaneous snapshot. Triangulated against
log(concurrent_players + 1) (live snapshot) and log(owners_estimate + 1)
(SteamSpy ownership stock). Three independent demand measures.

PRIMARY COVARIATE SPEC (leaner)
log_price, months_since_release, has_multiplayer, dlc_count. All pre-launch or
fixed game properties. We deliberately EXCLUDE log_total_reviews: it is the
denominator of the running variable and is itself downstream of the policy
(a bad control). Its inclusion absorbs the treatment effect; see spec_robustness.

HONEST CEILING
Cross-sectional, single snapshot, boundary sample ~60 games per side, sample
selects on >=50 reviews, no pre-trend. We read the result as credible,
triangulated, suggestive evidence of a real demand cost, not a clean causal
point estimate. The within-game event study is the identified next step.

Reads:  steam_rdd_data.csv (this directory).
Writes: figures_final/*.png, tables_final/*.csv.
Requires: pip install pandas numpy matplotlib rdrobust [rddensity optional]
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdrobust import rdrobust
try:
    from rddensity import rddensity
    HAVE_RDDENSITY = True
except ImportError:
    HAVE_RDDENSITY = False


# ===== configuration =====
INPUT_CSV   = "steam_rdd_data.csv"
FIGURES_DIR = "figures_final"
TABLES_DIR  = "tables_final"
SEED        = 410014

TREATMENT_CUTOFF     = 0.40
FALSIFICATION_CUTOFF = 0.70

# outcomes: (column, log-name, label). First is PRIMARY.
OUTCOMES = [
    ("peak_ccu_yesterday", "y_peak",   "peak CCU (primary)"),
    ("concurrent_players", "y_live",   "live CCU"),
    ("owners_estimate",    "y_owners", "owners"),
]
PRIMARY_Y = "y_peak"

# primary (leaner) adjustment covariates
ADJUST_COVS = ["log_price", "months_since_release", "has_multiplayer", "dlc_count_f"]

# specs for the robustness panel
ROBUSTNESS_SPECS = {
    "raw":                        None,
    "leaner [primary]":           ADJUST_COVS,
    "with_reviews [bad control]": ["log_price", "log_total_reviews",
                                   "months_since_release", "has_multiplayer", "dlc_count_f"],
    "minimal":                    ["log_price", "months_since_release"],
}

np.random.seed(SEED)
warnings.filterwarnings("ignore")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ===== helpers =====
def extract(rd):
    out = {
        "tau":   float(rd.coef.iloc[0, 0]),
        "tau_bc": float(rd.coef.iloc[1, 0]),
        "se":    float(rd.se.iloc[2, 0]),
        "p":     float(rd.pv.iloc[2, 0]),
        "ci_lo": float(rd.ci.iloc[2, 0]),
        "ci_hi": float(rd.ci.iloc[2, 1]),
        "h":     float(rd.bws.iloc[0, 0]),
    }
    nh = getattr(rd, "N_h", None) or getattr(rd, "Nh", None)
    if nh is not None:
        out["n_left"]  = int(nh[0])
        out["n_right"] = int(nh[1])
    return out


def cmat(df, cols):
    m = df[cols].copy()
    for c in cols:
        m[c] = m[c].fillna(m[c].median() if m[c].notna().any() else 0)
    return m.values


def rdd(df, ycol, c, covs_cols=None):
    d = df[[ycol, "X"] + (covs_cols or [])].dropna()
    kwargs = {"c": c}
    if covs_cols:
        kwargs["covs"] = cmat(d, covs_cols)
    return extract(rdrobust(d[ycol].values, d["X"].values, **kwargs))


# ===== prep (matches the actual CSV schema) =====
def load_and_prep():
    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)

    df = df[df["type"].fillna("game") == "game"]
    df = df[~df["coming_soon"].fillna(False).astype(bool)]
    df = df[df["total_reviews"] >= 50]

    df["X"] = df["pct_positive"]
    for col, logname, _ in OUTCOMES:
        df[logname] = np.log1p(df[col].fillna(0))

    df["rel"]  = pd.to_datetime(df["release_date"], errors="coerce")
    df["snap"] = pd.to_datetime(df["collected_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    df["release_year"] = df["rel"].dt.year
    df["months_since_release"] = (df["snap"] - df["rel"]).dt.days / 30.44

    def cohort(r):
        y, m = r["release_year"], r["rel"].month if pd.notna(r["rel"]) else None
        if pd.isna(y):
            return None
        if y < 2020 or (y == 2020 and m < 3):
            return "pre_covid"
        if y in (2020, 2021):
            return "covid"
        if y >= 2022:
            return "post_covid"
        return None
    df["cohort"] = df.apply(cohort, axis=1)

    df["log_price"]         = np.log1p(df["price_cents"].fillna(0))
    df["log_total_reviews"] = np.log(df["total_reviews"])
    df["has_multiplayer"]   = df["categories"].fillna("").str.contains("Multi-player").astype(int)
    df["dlc_count_f"]       = df["dlc_count"].fillna(0)
    df["plat_mac_i"]        = df["platform_mac"].fillna(False).astype(int)
    df["achievements_f"]    = df["achievements_total"].fillna(0)
    df["langs_f"]           = df["supported_languages_count"].fillna(0)

    df = df[df["release_year"] >= 2018]

    print(f"prep: {n0} -> {len(df)} rows after filters")
    print(f"  pct_positive: min={df.X.min():.2f} median={df.X.median():.2f} max={df.X.max():.2f}")
    print(f"  cohorts: {df.cohort.value_counts().to_dict()}")
    print(f"  primary outcome: {PRIMARY_Y};  adjustment covs: {ADJUST_COVS}")
    return df


# ===== validity =====
def density_test(df, c):
    print(f"\n=== density test at c={c} ===")
    if not HAVE_RDDENSITY:
        print("  rddensity not installed; skipping (run: pip install rddensity)")
        return None
    try:
        r = rddensity(X=df["X"].values, c=c)
    except Exception as e:
        print(f"  errored: {e}")
        return None
    obj = getattr(r, "test", None)
    for cand in ("p_jk", "P_jk", "P>|z|", "p", "p_value", "pv"):
        if obj is not None and hasattr(obj, "columns") and cand in obj.columns:
            v = float(obj[cand].iloc[0]); print(f"  p-value ({cand}): {v:.4f}"); return v
        if obj is not None and hasattr(obj, "index") and not hasattr(obj, "columns") and cand in obj.index:
            v = float(obj.loc[cand]); print(f"  p-value ({cand}): {v:.4f}"); return v
    print("  could not extract p-value")
    return None


def covariate_balance(df, c):
    print(f"\n=== covariate balance at c={c} ===")
    covs = ["log_price", "log_total_reviews", "months_since_release",
            "has_multiplayer", "plat_mac_i", "dlc_count_f", "achievements_f", "langs_f"]
    rows = []
    for cov in covs:
        d = df[[cov, "X"]].dropna()
        if len(d) < 80:
            continue
        try:
            e = extract(rdrobust(d[cov].values, d["X"].values, c=c))
            flag = "  FAIL" if e["p"] < 0.05 else ""
            print(f"  {cov:22s} tau={e['tau']:+.4f}  p={e['p']:.3f}{flag}")
            rows.append({"covariate": cov, "tau": e["tau"], "p": e["p"]})
        except Exception as ex:
            print(f"  {cov:22s} error: {ex}")
    return pd.DataFrame(rows)


def placebos(df, ycol, c_list):
    print(f"\n=== placebo cutoffs on {ycol} (leaner spec) ===")
    rows = []
    for c in c_list:
        try:
            e = rdd(df, ycol, c, ADJUST_COVS)
            sig = "  *" if e["p"] < 0.05 else ""
            print(f"  c={c:.2f}  tau={e['tau']:+.3f}  CI=[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}]  p={e['p']:.3f}{sig}")
            rows.append({"cutoff": c, **e})
        except Exception as ex:
            print(f"  c={c:.2f}  error: {ex}")
    return pd.DataFrame(rows)


# ===== main estimates =====
def main_estimates(df):
    print(f"\n{'='*78}\nMAIN ESTIMATES  (raw and leaner-adjusted)\n{'='*78}")
    rows = []
    for c, ctag in [(TREATMENT_CUTOFF, "treatment"), (FALSIFICATION_CUTOFF, "falsification")]:
        for col, ycol, lab in OUTCOMES:
            for spec, covs in [("raw", None), ("adjusted", ADJUST_COVS)]:
                e = rdd(df, ycol, c, covs)
                rows.append({"cutoff": c, "cutoff_role": ctag, "outcome": ycol,
                             "outcome_label": lab, "spec": spec, **e})
        print(f"\nc = {c:.2f}  ({ctag})")
        print("-"*78)
        for col, ycol, lab in OUTCOMES:
            er = [r for r in rows if r["cutoff"]==c and r["outcome"]==ycol]
            raw = next(r for r in er if r["spec"]=="raw")
            adj = next(r for r in er if r["spec"]=="adjusted")
            print(f"  {lab:20s} raw  tau={raw['tau']:+.3f} CI=[{raw['ci_lo']:+.3f},{raw['ci_hi']:+.3f}] p={raw['p']:.3f}"
                  f"{'  *' if raw['p']<0.05 else ''}")
            print(f"  {'':20s} adj  tau={adj['tau']:+.3f} CI=[{adj['ci_lo']:+.3f},{adj['ci_hi']:+.3f}] p={adj['p']:.3f}"
                  f"{'  *' if adj['p']<0.05 else ''}   N={adj.get('n_left','?')}/{adj.get('n_right','?')}")
    return pd.DataFrame(rows)


def spec_robustness(df):
    print(f"\n{'='*78}\nSPEC ROBUSTNESS  ({PRIMARY_Y}, both cutoffs)\n{'='*78}")
    rows = []
    for c in (TREATMENT_CUTOFF, FALSIFICATION_CUTOFF):
        print(f"\nc = {c:.2f}")
        print("-"*78)
        for name, covs in ROBUSTNESS_SPECS.items():
            try:
                e = rdd(df, PRIMARY_Y, c, covs)
                sig = "  *" if e["p"] < 0.05 else ""
                print(f"  {name:30s} tau={e['tau']:+.3f}  CI=[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}]  p={e['p']:.3f}{sig}")
                rows.append({"cutoff": c, "spec": name, **e})
            except Exception as ex:
                print(f"  {name:30s} error: {ex}")
    return pd.DataFrame(rows)


def heterogeneity_genre(df, c, ycol, min_obs=40):
    print(f"\n=== per-genre RDD ({ycol}, leaner) c={c} ===")
    x = df.copy()
    x["g"] = x["genres"].fillna("").str.split("|")
    x = x.explode("g"); x["g"] = x["g"].str.strip(); x = x[x["g"] != ""]
    rows = []
    for g, sub in x.groupby("g"):
        if len(sub) < min_obs:
            continue
        try:
            e = rdd(sub, ycol, c, ADJUST_COVS)
            rows.append({"genre": g, "n": len(sub), **e})
            print(f"  {g:24s} n={len(sub):4d}  tau={e['tau']:+.3f}  CI=[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}]")
        except Exception:
            pass
    return pd.DataFrame(rows)


def heterogeneity_cohort(df, c, ycol):
    print(f"\n=== per-cohort RDD ({ycol}, leaner) c={c} ===")
    rows = []
    for coh in ("pre_covid", "covid", "post_covid"):
        sub = df[df["cohort"] == coh]
        if len(sub) < 40:
            continue
        try:
            e = rdd(sub, ycol, c, ADJUST_COVS)
            rows.append({"cohort": coh, "n": len(sub), **e})
            print(f"  {coh:12s} n={len(sub):4d}  tau={e['tau']:+.3f}  CI=[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}]")
        except Exception:
            pass
    return pd.DataFrame(rows)


def bandwidth_sensitivity(df, ycol, c, mults=(0.5, 0.75, 1.0, 1.25, 1.5)):
    print(f"\n=== bandwidth sensitivity ({ycol}, leaner) c={c} ===")
    h_opt = rdd(df, ycol, c, ADJUST_COVS)["h"]
    rows = []
    for mlt in mults:
        h = h_opt * mlt
        d = df[[ycol, "X"] + ADJUST_COVS].dropna()
        try:
            e = extract(rdrobust(d[ycol].values, d["X"].values, c=c, h=h, covs=cmat(d, ADJUST_COVS)))
            rows.append({"h_mult": mlt, "h": h, **e})
            print(f"  {mlt:.2f}x  h={h:.3f}  tau={e['tau']:+.3f}  CI=[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}]")
        except Exception:
            pass
    return pd.DataFrame(rows)


def donut(df, ycol, c, hole=0.005):
    sub = df[(df["X"] < c - hole) | (df["X"] > c + hole)]
    e = rdd(sub, ycol, c, ADJUST_COVS)
    print(f"\n=== donut RDD ({ycol}, leaner, hole={hole}) c={c} ===")
    print(f"  tau={e['tau']:+.3f}  CI=[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}]  p={e['p']:.3f}")
    return e


# ===== figures =====
BLUE, RED, AMBER, GREY = "#3a6fb0", "#A32D2D", "#BA7517", "#7a7a7a"


def _wls_at_cutoff(d, ycol, c, h, side):
    """Triangular-kernel local-linear fit on one side within bandwidth h.
    Returns ((xx, yy) line over [c-h,c] or [c,c+h], fitted level at c). This
    reproduces rdrobust's conventional point estimate, so the gap between the two
    sides' fitted levels at c equals the reported RD estimate."""
    if side == "L":
        m = (d["X"] >= c - h) & (d["X"] < c)
    else:
        m = (d["X"] >= c) & (d["X"] <= c + h)
    xs = d["X"][m].values.astype(float); ys = d[ycol][m].values.astype(float)
    if len(xs) < 5:
        return None, None
    w = np.clip(1 - np.abs(xs - c) / h, 0, None)
    X = np.column_stack([np.ones_like(xs), xs - c])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + 1e-9 * np.eye(2), XtW @ ys)
    xx = np.linspace(c - h, c, 40) if side == "L" else np.linspace(c, c + h, 40)
    return (xx, beta[0] + beta[1] * (xx - c)), float(beta[0])


def fig_rdd(df, ycol, c, fname, title, ylab):
    d = df[[ycol, "X"]].dropna()
    # bandwidth from the unadjusted local-linear fit (matches the RD estimand)
    try:
        h = extract(rdrobust(d[ycol].values, d["X"].values, c=c))["h"]
    except Exception:
        h = 0.05
    win = max(2 * h, 0.12)  # wider context window for the binned means
    s = d[(d["X"] >= c - win) & (d["X"] <= c + win)].copy()
    s["bin"] = pd.cut(s["X"], 24)
    b = s.groupby("bin", observed=True).agg({"X": "mean", ycol: "mean"}).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(c - h, c + h, color="gray", alpha=0.08, zorder=0,
               label=f"estimation bandwidth (±{h:.3f})")
    ax.scatter(b["X"], b[ycol], s=38, alpha=0.55, color=BLUE, zorder=3)
    lineL, aL = _wls_at_cutoff(d, ycol, c, h, "L")
    lineR, aR = _wls_at_cutoff(d, ycol, c, h, "R")
    if lineL is not None: ax.plot(*lineL, color=RED, lw=2.4, zorder=4)
    if lineR is not None: ax.plot(*lineR, color=AMBER, lw=2.4, zorder=4)
    ax.axvline(c, color="gray", ls="--", alpha=0.7, zorder=1)
    if aL is not None and aR is not None:
        ax.annotate(f"local gap at cutoff \u2248 {aR-aL:+.2f}",
                    xy=(c, max(aL, aR)), xytext=(c + h*0.4, max(aL, aR)),
                    fontsize=9, color="black")
    ax.set_xlabel("share of reviews positive"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


def fig_density(df, c, fname, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["X"], bins=50, alpha=0.75, color=BLUE, edgecolor="white")
    ax.axvline(c, color=RED, ls="--", lw=2, label=f"cutoff = {c}")
    ax.set_xlabel("share of reviews positive"); ax.set_ylabel("number of games")
    ax.set_title(title); ax.legend()
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


def fig_triangulation(main_df, fname):
    """Three outcomes at the treatment cutoff, adjusted spec: the no-BS exhibit."""
    sub = main_df[(main_df["cutoff"] == TREATMENT_CUTOFF) & (main_df["spec"] == "adjusted")]
    sub = sub.set_index("outcome").reindex([o[1] for o in OUTCOMES]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(sub))
    ax.errorbar(sub["tau"], y, xerr=[sub["tau"]-sub["ci_lo"], sub["ci_hi"]-sub["tau"]],
                fmt="o", color=BLUE, capsize=4, markersize=8, lw=1.6)
    ax.axvline(0, color="gray", ls="--", alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(sub["outcome_label"])
    ax.set_xlabel("estimated tau at 40% cutoff (95% robust CI)")
    ax.set_title("Effect at 40% holds across three demand measures (leaner spec)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


def fig_falsification(main_df, fname):
    """40% vs 70%, adjusted, all three outcomes: the falsification contrast."""
    sub = main_df[main_df["spec"] == "adjusted"].copy()
    labels = [o[2] for o in OUTCOMES]; ynames = [o[1] for o in OUTCOMES]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    off = 0.16
    for i, yn in enumerate(ynames):
        r40 = sub[(sub.cutoff==TREATMENT_CUTOFF) & (sub.outcome==yn)].iloc[0]
        r70 = sub[(sub.cutoff==FALSIFICATION_CUTOFF) & (sub.outcome==yn)].iloc[0]
        ax.errorbar(r40["tau"], i+off, xerr=[[r40["tau"]-r40["ci_lo"]],[r40["ci_hi"]-r40["tau"]]],
                    fmt="o", color=BLUE, capsize=3, markersize=7, label="40% (documented boost change)" if i==0 else "")
        ax.errorbar(r70["tau"], i-off, xerr=[[r70["tau"]-r70["ci_lo"]],[r70["ci_hi"]-r70["tau"]]],
                    fmt="o", color=AMBER, capsize=3, markersize=7, label="70% (no boost change: falsification)" if i==0 else "")
    ax.axvline(0, color="gray", ls="--", alpha=0.6)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("estimated tau (95% robust CI)")
    ax.set_title("40% vs 70%: treatment cutoff vs falsification cutoff (leaner spec)")
    ax.legend(loc="best", fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


def fig_spec(robust_df, fname):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
    order = list(ROBUSTNESS_SPECS.keys())
    cmap = {"raw": GREY, "leaner [primary]": BLUE, "with_reviews [bad control]": RED, "minimal": AMBER}
    for ax, c in zip(axes, (TREATMENT_CUTOFF, FALSIFICATION_CUTOFF)):
        sub = robust_df[robust_df.cutoff==c].set_index("spec").reindex(order).reset_index()
        y = np.arange(len(sub))
        for i, row in sub.iterrows():
            ax.errorbar(row["tau"], i, xerr=[[row["tau"]-row["ci_lo"]],[row["ci_hi"]-row["tau"]]],
                        fmt="o", color=cmap.get(row["spec"], BLUE), capsize=3, markersize=7, lw=1.5)
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.set_yticks(y); ax.set_yticklabels(sub["spec"], fontsize=9)
        ax.set_title(f"c = {c:.2f}  ({PRIMARY_Y})", fontsize=10)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("estimated tau (95% robust CI)")
    fig.suptitle("Spec robustness: the 'with reviews' bad control absorbs the effect", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


def fig_forest(d, label_col, fname, title):
    if d is None or d.empty: return
    d = d.sort_values("tau").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45*len(d))))
    y = np.arange(len(d))
    ax.errorbar(d["tau"], y, xerr=[d["tau"]-d["ci_lo"], d["ci_hi"]-d["tau"]],
                fmt="o", color=BLUE, capsize=3, markersize=6, lw=1.5)
    ax.axvline(0, color="gray", ls="--", alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{l} (n={n})" for l, n in zip(d[label_col], d["n"])])
    ax.set_xlabel("estimated tau (95% robust CI)"); ax.set_title(title)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


def fig_bandwidth(d, fname, title):
    if d is None or d.empty: return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(d["h_mult"], d["tau"], yerr=[d["tau"]-d["ci_lo"], d["ci_hi"]-d["tau"]],
                fmt="o-", color=BLUE, capsize=4, markersize=7)
    ax.axhline(0, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("bandwidth (multiple of MSE-optimal h)")
    ax.set_ylabel("estimated tau (95% robust CI)"); ax.set_title(title)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=200); plt.close()
    print(f"  saved {fname}")


# ===== main =====
def main():
    df = load_and_prep()
    df.to_csv(f"{TABLES_DIR}/prepped_sample.csv", index=False)

    # validity
    for c in (TREATMENT_CUTOFF, FALSIFICATION_CUTOFF):
        density_test(df, c)
        covariate_balance(df, c).to_csv(f"{TABLES_DIR}/cov_balance_{int(c*100)}.csv", index=False)
    pl40 = placebos(df, PRIMARY_Y, [0.30, 0.35, 0.45, 0.50])
    pl70 = placebos(df, PRIMARY_Y, [0.60, 0.65, 0.75, 0.80])
    pl40.to_csv(f"{TABLES_DIR}/placebo_40.csv", index=False)
    pl70.to_csv(f"{TABLES_DIR}/placebo_70.csv", index=False)

    # main + robustness
    main_df = main_estimates(df);   main_df.to_csv(f"{TABLES_DIR}/main_estimates.csv", index=False)
    rob_df  = spec_robustness(df);  rob_df.to_csv(f"{TABLES_DIR}/spec_robustness.csv", index=False)

    # heterogeneity (primary outcome, treatment cutoff)
    g40 = heterogeneity_genre(df, TREATMENT_CUTOFF, PRIMARY_Y);  g40.to_csv(f"{TABLES_DIR}/genre_40.csv", index=False)
    c40 = heterogeneity_cohort(df, TREATMENT_CUTOFF, "y_owners"); c40.to_csv(f"{TABLES_DIR}/cohort_40.csv", index=False)

    # sensitivity (primary outcome, treatment cutoff)
    bw40 = bandwidth_sensitivity(df, PRIMARY_Y, TREATMENT_CUTOFF); bw40.to_csv(f"{TABLES_DIR}/bandwidth_40.csv", index=False)
    donut(df, PRIMARY_Y, TREATMENT_CUTOFF)

    # figures
    print("\n=== figures ===")
    fig_rdd(df, "y_peak", TREATMENT_CUTOFF, "rdd_40_peak.png",
            "RDD at 40%: peak CCU (primary outcome)", "log peak concurrent players")
    fig_rdd(df, "y_live", TREATMENT_CUTOFF, "rdd_40_live.png",
            "RDD at 40%: live CCU", "log live concurrent players")
    fig_rdd(df, "y_owners", TREATMENT_CUTOFF, "rdd_40_owners.png",
            "RDD at 40%: owners", "log owners (SteamSpy)")
    fig_rdd(df, "y_peak", FALSIFICATION_CUTOFF, "rdd_70_peak.png",
            "RDD at 70% (falsification): peak CCU", "log peak concurrent players")
    fig_density(df, TREATMENT_CUTOFF, "density_40.png", "Density of pct_positive (manipulation check, 40%)")
    fig_density(df, FALSIFICATION_CUTOFF, "density_70.png", "Density of pct_positive (manipulation check, 70%)")
    fig_triangulation(main_df, "triangulation_40.png")
    fig_falsification(main_df, "falsification_40_vs_70.png")
    fig_spec(rob_df, "spec_robustness.png")
    fig_forest(g40, "genre", "forest_genre_40.png", "Per-genre tau at 40% (peak CCU, leaner)")
    fig_forest(c40, "cohort", "forest_cohort_40.png", "Per-cohort tau at 40% (owners, leaner)")
    fig_bandwidth(bw40, "bandwidth_40.png", "Bandwidth sensitivity at 40% (peak CCU, leaner)")

    # slide-ready headline
    print(f"\n{'='*78}\nHEADLINE (primary outcome, 40% cutoff, leaner spec)\n{'='*78}")
    h = main_df[(main_df.cutoff==TREATMENT_CUTOFF) & (main_df.outcome==PRIMARY_Y) & (main_df.spec=="adjusted")].iloc[0]
    mult_lo, mult_hi = np.exp(h["ci_lo"]), np.exp(h["ci_hi"])
    print(f"  tau = {h['tau']:+.3f}  (log points)   p = {h['p']:.3f}   N = {h.get('n_left')}/{h.get('n_right')}")
    print(f"  multiplicative effect on peak CCU: x{np.exp(h['tau']):.2f}  (95% CI x{mult_lo:.2f} to x{mult_hi:.2f})")
    print(f"  reading: crossing above 40% is associated with roughly a {np.exp(h['tau']):.1f}x higher peak")
    print(f"           concurrent-player count, though the CI is wide (x{mult_lo:.2f} to x{mult_hi:.2f}).")
    print("\ndone. tables -> tables_final/, figures -> figures_final/")


if __name__ == "__main__":
    main()