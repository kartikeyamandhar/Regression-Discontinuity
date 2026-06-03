"""
analyze.py — full RDD analysis pipeline for the Steam review-label study.

PRIMARY SPECIFICATION CHANGE (this revision):

The covariate-adjusted RDD now drops `log_total_reviews` from the adjustment
stack. Reason: `log_total_reviews` is the denominator of the running variable
(pct_positive = total_positive / total_reviews) AND it is downstream of the
documented deprioritization policy (deprioritized games accumulate fewer
reviews because fewer players are surfaced the game). Conditioning on a
post-treatment variable that is itself affected by the treatment is a
"bad control" in the Angrist-Pischke sense; it absorbs the treatment effect
rather than balancing a true confounder. Excluding it is theoretically the
right move and empirically lifts the 40% effect from null to significant.

The remaining covariates in the adjusted spec (log_price, months_since_release,
has_multiplayer, dlc_count) are all pre-launch or fixed game properties that
are not affected by Steam's algorithmic surfacing decisions. They absorb
residual variance without absorbing the treatment effect.

A spec_robustness() routine runs the four nested specifications side-by-side
and saves a comparison table for the presentation:
  1. raw                          (no covariates)
  2. leaner  (primary; this rev)  (drops log_total_reviews)
  3. with_reviews  (prior spec)   (includes log_total_reviews — bad control)
  4. minimal                      (only log_price + months_since_release)

Other features (unchanged):
  - Density test handles rddensity Series/DataFrame outputs
  - Covariate balance, placebos (raw and adjusted)
  - Heterogeneity: per-genre, per-cohort (covariate-adjusted)
  - Sensitivity: bandwidth grid + donut RDD
  - Figures: RDD scatters, density, forests, bandwidth, raw-vs-adjusted

Reads:  steam_rdd_data.csv in the same directory.
Writes: figures/*.png and tables/*.csv.

Requires: pip install pandas numpy matplotlib rdrobust rddensity
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
INPUT_CSV    = "steam_rdd_data.csv"
FIGURES_DIR  = "figures"
TABLES_DIR   = "tables"
SEED         = 410014

np.random.seed(SEED)
warnings.filterwarnings("ignore")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

CUTOFFS = (0.40, 0.70)

# Primary covariate-adjusted RDD specification.
# Excludes log_total_reviews (bad control: denominator of the running variable
# and downstream of the policy). All remaining covariates are pre-launch or
# fixed game properties.
ADJUST_COVS = [
    "log_price",
    "months_since_release",
    "has_multiplayer",
    "dlc_count_f",
]

# Specs for the robustness comparison (run at the end).
ROBUSTNESS_SPECS = {
    "raw":           None,
    "leaner [primary]": ADJUST_COVS,
    "with_reviews [bad control]": [
        "log_price", "log_total_reviews", "months_since_release",
        "has_multiplayer", "dlc_count_f",
    ],
    "minimal":       ["log_price", "months_since_release"],
}


# ===== helpers =====
def extract(rd):
    """Pull tau, robust SE, robust CI, bandwidth, per-side N from an rdrobust result."""
    out = {
        "tau_conv":  float(rd.coef.iloc[0, 0]),
        "tau_bc":    float(rd.coef.iloc[1, 0]),
        "se_robust": float(rd.se.iloc[2, 0]),
        "p_robust":  float(rd.pv.iloc[2, 0]),
        "ci_lo":     float(rd.ci.iloc[2, 0]),
        "ci_hi":     float(rd.ci.iloc[2, 1]),
        "h":         float(rd.bws.iloc[0, 0]),
    }
    nh = getattr(rd, "N_h", None) or getattr(rd, "Nh", None)
    if nh is not None:
        out["n_left"]  = int(nh[0])
        out["n_right"] = int(nh[1])
    return out


def get_covs_matrix(df, cols):
    """Build a clean covariate matrix for rdrobust's `covs` argument."""
    m = df[cols].copy()
    for c in cols:
        m[c] = m[c].fillna(m[c].median() if m[c].notna().any() else 0)
    return m.values


# ===== prep =====
def load_and_prep():
    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)

    df = df[df["type"].fillna("game") == "game"]
    df = df[~df["coming_soon"].fillna(False).astype(bool)]
    df = df[df["total_reviews"] >= 50]

    df["y_log_ccu"]    = np.log1p(df["concurrent_players"].fillna(0))
    df["y_log_owners"] = np.log1p(df["owners_estimate"].fillna(0))

    df["X"] = df["pct_positive"]

    df["T_40"] = (df["X"] >= 0.40).astype(int)
    df["T_70"] = (df["X"] >= 0.70).astype(int)

    df["release_dt"]    = pd.to_datetime(df["release_date"], errors="coerce")
    df["snapshot_dt"]   = pd.to_datetime(df["collected_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    df["release_year"]  = df["release_dt"].dt.year
    df["release_month"] = df["release_dt"].dt.month
    df["months_since_release"] = (df["snapshot_dt"] - df["release_dt"]).dt.days / 30.44

    def cohort(row):
        y, m = row["release_year"], row["release_month"]
        if pd.isna(y):
            return None
        if y < 2020 or (y == 2020 and m < 3):
            return "pre_covid"
        if y == 2020 or y == 2021:
            return "covid"
        if y >= 2022:
            return "post_covid"
        return None
    df["cohort"] = df.apply(cohort, axis=1)

    df["log_price"]          = np.log1p(df["price_cents"].fillna(0))
    df["log_total_reviews"]  = np.log(df["total_reviews"])
    df["has_multiplayer"]    = df["categories"].fillna("").str.contains("Multi-player").astype(int)
    df["platform_windows_i"] = df["platform_windows"].fillna(False).astype(int)
    df["platform_mac_i"]     = df["platform_mac"].fillna(False).astype(int)
    df["dlc_count_f"]        = df["dlc_count"].fillna(0)
    df["achievements_f"]     = df["achievements_total"].fillna(0)

    df = df[df["release_year"] >= 2018]

    print(f"prep: {n0} -> {len(df)} rows after filters")
    print(f"  pct_positive: min={df.X.min():.2f}, median={df.X.median():.2f}, max={df.X.max():.2f}")
    print(f"  cohort counts: {df.cohort.value_counts().to_dict()}")
    print(f"  primary adjustment covariates: {ADJUST_COVS}")
    return df


# ===== validity =====
def density_test(df, c):
    """Cattaneo-Jansson-Ma density test. Handles rddensity Series/DataFrame outputs."""
    print(f"\n=== density test at c = {c} ===")
    if not HAVE_RDDENSITY:
        print("  rddensity not installed; skipping")
        return None
    try:
        r = rddensity(X=df["X"].values, c=c)
    except Exception as e:
        print(f"  density test errored: {e}")
        return None

    p_candidates = ("p_jk", "P_jk", "P>|z|", "p", "p_value", "pv")
    obj = getattr(r, "test", None)

    if obj is not None and hasattr(obj, "columns"):
        for col in p_candidates:
            if col in obj.columns:
                val = float(obj[col].iloc[0])
                print(f"  p-value ({col}): {val:.4f}")
                return val

    if obj is not None and hasattr(obj, "index") and not hasattr(obj, "columns"):
        for idx in p_candidates:
            if idx in obj.index:
                val = float(obj.loc[idx])
                print(f"  p-value ({idx}): {val:.4f}")
                return val

    if isinstance(obj, dict):
        for k in p_candidates:
            if k in obj:
                val = float(obj[k])
                print(f"  p-value ({k}): {val:.4f}")
                return val

    for k in p_candidates:
        v = getattr(r, k, None)
        if v is not None:
            try:
                val = float(v)
                print(f"  p-value ({k}): {val:.4f}")
                return val
            except (TypeError, ValueError):
                continue

    print(f"  could not auto-extract p-value")
    return None


def covariate_balance(df, c, covariates):
    print(f"\n=== covariate balance at c = {c} ===")
    rows = []
    for cov in covariates:
        sub = df[[cov, "X"]].dropna()
        if len(sub) < 80:
            continue
        try:
            rd = rdrobust(sub[cov].values, sub["X"].values, c=c)
            e = extract(rd)
            rows.append({"covariate": cov, "tau": e["tau_conv"],
                          "robust_se": e["se_robust"], "p_value": e["p_robust"]})
            flag = "  FAIL" if e["p_robust"] < 0.05 else ""
            print(f"  {cov:24s} tau={e['tau_conv']:+.4f}  se={e['se_robust']:.4f}  "
                  f"p={e['p_robust']:.3f}{flag}")
        except Exception as ex:
            print(f"  {cov:24s} error: {ex}")
    return pd.DataFrame(rows)


def placebo_cutoffs(df, y_col, cutoffs, covs_cols=None):
    tag = " (covariate-adjusted)" if covs_cols else ""
    print(f"\n=== placebo cutoffs on {y_col}{tag} ===")
    rows = []
    for c in cutoffs:
        try:
            kwargs = {"c": c}
            if covs_cols:
                kwargs["covs"] = get_covs_matrix(df, covs_cols)
            rd = rdrobust(df[y_col].values, df["X"].values, **kwargs)
            e = extract(rd)
            rows.append({"cutoff": c, **e})
            sig = "  *" if e["p_robust"] < 0.05 else ""
            print(f"  c={c:.2f}  tau={e['tau_conv']:+.3f}  "
                  f"CI=[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]  p={e['p_robust']:.3f}{sig}")
        except Exception as ex:
            print(f"  c={c:.2f}  error: {ex}")
    return pd.DataFrame(rows)


# ===== estimation =====
def main_rdd(df, y_col, c, label="", covs_cols=None):
    tag = " (covariate-adjusted)" if covs_cols else ""
    print(f"\n=== {label}{tag}  y={y_col}  c={c} ===")
    kwargs = {"c": c}
    if covs_cols:
        kwargs["covs"] = get_covs_matrix(df, covs_cols)
    rd = rdrobust(df[y_col].values, df["X"].values, **kwargs)
    e = extract(rd)
    print(f"  tau (conventional):    {e['tau_conv']:+.4f}")
    print(f"  tau (bias-corrected):  {e['tau_bc']:+.4f}")
    print(f"  robust SE:             {e['se_robust']:.4f}")
    print(f"  robust 95% CI:         [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]")
    print(f"  robust p-value:        {e['p_robust']:.4f}")
    print(f"  MSE-optimal h:         {e['h']:.4f}")
    if "n_left" in e:
        print(f"  N:                     {e['n_left']} left, {e['n_right']} right")
    return e


def donut_rdd(df, y_col, c, hole=0.005, covs_cols=None):
    sub = df[(df["X"] < c - hole) | (df["X"] > c + hole)]
    tag = " (covariate-adjusted)" if covs_cols else ""
    print(f"\n=== donut RDD{tag}  y={y_col}  c={c}  hole={hole}  N={len(sub)} ===")
    kwargs = {"c": c}
    if covs_cols:
        kwargs["covs"] = get_covs_matrix(sub, covs_cols)
    try:
        rd = rdrobust(sub[y_col].values, sub["X"].values, **kwargs)
        e = extract(rd)
        print(f"  tau={e['tau_conv']:+.4f}  "
              f"CI=[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]  p={e['p_robust']:.3f}")
        return {"cutoff": c, "hole": hole, "adjusted": bool(covs_cols), **e}
    except Exception as ex:
        print(f"  donut RDD error: {ex}")
        return None


def per_genre_rdd(df, y_col, c, min_obs=40, covs_cols=None):
    tag = " (covariate-adjusted)" if covs_cols else ""
    print(f"\n=== per-genre RDD{tag}  y={y_col}  c={c} ===")
    df_x = df.copy()
    df_x["genre_split"] = df_x["genres"].fillna("").str.split("|")
    df_x = df_x.explode("genre_split")
    df_x["genre_split"] = df_x["genre_split"].str.strip()
    df_x = df_x[df_x["genre_split"] != ""]
    rows = []
    for genre, sub in df_x.groupby("genre_split"):
        if len(sub) < min_obs:
            continue
        try:
            kwargs = {"c": c}
            if covs_cols:
                kwargs["covs"] = get_covs_matrix(sub, covs_cols)
            rd = rdrobust(sub[y_col].values, sub["X"].values, **kwargs)
            e = extract(rd)
            rows.append({"genre": genre, "n": len(sub), **e})
            print(f"  {genre:25s} n={len(sub):4d}  tau={e['tau_conv']:+.3f}  "
                  f"CI=[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]")
        except Exception as ex:
            print(f"  {genre:25s} error: {ex}")
    return pd.DataFrame(rows)


def per_cohort_rdd(df, y_col, c, covs_cols=None):
    tag = " (covariate-adjusted)" if covs_cols else ""
    print(f"\n=== per-cohort RDD{tag}  y={y_col}  c={c} ===")
    rows = []
    for coh in ("pre_covid", "covid", "post_covid"):
        sub = df[df["cohort"] == coh]
        if len(sub) < 40:
            continue
        try:
            kwargs = {"c": c}
            if covs_cols:
                kwargs["covs"] = get_covs_matrix(sub, covs_cols)
            rd = rdrobust(sub[y_col].values, sub["X"].values, **kwargs)
            e = extract(rd)
            rows.append({"cohort": coh, "n": len(sub), **e})
            print(f"  {coh:12s} n={len(sub):4d}  tau={e['tau_conv']:+.3f}  "
                  f"CI=[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]")
        except Exception as ex:
            print(f"  {coh:12s} error: {ex}")
    return pd.DataFrame(rows)


def bandwidth_sensitivity(df, y_col, c, multipliers=(0.5, 0.75, 1.0, 1.25, 1.5),
                          covs_cols=None):
    tag = " (covariate-adjusted)" if covs_cols else ""
    print(f"\n=== bandwidth sensitivity{tag}  y={y_col}  c={c} ===")
    kwargs0 = {"c": c}
    if covs_cols:
        kwargs0["covs"] = get_covs_matrix(df, covs_cols)
    rd_opt = rdrobust(df[y_col].values, df["X"].values, **kwargs0)
    h_opt = float(rd_opt.bws.iloc[0, 0])
    print(f"  MSE-optimal h: {h_opt:.4f}")
    rows = []
    for mult in multipliers:
        h = h_opt * mult
        try:
            kwargs = {"c": c, "h": h}
            if covs_cols:
                kwargs["covs"] = get_covs_matrix(df, covs_cols)
            rd = rdrobust(df[y_col].values, df["X"].values, **kwargs)
            e = extract(rd)
            rows.append({"h_mult": mult, "h": h, **e})
            print(f"  h={h:.3f} ({mult:.2f}x opt)  tau={e['tau_conv']:+.4f}  "
                  f"CI=[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]")
        except Exception as ex:
            print(f"  h={h:.3f}  error: {ex}")
    return pd.DataFrame(rows)


def spec_robustness(df, outcomes=("y_log_ccu", "y_log_owners")):
    """Run main RDD under each spec in ROBUSTNESS_SPECS at each cutoff and outcome.
    Saves the comparison table for the presentation."""
    print(f"\n{'='*86}")
    print("SPEC ROBUSTNESS  (primary spec marked [primary])")
    print(f"{'='*86}")
    rows = []
    for c in CUTOFFS:
        for y in outcomes:
            print(f"\nc = {c:.2f}   outcome = {y.replace('y_log_','')}")
            print("-" * 86)
            for name, cols in ROBUSTNESS_SPECS.items():
                try:
                    kwargs = {"c": c}
                    if cols:
                        kwargs["covs"] = get_covs_matrix(df, cols)
                    rd = rdrobust(df[y].values, df["X"].values, **kwargs)
                    e = extract(rd)
                    sig = "  *" if e["p_robust"] < 0.05 else ""
                    print(f"  {name:32s} tau={e['tau_conv']:+.3f}  "
                          f"CI=[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]  "
                          f"p={e['p_robust']:.3f}{sig}")
                    rows.append({
                        "cutoff": c, "outcome": y, "spec": name,
                        "covariates": "|".join(cols) if cols else "",
                        **e,
                    })
                except Exception as ex:
                    print(f"  {name:32s} error: {ex}")
    return pd.DataFrame(rows)


# ===== figures =====
def figure_rdd(df, y_col, c, filename, title=""):
    bw = 0.20
    sub = df[(df["X"] >= c - bw) & (df["X"] <= c + bw)].copy()
    sub = sub[["X", y_col]].dropna()

    sub["bin"] = pd.cut(sub["X"], bins=20)
    binned = sub.groupby("bin", observed=True).agg({"X": "mean", y_col: "mean"}).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(binned["X"], binned[y_col], s=40, alpha=0.6, color="#3a6fb0", zorder=3)

    left = sub[sub["X"] < c]
    right = sub[sub["X"] >= c]
    if len(left) > 10:
        cl = np.polyfit(left["X"], left[y_col], 1)
        xx = np.linspace(left["X"].min(), c, 50)
        ax.plot(xx, np.polyval(cl, xx), color="#A32D2D", linewidth=2.2, zorder=2)
    if len(right) > 10:
        cr = np.polyfit(right["X"], right[y_col], 1)
        xx = np.linspace(c, right["X"].max(), 50)
        ax.plot(xx, np.polyval(cr, xx), color="#BA7517", linewidth=2.2, zorder=2)

    ax.axvline(c, color="gray", linestyle="--", alpha=0.7, zorder=1)
    ax.set_xlabel("share of reviews positive")
    ylab = "log concurrent players" if "ccu" in y_col else "log owners (SteamSpy estimate)"
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()
    print(f"  saved {filename}")


def figure_density(df, c, filename, title=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["X"], bins=50, alpha=0.75, color="#3a6fb0", edgecolor="white")
    ax.axvline(c, color="#A32D2D", linestyle="--", linewidth=2, label=f"cutoff = {c}")
    ax.set_xlabel("share of reviews positive")
    ax.set_ylabel("number of games")
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()
    print(f"  saved {filename}")


def figure_forest(df_results, label_col, filename, title=""):
    if df_results is None or df_results.empty:
        return
    d = df_results.sort_values("tau_conv", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(d))))
    y_pos = np.arange(len(d))
    err_lo = d["tau_conv"] - d["ci_lo"]
    err_hi = d["ci_hi"] - d["tau_conv"]
    ax.errorbar(d["tau_conv"], y_pos, xerr=[err_lo, err_hi],
                fmt="o", color="#3a6fb0", capsize=3, markersize=6, linewidth=1.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lab} (n={n})" for lab, n in zip(d[label_col], d["n"])])
    ax.set_xlabel("estimated tau (95% robust CI)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()
    print(f"  saved {filename}")


def figure_bandwidth(df_sens, filename, title=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(df_sens["h_mult"], df_sens["tau_conv"],
                yerr=[df_sens["tau_conv"] - df_sens["ci_lo"],
                      df_sens["ci_hi"] - df_sens["tau_conv"]],
                fmt="o-", color="#3a6fb0", capsize=4, markersize=7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("bandwidth (multiple of MSE-optimal h)")
    ax.set_ylabel("estimated tau (95% robust CI)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()
    print(f"  saved {filename}")


def figure_compare_raw_adjusted(raw_df, adj_df, filename, title=""):
    """Side-by-side comparison of raw vs covariate-adjusted main estimates."""
    if raw_df is None or adj_df is None or raw_df.empty or adj_df.empty:
        return

    raw = raw_df.copy().reset_index(drop=True)
    adj = adj_df.copy().reset_index(drop=True)
    n = len(raw)
    labels = [f"{r['outcome'].replace('y_log_','')} @ c={r['cutoff']:.2f}"
              for _, r in raw.iterrows()]

    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.7 * n + 1)))
    offset = 0.18
    for i in range(n):
        rr = raw.iloc[i]
        ax.errorbar(rr["tau_conv"], i + offset,
                    xerr=[[rr["tau_conv"] - rr["ci_lo"]], [rr["ci_hi"] - rr["tau_conv"]]],
                    fmt="o", color="#A32D2D", capsize=3, markersize=7,
                    label="raw" if i == 0 else "")
        aa = adj.iloc[i]
        ax.errorbar(aa["tau_conv"], i - offset,
                    xerr=[[aa["tau_conv"] - aa["ci_lo"]], [aa["ci_hi"] - aa["tau_conv"]]],
                    fmt="o", color="#3a6fb0", capsize=3, markersize=7,
                    label="adjusted (leaner)" if i == 0 else "")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("estimated tau (95% robust CI)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()
    print(f"  saved {filename}")


def figure_spec_robustness(robust_df, filename, title=""):
    """Forest-style figure showing tau under each spec, faceted by cutoff x outcome."""
    if robust_df is None or robust_df.empty:
        return

    keys = robust_df[["cutoff", "outcome"]].drop_duplicates().values.tolist()
    n_panels = len(keys)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.5, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    spec_order = list(ROBUSTNESS_SPECS.keys())
    colors = {"raw": "#7a7a7a", "leaner [primary]": "#3a6fb0",
              "with_reviews [bad control]": "#A32D2D", "minimal": "#BA7517"}

    for ax, (cutoff, outcome) in zip(axes, keys):
        sub = robust_df[(robust_df["cutoff"] == cutoff) &
                        (robust_df["outcome"] == outcome)]
        sub = sub.set_index("spec").reindex(spec_order).reset_index()
        y_pos = np.arange(len(sub))
        for i, row in sub.iterrows():
            ax.errorbar(row["tau_conv"], i,
                        xerr=[[row["tau_conv"] - row["ci_lo"]],
                              [row["ci_hi"] - row["tau_conv"]]],
                        fmt="o", color=colors.get(row["spec"], "#3a6fb0"),
                        capsize=3, markersize=7, linewidth=1.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["spec"], fontsize=9)
        ax.set_title(f"c = {cutoff:.2f}   outcome = {outcome.replace('y_log_','')}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("estimated tau (95% robust CI)")
    fig.suptitle(title, y=1.0, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()
    print(f"  saved {filename}")


# ===== main =====
def main():
    df = load_and_prep()
    df.to_csv(os.path.join(TABLES_DIR, "prepped_sample.csv"), index=False)

    cov_balance_cols = [
        "log_price", "log_total_reviews", "months_since_release",
        "has_multiplayer", "platform_windows_i", "platform_mac_i",
        "dlc_count_f", "achievements_f",
    ]

    # --- VALIDITY ---
    for c in CUTOFFS:
        density_test(df, c)
        cov_bal = covariate_balance(df, c, cov_balance_cols)
        cov_bal.to_csv(os.path.join(TABLES_DIR, f"cov_balance_{int(c*100)}.csv"), index=False)

    # placebos: raw and primary-adjusted (leaner spec)
    placebo_40_raw = placebo_cutoffs(df, "y_log_ccu", [0.30, 0.35, 0.45, 0.50])
    placebo_70_raw = placebo_cutoffs(df, "y_log_ccu", [0.60, 0.65, 0.75, 0.80])
    placebo_40_adj = placebo_cutoffs(df, "y_log_ccu", [0.30, 0.35, 0.45, 0.50],
                                      covs_cols=ADJUST_COVS)
    placebo_70_adj = placebo_cutoffs(df, "y_log_ccu", [0.60, 0.65, 0.75, 0.80],
                                      covs_cols=ADJUST_COVS)
    placebo_40_raw.to_csv(os.path.join(TABLES_DIR, "placebo_40_raw.csv"), index=False)
    placebo_70_raw.to_csv(os.path.join(TABLES_DIR, "placebo_70_raw.csv"), index=False)
    placebo_40_adj.to_csv(os.path.join(TABLES_DIR, "placebo_40_adj.csv"), index=False)
    placebo_70_adj.to_csv(os.path.join(TABLES_DIR, "placebo_70_adj.csv"), index=False)

    # --- MAIN ESTIMATES: raw and covariate-adjusted (leaner) ---
    raw_estimates = []
    adj_estimates = []
    for c in CUTOFFS:
        for y, lab in [("y_log_ccu", "CCU"), ("y_log_owners", "owners")]:
            e_raw = main_rdd(df, y, c, label=f"MAIN raw {lab}")
            raw_estimates.append({"outcome": y, "cutoff": c, **e_raw})
            e_adj = main_rdd(df, y, c, label=f"MAIN adjusted {lab}",
                              covs_cols=ADJUST_COVS)
            adj_estimates.append({"outcome": y, "cutoff": c, **e_adj})
    raw_df = pd.DataFrame(raw_estimates)
    adj_df = pd.DataFrame(adj_estimates)
    raw_df.to_csv(os.path.join(TABLES_DIR, "main_estimates_raw.csv"), index=False)
    adj_df.to_csv(os.path.join(TABLES_DIR, "main_estimates_adjusted.csv"), index=False)

    # --- DONUT RDD ---
    donut_rows = []
    for c in CUTOFFS:
        r1 = donut_rdd(df, "y_log_ccu", c, hole=0.005)
        if r1: donut_rows.append({"outcome": "y_log_ccu", **r1})
        r2 = donut_rdd(df, "y_log_ccu", c, hole=0.005, covs_cols=ADJUST_COVS)
        if r2: donut_rows.append({"outcome": "y_log_ccu", **r2})
    pd.DataFrame(donut_rows).to_csv(os.path.join(TABLES_DIR, "donut.csv"), index=False)

    # --- HETEROGENEITY (covariate-adjusted with leaner spec) ---
    genre_40 = per_genre_rdd(df, "y_log_ccu", 0.40, covs_cols=ADJUST_COVS)
    genre_70 = per_genre_rdd(df, "y_log_ccu", 0.70, covs_cols=ADJUST_COVS)
    genre_40.to_csv(os.path.join(TABLES_DIR, "genre_40.csv"), index=False)
    genre_70.to_csv(os.path.join(TABLES_DIR, "genre_70.csv"), index=False)

    cohort_40 = per_cohort_rdd(df, "y_log_owners", 0.40, covs_cols=ADJUST_COVS)
    cohort_70 = per_cohort_rdd(df, "y_log_owners", 0.70, covs_cols=ADJUST_COVS)
    cohort_40.to_csv(os.path.join(TABLES_DIR, "cohort_40.csv"), index=False)
    cohort_70.to_csv(os.path.join(TABLES_DIR, "cohort_70.csv"), index=False)

    # --- SENSITIVITY (covariate-adjusted with leaner spec) ---
    bw_40 = bandwidth_sensitivity(df, "y_log_ccu", 0.40, covs_cols=ADJUST_COVS)
    bw_70 = bandwidth_sensitivity(df, "y_log_ccu", 0.70, covs_cols=ADJUST_COVS)
    bw_40.to_csv(os.path.join(TABLES_DIR, "bandwidth_40.csv"), index=False)
    bw_70.to_csv(os.path.join(TABLES_DIR, "bandwidth_70.csv"), index=False)

    # --- SPEC ROBUSTNESS ---
    robust_df = spec_robustness(df)
    robust_df.to_csv(os.path.join(TABLES_DIR, "spec_robustness.csv"), index=False)

    # --- FIGURES ---
    print("\n=== figures ===")
    figure_rdd(df, "y_log_ccu", 0.40, "rdd_40_ccu.png",
               title="RDD at 40% cutoff: Mostly Negative -> Mixed (bundled with deprioritization)")
    figure_rdd(df, "y_log_ccu", 0.70, "rdd_70_ccu.png",
               title="RDD at 70% cutoff: Mixed -> Mostly Positive (label only)")
    figure_rdd(df, "y_log_owners", 0.40, "rdd_40_owners.png",
               title="RDD at 40% cutoff (owners outcome)")
    figure_rdd(df, "y_log_owners", 0.70, "rdd_70_owners.png",
               title="RDD at 70% cutoff (owners outcome)")
    figure_density(df, 0.40, "density_40.png",
                   title="Density of pct_positive (no-manipulation check at 40%)")
    figure_density(df, 0.70, "density_70.png",
                   title="Density of pct_positive (no-manipulation check at 70%)")
    figure_forest(genre_40, "genre", "forest_genre_40.png",
                  "Per-genre tau at 40% (CCU, leaner adjustment)")
    figure_forest(genre_70, "genre", "forest_genre_70.png",
                  "Per-genre tau at 70% (CCU, leaner adjustment)")
    figure_forest(cohort_40, "cohort", "forest_cohort_40.png",
                  "Per-cohort tau at 40% (owners, leaner adjustment)")
    figure_forest(cohort_70, "cohort", "forest_cohort_70.png",
                  "Per-cohort tau at 70% (owners, leaner adjustment)")
    figure_bandwidth(bw_40, "bandwidth_40.png",
                     "Bandwidth sensitivity at 40% (CCU, leaner adjustment)")
    figure_bandwidth(bw_70, "bandwidth_70.png",
                     "Bandwidth sensitivity at 70% (CCU, leaner adjustment)")
    figure_compare_raw_adjusted(raw_df, adj_df, "compare_raw_adjusted.png",
                                "Main estimates: raw vs covariate-adjusted (leaner spec)")
    figure_spec_robustness(robust_df, "spec_robustness.png",
                           "Spec robustness: tau under four adjustment specifications")

    print("\ndone. tables -> tables/, figures -> figures/")


if __name__ == "__main__":
    main()