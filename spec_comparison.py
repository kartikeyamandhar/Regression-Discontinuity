"""
spec_comparison.py — compare RDD estimates under four adjustment specs.

Tests Zahid's hypothesis: log_total_reviews is the denominator of pct_positive
AND is downstream of the policy (deprioritized games accumulate fewer reviews).
Including it in the covariate stack may absorb the treatment effect.

This script reruns the main RDD at both cutoffs and both outcomes under:
  1. raw                     — no covariates
  2. current adjustment      — log_price, log_total_reviews, months_since_release,
                               has_multiplayer, dlc_count  (the analyze.py spec)
  3. leaner adjustment       — drop log_total_reviews
  4. minimal adjustment      — only pre-launch covariates: price + release timing

Run from comp_project/:
    python spec_comparison.py
"""

import numpy as np
import pandas as pd
from rdrobust import rdrobust


INPUT_CSV = "steam_rdd_data.csv"


def covs_mat(df, cols):
    m = df[cols].copy()
    for c in cols:
        m[c] = m[c].fillna(m[c].median() if m[c].notna().any() else 0)
    return m.values


def load_and_prep():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["type"].fillna("game") == "game"]
    df = df[~df["coming_soon"].fillna(False).astype(bool)]
    df = df[df["total_reviews"] >= 50]
    df["y_log_ccu"]    = np.log1p(df["concurrent_players"].fillna(0))
    df["y_log_owners"] = np.log1p(df["owners_estimate"].fillna(0))
    df["X"] = df["pct_positive"]
    df["release_dt"]   = pd.to_datetime(df["release_date"], errors="coerce")
    df["snapshot_dt"]  = pd.to_datetime(df["collected_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    df["months_since_release"] = (df["snapshot_dt"] - df["release_dt"]).dt.days / 30.44
    df["log_price"]         = np.log1p(df["price_cents"].fillna(0))
    df["log_total_reviews"] = np.log(df["total_reviews"])
    df["has_multiplayer"]   = df["categories"].fillna("").str.contains("Multi-player").astype(int)
    df["dlc_count_f"]       = df["dlc_count"].fillna(0)
    df = df[df["release_dt"].dt.year >= 2018]
    return df


SPECS = {
    "raw (no adjustment)":            None,
    "current (with reviews)":         ["log_price", "log_total_reviews", "months_since_release", "has_multiplayer", "dlc_count_f"],
    "leaner (drop log_total_reviews)":["log_price", "months_since_release", "has_multiplayer", "dlc_count_f"],
    "minimal (price + timing only)":  ["log_price", "months_since_release"],
}


def main():
    df = load_and_prep()
    print(f"n = {len(df)} after filters\n")
    print("=" * 86)
    print(f"{'spec':<35s} {'tau':>9s}  {'robust 95% CI':>22s}  {'p':>7s}  {'sig':>4s}")
    print("=" * 86)

    for cutoff in (0.40, 0.70):
        for y in ("y_log_ccu", "y_log_owners"):
            outcome = y.replace("y_log_", "")
            print(f"\nc = {cutoff:.2f}   outcome = {outcome}")
            print("-" * 86)
            for name, cols in SPECS.items():
                try:
                    kwargs = {"c": cutoff}
                    if cols:
                        kwargs["covs"] = covs_mat(df, cols)
                    rd = rdrobust(df[y].values, df["X"].values, **kwargs)
                    tau = float(rd.coef.iloc[0, 0])
                    ci_lo = float(rd.ci.iloc[2, 0])
                    ci_hi = float(rd.ci.iloc[2, 1])
                    pv = float(rd.pv.iloc[2, 0])
                    sig = "  *" if pv < 0.05 else ""
                    print(f"{name:<35s} {tau:+8.3f}  [{ci_lo:+7.3f}, {ci_hi:+7.3f}]  {pv:7.3f} {sig}")
                except Exception as e:
                    print(f"{name:<35s} error: {e}")
    print()


if __name__ == "__main__":
    main()