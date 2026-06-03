import pandas as pd

df = pd.read_csv("steam_rdd_data.csv")
df["release_dt"]  = pd.to_datetime(df["release_date"],   errors="coerce")
df["snapshot_dt"] = pd.to_datetime(df["collected_utc"],  errors="coerce", utc=True).dt.tz_localize(None)
df["days_since_release"] = (df["snapshot_dt"] - df["release_dt"]).dt.days

print(f"snapshot date range: {df.snapshot_dt.min()} to {df.snapshot_dt.max()}")
print(f"release date range:  {df.release_dt.min()} to {df.release_dt.max()}\n")

for max_days in [90, 180, 365, 730, 1095]:
    sub = df[(df["days_since_release"] <= max_days) & (df["days_since_release"] >= 0)]
    n_total = len(sub)
    # focal RDD window at 40%: pct_positive in [0.20, 0.60)
    focal = sub[(sub["pct_positive"] >= 0.20) & (sub["pct_positive"] < 0.60)]
    n_left  = ((focal["pct_positive"] >= 0.20) & (focal["pct_positive"] < 0.40) & (focal["total_reviews"] >= 20)).sum()
    n_right = ((focal["pct_positive"] >= 0.40) & (focal["pct_positive"] < 0.60) & (focal["total_reviews"] >= 20)).sum()
    n_left_strict  = ((focal["pct_positive"] >= 0.20) & (focal["pct_positive"] < 0.40) & (focal["total_reviews"] >= 50)).sum()
    n_right_strict = ((focal["pct_positive"] >= 0.40) & (focal["pct_positive"] < 0.60) & (focal["total_reviews"] >= 50)).sum()
    print(f"<= {max_days:4d} days since release: {n_total:4d} total | "
          f"focal 40% window: L={n_left:3d} R={n_right:3d} (reviews>=20)  "
          f"L={n_left_strict:3d} R={n_right_strict:3d} (reviews>=50)")