# Regression Discontinuity — Steam Review Labels

A regression-discontinuity (RDD) study of how **Steam's review-summary label**
affects a game's visibility and player base. Steam assigns a discrete label
(e.g. *Mostly Negative*, *Mixed*, *Mostly Positive*) based on the share of
positive reviews. Because the label flips at sharp thresholds of that share,
games that land just below versus just above a cutoff are otherwise comparable —
a natural experiment for the causal effect of the label itself.

- **Running variable:** `pct_positive = total_positive / total_reviews` (Steam's canonical share)
- **Cutoffs:**
  - `0.40` — *Mostly Negative → Mixed* (label change bundled with algorithmic deprioritization)
  - `0.70` — *Mixed → Mostly Positive* (label-only change)
- **Outcomes:** concurrent players (live), SteamSpy owner estimate, peak CCU
- **Covariates:** price, release date, genres, categories, tags, developer,
  publisher, platforms, language count, age rating, DLC count, Metacritic, etc.

## Repository layout

| Path | What it is |
|------|------------|
| `main.py` | Two-phase data collection: screen SteamSpy's `all` pages by review share, then enrich each candidate via the Steam store / web APIs. Writes `steam_rdd_data.csv`. |
| `analyze.py` | Full RDD pipeline: cleaning, main estimates, covariate-adjusted estimates, placebo cutoffs, bandwidth sensitivity, density (manipulation) tests, donut RDD, and per-genre / per-cohort heterogeneity. Writes `figures/*.png` and `tables/*.csv`. |
| `steam_rdd_data.csv` | The raw collected dataset (already included so the analysis runs without re-fetching). |
| `figures/` | Generated plots (RDD plots, density tests, bandwidth curves, forest plots, raw-vs-adjusted comparison). |
| `tables/` | Generated result tables, plus `prepped_sample.csv` (the cleaned analysis sample). |
| `requirements.txt` | Pinned Python dependencies. |

## Recreate the environment

Requires **Python 3.14** (the pinned versions were built against it; 3.11+ will
generally work too).

```bash
# 1. Clone
git clone https://github.com/kartikeyamandhar/Regression-Discontinuity.git
cd Regression-Discontinuity

# 2. Create and activate a virtual environment
python3 -m venv comp_env
source comp_env/bin/activate        # Windows: comp_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Reproduce the analysis

The dataset is committed, so you can go straight to the analysis:

```bash
python analyze.py
```

This reads `steam_rdd_data.csv` and regenerates everything under `figures/` and
`tables/`.

## (Optional) Re-collect the data from scratch

Only needed if you want a fresh pull from the live Steam / SteamSpy APIs. This
makes many rate-limited network calls and takes a while.

```bash
# Delete the existing CSV first — the script appends and checks the schema.
rm steam_rdd_data.csv
python main.py
```

SteamSpy's bulk `all` endpoint is limited to one request per 60 seconds, so the
seeding phase is intentionally slow. Pacing knobs (`SLEEP_BETWEEN`,
`SEED_MAX_PAGES`, `TARGET_WINDOWS`, etc.) are at the top of `main.py`.

## Notes

- No API keys are required — all endpoints used are public.
- The random seed in `analyze.py` (`SEED = 410014`) fixes any stochastic steps.
- The `comp_env/` virtual environment is intentionally **not** committed; rebuild
  it from `requirements.txt`.
