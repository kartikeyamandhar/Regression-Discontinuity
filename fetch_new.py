"""
fetch_review_streams.py

Foundation for the crossing event-study design.

WHY THIS DESIGN
The cross-sectional RDD cannot cleanly identify Steam's documented 40%
deprioritization: it uses a stock outcome for a flow treatment on a sample that
selects on the post-treatment outcome. The fix is a within-game event study
around the moment a game crosses pct_positive = 0.40, using the game as its own
control (game fixed effects) and calendar-week fixed effects.

Valve's own statement (PCGamesN / Valve blog): the visibility boost is flat
across all Mixed-or-above buckets, with a >500% drop below 40%. So 40% is the
only real algorithmic discontinuity; 70% becomes a falsification cutoff.

TARGET POPULATION (where 40% crossings actually happen, per the literature):
  - games whose CURRENT pct_positive is in [0.30, 0.55] (likely to have crossed), OR
  - Early Access games (documented as the most score-volatile; crossings are
    patch-driven, giving quasi-random crossing timing)
drawn from app IDs already collected in steam_rdd_data.csv (+ steam_rdd_recent.csv).

WHAT THIS SCRIPT DOES
For each target game, paginate the FULL dated review stream (Steam appreviews,
cursor pagination) and save raw reviews to review_streams/{appid}.csv with:
    timestamp_created, voted_up, written_during_early_access, votes_up

From this ONE pull, the reconstruction step (next script) builds:
  - weekly cumulative pct_positive       (running variable: overall bucket)
  - weekly trailing-90-day pct_positive  (dynamic bucket Steam surfaces)
  - weekly review velocity               (demand proxy / outcome)
and locates the 40% crossing week for each game.

ToS-clean: reads only the public review stream. No third-party scraping.
Resumable: re-run after Ctrl-C; finished games are skipped.

Run from comp_project/:
    python fetch_review_streams.py
"""

import os
import csv
import time
import signal
import datetime as dt

import requests
import pandas as pd


# ===== config =====
INPUT_CSVS           = ["steam_rdd_data.csv", "steam_rdd_recent.csv"]
OUTPUT_DIR           = "review_streams"
PROGRESS_FILE        = "review_streams_done.txt"

# target selection
PCT_LOW, PCT_HIGH    = 0.30, 0.55     # near-threshold window (likely crossers)
INCLUDE_EARLY_ACCESS = True
MIN_TOTAL_REVIEWS    = 120            # need enough reviews for weekly resolution

# fetch params
NUM_PER_PAGE         = 100
MAX_PAGES_PER_GAME   = 150            # cap ~15k reviews/game; bounds runtime
SLEEP_BETWEEN_PAGES  = 1.2            # appreviews is rate-limited
TIMEOUT              = 25
USER_AGENT           = "rdd-research-project/1.0 (academic; MGMTMSA 409)"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ===== graceful Ctrl-C =====
_stop = False
def _sigint(sig, frame):
    global _stop
    print("\n[stop requested] finishing current game then exiting")
    _stop = True
signal.signal(signal.SIGINT, _sigint)


# ===== target list =====
def build_target_list():
    frames = []
    for path in INPUT_CSVS:
        if os.path.exists(path):
            f = pd.read_csv(path)
            frames.append(f)
            print(f"  loaded {path}: {len(f)} rows")
    if not frames:
        raise SystemExit("no input CSVs found; need steam_rdd_data.csv at minimum")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["appid"], keep="last")
    df = df[df["total_reviews"] >= MIN_TOTAL_REVIEWS]

    near = (df["pct_positive"] >= PCT_LOW) & (df["pct_positive"] <= PCT_HIGH)
    ea = df["genres"].fillna("").str.contains("Early Access") if INCLUDE_EARLY_ACCESS \
        else pd.Series(False, index=df.index)
    mask = near | ea

    targets = df[mask][["appid", "name", "pct_positive", "total_reviews", "genres"]].copy()
    targets = targets.sort_values("pct_positive").reset_index(drop=True)
    print(f"  target games: {len(targets)}  "
          f"(near-threshold {int(near.sum())}, early-access {int(ea.sum())}, "
          f"overlap counted once)")
    return targets


# ===== review-stream fetch =====
def fetch_stream(appid):
    """Paginate the full dated review stream. Returns list of dicts (newest-first)."""
    reviews = []
    cursor = "*"
    seen = set()
    for _ in range(MAX_PAGES_PER_GAME):
        params = {
            "json": 1,
            "filter": "recent",        # chronological; paginate backward through time
            "language": "all",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": NUM_PER_PAGE,
            "cursor": cursor,          # requests URL-encodes this once (correct)
        }
        r = session.get(f"https://store.steampowered.com/appreviews/{appid}",
                        params=params, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        if not j.get("success"):
            break
        batch = j.get("reviews", [])
        if not batch:
            break
        for rev in batch:
            reviews.append({
                "timestamp_created": rev.get("timestamp_created"),
                "voted_up": int(bool(rev.get("voted_up"))),
                "written_during_early_access": int(bool(rev.get("written_during_early_access"))),
                "votes_up": rev.get("votes_up", 0),
            })
        nxt = j.get("cursor", "")
        if not nxt or nxt in seen:
            break
        seen.add(nxt)
        cursor = nxt
        time.sleep(SLEEP_BETWEEN_PAGES)
    return reviews


# ===== resume bookkeeping =====
def load_done():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        return {int(x.strip()) for x in f if x.strip().isdigit()}


def mark_done(appid):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{appid}\n")


# ===== main =====
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("building target list ...")
    targets = build_target_list()

    done = load_done()
    todo = targets[~targets["appid"].isin(done)].reset_index(drop=True)
    print(f"\n{len(done)} already fetched, {len(todo)} to go\n")

    n_crossers_hint = 0
    for i, row in enumerate(todo.itertuples()):
        if _stop:
            break
        appid = int(row.appid)
        out_path = os.path.join(OUTPUT_DIR, f"{appid}.csv")
        try:
            stream = fetch_stream(appid)
            if not stream:
                print(f"  [{i+1:4d}/{len(todo)}] appid={appid:>9}  no reviews; skip")
                mark_done(appid)
                continue

            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "timestamp_created", "voted_up",
                    "written_during_early_access", "votes_up"])
                w.writeheader()
                w.writerows(stream)
            mark_done(appid)

            # quick crossing hint on the CUMULATIVE share (post-stabilization)
            s = pd.DataFrame(stream).sort_values("timestamp_created").reset_index(drop=True)
            cum_pos = s["voted_up"].cumsum().values
            cum_tot = (s.index + 1).values
            cum_pct = cum_pos / cum_tot
            final_pct = cum_pct[-1]
            crossed = False
            start = 30  # ignore noisy first reviews
            if len(cum_pct) > start + 5:
                seg = cum_pct[start:]
                up = ((seg[:-1] < 0.40) & (seg[1:] >= 0.40)).any()
                dn = ((seg[:-1] >= 0.40) & (seg[1:] < 0.40)).any()
                crossed = bool(up or dn)
            if crossed:
                n_crossers_hint += 1
            print(f"  [{i+1:4d}/{len(todo)}] appid={appid:>9}  "
                  f"{len(stream):>5} reviews  final_cum_pct={final_pct:.3f}  "
                  f"{'CROSSES_40(cum)' if crossed else ''}")

        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                print(f"  [{i+1}/{len(todo)}] appid={appid} 429; sleeping 60s")
                time.sleep(60)
            else:
                print(f"  [{i+1}/{len(todo)}] appid={appid} HTTP {code}; skip")
            continue
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] appid={appid} error: {type(e).__name__}: {e}")
            time.sleep(2)
            continue

    total_files = len([x for x in os.listdir(OUTPUT_DIR) if x.endswith(".csv")])
    print(f"\ndone. {total_files} review-stream files in {OUTPUT_DIR}/")
    print(f"cumulative-crossing hint this run: {n_crossers_hint} games "
          f"(the trailing-window reconstruction will find more)")


if __name__ == "__main__":
    main()