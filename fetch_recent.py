"""
fetch_recent.py — pull recent Steam releases for the launch-window subsample.

Robust against Steam store search 429 rate-limiting:
  - Phase 1 saves candidate app IDs to phase1_candidates.txt as they're collected
  - 429 handling uses exponential backoff (60s, 120s, 240s), then bails Phase 1
    and proceeds to Phase 2 with whatever has been collected so far
  - Fully resumable: Ctrl-C and rerun; both phases pick up where they left off

Outputs:
  phase1_candidates.txt   one app ID per line; appended-to incrementally
  steam_rdd_recent.csv    enriched, filtered rows matching steam_rdd_data.csv schema

Run from comp_project/:
    python fetch_recent.py
"""

import os
import re
import time
import signal
import datetime as dt

import requests
import pandas as pd


# ===== config =====
OUTPUT_CSV             = "steam_rdd_recent.csv"
CANDIDATES_FILE        = "phase1_candidates.txt"
MAX_DAYS_SINCE_RELEASE = 365
MIN_REVIEWS_SCREEN     = 20
SLEEP_BETWEEN_API      = 0.75
SLEEP_BETWEEN_STORE    = 1.5     # bumped up; Steam search is touchy
MAX_PAGES_STORE        = 200
TIMEOUT                = 25
USER_AGENT             = "rdd-research-project/1.0 (academic; MGMTMSA 409)"
FETCH_STEAMSPY         = True

# 429 backoff schedule for store search (seconds)
BACKOFF_SCHEDULE       = [60, 120, 240]   # bail after 3 consecutive 429s exhaust this

SCREEN_WINDOWS = [(0.20, 0.60), (0.55, 0.85)]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

NOW = dt.datetime.utcnow()


# ===== graceful Ctrl-C =====
_stop = False
def _sigint(sig, frame):
    global _stop
    print("\n[stop requested] will exit after current item")
    _stop = True
signal.signal(signal.SIGINT, _sigint)


# ===== Phase 1: collect recent app IDs from store search =====
def load_existing_candidates():
    if not os.path.exists(CANDIDATES_FILE):
        return []
    with open(CANDIDATES_FILE) as f:
        return [int(x.strip()) for x in f if x.strip() and x.strip().isdigit()]


def append_candidates(new_ids):
    """Append new IDs to disk immediately so Ctrl-C / 429 can't lose them."""
    if not new_ids:
        return
    with open(CANDIDATES_FILE, "a") as f:
        for aid in new_ids:
            f.write(f"{aid}\n")


def collect_recent_appids():
    seen = set(load_existing_candidates())
    print(f"phase 1: walking store search (Released_DESC), {len(seen)} IDs already on disk")
    if seen:
        print(f"  resume mode: will append new IDs not seen before")

    url = "https://store.steampowered.com/search/results/"
    consecutive_429 = 0

    for page in range(MAX_PAGES_STORE):
        if _stop:
            break
        start = page * 50
        params = {
            "query":         "",
            "start":         start,
            "count":         50,
            "dynamic_data":  "",
            "sort_by":       "Released_DESC",
            "category1":     998,
            "supportedlang": "english",
            "infinite":      1,
        }
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            consecutive_429 = 0  # reset on success
            j = r.json()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                if consecutive_429 >= len(BACKOFF_SCHEDULE):
                    print(f"  page {page:3d}  429: max retries exhausted; bailing Phase 1 with {len(seen)} IDs")
                    break
                wait = BACKOFF_SCHEDULE[consecutive_429]
                print(f"  page {page:3d}  429: backing off {wait}s ({consecutive_429+1}/{len(BACKOFF_SCHEDULE)})")
                consecutive_429 += 1
                # sleep in 10s chunks so Ctrl-C is responsive
                slept = 0
                while slept < wait and not _stop:
                    time.sleep(min(10, wait - slept))
                    slept += 10
                continue
            print(f"  page {page:3d}  HTTP {code}: sleeping 10s and continuing")
            time.sleep(10)
            continue
        except Exception as e:
            print(f"  page {page:3d}  unexpected error: {e}; sleeping 10s")
            time.sleep(10)
            continue

        html = j.get("results_html", "")
        ids = re.findall(r'data-ds-appid="(\d+)"', html)
        new_ids = []
        for aid in ids:
            aid_i = int(aid)
            if aid_i in seen:
                continue
            seen.add(aid_i)
            new_ids.append(aid_i)
        append_candidates(new_ids)   # persist immediately
        print(f"  page {page:3d}  +{len(new_ids)} new (cum {len(seen)})")
        if not new_ids:
            print("  page yielded zero new IDs; assuming end of catalog or fully resumed; stopping")
            break
        time.sleep(SLEEP_BETWEEN_STORE)

    return sorted(seen)


# ===== enrichment endpoints (Phase 2) =====
def get_appreviews(appid):
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {"json": 1, "language": "all", "purchase_type": "all", "num_per_page": 0}
    r = session.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if not j.get("success"):
        return None
    qs = j.get("query_summary", {})
    total = qs.get("total_reviews", 0)
    if total == 0:
        return None
    pos = qs.get("total_positive", 0)
    neg = qs.get("total_negative", 0)
    return {
        "total_positive":    pos,
        "total_negative":    neg,
        "total_reviews":     total,
        "pct_positive":      pos / total,
        "review_score":      qs.get("review_score"),
        "review_score_desc": qs.get("review_score_desc", ""),
    }


def get_appdetails(appid):
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": appid, "cc": "us", "l": "english"}
    r = session.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json().get(str(appid), {})
    if not payload.get("success"):
        return None
    d = payload.get("data", {})
    rd = d.get("release_date", {}) or {}
    price = (d.get("price_overview") or {}).get("final", 0) or 0
    meta = (d.get("metacritic") or {}).get("score")
    recs = (d.get("recommendations") or {}).get("total")
    ach = (d.get("achievements") or {}).get("total")
    plats = d.get("platforms", {}) or {}
    cats = d.get("categories", []) or []
    gens = d.get("genres", []) or []
    langs = d.get("supported_languages", "") or ""
    return {
        "name":               d.get("name", ""),
        "type":               d.get("type", ""),
        "coming_soon":        rd.get("coming_soon", False),
        "release_date":       rd.get("date", ""),
        "price_cents":        price,
        "is_free":            d.get("is_free", False),
        "metacritic_score":   meta,
        "recommendations":    recs,
        "achievements_total": ach,
        "platform_windows":   plats.get("windows", False),
        "platform_mac":       plats.get("mac", False),
        "platform_linux":     plats.get("linux", False),
        "n_languages":        len([s for s in langs.split(",") if s.strip()]) if langs else 0,
        "dlc_count":          len(d.get("dlc") or []),
        "categories":         "|".join([c.get("description","") for c in cats]),
        "genres":             "|".join([g.get("description","") for g in gens]),
        "controller_support": d.get("controller_support", ""),
        "developers":         "|".join(d.get("developers", []) or []),
        "publishers":         "|".join(d.get("publishers", []) or []),
    }


def get_current_players(appid):
    url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
    r = session.get(url, params={"appid": appid}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", {}).get("player_count", 0) or 0


def get_steamspy(appid):
    url = "https://steamspy.com/api.php"
    r = session.get(url, params={"request": "appdetails", "appid": appid}, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json() or {}
    owners_str = j.get("owners", "") or ""
    nums = re.findall(r"[\d,]+", owners_str)
    lo = hi = 0
    if len(nums) >= 2:
        try:
            lo = int(nums[0].replace(",", ""))
            hi = int(nums[1].replace(",", ""))
        except ValueError:
            pass
    tags = j.get("tags") or {}
    tags_str = "|".join(tags.keys()) if isinstance(tags, dict) else ""
    langs = j.get("languages") or ""
    return {
        "owners_lo":             lo,
        "owners_hi":             hi,
        "owners_estimate":       (lo + hi) // 2,
        "steamspy_tags":         tags_str,
        "steamspy_n_languages":  len([s for s in langs.split(",") if s.strip()]) if langs else 0,
    }


def parse_release_date(s):
    if not s:
        return None
    s = s.strip()
    for fmt in ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%d %B, %Y", "%b %Y", "%Y"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ===== main =====
def main():
    candidates = collect_recent_appids()
    print(f"\nphase 1 complete: {len(candidates)} unique candidate app IDs on disk\n")

    existing = pd.DataFrame()
    done_ids = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(existing["appid"].astype(int).tolist())
        print(f"phase 2 resume: {len(done_ids)} app IDs already enriched in {OUTPUT_CSV}")

    todo = [a for a in candidates if a not in done_ids]
    print(f"phase 2: enriching {len(todo)} candidates ...\n")

    rows = []
    kept = 0
    for i, appid in enumerate(todo):
        if _stop:
            break
        try:
            rev = get_appreviews(appid)
            time.sleep(SLEEP_BETWEEN_API)
            if rev is None or rev["total_reviews"] < MIN_REVIEWS_SCREEN:
                continue
            pct = rev["pct_positive"]
            if not any(lo <= pct < hi for lo, hi in SCREEN_WINDOWS):
                continue

            det = get_appdetails(appid)
            time.sleep(SLEEP_BETWEEN_API)
            if det is None or det["type"] != "game" or det["coming_soon"]:
                continue
            rd = parse_release_date(det["release_date"])
            if rd is None:
                continue
            days_since = (NOW - rd).days
            if days_since < 0 or days_since > MAX_DAYS_SINCE_RELEASE:
                continue

            ccu = get_current_players(appid)
            time.sleep(SLEEP_BETWEEN_API)

            if FETCH_STEAMSPY:
                try:
                    spy = get_steamspy(appid)
                except Exception:
                    spy = None
                time.sleep(SLEEP_BETWEEN_API)
            else:
                spy = None
            if spy is None:
                spy = {"owners_lo": 0, "owners_hi": 0, "owners_estimate": 0,
                       "steamspy_tags": "", "steamspy_n_languages": 0}

            row = {
                "appid":              appid,
                "collected_utc":      NOW.isoformat(),
                **rev, **det, **spy,
                "concurrent_players": ccu,
            }
            rows.append(row)
            kept += 1
            print(f"  [{i+1:4d}/{len(todo)}] appid={appid:>10} "
                  f"pct={pct:.3f}  N_rev={rev['total_reviews']:>5}  "
                  f"released={det['release_date']:>15} ({days_since}d ago)  "
                  f"CCU={ccu:>6}  [kept {kept}]")

            if len(rows) % 10 == 0:
                save(rows, existing)

        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                print(f"  [{i+1}/{len(todo)}] appid={appid} 429; sleeping 60s")
                time.sleep(60)
            else:
                print(f"  [{i+1}/{len(todo)}] appid={appid} HTTP {code}; skipping")
                time.sleep(SLEEP_BETWEEN_API)
            continue
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] appid={appid} error: {type(e).__name__}: {e}")
            time.sleep(SLEEP_BETWEEN_API)
            continue

    save(rows, existing)
    total = len(existing) + len(rows)
    print(f"\ndone. new rows: {len(rows)}.  total in {OUTPUT_CSV}: {total}")


def save(rows, existing):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    df = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()