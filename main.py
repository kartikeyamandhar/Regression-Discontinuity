"""
steam_rdd_fetch.py

Two-phase fetch for the Steam review-label RDD design.

Phase 1: SCREEN. Walk SteamSpy 'all' pages, compute pct_positive from the
positive/negative counts in each row, and keep only games whose share lands
in a target window for the 40% or 70% cutoff. This cuts wasted enrichment
calls on the right tail of Overwhelmingly Positive games that contribute
nothing to the RDD.

Phase 2: ENRICH. For each screened candidate, hit Steam appreviews (for the
canonical running variable and the displayed label), appdetails (for the
release-year filter and the covariate panel), GetNumberOfCurrentPlayers (the
live outcome), and SteamSpy appdetails (tags and language list only; the
bulk SteamSpy fields are already captured from the seed). Pre-2018 games
are short-circuited after appdetails.

  Running variable : pct_positive = total_positive / total_reviews (Steam canonical)
  Treatment        : Steam review summary label
                     cutoff 0.40 (Mostly Negative -> Mixed; bundled with
                                  algorithmic deprioritization)
                     cutoff 0.70 (Mixed -> Mostly Positive; label-only)
  Outcomes:
    primary        : concurrent_players (live)
    cross-check    : owners_estimate (SteamSpy bucketed midpoint)
                     peak_ccu_yesterday
  Covariates       : price, release date, genres, categories, tags,
                     developer, publisher, platforms, languages, age,
                     dlc count, metacritic, recommendations, achievements

Run LOCALLY. Schema differs from prior versions; if an old steam_rdd_data.csv
exists in the working directory, delete it before running.
Requires: pip install requests
"""

import csv
import json
import os
import random
import re
import sys
import time

import requests

# ===== configuration =====
OUTPUT_CSV             = "steam_rdd_data.csv"
REQUEST_TIMEOUT        = 20

# Per-app pacing in the enrichment phase. Each app hits two store-endpoint
# calls (appreviews + appdetails) which share the rate limit; the
# backoff in get_json absorbs occasional 429s. Raise this by 0.25 if 429s
# start firing visibly in the log.
SLEEP_BETWEEN          = 0.75

# SteamSpy 'all' is rate-limited to one request per 60 seconds.
SLEEP_SEED_PAGE        = 61

CHECKPOINT_EVERY       = 5
MIN_RELEASE_YEAR       = 2018
MIN_REVIEWS_FOR_SCREEN = 20         # cheap noise filter at seed time

# Rating windows. Each window is sampled up to PER_WINDOW_CAP candidates.
# Window 0 covers the 40% cutoff with bandwidth; window 1 covers the 70%.
TARGET_WINDOWS         = [(0.20, 0.55), (0.55, 0.85)]
PER_WINDOW_CAP         = 1500
PER_GENRE_CAP          = None       # optional balance; None = off

# How deep to walk SteamSpy 'all'. Each page is ~1000 games (owner-sorted,
# descending). Deeper pages have more low-rated and low-owner titles, which
# is where the 40% cutoff mass lives.
SEED_MAX_PAGES         = 20

# Tags and the language list come from SteamSpy appdetails (one call per app,
# ~1 second). All other SteamSpy fields are read from the seed bulk data.
FETCH_STEAMSPY_DETAILS = True

HEADERS = {"User-Agent": "rdd-research-script/1.0"}

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")


# ===== helpers =====

def get_json(url, params=None, max_retries=5):
    """GET parsed JSON with exponential backoff on 429s and transient errors."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS,
                              timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep((2 ** attempt) + random.random())
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError):
            time.sleep((2 ** attempt) + random.random())
    return None


def parse_release_year(s):
    if not s:
        return None
    m = YEAR_RE.search(s)
    return int(m.group()) if m else None


def safe_int(x):
    try:
        return int(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def count_languages(s):
    if not s:
        return None
    clean = HTML_TAG_RE.sub("", s).replace("*", "")
    return len([p for p in clean.split(",") if p.strip()])


def parse_owners_range(s):
    if not s:
        return None
    try:
        lo_s, hi_s = s.replace(",", "").split("..")
        return (int(lo_s.strip()) + int(hi_s.strip())) // 2
    except (ValueError, AttributeError):
        return None


def format_top_tags(tags_obj, top_n=10):
    if not isinstance(tags_obj, dict) or not tags_obj:
        return ""
    items = sorted(tags_obj.items(), key=lambda kv: -(kv[1] or 0))[:top_n]
    return "|".join(f"{t}:{c}" for t, c in items)


# ===== Phase 1: seed and screen =====

def seed_and_screen():
    """Walk SteamSpy 'all' pages; screen each row by pct_positive against
    TARGET_WINDOWS. Returns dict[appid -> seed_record]. Stops early once
    every target window has hit PER_WINDOW_CAP.
    """
    seed = {}
    window_counts = [0] * len(TARGET_WINDOWS)
    genre_counts = {}

    print(f"phase 1: seeding from SteamSpy with windows={TARGET_WINDOWS}, "
          f"cap={PER_WINDOW_CAP}/window, max {SEED_MAX_PAGES} pages",
          file=sys.stderr)

    for page in range(SEED_MAX_PAGES):
        if all(c >= PER_WINDOW_CAP for c in window_counts):
            print(f"all windows full; stopping seed at page {page}", file=sys.stderr)
            break

        data = get_json("https://steamspy.com/api.php",
                        {"request": "all", "page": page})
        if not data:
            print(f"seed page {page}: empty / failed; stopping", file=sys.stderr)
            break

        added = 0
        for appid_s, info in data.items():
            try:
                pos = int(info.get("positive") or 0)
                neg = int(info.get("negative") or 0)
                total = pos + neg
                if total < MIN_REVIEWS_FOR_SCREEN:
                    continue
                pct = pos / total
                widx = next((i for i, (lo, hi) in enumerate(TARGET_WINDOWS)
                             if lo <= pct < hi), None)
                if widx is None:
                    continue
                if window_counts[widx] >= PER_WINDOW_CAP:
                    continue
                if PER_GENRE_CAP is not None:
                    g = (info.get("genre") or "").split(",")[0].strip()
                    if genre_counts.get(g, 0) >= PER_GENRE_CAP:
                        continue
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                appid = int(appid_s)
                if appid in seed:
                    continue
                seed[appid] = info
                window_counts[widx] += 1
                added += 1
            except (ValueError, TypeError, ZeroDivisionError):
                continue

        totals = " | ".join(f"w{i}={c}" for i, c in enumerate(window_counts))
        print(f"  seed page {page}: +{added} candidates | {totals}",
              file=sys.stderr)

        # do not sleep after the final page or if every window is full
        if page + 1 < SEED_MAX_PAGES and not all(c >= PER_WINDOW_CAP for c in window_counts):
            time.sleep(SLEEP_SEED_PAGE)

    print(f"seed phase done: {len(seed)} screened candidates", file=sys.stderr)
    return seed


# ===== Phase 2: per-app enrichment =====

def get_review_summary(appid):
    url = f"https://store.steampowered.com/appreviews/{appid}"
    data = get_json(url, {"json": 1, "language": "all",
                          "purchase_type": "all", "num_per_page": 0})
    if not data or data.get("success") != 1:
        return None
    q = data.get("query_summary", {})
    pos, neg, total = (q.get("total_positive", 0),
                       q.get("total_negative", 0),
                       q.get("total_reviews", 0))
    if total == 0:
        return None
    return {
        "total_positive": pos,
        "total_negative": neg,
        "total_reviews": total,
        "pct_positive": pos / total,
        "review_score": q.get("review_score"),
        "review_label": q.get("review_score_desc"),
    }


def get_app_details(appid):
    url = "https://store.steampowered.com/api/appdetails"
    data = get_json(url, {"appids": appid, "cc": "us", "l": "en"})
    if not data or str(appid) not in data:
        return None
    entry = data[str(appid)]
    if not entry.get("success"):
        return None
    d = entry["data"]
    rel = d.get("release_date", {}) or {}
    platforms = d.get("platforms", {}) or {}
    metacritic = d.get("metacritic", {}) or {}
    recs = d.get("recommendations", {}) or {}
    achv = d.get("achievements", {}) or {}
    return {
        "type": d.get("type"),
        "is_free": d.get("is_free"),
        "release_date": rel.get("date", ""),
        "coming_soon": rel.get("coming_soon"),
        "price_cents": d.get("price_overview", {}).get("initial"),
        "genres": "|".join(g["description"] for g in d.get("genres", []) or []),
        "categories": "|".join(c["description"] for c in d.get("categories", []) or []),
        "developer": "|".join(d.get("developers", []) or []),
        "publisher": "|".join(d.get("publishers", []) or []),
        "metacritic_score":      safe_int(metacritic.get("score")),
        "recommendations_total": safe_int(recs.get("total")),
        "achievements_total":    safe_int(achv.get("total")),
        "platform_windows":      bool(platforms.get("windows")),
        "platform_mac":          bool(platforms.get("mac")),
        "platform_linux":        bool(platforms.get("linux")),
        "supported_languages_count": count_languages(d.get("supported_languages", "")),
        "required_age":          safe_int(d.get("required_age")),
        "dlc_count":             len(d.get("dlc", []) or []),
        "controller_support":    d.get("controller_support") or None,
    }


def get_concurrent_players(appid):
    url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
    data = get_json(url, {"appid": appid})
    if not data:
        return None
    resp = data.get("response", {})
    return resp.get("player_count") if resp.get("result") == 1 else None


def get_steamspy_details(appid):
    """Only the fields not exposed by the SteamSpy 'all' bulk: tags + languages."""
    data = get_json("https://steamspy.com/api.php",
                    {"request": "appdetails", "appid": appid})
    if not data or "appid" not in data:
        return None
    return {
        "steamspy_languages": (data.get("languages") or "").strip() or None,
        "steamspy_tags":      format_top_tags(data.get("tags")),
    }


FIELDS = [
    "appid", "name",
    "type", "is_free", "release_date", "coming_soon",
    "price_cents", "genres", "categories", "developer", "publisher",
    "metacritic_score", "recommendations_total", "achievements_total",
    "platform_windows", "platform_mac", "platform_linux",
    "supported_languages_count", "required_age", "dlc_count",
    "controller_support",
    "total_positive", "total_negative", "total_reviews",
    "pct_positive", "review_score", "review_label",
    "concurrent_players",
    "owners_estimate", "peak_ccu_yesterday",
    "steamspy_score_rank", "steamspy_languages", "steamspy_tags",
    "collected_utc",
]


def load_done_appids():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            sys.exit("existing CSV has a different column schema; "
                     "delete it and re-run to start fresh")
        return {int(row["appid"]) for row in reader if row.get("appid")}


def main():
    done = load_done_appids()
    seed = seed_and_screen()
    candidates = [(appid, info.get("name", "")) for appid, info in seed.items()]
    print(f"\nphase 2: enrichment | {len(candidates)} candidates "
          f"({len(done)} already done)\n", file=sys.stderr)

    new_file = not os.path.exists(OUTPUT_CSV)
    kept = skipped_old = skipped_other = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for i, (appid, name) in enumerate(candidates, 1):
            if appid in done:
                continue

            summary = get_review_summary(appid)
            if summary is None:
                skipped_other += 1
                time.sleep(SLEEP_BETWEEN)
                continue

            details = get_app_details(appid) or {}
            if details.get("type") not in (None, "game"):
                skipped_other += 1
                time.sleep(SLEEP_BETWEEN)
                continue

            year = parse_release_year(details.get("release_date", ""))
            if year is None or year < MIN_RELEASE_YEAR:
                skipped_old += 1
                time.sleep(SLEEP_BETWEEN)
                continue

            ss_info = seed[appid]
            ss_extra = (get_steamspy_details(appid) if FETCH_STEAMSPY_DETAILS else {}) or {}

            row = {
                "appid": appid,
                "name": name or "",
                **details,
                **summary,
                "concurrent_players":  get_concurrent_players(appid),
                "owners_estimate":     parse_owners_range(ss_info.get("owners", "")),
                "peak_ccu_yesterday":  safe_int(ss_info.get("ccu")),
                "steamspy_score_rank": ss_info.get("score_rank") or None,
                "steamspy_languages":  ss_extra.get("steamspy_languages"),
                "steamspy_tags":       ss_extra.get("steamspy_tags", ""),
                "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            writer.writerow(row)
            kept += 1

            if i % CHECKPOINT_EVERY == 0:
                f.flush()
                print(f"  {i}/{len(candidates)} processed | "
                      f"{kept} kept | {skipped_old} pre-{MIN_RELEASE_YEAR} | "
                      f"{skipped_other} other skips",
                      file=sys.stderr)

            time.sleep(SLEEP_BETWEEN + random.random())

    print(f"\ndone. {kept} new rows | {skipped_old} pre-{MIN_RELEASE_YEAR} | "
          f"{skipped_other} other skips. Output: {OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()