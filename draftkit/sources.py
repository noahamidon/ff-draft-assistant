"""External projection sources, blended with ESPN's.

Why more than one: across 12 seasons, a simple average of projection sources
beat individual sources in 69% of head-to-head comparisons, and equal weights
did as well as historically tuned ones (Fantasy Football Analytics, 2026).
So the model averages every available source with equal weight.

Sources (all free, no login):
  * ESPN            -- this week's + season projection, already in your league's
                       scoring (comes with the league pull).
  * Sleeper         -- RotoWire's weekly projections for every remaining week
                       (api.sleeper.com). Raw stat lines are rescored with YOUR
                       league's ESPN scoring settings.
  * FantasyPros     -- expert consensus rankings (ECR): weekly and rest-of-season
                       ("redraft"), via DynastyProcess's daily open-data scrape.
                       ECR is a ranking, so it's converted to points by rank-
                       matching onto the other sources' points at that position.

Player ids are joined through DynastyProcess's id crosswalk (ESPN <-> Sleeper <->
FantasyPros); D/STs join by team; anything else falls back to name + position.
Downloads are cached in data/cache/ (gitignored).
"""

from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

CACHE_DIR = os.path.join("data", "cache")
_UA = {"User-Agent": "Mozilla/5.0 (draftkit)"}
_DP = "https://github.com/dynastyprocess/data/raw/master/files/"
_SLEEPER = ("https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular"
            "&position%5B%5D=QB&position%5B%5D=RB&position%5B%5D=WR&position%5B%5D=TE"
            "&position%5B%5D=K&position%5B%5D=DEF&order_by=pts_ppr")

SOURCES = ("espn", "sleeper", "fantasypros")
SOURCE_LABELS = {"espn": "ESPN", "sleeper": "Sleeper (RotoWire)", "fantasypros": "FantasyPros consensus"}

# one spelling per NFL team
_TEAM_ALIAS = {"WAS": "WSH", "JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC",
               "OAK": "LV", "LVR": "LV", "KCC": "KC", "GBP": "GB", "NEP": "NE",
               "NOS": "NO", "SFO": "SF", "TBB": "TB", "ARZ": "ARI", "BLT": "BAL",
               "CLV": "CLE", "HST": "HOU"}


def norm_team(t) -> str:
    t = str(t or "").upper().strip()
    return _TEAM_ALIAS.get(t, t)


def norm_name(s) -> str:
    s = str(s or "").lower()
    for suf in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv", " v"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return "".join(c for c in s if c.isalnum())


# --------------------------------------------------------------------------
# Download helpers (with a small on-disk cache)
# --------------------------------------------------------------------------
def _cached_get(url: str, name: str, max_age_s: float, timeout: int = 30) -> bytes:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_s:
        with open(path, "rb") as fh:
            return fh.read()
    try:
        resp = requests.get(url, headers=_UA, timeout=timeout)
        resp.raise_for_status()
        body = resp.content
    except Exception:
        if os.path.exists(path):                 # stale beats nothing
            with open(path, "rb") as fh:
                return fh.read()
        raise
    with open(path, "wb") as fh:
        fh.write(body)
    return body


def _csv(url: str, name: str, max_age_s: float) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_cached_get(url, name, max_age_s)), low_memory=False)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def fetch_all(season: int, weeks: Iterable[int], enabled: Iterable[str] = SOURCES) -> dict:
    """Raw data for every enabled external source. Failures are recorded in
    out['errors'] and the model simply runs without that source."""
    enabled = set(enabled)
    out: dict = {"errors": {}, "sleeper": {}, "fp_week": None, "fp_ros": None, "ids": None}
    try:
        ids = _csv(_DP + "db_playerids.csv", "db_playerids.csv", 3 * 86400)
        out["ids"] = ids
    except Exception as exc:  # noqa: BLE001
        out["errors"]["ids"] = str(exc)

    if "sleeper" in enabled:
        def one(w):
            body = _cached_get(_SLEEPER.format(season=season, week=w),
                               f"sleeper_{season}_{w}.json", 3 * 3600)
            return w, json.loads(body)
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                for w, data in pool.map(one, list(weeks)):
                    out["sleeper"][int(w)] = data
        except Exception as exc:  # noqa: BLE001
            out["errors"]["sleeper"] = str(exc)

    if "fantasypros" in enabled:
        try:
            out["fp_week"] = _csv(_DP + "fp_latest_weekly.csv", "fp_latest_weekly.csv", 6 * 3600)
        except Exception as exc:  # noqa: BLE001
            out["errors"]["fantasypros_weekly"] = str(exc)
        try:
            ecr = _csv(_DP + "db_fpecr_latest.csv", "db_fpecr_latest.csv", 12 * 3600)
            out["fp_ros"] = ecr[ecr["ecr_type"] == "rp"]          # redraft (= ROS in season)
        except Exception as exc:  # noqa: BLE001
            out["errors"]["fantasypros_ros"] = str(exc)
    return out


# --------------------------------------------------------------------------
# League scoring for Sleeper stat lines
# --------------------------------------------------------------------------
# ESPN scoring statId -> Sleeper stat key(s). Anything unmapped (yardage
# bonuses, long-TD bonuses) is absorbed by the per-position calibration below.
_ESPN_TO_SLEEPER = {
    0: ["pass_att"], 1: ["pass_cmp"], 2: ["pass_inc"], 3: ["pass_yd"], 4: ["pass_td"],
    19: ["pass_2pt"], 20: ["pass_int"], 23: ["rush_att"], 24: ["rush_yd"], 25: ["rush_td"],
    26: ["rush_2pt"], 41: ["rec"], 42: ["rec_yd"], 43: ["rec_td"], 44: ["rec_2pt"],
    53: ["rec"], 58: ["rec_tgt"], 64: ["pass_sack"], 68: ["fum"], 72: ["fum_lost"],
    74: ["fgm_50p"], 76: ["fgmiss_50p"], 77: ["fgm_40_49"], 79: ["fgmiss_40_49"],
    80: ["fgm_0_19", "fgm_20_29", "fgm_30_39"],
    82: ["fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39"],
    85: ["fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49", "fgmiss_50p"],
    86: ["xpm"], 88: ["xpmiss"],
}


def score_stats(stats: dict, pos: str, scoring_items: Dict[int, float], ppr: float) -> Optional[float]:
    """League-scored points for one Sleeper projection line."""
    if not stats:
        return None
    key = "pts_ppr" if ppr >= 0.75 else ("pts_half_ppr" if ppr >= 0.25 else "pts_std")
    fallback = stats.get(key, stats.get("pts_std"))
    fallback = None if fallback is None else float(fallback)
    if pos == "DST" or not scoring_items:
        return fallback
    # don't double count: 41 and 53 are both receptions; ESPN uses one of them
    items = dict(scoring_items)
    if 53 in items and 41 in items:
        items.pop(41)
    total, hit = 0.0, False
    for sid, pts in items.items():
        keys = _ESPN_TO_SLEEPER.get(int(sid))
        if not keys or not pts:
            continue
        val = sum(float(stats.get(k) or 0.0) for k in keys)
        if val:
            hit = True
        total += val * float(pts)
    return total if hit else fallback


# --------------------------------------------------------------------------
# Join sources onto snapshot player rows
# --------------------------------------------------------------------------
def _id_maps(ids: Optional[pd.DataFrame]):
    sl, fp = {}, {}
    if ids is None or ids.empty:
        return sl, fp
    sub = ids.dropna(subset=["espn_id"])
    for _, r in sub.iterrows():
        e = str(int(r["espn_id"]))
        if pd.notna(r.get("sleeper_id")):
            sl[str(r["sleeper_id"]).split(".")[0]] = e
        if pd.notna(r.get("fantasypros_id")):
            fp[str(r["fantasypros_id"]).split(".")[0]] = e
    return sl, fp


def attach(rows: List[dict], raw: dict, scoring_items: Dict[int, float], ppr: float) -> Dict[str, int]:
    """Write each source's numbers onto rows as row['src'] (in place):
        sleeper  {week: league-scored pts}
        fp_week  FantasyPros weekly ECR (positional rank, lower = better)
        fp_ros   FantasyPros rest-of-season ECR
    Returns {source: players matched} for display."""
    by_id = {r["player_id"]: r for r in rows}
    by_name = {(norm_name(r["name"]), r["pos"]): r for r in rows if r["pos"] != "DST"}
    by_dst = {norm_team(r["pro_team"]): r for r in rows if r["pos"] == "DST"}
    for r in rows:
        r["src"] = {"sleeper": {}, "fp_week": None, "fp_ros": None}
    sl_map, fp_map = _id_maps(raw.get("ids"))

    def find(pid_espn, name, pos, team):
        if pos == "DST":
            return by_dst.get(norm_team(team))
        r = by_id.get(pid_espn) if pid_espn else None
        return r or by_name.get((norm_name(name), pos))

    matched = {"sleeper": set(), "fantasypros": set()}
    for week, entries in (raw.get("sleeper") or {}).items():
        for e in entries or []:
            p = e.get("player") or {}
            pos = {"DEF": "DST"}.get(p.get("position"), p.get("position"))
            if pos not in ("QB", "RB", "WR", "TE", "K", "DST"):
                continue
            name = f"{p.get('first_name', '')} {p.get('last_name', '')}"
            r = find(sl_map.get(str(e.get("player_id"))), name, pos, e.get("team") or p.get("team"))
            if r is None:
                continue
            pts = score_stats(e.get("stats") or {}, pos, scoring_items, ppr)
            if pts is not None:
                r["src"]["sleeper"][int(week)] = pts
                matched["sleeper"].add(r["player_id"])

    _POS_FP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DST": "DST"}
    for key, df, id_col, name_col in (("fp_week", raw.get("fp_week"), "fantasypros_id", "player_name"),
                                      ("fp_ros", raw.get("fp_ros"), "id", "player")):
        if df is None or df.empty:
            continue
        for _, e in df.iterrows():
            pos = _POS_FP.get(str(e.get("pos", "")).upper())
            if pos is None or pd.isna(e.get("ecr")):
                continue
            fid = str(e.get(id_col)).split(".")[0]
            team = e.get("team") if "team" in df else e.get("tm")
            r = find(fp_map.get(fid), e.get(name_col), pos, team)
            if r is not None:
                r["src"][key] = float(e["ecr"])
                matched["fantasypros"].add(r["player_id"])
    return {k: len(v) for k, v in matched.items()}


def rank_to_points(rows: List[dict], rank_key: str, value_fn) -> Dict[str, float]:
    """Convert a ranking into points: within each position, the k-th ranked
    player gets the k-th highest value from the other sources (rank matching).
    Keeps the ranking's opinion while speaking your league's scoring."""
    out: Dict[str, float] = {}
    by_pos: Dict[str, List[dict]] = {}
    for r in rows:
        by_pos.setdefault(r["pos"], []).append(r)
    for pos, rs in by_pos.items():
        vals = sorted((v for v in (value_fn(r) for r in rs) if v is not None), reverse=True)
        ranked = sorted((r for r in rs if r["src"].get(rank_key) is not None),
                        key=lambda r: r["src"][rank_key])
        for i, r in enumerate(ranked[: len(vals)]):
            out[r["player_id"]] = vals[i]
    return out


# --------------------------------------------------------------------------
# Draft-time consensus (season totals)
# --------------------------------------------------------------------------
_SLEEPER_SEASON = ("https://api.sleeper.com/projections/nfl/{season}?season_type=regular"
                   "&position%5B%5D=QB&position%5B%5D=RB&position%5B%5D=WR&position%5B%5D=TE"
                   "&position%5B%5D=K&position%5B%5D=DEF&order_by=pts_ppr")


def consensus_season(espn_df: pd.DataFrame, season: int, scoring_items: Dict[int, float],
                     ppr: float, enabled: Iterable[str] = SOURCES) -> pd.DataFrame:
    """Season projections averaged across ESPN, Sleeper and FantasyPros (equal
    weights), for the draft board. Input/output: [player_id, name, pos, team,
    proj, adp] with ESPN player ids; adds proj_espn / proj_sleeper / proj_fp."""
    enabled = set(enabled)
    raw = {"errors": {}, "sleeper": {}, "fp_week": None, "fp_ros": None, "ids": None}
    try:
        raw["ids"] = _csv(_DP + "db_playerids.csv", "db_playerids.csv", 3 * 86400)
    except Exception as exc:  # noqa: BLE001
        raw["errors"]["ids"] = str(exc)
    if "sleeper" in enabled:
        try:
            raw["sleeper"][0] = json.loads(_cached_get(_SLEEPER_SEASON.format(season=season),
                                                       f"sleeper_{season}_season.json", 12 * 3600))
        except Exception as exc:  # noqa: BLE001
            raw["errors"]["sleeper"] = str(exc)
    if "fantasypros" in enabled:
        try:
            ecr = _csv(_DP + "db_fpecr_latest.csv", "db_fpecr_latest.csv", 12 * 3600)
            raw["fp_ros"] = ecr[ecr["ecr_type"] == "rp"]
        except Exception as exc:  # noqa: BLE001
            raw["errors"]["fantasypros"] = str(exc)

    rows = [{"player_id": str(r["player_id"]), "name": r["name"], "pos": r["pos"],
             "pro_team": r["team"], "espn": float(r["proj"])} for _, r in espn_df.iterrows()]
    attach(rows, raw, scoring_items, ppr)
    # calibrate Sleeper to ESPN's league scoring per position (median ratio)
    scale = {}
    for pos in {r["pos"] for r in rows}:
        pairs = sorted(((r["espn"], r["src"]["sleeper"].get(0)) for r in rows
                        if r["pos"] == pos and r["espn"] > 0 and r["src"]["sleeper"].get(0)),
                       key=lambda p: -p[0])[:40]
        if len(pairs) >= 5:
            e = sorted(p[0] for p in pairs)[len(pairs) // 2]
            s = sorted(p[1] for p in pairs)[len(pairs) // 2]
            scale[pos] = min(max(e / s, 0.7), 1.4) if s > 0 else 1.0
    for r in rows:
        v = r["src"]["sleeper"].get(0)
        r["sleeper"] = v * scale.get(r["pos"], 1.0) if v is not None else None
    def points_view(r):
        vals = [v for v in ((r["espn"] if "espn" in enabled else None), r["sleeper"]) if v is not None]
        return sum(vals) / len(vals) if vals else None

    fp = rank_to_points(rows, "fp_ros", points_view)
    out = espn_df.copy()
    out["proj_espn"] = [r["espn"] for r in rows]
    out["proj_sleeper"] = [r["sleeper"] for r in rows]
    out["proj_fp"] = [fp.get(r["player_id"]) for r in rows]
    cols = [c for c, s in (("proj_espn", "espn"), ("proj_sleeper", "sleeper"), ("proj_fp", "fantasypros"))
            if s in enabled]
    out["proj"] = out[cols].mean(axis=1, skipna=True).fillna(out["proj_espn"]).round(1)
    out.attrs["errors"] = raw["errors"]
    return out.sort_values("proj", ascending=False).reset_index(drop=True)
