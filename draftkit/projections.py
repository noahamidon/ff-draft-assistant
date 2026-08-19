"""Projections -- the single biggest lever on quality.

The model consumes a tidy DataFrame with these columns:
    player_id  str   stable id (name-slug is fine if you have no real id)
    name       str
    pos        str   one of QB/RB/WR/TE/K/DST
    team       str
    proj       float league-scored projected season points
    adp        float average draft position (overall). Lower = earlier.
    std        float projection standard deviation (risk). Optional.

Two input styles are supported by load_projections():

  A) PRE-SCORED: your CSV already has a `proj` (or `fpts`) column. Simplest.
     If it also has `receptions`, we can nudge for your league's PPR value.

  B) RAW STATS: your CSV has stat columns (pass_yds, rush_td, receptions,...)
     and we compute `proj` from LeagueConfig.scoring. This is the most
     accurate path for weird scoring -- true ROI on your exact rules.

Blending: pass several DataFrames to blend_projections() to average them
(a consensus of 2-3 sources beats any single source).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import LeagueConfig

_REQUIRED = ["name", "pos", "team"]

# stat column -> scoring key in LeagueConfig.scoring
_STAT_TO_SCORE = {
    "pass_yds": "pass_yds",
    "pass_td": "pass_td",
    "interceptions": "interceptions",
    "rush_yds": "rush_yds",
    "rush_td": "rush_td",
    "rec_yds": "rec_yds",
    "rec_td": "rec_td",
    "receptions": "receptions",
}


def _slug(name: str, pos: str) -> str:
    return f"{pos}:{''.join(ch.lower() for ch in str(name) if ch.isalnum())}"


def score_from_raw(df: pd.DataFrame, config: LeagueConfig) -> pd.Series:
    """Compute league-scored points from raw stat columns present in df."""
    pts = pd.Series(0.0, index=df.index)
    for stat_col, score_key in _STAT_TO_SCORE.items():
        if stat_col in df.columns and score_key in config.scoring:
            pts = pts + df[stat_col].fillna(0.0) * float(config.scoring[score_key])
    return pts


def load_projections(
    path: str,
    config: Optional[LeagueConfig] = None,
    rescore_if_raw: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"projections CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df["pos"] = df["pos"].str.upper().str.replace("D/ST", "DST", regex=False)

    # resolve projected points
    has_raw = any(c in df.columns for c in _STAT_TO_SCORE)
    if "proj" not in df.columns and "fpts" in df.columns:
        df["proj"] = df["fpts"]
    if ("proj" not in df.columns or rescore_if_raw) and has_raw and config is not None:
        raw_pts = score_from_raw(df, config)
        # only overwrite where we actually accumulated something
        if raw_pts.abs().sum() > 0:
            df["proj"] = raw_pts
    if "proj" not in df.columns:
        raise ValueError(
            "Could not determine `proj`. Provide a `proj`/`fpts` column, or "
            "raw stat columns plus a LeagueConfig with scoring."
        )

    if "adp" not in df.columns:
        # fall back to proj-rank as a crude ADP so the sim still runs
        df["adp"] = df["proj"].rank(ascending=False, method="first")
    if "std" not in df.columns:
        # default risk: heavier for skill positions, scaled to projection
        pos_cv = {"QB": 0.16, "RB": 0.28, "WR": 0.26, "TE": 0.30, "K": 0.35, "DST": 0.40}
        df["std"] = df.apply(
            lambda r: max(1.0, r["proj"] * pos_cv.get(r["pos"], 0.25)), axis=1
        )
    if "player_id" not in df.columns:
        df["player_id"] = [
            _slug(n, p) for n, p in zip(df["name"], df["pos"])
        ]

    df["proj"] = df["proj"].astype(float)
    df["adp"] = df["adp"].astype(float)
    df["std"] = df["std"].astype(float)
    df["player_id"] = df["player_id"].astype(str)

    cols = ["player_id", "name", "pos", "team", "proj", "adp", "std"]
    return df[cols].sort_values("proj", ascending=False).reset_index(drop=True)


def blend_projections(frames: List[pd.DataFrame], weights: Optional[List[float]] = None) -> pd.DataFrame:
    """Average `proj`/`adp`/`std` across sources, joined on player_id."""
    if len(frames) == 1:
        return frames[0]
    if weights is None:
        weights = [1.0] * len(frames)
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    base = frames[0][["player_id", "name", "pos", "team"]].copy()
    proj = pd.DataFrame({"player_id": base["player_id"]}).set_index("player_id")
    for metric in ("proj", "adp", "std"):
        acc = None
        for wi, f in zip(w, frames):
            s = f.set_index("player_id")[metric] * wi
            acc = s if acc is None else acc.add(s, fill_value=0.0)
        proj[metric] = acc
    out = base.set_index("player_id").join(proj).reset_index()
    return out.sort_values("proj", ascending=False).reset_index(drop=True)
