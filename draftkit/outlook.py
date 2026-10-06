"""Rest-of-season outlook -- projected strength, final record, playoff odds.

Each team's expected score for every remaining week is its best lineup that
week (byes + injuries handled by season.roster_week_values); weekly spread
comes from its current best lineup's player volatility. Remaining
regular-season matchups are then simulated many times; seeds go by wins,
then points for. Divisions and custom tiebreakers are ignored.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .lineup import team_week_strength
from .season import Snapshot, roster_week_values


def season_outlook(snap: Snapshot, n_sims: int = 4000, seed: int = 0) -> pd.DataFrame:
    tids = sorted(snap.teams)
    if not tids:
        return pd.DataFrame()
    reg_weeks = {w for p in range(1, snap.reg_season_periods + 1)
                 for w in snap.matchup_periods.get(p, [p])}
    week_vals: Dict[int, Dict[int, float]] = {}
    sd: Dict[int, float] = {}
    for t in tids:
        rows = snap.rosters.get(t, [])
        week_vals[t] = roster_week_values(rows, snap)
        sd[t] = max(team_week_strength(rows, snap)[1], 8.0)

    remaining = [m for m in snap.schedule
                 if not m["playoff"] and m["away"] is not None
                 and m["period"] <= snap.reg_season_periods
                 and m["winner"] == "UNDECIDED"]

    def period_mean(t, period):
        weeks = snap.matchup_periods.get(period, [period])
        vals = [week_vals[t].get(w) for w in weeks]
        vals = [v for v in vals if v is not None]
        if vals:
            return float(sum(vals))
        future = [v for w, v in week_vals[t].items() if w in reg_weeks]
        return float(np.mean(future)) if future else 0.0

    idx = {t: i for i, t in enumerate(tids)}
    n = len(tids)
    rng = np.random.default_rng(seed)
    wins = np.tile(np.array([snap.teams[t]["wins"] + 0.5 * snap.teams[t]["ties"] for t in tids], float), (n_sims, 1))
    pf = np.tile(np.array([snap.teams[t]["pf"] for t in tids], float), (n_sims, 1))
    for m in remaining:
        h, a = m["home"], m["away"]
        if h not in idx or a not in idx:
            continue
        sh = rng.normal(period_mean(h, m["period"]), sd[h], n_sims)
        sa = rng.normal(period_mean(a, m["period"]), sd[a], n_sims)
        wins[:, idx[h]] += sh > sa
        wins[:, idx[a]] += sa > sh
        pf[:, idx[h]] += sh
        pf[:, idx[a]] += sa

    # rank within each sim: wins, then points for
    key = wins * 1e6 + pf
    order = np.argsort(-key, axis=1)
    seeds = np.empty_like(order)
    rows_ix = np.arange(n_sims)[:, None]
    seeds[rows_ix, order] = np.arange(n)[None, :]
    playoff = (seeds < snap.playoff_teams).mean(axis=0)
    first = (seeds == 0).mean(axis=0)

    future_weeks = [w for w in snap.remaining_weeks if w in reg_weeks] or snap.remaining_weeks
    rows = []
    for t in tids:
        tm = snap.teams[t]
        wk = [week_vals[t].get(w, 0.0) for w in future_weeks]
        rows.append({
            "team": tm["name"] + ("  (you)" if t == snap.my_team_id else ""),
            "record": f"{tm['wins']}-{tm['losses']}" + (f"-{tm['ties']}" if tm["ties"] else ""),
            "points_for": round(tm["pf"], 1),
            "proj_weekly": round(float(np.mean(wk)) if wk else 0.0, 1),
            "proj_wins": round(float(wins[:, idx[t]].mean()), 1),
            "playoff_%": round(100 * float(playoff[idx[t]])),
            "top_seed_%": round(100 * float(first[idx[t]])),
            "espn_playoff_%": round(tm["espn_playoff_pct"]),
            "remaining_games": sum(1 for m in remaining if t in (m["home"], m["away"])),
        })
    df = pd.DataFrame(rows).sort_values(["playoff_%", "proj_wins"], ascending=False)
    df.insert(0, "power_rank", df["proj_weekly"].rank(ascending=False, method="min").astype(int))
    return df.reset_index(drop=True)
