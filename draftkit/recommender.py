"""Recommender -- fold every signal into one ranked board with reasoning.

Signals:
  vorp     value over replacement (league-rule baseline)
  vona     value over what survives to your next pick (draft context)
  need     do you still need a starter at this position?
  tier     projection-gap tiers within a position (cliff awareness)
  sim_ev   Monte Carlo expected final lineup value (the heavy gun)

The headline recommendation ranks by sim_ev when a simulation is supplied,
otherwise by a blended VORP/VONA/need score.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import LeagueConfig
from .draft_state import DraftState
from .valuation import add_vona, add_vorp


def assign_tiers(players: pd.DataFrame, gap_mult: float = 1.0) -> pd.DataFrame:
    """Tier players within each position by projection cliffs.

    A new tier starts when the drop to the next player exceeds the position's
    typical gap (mean gap * gap_mult).
    """
    out = players.copy()
    out["tier"] = 0
    for pos, grp in out.groupby("pos"):
        grp = grp.sort_values("proj", ascending=False)
        projs = grp["proj"].to_numpy()
        if len(projs) <= 1:
            out.loc[grp.index, "tier"] = 1
            continue
        gaps = -np.diff(projs)
        thresh = max(gaps.mean() * gap_mult, 1e-9)
        tier = 1
        tiers = [1]
        for g in gaps:
            if g > thresh:
                tier += 1
            tiers.append(tier)
        out.loc[grp.index, "tier"] = tiers
    return out


def positional_need(state: DraftState, config: LeagueConfig) -> Dict[str, float]:
    """0..1 need score per position: unmet starting demand for YOUR roster.

    K / DST / IDP are deliberately excluded (returned 0): they should never
    create early-round urgency. When to actually draft them is handled by the
    late-round gate in build_board, not by a need spike.
    """
    counts = defaultdict(int)
    for p in state.my_roster():
        counts[p.pos] += 1

    LATE = {"K", "DST", "IDP"}
    need: Dict[str, float] = {}
    flex_elig = set().union(*[e for _, e in config.flex_slots]) if config.flex_slots else set()
    flex_slots = config.flex_count
    surplus = 0
    for pos, dem in config.starters.items():
        have = counts.get(pos, 0)
        need[pos] = max(0.0, dem - have)
        if pos in flex_elig and have > dem:
            surplus += have - dem
    remaining_flex = max(0, flex_slots - surplus)

    out: Dict[str, float] = {}
    for pos in config.positions:
        if pos in LATE:
            out[pos] = 0.0
            continue
        base = need.get(pos, 0.0)
        if pos in flex_elig and remaining_flex > 0:
            base += 0.5 * remaining_flex / max(1, len(flex_elig))
        dem = config.starters.get(pos, 0) + (remaining_flex if pos in flex_elig else 0)
        out[pos] = min(1.0, base / max(1.0, dem)) if dem else 0.0
    return out


def build_board(
    state: DraftState,
    players: pd.DataFrame,
    config: LeagueConfig,
    sim_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """The full recommendation table over currently-available players."""
    avail = state.available(players)
    if avail.empty:
        return avail

    avail = add_vorp(avail, config)
    next_pick = state.my_next_pick() or state.next_overall
    avail = add_vona(avail, next_pick, config)
    avail = assign_tiers(avail)

    need = positional_need(state, config)
    avail["need"] = avail["pos"].map(need).fillna(0.0)

    # Drop positions your league doesn't start (e.g. D/ST when you run IDP).
    unused = [p for p in ("DST", "K", "IDP") if config.starters.get(p, 0) == 0]
    if unused:
        avail = avail[~avail["pos"].isin(unused)]

    # blended fallback score (used when no sim). Normalize vorp/vona to compare.
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    avail["blended"] = (
        _z(avail["vorp"]) * 1.0
        + _z(avail["vona"]) * 0.6
        + avail["need"] * 1.2
    )

    # -- late-round gate ----------------------------------------------------
    # K / DST / IDP are near-streamable and tightly bunched, so drafting them
    # early is wasteful. Suppress them until you're within a few rounds of the
    # end AND still short a starter there; otherwise sink them to the bottom.
    import math as _math
    my_counts = defaultdict(int)
    for p in state.my_roster():
        my_counts[p.pos] += 1
    picks_left = state.total_picks - state.next_overall + 1
    rounds_left = _math.ceil(picks_left / max(1, config.team_count))
    LATE_WITHIN = {"K": 2, "DST": 2, "IDP": 5}

    def _penalty(row) -> float:
        pos = row["pos"]
        within = LATE_WITHIN.get(pos)
        if within is None:
            return 0.0
        still_need = my_counts.get(pos, 0) < config.starters.get(pos, 0)
        if still_need and rounds_left <= within:
            return 0.0                      # OK to consider now
        return 1e6                          # otherwise bury it

    avail["_penalty"] = avail.apply(_penalty, axis=1)

    if sim_results is not None and not sim_results.empty:
        sim_map = sim_results.set_index("player_id")["mean_value"].to_dict()
        std_map = sim_results.set_index("player_id")["std_value"].to_dict()
        avail["sim_ev"] = avail["player_id"].astype(str).map(sim_map)
        avail["sim_std"] = avail["player_id"].astype(str).map(std_map)
        avail["_rank_key"] = avail["sim_ev"].fillna(-1e18) - avail["_penalty"]
        avail = avail.sort_values(
            ["_rank_key", "blended"], ascending=[False, False]
        ).drop(columns="_rank_key")
    else:
        avail["sim_ev"] = np.nan
        avail["sim_std"] = np.nan
        avail["_score"] = avail["blended"] - avail["_penalty"]
        avail = avail.sort_values("_score", ascending=False).drop(columns="_score")

    return avail.drop(columns="_penalty").reset_index(drop=True)


def reason_for(row: pd.Series, config: LeagueConfig) -> str:
    bits: List[str] = []
    if row.get("need", 0) >= 0.6:
        bits.append(f"fills a {row['pos']} starting need")
    if row.get("vona", 0) > 0 and row.get("vona") >= row.get("vorp", 0) * 0.5:
        bits.append("unlikely to survive to your next pick")
    if row.get("tier", 99) == 1:
        bits.append(f"top tier at {row['pos']}")
    if row.get("vorp", 0) > 0:
        bits.append(f"+{row['vorp']:.0f} pts over replacement")
    if not bits:
        bits.append("best value on the board")
    return "; ".join(bits)
