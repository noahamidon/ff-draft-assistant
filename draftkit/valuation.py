"""Valuation -- turning projections into league-aware value.

optimal_lineup_value : the objective. Points only count if they can START.
replacement_levels   : baseline per position, set by YOUR league rules (ROI).
vorp                 : proj - replacement (static value-based drafting).
vona                 : proj - best you'd expect at your NEXT pick (draft context).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .config import LeagueConfig


# ---------------------------------------------------------------------------
# Optimal starting lineup value
# ---------------------------------------------------------------------------
def optimal_lineup_value(roster: List[dict], config: LeagueConfig) -> float:
    """Best achievable starting-lineup points from a roster.

    Fill dedicated slots with the top players at each position, then fill flex
    slots from the best remaining eligible players. For additive point values
    and nested flex eligibility this greedy fill is optimal.
    """
    by_pos: Dict[str, List[float]] = defaultdict(list)
    for p in roster:
        by_pos[p["pos"]].append(float(p["proj"]))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    used = defaultdict(int)
    total = 0.0

    for pos, count in config.starters.items():
        avail = by_pos.get(pos, [])
        take = avail[:count]
        total += sum(take)
        used[pos] += len(take)

    leftovers: List[Tuple[float, str]] = []
    for pos, vals in by_pos.items():
        leftovers += [(v, pos) for v in vals[used[pos]:]]
    leftovers.sort(reverse=True)

    for count, elig in config.flex_slots:
        taken = 0
        remaining: List[Tuple[float, str]] = []
        for v, pos in leftovers:
            if taken < count and pos in elig:
                total += v
                taken += 1
            else:
                remaining.append((v, pos))
        leftovers = remaining

    return total


def marginal_lineup_value(roster: List[dict], candidate: dict, config: LeagueConfig) -> float:
    """How much `candidate` improves the optimal starting lineup."""
    base = optimal_lineup_value(roster, config)
    return optimal_lineup_value(roster + [candidate], config) - base


# ---------------------------------------------------------------------------
# Replacement levels + VORP
# ---------------------------------------------------------------------------
def replacement_levels(
    players: pd.DataFrame, config: LeagueConfig
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Replacement projection per position, given league-wide starter demand.

    Walk players best-first, filling each team's dedicated then flex starting
    demand across the whole league. Replacement level for a position is the
    projection of the best player who does NOT crack a starting slot.
    """
    ranked = players.sort_values("proj", ascending=False)

    dedicated_demand = {p: config.team_count * c for p, c in config.starters.items()}
    flex_demand = sum(config.team_count * c for c, _ in config.flex_slots)
    flex_elig = set().union(*[e for _, e in config.flex_slots]) if config.flex_slots else set()

    filled = defaultdict(int)
    starters_by_pos = defaultdict(int)
    flex_used = 0

    for _, row in ranked.iterrows():
        pos = row["pos"]
        if pos in dedicated_demand and filled[pos] < dedicated_demand[pos]:
            filled[pos] += 1
            starters_by_pos[pos] += 1
        elif pos in flex_elig and flex_used < flex_demand:
            flex_used += 1
            starters_by_pos[pos] += 1

    repl: Dict[str, float] = {}
    for pos in config.positions:
        pos_proj = ranked.loc[ranked["pos"] == pos, "proj"].tolist()
        if not pos_proj:
            repl[pos] = 0.0
            continue
        idx = min(starters_by_pos.get(pos, 0), len(pos_proj) - 1)
        repl[pos] = float(pos_proj[idx])
    return repl, dict(starters_by_pos)


def add_vorp(players: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    repl, _ = replacement_levels(players, config)
    out = players.copy()
    out["replacement"] = out["pos"].map(repl).fillna(0.0)
    out["vorp"] = out["proj"] - out["replacement"]
    return out


# ---------------------------------------------------------------------------
# VONA -- value over next available
# ---------------------------------------------------------------------------
def prob_available(adp: float, pick_no: int, scale: float = 8.0) -> float:
    """P(player still on the board at overall pick `pick_no`), from ADP.

    Logistic in (pick_no - adp): a player whose ADP is later than this pick is
    likely still available; earlier ADP means likely gone.
    """
    return 1.0 / (1.0 + math.exp((pick_no - adp) / max(scale, 1e-6)))


def add_vona(
    players: pd.DataFrame,
    next_pick_no: int,
    config: LeagueConfig,
    scale: float = 8.0,
    survive_threshold: float = 0.5,
) -> pd.DataFrame:
    """VORP-style comparison against the best player likely to survive to your
    next pick, per position. High VONA => grab now, it won't come back.
    """
    out = players.copy()
    best_next: Dict[str, float] = {}
    for pos in config.positions:
        pool = out[out["pos"] == pos]
        survivors = pool[
            pool["adp"].apply(lambda a: prob_available(a, next_pick_no, scale)) >= survive_threshold
        ]
        best_next[pos] = float(survivors["proj"].max()) if len(survivors) else 0.0
    out["best_avail_next"] = out["pos"].map(best_next)
    out["vona"] = out["proj"] - out["best_avail_next"]
    return out
