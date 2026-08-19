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
from .valuation import add_vona, add_vorp, optimal_lineup_value


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


def positional_need(state: DraftState, config: LeagueConfig, team: Optional[int] = None) -> Dict[str, float]:
    """0..1 need score per position: unmet starting demand for a team's roster.

    K / DST / IDP are deliberately excluded (returned 0): they should never
    create early-round urgency. When to actually draft them is handled by the
    late-round gate, not by a need spike.
    """
    team = team if team is not None else state.my_team
    counts = defaultdict(int)
    for p in state.roster_of(team):
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


def suppressed_positions(state: DraftState, config: LeagueConfig, team: Optional[int] = None) -> set:
    """Positions that should NOT be recommended/simulated yet for a team.

    K / DST / IDP are near-streamable, so they're deferred until within a few
    rounds of the end AND still short a starter. Positions the league doesn't
    start at all are always suppressed.
    """
    import math
    team = team if team is not None else state.my_team
    counts = defaultdict(int)
    for p in state.roster_of(team):
        counts[p.pos] += 1
    picks_left = state.total_picks - state.next_overall + 1
    rounds_left = math.ceil(picks_left / max(1, config.team_count))
    within = {"K": 2, "DST": 2, "IDP": 5}
    supp = set()
    for pos in ("K", "DST", "IDP"):
        req = config.starters.get(pos, 0)
        if req == 0:
            supp.add(pos)
            continue
        still_need = counts.get(pos, 0) < req
        if not (still_need and rounds_left <= within[pos]):
            supp.add(pos)
    return supp


def candidate_pool(
    state: DraftState, players: pd.DataFrame, config: LeagueConfig, n: int,
    team: Optional[int] = None,
) -> pd.DataFrame:
    """Candidates to simulate: the best available by value, EXCLUDING deferred
    positions, and guaranteeing the best option at each unmet-need position is
    included (so e.g. WR is considered when you still need WRs)."""
    team = team if team is not None else state.my_team
    supp = suppressed_positions(state, config, team)
    av = add_vorp(state.available(players), config)
    av = av[~av["pos"].isin(supp)]
    if av.empty:
        return av
    cand = av.sort_values("vorp", ascending=False).head(n)

    counts = defaultdict(int)
    for p in state.roster_of(team):
        counts[p.pos] += 1
    flex_elig = set().union(*[e for _, e in config.flex_slots]) if config.flex_slots else set()
    for pos, req in config.starters.items():
        if pos in supp or req == 0:
            continue
        if counts.get(pos, 0) < req or pos in flex_elig:
            best = av[av["pos"] == pos].sort_values("vorp", ascending=False).head(1)
            cand = pd.concat([cand, best])
    return cand.drop_duplicates("player_id").reset_index(drop=True)


def build_board(
    state: DraftState,
    players: pd.DataFrame,
    config: LeagueConfig,
    sim_results: Optional[pd.DataFrame] = None,
    team: Optional[int] = None,
    defense: float = 0.0,
) -> pd.DataFrame:
    """The recommendation table for a team's perspective (defaults to you).

    `defense` (0..~0.5) is the draft-defense dial: how much of a bench-only
    player's scarcity value still counts (insurance + denying rivals). 0 =
    pure starting-lineup optimization; higher = more willing to grab scarce
    depth. It's deliberately a fraction, so it never outweighs filling starters.
    """
    team = team if team is not None else state.my_team
    avail = state.available(players)
    if avail.empty:
        return avail

    avail = add_vorp(avail, config)
    next_pick = (state.my_next_pick() or state.next_overall) if team == state.my_team else state.next_overall
    avail = add_vona(avail, next_pick, config)
    avail = assign_tiers(avail)

    need = positional_need(state, config, team)
    avail["need"] = avail["pos"].map(need).fillna(0.0)

    # Drop positions the league doesn't start, and defer K/DST/IDP.
    supp = suppressed_positions(state, config, team)
    if supp:
        avail = avail[~avail["pos"].isin(supp)]
    if avail.empty:
        avail["sim_ev"] = []
        return avail

    # marginal starting-lineup value: how much this player improves the team's
    # optimal starting lineup right now. A 3rd RB when RB slots are full adds
    # ~0; a WR when they have none adds a lot. Makes the board roster-aware.
    proj_lookup = dict(zip(players["player_id"].astype(str), players["proj"].astype(float)))
    my_rows = [{"pos": p.pos, "proj": proj_lookup.get(str(p.player_id), 0.0)}
               for p in state.roster_of(team)]
    base_val = optimal_lineup_value(my_rows, config)
    avail["marginal"] = [
        optimal_lineup_value(my_rows + [{"pos": pos, "proj": pr}], config) - base_val
        for pos, pr in zip(avail["pos"], avail["proj"])
    ]

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    # "Startable headroom": positional value (VORP/VONA) should only count if
    # the player can still claim a starting or flex slot. Once a position's
    # dedicated + flex slots are full (e.g. a 3rd QB in a 2-QB superflex), extra
    # players there can only ride the bench, so scarcity value is switched off
    # and only genuine upgrade value (marginal) can lift them.
    counts = defaultdict(int)
    for p in state.roster_of(team):
        counts[p.pos] += 1
    flex_elig = set().union(*[e for _, e in config.flex_slots]) if config.flex_slots else set()
    flex_surplus = sum(max(0, counts.get(p, 0) - config.starters.get(p, 0)) for p in flex_elig)
    flex_open = max(0, config.flex_count - flex_surplus)

    def _startable(pos: str) -> float:
        ded_open = config.starters.get(pos, 0) - counts.get(pos, 0) > 0
        return 1.0 if (ded_open or (pos in flex_elig and flex_open > 0)) else 0.0

    startable = avail["pos"].map(_startable)
    avail["bench_only"] = 1.0 - startable
    # bench-only positions keep `defense` fraction of scarcity value (draft
    # defense), startable positions keep it all.
    mult = startable * (1.0 - defense) + defense

    avail["blended"] = (
        _z(avail["marginal"]) * 1.3
        + _z(avail["vorp"]) * 0.7 * mult
        + _z(avail["vona"]) * 0.4 * mult
        + avail["need"] * 0.5
    )

    if sim_results is not None and not sim_results.empty:
        sim_map = sim_results.set_index("player_id")["mean_value"].to_dict()
        std_map = sim_results.set_index("player_id")["std_value"].to_dict()
        avail["sim_ev"] = avail["player_id"].astype(str).map(sim_map)
        avail["sim_std"] = avail["player_id"].astype(str).map(std_map)
        avail["_rank_key"] = avail["sim_ev"].fillna(-1e18)
        avail = avail.sort_values(
            ["_rank_key", "blended"], ascending=[False, False]
        ).drop(columns="_rank_key")
    else:
        avail["sim_ev"] = np.nan
        avail["sim_std"] = np.nan
        avail = avail.sort_values("blended", ascending=False)

    return avail.reset_index(drop=True)


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
