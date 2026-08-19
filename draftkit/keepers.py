"""Keeper analysis.

Decision rule for THIS league: keep up to 3 players; the 1st costs your round-1
pick, the 2nd your round-2 pick, the 3rd your round-3 pick.

Keeping is worth it only when the player is worth more than the pick you forfeit
to keep him. Concretely, keeping k players means giving up the picks you would
otherwise make in rounds 1..k, so:

    net_surplus(keep k) = sum(value of your best k keepers)
                          - sum(value of the player you'd draft at your
                                round-1..k picks, given every keeper leaguewide
                                is already off the board)

We evaluate k = 0,1,2,3 and recommend the k with the highest net surplus, and
show each keeper's individual surplus versus the pick it costs. "Value" is VORP
(replacement-aware), so it already reflects superflex + your scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .config import LeagueConfig
from .valuation import add_vorp, prob_available


def pick_overall_numbers(my_slot: int, team_count: int, rounds: int) -> List[int]:
    """Overall pick numbers for a given draft slot in a snake draft."""
    picks = []
    for r in range(rounds):                      # 0-indexed round
        if r % 2 == 0:
            picks.append(r * team_count + my_slot)
        else:
            picks.append(r * team_count + (team_count - my_slot + 1))
    return picks


def best_available_value(
    players: pd.DataFrame,
    gone_ids: set,
    overall_pick: int,
    value_col: str = "vorp",
    survive_threshold: float = 0.5,
    adp_scale: float = 10.0,
) -> Optional[pd.Series]:
    """The player you'd realistically draft at `overall_pick` — i.e. the
    opportunity cost of spending that pick on a keeper.

    In a value-based draft, roughly (overall_pick - 1) players are gone by the
    time you pick (whether taken live or held as keepers), so the best player
    still available is about the (overall_pick)-th best by value. We therefore
    walk the value board from that slot and return the first player not already
    kept/taken. This is ADP-independent, which matters in superflex leagues
    where ADP badly lags true QB value and would otherwise invent a phantom
    "someone just as good is still there" alternative.
    """
    board = players.sort_values(value_col, ascending=False).reset_index(drop=True)
    n = len(board)
    if n == 0:
        return None
    gone = {str(g) for g in gone_ids}
    start = min(max(overall_pick - 1, 0), n - 1)
    for j in range(start, n):
        if str(board.iloc[j]["player_id"]) not in gone:
            return board.iloc[j]
    for j in range(n):                     # fallback: best non-gone anywhere
        if str(board.iloc[j]["player_id"]) not in gone:
            return board.iloc[j]
    return board.iloc[-1]


@dataclass
class KeeperResult:
    recommended_keep: int
    per_keeper: pd.DataFrame       # each candidate vs the pick it would cost
    by_count: pd.DataFrame         # net surplus for keeping k = 0..3


def evaluate_keepers(
    candidate_names: List[str],
    players: pd.DataFrame,
    config: LeagueConfig,
    my_slot: int,
    other_keeper_ids: Optional[set] = None,
    max_keep: int = 3,
    value_col: str = "vorp",
    candidate_ids: Optional[List[str]] = None,
    other_keeper_overalls: Optional[List[int]] = None,
) -> KeeperResult:
    """Rank your keeper candidates and recommend how many to keep.

    candidate_names       : your keeper options (matched against players["name"])
    candidate_ids         : optional player_ids (matched first; more robust)
    other_keeper_ids      : player_ids kept by OTHER teams (removed from pool)
    other_keeper_overalls : the overall pick slots those rivals' keepers occupy.
                            Keepers burn picks, so FEWER live picks happen before
                            your slot -- meaning better players fall to you, which
                            raises your draft alternative and lowers keeper value.
                            Without this we fall back to assuming a normal draft.
    my_slot               : your draft slot, 1..team_count
    """
    players = add_vorp(players, config) if value_col == "vorp" and "vorp" not in players else players.copy()
    if value_col not in players:
        players = add_vorp(players, config)

    other_keeper_ids = set(str(x) for x in (other_keeper_ids or set()))
    okos = sorted(other_keeper_overalls) if other_keeper_overalls else None

    if candidate_ids:
        cands = players[players["player_id"].astype(str).isin([str(c) for c in candidate_ids])].copy()
    else:
        cands = players[players["name"].isin(candidate_names)].copy()
    cands = cands.sort_values(value_col, ascending=False).reset_index(drop=True)

    my_picks = pick_overall_numbers(my_slot, config.team_count, config.roster_size)

    per_rows = []
    for i, row in cands.iterrows():
        if i >= max_keep:
            break
        cost_round = i + 1                       # 1st keeper -> round 1, etc.
        overall = my_picks[cost_round - 1]

        # keeper-free pool = players still draftable (exclude all rivals' keepers
        # and the keepers you're keeping in this scenario)
        my_kept = {str(cands.iloc[j]["player_id"]) for j in range(cost_round)}
        gone = other_keeper_ids | my_kept
        pool = players[~players["player_id"].astype(str).isin(gone)]
        pool = pool.sort_values(value_col, ascending=False).reset_index(drop=True)

        # how many LIVE picks happen before your slot? Total prior picks minus
        # the keeper slots sitting before you (rivals' + your own earlier keepers).
        if okos is not None:
            keepers_before = sum(1 for o in okos if o < overall) + (cost_round - 1)
            live_before = max(0, (overall - 1) - keepers_before)
        else:
            live_before = overall - 1            # fallback: assume a full draft
        idx = min(live_before, len(pool) - 1) if len(pool) else 0

        alt = pool.iloc[idx] if len(pool) else None
        alt_val = float(alt[value_col]) if alt is not None else 0.0
        alt_name = alt["name"] if alt is not None else "(none)"
        keeper_val = float(row[value_col])
        per_rows.append({
            "keeper": row["name"],
            "pos": row["pos"],
            "keeper_value": round(keeper_val, 1),
            "cost_round": cost_round,
            "cost_pick": overall,
            "best_available_there": alt_name,
            "pick_value": round(alt_val, 1),
            "surplus": round(keeper_val - alt_val, 1),
        })

    per_keeper = pd.DataFrame(per_rows)

    # net surplus for keeping exactly k (cumulative)
    by_rows = [{"keep": 0, "net_surplus": 0.0}]
    running = 0.0
    for k in range(1, len(per_keeper) + 1):
        running += per_keeper.iloc[k - 1]["surplus"]
        by_rows.append({"keep": k, "net_surplus": round(running, 1)})
    by_count = pd.DataFrame(by_rows)

    recommended = int(by_count.loc[by_count["net_surplus"].idxmax(), "keep"])
    return KeeperResult(recommended, per_keeper, by_count)
