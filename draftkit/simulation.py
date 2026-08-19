"""Monte Carlo over the rest of the draft -- the strong recommendation engine.

For each candidate you could take right now:
  * simulate the remaining draft many times
  * opponents pick by sampling from ADP (softmax with temperature + noise)
  * you fill your future picks greedily by marginal starting-lineup value
  * score the sim by your final optimal starting-lineup value
Recommend the candidate with the best mean outcome. Reporting the spread
(std) across sims gives you a risk read for free.

This naturally captures snake dynamics, positional runs, and scarcity without
hand-tuned heuristics -- it's "draft context" done properly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import LeagueConfig
from .draft_state import DraftState
from .valuation import marginal_lineup_value, optimal_lineup_value


@dataclass
class SimResult:
    player_id: str
    name: str
    pos: str
    mean_value: float
    std_value: float
    n_sims: int


def pick_probabilities(sim_df: pd.DataFrame, n_samples: int = 20000, seed: int = 0) -> dict:
    """P(each candidate yields the best final lineup), from the sim's mean/std.

    Samples each candidate's final-roster value from Normal(mean, std) and counts
    how often each is the argmax. Big, clear gaps -> the top pick approaches
    ~100%; tightly bunched candidates split the probability (e.g. 30% / 29%),
    which is exactly the confidence signal we want to surface.
    """
    if sim_df is None or sim_df.empty:
        return {}
    rng = np.random.default_rng(seed)
    means = sim_df["mean_value"].to_numpy(dtype=float)
    stds = np.maximum(sim_df["std_value"].to_numpy(dtype=float), 1e-6)
    draws = rng.normal(means, stds, size=(n_samples, len(means)))
    winners = draws.argmax(axis=1)
    counts = np.bincount(winners, minlength=len(means))
    probs = counts / float(n_samples)
    return {str(pid): float(p) for pid, p in zip(sim_df["player_id"], probs)}


def availability_until(
    state: DraftState,
    players: pd.DataFrame,
    config: LeagueConfig,
    until_overall: int,
    target_ids: list,
    n_sims: int = 300,
    opp_temp: float = 5.0,
    opp_pool: int = 12,
    seed: int = 0,
) -> dict:
    """P(each target player is still available at `until_overall`).

    Simulates only the opponent picks between now and your next pick (fast),
    using the same need-aware opponent model, and counts how often each target
    survives. If it's already your turn, everyone is available (1.0).
    """
    from collections import Counter

    from .valuation import add_vorp

    targets = {str(t) for t in target_ids}
    if not targets or until_overall <= state.next_overall:
        return {t: 1.0 for t in targets}

    rng = np.random.default_rng(seed)
    avail = add_vorp(state.available(players), config).sort_values("adp").reset_index(drop=True)
    span = until_overall - state.next_overall
    cap = min(len(avail), max(60, span + 40))
    base_board = _rows_to_records(avail.head(cap))
    need_ctx = _make_need_ctx(config)

    seed_counts = {t: Counter() for t in range(1, state.team_count + 1)}
    for t, picks in state.rosters().items():
        seed_counts[t] = Counter(pk.pos for pk in picks)

    survived = {t: 0 for t in targets}
    for _ in range(n_sims):
        board = [dict(r) for r in base_board]
        counts = {t: Counter(c) for t, c in seed_counts.items()}
        overall = state.next_overall
        while overall < until_overall and board:
            team = state.team_on_clock(overall)
            if team != state.my_team:
                idx = _opponent_pick(board, overall, counts[team], need_ctx, rng, opp_temp, opp_pool)
                if idx >= 0:
                    counts[team][board.pop(idx)["pos"]] += 1
            overall += 1
        remaining = {r["player_id"] for r in board}
        for t in targets:
            if t in remaining:
                survived[t] += 1
    return {t: survived[t] / n_sims for t in targets}


def _rows_to_records(df: pd.DataFrame) -> List[dict]:
    cols = ["player_id", "name", "pos", "proj", "adp"]
    if "vorp" in df.columns:
        cols.append("vorp")
    return df[cols].to_dict("records")


def _make_need_ctx(config: LeagueConfig) -> dict:
    """Precompute the position requirements ONCE (recomputing the flex set on
    every need lookup was the main bottleneck)."""
    flex_elig = (
        set().union(*[e for _, e in config.flex_slots]) if config.flex_slots else set()
    )
    return {
        "starters": dict(config.starters),
        "flex_elig": flex_elig,
        "flex_count": config.flex_count,
    }


def _need_mult(ctx: dict, counts: dict, pos: str) -> float:
    """Positional-need multiplier using precomputed context. O(1), no allocs."""
    have = counts.get(pos, 0)
    if pos in ("K", "DST", "IDP"):
        return 0.10 if have >= 1 else 0.5
    req = ctx["starters"].get(pos, 0)
    if have < req:
        return 3.0
    if pos in ctx["flex_elig"] and have < req + ctx["flex_count"]:
        return 1.3
    return 0.35


def _opponent_pick(board, overall, counts, ctx, rng, temp, pool_size) -> int:
    """ADP + need opponent model, pure-Python weighted sample (fast for small k)."""
    n = len(board)
    if n == 0:
        return -1
    k = min(pool_size, n)
    weights = [0.0] * k
    tot = 0.0
    inv_temp = 1.0 / max(temp, 1e-6)
    for i in range(k):
        p = board[i]
        w = math.exp((overall - p["adp"]) * inv_temp) * _need_mult(ctx, counts, p["pos"])
        weights[i] = w
        tot += w
    if tot <= 0:
        return int(rng.integers(k))
    r = rng.random() * tot
    cum = 0.0
    for i in range(k):
        cum += weights[i]
        if r <= cum:
            return i
    return k - 1


def _greedy_my_pick(board, counts, ctx) -> int:
    """Fast pick for future-me: best VORP weighted by need. O(horizon)."""
    best_idx, best = -1, -1e18
    horizon = min(len(board), 30)
    for i in range(horizon):
        p = board[i]
        score = p.get("vorp", p["proj"]) * _need_mult(ctx, counts, p["pos"])
        if score > best:
            best, best_idx = score, i
    return best_idx


def simulate_candidates(
    state: DraftState,
    players: pd.DataFrame,
    config: LeagueConfig,
    candidates: Optional[pd.DataFrame] = None,
    n_sims: int = 150,
    opp_temp: float = 5.0,
    opp_pool: int = 12,
    max_candidates: int = 12,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return a DataFrame of SimResult rows, best mean_value first."""
    rng = np.random.default_rng(seed)

    # compute VORP once on the available pool so the inner loop is cheap
    from collections import Counter

    from .valuation import add_vorp
    avail = add_vorp(state.available(players), config).sort_values("adp").reset_index(drop=True)
    if candidates is None:
        candidates = avail.head(max_candidates)
    else:
        candidates = candidates.head(max_candidates)

    proj_by_id = dict(zip(players["player_id"].astype(str), players["proj"].astype(float)))
    my_current = [
        {"pos": p.pos, "proj": proj_by_id.get(p.player_id, 0.0)}
        for p in state.my_roster()
    ]

    total_picks = state.total_picks
    start_overall = state.next_overall
    picks_remaining = total_picks - start_overall + 1
    need_ctx = _make_need_ctx(config)

    # Cap the board to the realistically-draftable pool: only the top players by
    # ADP ever get picked, so simulating the 400th-ranked player is wasted work.
    cap = min(len(avail), max(160, picks_remaining + 40))
    base_board = _rows_to_records(avail.head(cap))

    # seed each team's position COUNTS from the actual draft so far (O(1) need)
    seed_counts: Dict[int, Counter] = {t: Counter() for t in range(1, state.team_count + 1)}
    for t, picks in state.rosters().items():
        seed_counts[t] = Counter(pk.pos for pk in picks)

    results: List[SimResult] = []
    for _, crow in candidates.iterrows():
        cand = {
            "player_id": str(crow["player_id"]),
            "name": crow["name"],
            "pos": crow["pos"],
            "proj": float(crow["proj"]),
        }
        cand_board = [dict(r) for r in base_board if r["player_id"] != cand["player_id"]]
        values = np.empty(n_sims, dtype=float)

        for s in range(n_sims):
            board = [dict(r) for r in cand_board]
            my_roster = list(my_current) + [{"pos": cand["pos"], "proj": cand["proj"]}]
            counts = {t: Counter(c) for t, c in seed_counts.items()}
            counts[state.my_team][cand["pos"]] += 1

            overall = start_overall + 1
            while overall <= total_picks and board:
                team = state.team_on_clock(overall)
                if team == state.my_team:
                    idx = _greedy_my_pick(board, counts[team], need_ctx)
                    if idx >= 0:
                        pick = board.pop(idx)
                        my_roster.append({"pos": pick["pos"], "proj": pick["proj"]})
                        counts[team][pick["pos"]] += 1
                else:
                    idx = _opponent_pick(
                        board, overall, counts[team], need_ctx, rng, opp_temp, opp_pool,
                    )
                    if idx >= 0:
                        counts[team][board.pop(idx)["pos"]] += 1
                overall += 1

            values[s] = optimal_lineup_value(my_roster, config)

        results.append(
            SimResult(
                player_id=cand["player_id"],
                name=cand["name"],
                pos=cand["pos"],
                mean_value=float(values.mean()),
                std_value=float(values.std()),
                n_sims=n_sims,
            )
        )

    out = pd.DataFrame([r.__dict__ for r in results])
    return out.sort_values("mean_value", ascending=False).reset_index(drop=True)
