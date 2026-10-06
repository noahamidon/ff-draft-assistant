"""Trades -- evaluate an offer, and find ones worth proposing.

Value = change in each team's rest-of-season starting-lineup points (the same
yardstick as waivers). Lineup-aware value is what makes win-win trades exist:
your 4th RB is worth ~0 to you but may start for a team with an RB hole.

Confidence ("your win %"): the traded players' ROS outlooks are uncertain, so
each is rescaled by a lognormal factor (sd ~30%) across many samples; the
share of samples where your lineup gains is P(the trade helps you).

Acceptance odds are a heuristic: real managers weigh both their lineup and
the raw points they see changing hands. We blend P(their lineup gains) with
a fairness read on raw ROS points. Treat it as "worth sending / long shot",
not a promise.

When a trade leaves a team over the roster limit, it drops its least
valuable remaining player (counted in the evaluation).
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .season import IR_SLOT, Snapshot, roster_week_values, ros_points, slot_value, week_mean

_OUTLOOK_SD = 0.30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50), -50)))


class _Ctx:
    """Precomputed weekly means so evaluating thousands of trades is cheap."""

    def __init__(self, snap: Snapshot, tids: Sequence[int]):
        self.snap = snap
        self.weeks = snap.remaining_weeks
        self.means: Dict[Tuple[str, int], float] = {}   # slot value (with stream fill-in)
        self.own: Dict[Tuple[str, int], float] = {}     # the player's own expected points
        self.raw: Dict[str, float] = {}
        self.rows: Dict[int, List[dict]] = {}
        for tid in tids:
            rows = list(snap.rosters.get(int(tid), []))
            self.rows[int(tid)] = rows
            for r in rows:
                for w in self.weeks:
                    self.means[(r["player_id"], w)] = slot_value(r, w, snap)
                    self.own[(r["player_id"], w)] = week_mean(r, w, snap)
                self.raw[r["player_id"]] = ros_points(r, snap)
        self.base = {tid: self.value(rows) for tid, rows in self.rows.items()}

    def value(self, rows: List[dict], means=None) -> Tuple[float, float]:
        wv = roster_week_values(rows, self.snap, self.weeks, means or self.means)
        return sum(wv.values()), wv.get(self.snap.week, 0.0)

    def after(self, tid: int, out_ids: set, in_rows: List[dict]) -> Tuple[List[dict], Optional[dict]]:
        """Roster after the trade, plus the forced drop if it's over the limit."""
        keep = [r for r in self.rows[tid] if r["player_id"] not in out_ids]
        new = keep + list(in_rows)
        active = [r for r in new if r.get("slot") != IR_SLOT]
        dropped = None
        if len(active) > self.snap.config.roster_size:
            in_ids = {r["player_id"] for r in in_rows}
            pool = [r for r in active if r["player_id"] not in in_ids] or active
            dropped = min(pool, key=lambda r: self.raw.get(r["player_id"], 0.0))
            new = [r for r in new if r is not dropped]
        return new, dropped


def _deltas(ctx: _Ctx, my_tid, their_tid, give, get, means=None):
    give_ids = {r["player_id"] for r in give}
    get_ids = {r["player_id"] for r in get}
    my_new, my_drop = ctx.after(my_tid, give_ids, get)
    their_new, their_drop = ctx.after(their_tid, get_ids, give)
    if means is None:
        my_base, their_base = ctx.base[my_tid], ctx.base[their_tid]
    else:
        my_base, their_base = ctx.value(ctx.rows[my_tid], means), ctx.value(ctx.rows[their_tid], means)
    m_tot, m_wk = ctx.value(my_new, means)
    t_tot, t_wk = ctx.value(their_new, means)
    return {
        "my_ros": m_tot - my_base[0], "my_week": m_wk - my_base[1],
        "their_ros": t_tot - their_base[0], "their_week": t_wk - their_base[1],
        "my_drop": my_drop, "their_drop": their_drop,
    }


def _accept_odds(p_their_gain: float, raw_in: float, raw_out: float) -> float:
    scale = max(20.0, 0.15 * max(raw_in, raw_out))
    perceived = _sigmoid((raw_in - raw_out) / scale * 2.0)
    return 0.6 * p_their_gain + 0.4 * perceived


def evaluate_trade(snap: Snapshot, my_tid: int, their_tid: int,
                   give_ids: Sequence[str], get_ids: Sequence[str],
                   n_samples: int = 300, seed: int = 0, ctx: Optional[_Ctx] = None) -> dict:
    my_tid, their_tid = int(my_tid), int(their_tid)
    ctx = ctx or _Ctx(snap, [my_tid, their_tid])
    give = [r for r in ctx.rows[my_tid] if r["player_id"] in set(map(str, give_ids))]
    get = [r for r in ctx.rows[their_tid] if r["player_id"] in set(map(str, get_ids))]
    d = _deltas(ctx, my_tid, their_tid, give, get)

    rng = np.random.default_rng(seed)
    traded = [r["player_id"] for r in give + get]
    my_wins = their_wins = 0
    my_samples = []
    for _ in range(n_samples):
        means = dict(ctx.means)
        for pid in traded:
            f = float(np.exp(rng.normal(-0.5 * _OUTLOOK_SD ** 2, _OUTLOOK_SD)))
            for w in ctx.weeks:
                # only his own production is uncertain; the stream fill-in isn't
                own = ctx.own[(pid, w)]
                means[(pid, w)] = ctx.means[(pid, w)] + own * (f - 1.0)
        s = _deltas(ctx, my_tid, their_tid, give, get, means)
        my_wins += s["my_ros"] > 0
        their_wins += s["their_ros"] > 0
        my_samples.append(s["my_ros"])

    raw_in = sum(ctx.raw[r["player_id"]] for r in give)     # what THEY receive
    raw_out = sum(ctx.raw[r["player_id"]] for r in get)
    p_me, p_them = my_wins / n_samples, their_wins / n_samples
    accept = _accept_odds(p_them, raw_in, raw_out)

    if d["my_ros"] <= 0:
        verdict = "Decline — this lowers your rest-of-season lineup."
    elif p_me >= 0.7 and accept >= 0.5:
        verdict = "Strong — helps you with high confidence and should appeal to them."
    elif p_me >= 0.7:
        verdict = "Good for you — but they may balk; expect a counter."
    elif p_me >= 0.55:
        verdict = "Lean accept — modest edge with real uncertainty."
    else:
        verdict = "Coin flip — the edge is within projection noise."

    return {
        **d, "p_my_gain": p_me, "p_their_gain": p_them, "accept_odds": accept,
        "my_ros_p10": float(np.percentile(my_samples, 10)) if my_samples else 0.0,
        "my_ros_p90": float(np.percentile(my_samples, 90)) if my_samples else 0.0,
        "raw_given": raw_in, "raw_received": raw_out, "verdict": verdict,
        "give": [r["name"] for r in give], "get": [r["name"] for r in get],
    }


def suggest_trades(snap: Snapshot, my_tid: int, per_side: int = 8, per_team: int = 3,
                   top: int = 12, n_samples: int = 200, min_confidence: float = 0.6,
                   min_accept: float = 0.35) -> pd.DataFrame:
    """Search 1-for-1, 2-for-1 and 1-for-2 deals with every rival; keep the ones
    that help you and plausibly help them, then score them with full confidence.
    Only trades with P(helps you) >= min_confidence and accept odds >=
    min_accept are returned."""
    my_tid = int(my_tid)
    rivals = [t for t in snap.teams if t != my_tid]
    ctx = _Ctx(snap, [my_tid] + rivals)

    def tradeable(tid):
        rows = [r for r in ctx.rows[tid] if r.get("slot") != IR_SLOT]
        return sorted(rows, key=lambda r: ctx.raw[r["player_id"]], reverse=True)[:per_side]

    mine = tradeable(my_tid)
    shortlist = []
    for tid in rivals:
        theirs = tradeable(tid)
        shapes = [(1, 1), (2, 1), (1, 2)]
        found = []
        for ng, nr in shapes:
            for give in itertools.combinations(mine, ng):
                for get in itertools.combinations(theirs, nr):
                    d = _deltas(ctx, my_tid, tid, list(give), list(get))
                    if d["my_ros"] <= 3.0 or d["their_ros"] < -5.0:
                        continue
                    raw_in = sum(ctx.raw[r["player_id"]] for r in give)
                    raw_out = sum(ctx.raw[r["player_id"]] for r in get)
                    quick = _accept_odds(_sigmoid(d["their_ros"] / 8.0), raw_in, raw_out)
                    found.append((d["my_ros"] * quick, tid, give, get))
        found.sort(key=lambda x: x[0], reverse=True)
        shortlist += found[:per_team]

    shortlist.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for _, tid, give, get in shortlist[: top * 3]:
        ev = evaluate_trade(snap, my_tid, tid, [r["player_id"] for r in give],
                            [r["player_id"] for r in get], n_samples=n_samples, ctx=ctx)
        if ev["p_my_gain"] < min_confidence or ev["accept_odds"] < min_accept:
            continue
        rows.append({
            "partner": snap.team_name(tid), "partner_id": tid,
            "you_give": ", ".join(ev["give"]), "you_get": ", ".join(ev["get"]),
            "your_gain_ros": round(ev["my_ros"], 1),
            "their_gain_ros": round(ev["their_ros"], 1),
            "your_win_%": round(100 * ev["p_my_gain"]),
            "accept_%": round(100 * ev["accept_odds"]),
            "_score": ev["my_ros"] * ev["accept_odds"] * ev["p_my_gain"],
            "give_ids": [r["player_id"] for r in give],
            "get_ids": [r["player_id"] for r in get],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("_score", ascending=False).head(top).drop(columns="_score").reset_index(drop=True)
