"""Start/sit -- pick the lineup that maximizes P(beating this week's opponent).

Maximizing projected points is the right call only when you're a coin flip.
Treat each side's week as roughly Normal:

    P(win) = Phi( (mu_me - mu_opp) / sqrt(sigma_me^2 + sigma_opp^2) )

* Favored (mu_me > mu_opp): shrinking sigma_me raises P(win) -> prefer safe floors.
* Underdog: raising sigma_me raises P(win) -> prefer boom/bust upside.
* The opponent's own risk (sigma_opp, from the lineup they've actually set)
  sits in the denominator: against a volatile opponent your variance matters
  less, and against a safe lineup it matters more.

Search: for a grid of risk appetites lambda, solve the exact slot assignment
maximizing sum(mu_i + lambda * sigma_i^2) (a linear objective, so the
Hungarian algorithm solves it exactly with real ESPN slot eligibility), then
keep whichever lineup has the highest true P(win). That traces the
mean-variance frontier, which is where the P(win)-optimal lineup lives.
Players whose games have kicked off are locked in their current slot.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .config import ESPN_SLOT_LABELS
from .season import BENCH_SLOTS, IR_SLOT, Snapshot, week_mean, week_std

_IDP_LABELS = {8: "DT", 9: "DE", 10: "LB", 11: "DL", 12: "CB", 13: "S", 14: "DB", 15: "DP"}
_SLOT_ORDER = [0, 1, 2, 4, 6, 3, 5, 23, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def slot_label(slot_id) -> str:
    if slot_id is None:
        return "—"
    sid = int(slot_id)
    return ESPN_SLOT_LABELS.get(sid) or _IDP_LABELS.get(sid) or str(sid)


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def win_prob(mu_me: float, sd_me: float, mu_opp: float, sd_opp: float) -> float:
    sd = math.sqrt(sd_me ** 2 + sd_opp ** 2)
    if sd < 1e-9:
        return 1.0 if mu_me > mu_opp else (0.5 if mu_me == mu_opp else 0.0)
    return phi((mu_me - mu_opp) / sd)


# --------------------------------------------------------------------------
# Exact assignment (Hungarian, O(n^2 m)), rows = slots, cols = players
# --------------------------------------------------------------------------
def _hungarian(cost: List[List[float]]) -> List[int]:
    n, m = len(cost), len(cost[0]) if cost else 0
    if n == 0:
        return []
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            ans[p[j] - 1] = j - 1
    return ans


def best_assignment(slots: List[int], players: List[dict], score) -> List[Optional[int]]:
    """For each slot, the index into `players` to start there (None = empty).
    `score(player) -> float` is maximized subject to eligibleSlots."""
    if not slots:
        return []
    n_real = len(players)
    EMPTY, BAD = 1000.0, 1e7          # leaving a slot empty is a big penalty
    cost = []
    for sid in slots:
        row = [(-score(p) if sid in p["elig"] else BAD) for p in players]
        row += [EMPTY] * len(slots)   # dummy "nobody" columns
        cost.append(row)
    ans = _hungarian(cost)
    return [j if 0 <= j < n_real and cost[i][j] < BAD else None
            for i, j in enumerate(ans)]


def expand_slots(lineup_slots: Dict[int, int]) -> List[int]:
    order = {s: i for i, s in enumerate(_SLOT_ORDER)}
    out = []
    for sid in sorted(lineup_slots, key=lambda s: order.get(s, 99)):
        out += [sid] * lineup_slots[sid]
    return out


# --------------------------------------------------------------------------
# Lineups
# --------------------------------------------------------------------------
def _lineup_stats(entries: List[dict]) -> Tuple[float, float]:
    mu = sum(e["mean"] for e in entries)
    sd = math.sqrt(sum(e["std"] ** 2 for e in entries))
    return mu, sd


def _entry(slot, p, snap) -> dict:
    return {"slot": slot, "slot_label": slot_label(slot), "player_id": p["player_id"],
            "name": p["name"], "pos": p["pos"], "pro_team": p["pro_team"],
            "injury": p["injury"], "mean": week_mean(p, snap.week, snap),
            "std": week_std(p, snap.week, snap), "locked": bool(p.get("locked"))}


def current_lineup(rows: List[dict], snap: Snapshot) -> List[dict]:
    """The lineup a team has actually set on ESPN right now."""
    return [_entry(r["slot"], r, snap) for r in rows
            if r.get("slot") is not None and int(r["slot"]) not in BENCH_SLOTS]


def solve_lineup(rows: List[dict], snap: Snapshot, lam: float = 0.0) -> List[dict]:
    """Best lineup for risk appetite `lam` (0 = max projected points).
    Locked starters stay put; locked bench players can't come in."""
    slots = expand_slots(snap.lineup_slots)
    fixed, free_players = [], []
    for r in rows:
        if r.get("slot") == IR_SLOT:
            continue
        in_lineup = r.get("slot") is not None and int(r["slot"]) not in BENCH_SLOTS
        if r.get("locked"):
            if in_lineup and int(r["slot"]) in slots:
                slots.remove(int(r["slot"]))
                fixed.append(_entry(int(r["slot"]), r, snap))
            continue
        free_players.append(r)

    def score(p):
        m, s = week_mean(p, snap.week, snap), week_std(p, snap.week, snap)
        return m + lam * s * s

    picks = best_assignment(slots, free_players, score)
    out = list(fixed)
    for sid, j in zip(slots, picks):
        if j is not None:
            out.append(_entry(sid, free_players[j], snap))
    order = {s: i for i, s in enumerate(_SLOT_ORDER)}
    return sorted(out, key=lambda e: order.get(e["slot"], 99))


_LAMBDAS = [x / 100.0 for x in range(-10, 11, 1)]


def start_sit(snap: Snapshot, my_tid: int, opp_tid: Optional[int] = None) -> dict:
    """Recommended lineup vs this week's opponent, plus the comparison set."""
    my_rows = snap.rosters.get(int(my_tid), [])
    opp_tid = opp_tid if opp_tid is not None else snap.opponent_of(int(my_tid))

    # opponent: the lineup they've actually set (that's the risk they're
    # taking); fall back to their best lineup if nothing is set yet.
    opp_lineup, opp_source = [], "none"
    if opp_tid is not None:
        opp_rows = snap.rosters.get(int(opp_tid), [])
        opp_lineup = current_lineup(opp_rows, snap)
        opp_source = "set lineup"
        if not opp_lineup:
            opp_lineup = solve_lineup(opp_rows, snap)
            opp_source = "projected best lineup"
    mu_o, sd_o = _lineup_stats(opp_lineup)

    cur = current_lineup(my_rows, snap)
    max_pts = solve_lineup(my_rows, snap, 0.0)

    def objective(lu):
        mu, sd = _lineup_stats(lu)
        return win_prob(mu, sd, mu_o, sd_o) if opp_tid is not None else mu

    # start from max points; a risk tilt must beat it outright to be chosen
    best, best_p, best_lam = max_pts, objective(max_pts), 0.0
    seen = {tuple(sorted(e["player_id"] for e in max_pts))}
    if opp_tid is not None:
        for lam in sorted(_LAMBDAS, key=abs):
            lu = solve_lineup(my_rows, snap, lam)
            key = tuple(sorted(e["player_id"] for e in lu))
            if key in seen:
                continue
            seen.add(key)
            p = objective(lu)
            if p > best_p + 1e-4:
                best, best_p, best_lam = lu, p, lam

    def summary(lu):
        mu, sd = _lineup_stats(lu)
        return {"lineup": lu, "mean": mu, "std": sd,
                "win_prob": win_prob(mu, sd, mu_o, sd_o) if opp_tid is not None else None}

    rec = summary(best)
    cur_ids = {e["player_id"] for e in cur}
    rec_ids = {e["player_id"] for e in best}
    names = {r["player_id"]: r["name"] for r in my_rows}
    changes = {"start": [names[i] for i in rec_ids - cur_ids if i in names],
               "bench": [names[i] for i in cur_ids - rec_ids if i in names]}

    edge = rec["mean"] - mu_o
    if opp_tid is None:
        stance = "No opponent this week — maximizing projected points."
    elif best_lam > 0:
        stance = (f"You're projected {abs(edge):.1f} pts behind, so the lineup "
                  f"leans toward upside (higher variance) to raise your odds.")
    elif best_lam < 0:
        stance = (f"You're projected {edge:.1f} pts ahead, so the lineup leans "
                  f"toward safe floors (lower variance) to protect the lead.")
    else:
        stance = "Max projected points is also your best shot at winning this week."

    return {
        "opponent": opp_tid, "opp_lineup": opp_lineup, "opp_source": opp_source,
        "opp_mean": mu_o, "opp_std": sd_o,
        "current": summary(cur), "max_points": summary(max_pts), "recommended": rec,
        "risk_lambda": best_lam, "changes": changes, "stance": stance,
    }


def team_week_strength(rows: List[dict], snap: Snapshot) -> Tuple[float, float]:
    """(mean, std) of a team's best lineup this week -- used by the outlook."""
    return _lineup_stats(solve_lineup(rows, snap, 0.0))
