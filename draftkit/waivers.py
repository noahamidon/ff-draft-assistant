"""Waivers -- who to add, who to drop, and what the rest of the league is doing.

A pickup's value is how much it raises your roster's rest-of-season lineup
points (see season.roster_ros_value) after making room by dropping the player
whose loss hurts least. Because this week is valued from ESPN's weekly
projection (byes and injuries included) and later weeks from ROS rates, a
streamer who plugs a bye/injury hole this week gets credit for exactly the
points he adds this week, while a long-term upgrade earns it every week.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .season import IR_SLOT, Snapshot, roster_ros_value, ros_points, week_mean


def rank_pickups(snap: Snapshot, my_tid: int, top: int = 25,
                 pool_size: int = 80, min_gain: float = 0.5) -> pd.DataFrame:
    rows = list(snap.rosters.get(int(my_tid), []))
    active = [r for r in rows if r.get("slot") != IR_SLOT]
    ir = [r for r in rows if r.get("slot") == IR_SLOT]
    base_total, base_week = roster_ros_value(rows, snap)
    open_spot = len(active) < snap.config.roster_size

    # candidates: best long-term + best this-week (streamers)
    fas = [f for f in snap.free_agents if f["pos"] in _started_positions(snap)]
    by_ros = sorted(fas, key=lambda f: ros_points(f, snap), reverse=True)[:pool_size]
    by_week = sorted(fas, key=lambda f: week_mean(f, snap.week, snap), reverse=True)[:20]
    pool = {f["player_id"]: f for f in by_ros + by_week}.values()

    out = []
    for fa in pool:
        options: List[Optional[dict]] = [None] if open_spot else []
        options += active
        best = None
        for drop in options:
            new_rows = [r for r in active if r is not drop] + ir + [fa]
            total, wk = roster_ros_value(new_rows, snap)
            gain = total - base_total
            if best is None or gain > best[0]:
                best = (gain, wk - base_week, drop)
        if best is None or best[0] < min_gain:
            continue
        gain, wgain, drop = best
        out.append({
            "add": fa["name"], "pos": fa["pos"], "team": fa["pro_team"],
            "status": "Waivers" if fa["status"] == "WAIVERS" else "Free agent",
            "owned_%": round(fa["pct_owned"], 1),
            "inj": "" if fa["injury"] == "ACTIVE" else fa["injury"].title().replace("_", " "),
            "this_wk_proj": round(week_mean(fa, snap.week, snap), 1),
            "ros_ppg": round(fa["rate"], 1),
            "gain_this_wk": round(wgain, 1),
            "gain_ros": round(gain, 1),
            "drop": drop["name"] if drop else "(open spot)",
            "drop_pos": drop["pos"] if drop else "",
            "player_id": fa["player_id"],
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("gain_ros", ascending=False).head(top).reset_index(drop=True)


def _started_positions(snap: Snapshot) -> set:
    cfg = snap.config
    pos = {p for p, c in cfg.starters.items() if c > 0}
    for _, elig in cfg.flex_slots:
        pos |= set(elig)
    return pos


def waiver_table(snap: Snapshot) -> pd.DataFrame:
    """Waiver priority, FAAB left, and move counts per team."""
    rows = []
    for tid, t in snap.teams.items():
        r = {"team": t["name"], "waiver_rank": t["waiver_rank"],
             "adds": t["acquisitions"], "drops": t["drops"], "trades": t["trades"]}
        if snap.faab:
            r["faab_left"] = snap.budget - t["faab_spent"]
        if tid == snap.my_team_id:
            r["team"] += "  (you)"
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("waiver_rank", na_position="last").reset_index(drop=True)


_TYPE_LABEL = {"WAIVER": "Waiver claim", "WAIVER_ERROR": "Failed claim",
               "FREEAGENT": "Free-agent add", "TRADE_ACCEPT": "Trade"}


def recent_activity(snap: Snapshot, limit: int = 60) -> pd.DataFrame:
    rows = []
    for t in snap.transactions[:limit]:
        adds = [snap.names.get(i["player_id"], f"#{i['player_id']}") for i in t["items"] if i["type"] == "ADD"]
        drops = [snap.names.get(i["player_id"], f"#{i['player_id']}") for i in t["items"] if i["type"] == "DROP"]
        status = (t["status"] or "").title().replace("_", " ")
        rows.append({
            "week": t["period"],
            "team": snap.team_name(t["team"]),
            "move": _TYPE_LABEL.get(t["type"], (t["type"] or "").title()),
            "added": ", ".join(adds),
            "dropped": ", ".join(drops),
            "bid": t["bid"] if snap.faab and t["bid"] is not None else None,
            "status": status,
            "when": pd.to_datetime(t["date"], unit="ms") if t["date"] else pd.NaT,
        })
    df = pd.DataFrame(rows)
    if not df.empty and not snap.faab:
        df = df.drop(columns=["bid"])
    return df
