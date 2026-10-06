"""In-season data: one Snapshot of the league that every in-season tool reads.

build_snapshot(client) makes ~5 ESPN calls (league, player cards, free agents,
NFL schedule, transactions) and parses them into plain dicts, so lineup,
waiver, trade and outlook logic never touch raw ESPN JSON.

Player row (dict) fields:
    player_id, name, pos, pro_team, slot (ESPN lineupSlotId or None), elig
    (eligible slot ids), injury, status (ROSTER/FREEAGENT/WAIVERS), pct_owned,
    week_proj, week_actual, season_proj, season_pts, games_played, ppg,
    weekly {period: pts}, rate (expected pts per game rest of season),
    std_week (weekly scoring std), locked (game already kicked off this week)

Rest-of-season (ROS) value of a roster = sum over the remaining weeks of its
best possible starting lineup that week -- this week from ESPN's weekly
projection, future weeks from `rate` -- skipping byes and injury absences.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import LeagueConfig
from .espn_client import player_pos, pro_team_abbrev
from .valuation import optimal_lineup_value

BENCH_SLOTS = {20, 21, 24, 25}       # bench, IR, (unused), rookie
IR_SLOT = 21

# weekly scoring volatility (std / mean) when a player has too little history
WEEKLY_CV = {"QB": 0.40, "RB": 0.55, "WR": 0.60, "TE": 0.65,
             "K": 0.50, "DST": 0.70, "IDP": 0.60}
_SHRINK_GAMES = 4                    # prior strength, in games, for weekly std
_PRIOR_GAMES = 6                     # prior strength, in games, for ROS rate
_GAME_LENGTH_MS = 3.5 * 3600 * 1000

# how many upcoming weeks an injury designation keeps a player out
_OUT_WEEKS = {"INJURY_RESERVE": 4, "OUT": 1, "SUSPENSION": 1}
# share of a normal game expected this week under a designation
_WEEK_FACTOR = {"DOUBTFUL": 0.25, "QUESTIONABLE": 0.9}


@dataclass
class Snapshot:
    config: LeagueConfig
    year: int
    week: int                                    # current scoring period
    matchup_period: int
    final_week: int                              # last scoring period of the season
    reg_season_periods: int
    matchup_periods: Dict[int, List[int]]        # matchup period -> scoring periods
    playoff_teams: int
    faab: bool
    budget: int
    lineup_slots: Dict[int, int]                 # starting slot id -> count
    teams: Dict[int, dict]
    rosters: Dict[int, List[dict]]
    free_agents: List[dict]
    schedule: List[dict]
    transactions: List[dict]
    games: Dict[str, Dict[int, int]]             # pro team -> {week: kickoff ms}
    my_team_id: Optional[int]
    names: Dict[str, str] = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)

    @property
    def remaining_weeks(self) -> List[int]:
        return list(range(self.week, self.final_week + 1))

    def team_name(self, tid) -> str:
        t = self.teams.get(int(tid)) if tid is not None else None
        return t["name"] if t else f"Team {tid}"

    def opponent_of(self, tid: int, matchup_period: Optional[int] = None) -> Optional[int]:
        mp = matchup_period or self.matchup_period
        for m in self.schedule:
            if m["period"] != mp:
                continue
            if m["home"] == tid:
                return m["away"]
            if m["away"] == tid:
                return m["home"]
        return None

    def has_game(self, pro_team: str, week: int) -> bool:
        sched = self.games.get(pro_team)
        if not sched:                                # no schedule data: assume yes
            return pro_team not in ("FA", "0")
        return week in sched


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_player_stats(player: dict, year: int, week: int) -> dict:
    """Pull projections + weekly history out of an ESPN player's `stats` list."""
    out = {"week_proj": None, "week_actual": None, "season_proj": None,
           "season_pts": None, "weekly": {}}
    played = set()
    for s in player.get("stats", []) or []:
        if s.get("seasonId") not in (year, None):
            continue
        total = s.get("appliedTotal")
        if total is None:
            continue
        src, split, per = s.get("statSourceId"), s.get("statSplitTypeId"), s.get("scoringPeriodId")
        if split == 1 and per:
            if src == 0:
                out["weekly"][int(per)] = float(total)
                if s.get("stats"):                   # empty breakdown = didn't play
                    played.add(int(per))
                if int(per) == week:
                    out["week_actual"] = float(total)
            elif src == 1 and int(per) == week:
                out["week_proj"] = float(total)
        elif split == 0:
            if src == 0:
                out["season_pts"] = float(total)
            elif src == 1:
                out["season_proj"] = float(total)
    # history = completed weeks only (the current week may be in progress)
    hist = {p: v for p, v in out["weekly"].items() if p < week and (p in played or v != 0)}
    out["weekly"] = hist
    out["games_played"] = len(hist)
    out["ppg"] = (sum(hist.values()) / len(hist)) if hist else None
    return out


def _player_row(player: dict, year: int, week: int, **extra) -> Optional[dict]:
    pos = player_pos(player)
    if pos is None:
        return None
    row = {
        "player_id": str(player.get("id")),
        "name": player.get("fullName", "Unknown"),
        "pos": pos,
        "pro_team": pro_team_abbrev(player.get("proTeamId")),
        "elig": [int(x) for x in player.get("eligibleSlots", []) or []],
        "injury": (player.get("injuryStatus") or "ACTIVE").upper(),
        "pct_owned": float((player.get("ownership") or {}).get("percentOwned", 0.0) or 0.0),
        "slot": None,
        "status": "ROSTER",
    }
    row.update(parse_player_stats(player, year, week))
    row.update(extra)
    return row


def _merge_stats(primary: dict, card: Optional[dict]) -> dict:
    """Combine stats from the roster entry and the player card (card wins ties)."""
    if not card:
        return primary
    merged = dict(primary)
    seen = {}
    for s in (primary.get("stats") or []) + (card.get("stats") or []):
        key = (s.get("seasonId"), s.get("statSourceId"), s.get("statSplitTypeId"),
               s.get("scoringPeriodId"))
        seen[key] = s
    merged["stats"] = list(seen.values())
    for k in ("injuryStatus", "ownership", "eligibleSlots", "proTeamId"):
        if card.get(k) is not None:
            merged[k] = card[k]
    return merged


def _parse_games(pro_sched: dict) -> Dict[str, Dict[int, int]]:
    games: Dict[str, Dict[int, int]] = {}
    for team in (pro_sched.get("settings", {}) or {}).get("proTeams", []) or []:
        if not team.get("id"):
            continue
        abbr = pro_team_abbrev(team["id"])
        wk = {}
        for per, glist in (team.get("proGamesByScoringPeriod") or {}).items():
            if glist:
                wk[int(per)] = int(glist[0].get("date", 0) or 0)
        games[abbr] = wk
    return games


def _parse_transactions(raws: List[dict]) -> List[dict]:
    out, seen = [], set()
    for raw in raws:
        for t in (raw or {}).get("transactions", []) or []:
            if t.get("id") in seen:
                continue
            seen.add(t.get("id"))
            items = [i for i in t.get("items", []) or [] if i.get("type") in ("ADD", "DROP")]
            if not items:
                continue
            out.append({
                "id": t.get("id"),
                "type": t.get("type"),
                "status": t.get("status"),
                "team": t.get("teamId"),
                "period": t.get("scoringPeriodId"),
                "bid": t.get("bidAmount"),
                "date": t.get("processDate") or t.get("proposedDate"),
                "items": [{"type": i["type"], "player_id": str(i.get("playerId")),
                           "from": i.get("fromTeamId"), "to": i.get("toTeamId")}
                          for i in items],
            })
    out.sort(key=lambda t: t["date"] or 0, reverse=True)
    return out


def snapshot_from_raw(
    league: dict,
    cards: dict,
    free_agents: dict,
    pro_sched: dict,
    transactions: List[dict],
    year: int,
    my_team_id: Optional[int] = None,
    now_ms: Optional[float] = None,
) -> Snapshot:
    """Pure parse of raw ESPN payloads -> Snapshot (no network)."""
    settings = league.get("settings", {}) or {}
    status = league.get("status", {}) or {}
    cfg = LeagueConfig.from_espn_settings(league)
    week = int(league.get("scoringPeriodId") or status.get("currentMatchupPeriod") or 1)
    sched_set = settings.get("scheduleSettings", {}) or {}
    acq = settings.get("acquisitionSettings", {}) or {}

    slot_counts = (settings.get("rosterSettings", {}) or {}).get("lineupSlotCounts", {}) or {}
    lineup_slots = {int(k): int(v) for k, v in slot_counts.items()
                    if int(v) > 0 and int(k) not in BENCH_SLOTS}

    card_by_id = {str(p.get("id")): p for p in (cards or {}).get("players", []) or []
                  if isinstance(p, dict)}
    # kona_playercard wraps as {"player": {...}} in some responses
    for p in (cards or {}).get("players", []) or []:
        if isinstance(p, dict) and "player" in p:
            card_by_id[str(p["player"].get("id"))] = p["player"]

    teams: Dict[int, dict] = {}
    rosters: Dict[int, List[dict]] = {}
    for t in league.get("teams", []) or []:
        tid = int(t["id"])
        name = (t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}").strip()
        rec = ((t.get("record") or {}).get("overall") or {})
        tc = t.get("transactionCounter", {}) or {}
        teams[tid] = {
            "id": tid,
            "name": name or t.get("abbrev") or f"Team {tid}",
            "abbrev": t.get("abbrev", ""),
            "owners": t.get("owners", []) or [],
            "wins": int(rec.get("wins", 0) or 0),
            "losses": int(rec.get("losses", 0) or 0),
            "ties": int(rec.get("ties", 0) or 0),
            "pf": float(rec.get("pointsFor", 0) or 0),
            "pa": float(rec.get("pointsAgainst", 0) or 0),
            "waiver_rank": t.get("waiverRank"),
            "faab_spent": int(tc.get("acquisitionBudgetSpent", 0) or 0),
            "acquisitions": int(tc.get("acquisitions", 0) or 0),
            "drops": int(tc.get("drops", 0) or 0),
            "trades": int(tc.get("trades", 0) or 0),
            "espn_playoff_pct": float((t.get("currentSimulationResults") or {}).get("playoffPct", 0) or 0) * 100,
        }
        rows = []
        for e in (t.get("roster", {}) or {}).get("entries", []) or []:
            player = (e.get("playerPoolEntry", {}) or {}).get("player", {}) or {}
            player = _merge_stats(player, card_by_id.get(str(player.get("id", e.get("playerId")))))
            row = _player_row(player, year, week, slot=e.get("lineupSlotId"), status="ROSTER")
            if row:
                rows.append(row)
        rosters[tid] = rows

    fas = []
    for e in (free_agents or {}).get("players", []) or []:
        player = e.get("player", e) or {}
        row = _player_row(player, year, week, status=(e.get("status") or "FREEAGENT").upper())
        if row:
            fas.append(row)

    schedule = []
    for m in league.get("schedule", []) or []:
        home = (m.get("home") or {}).get("teamId")
        away = (m.get("away") or {}).get("teamId")
        if home is None:
            continue
        schedule.append({
            "period": int(m.get("matchupPeriodId", 0) or 0),
            "home": int(home), "away": int(away) if away is not None else None,
            "home_pts": float((m.get("home") or {}).get("totalPoints", 0) or 0),
            "away_pts": float((m.get("away") or {}).get("totalPoints", 0) or 0),
            "winner": m.get("winner", "UNDECIDED"),
            "playoff": (m.get("playoffTierType") or "NONE") != "NONE",
        })

    mp_raw = sched_set.get("matchupPeriods", {}) or {}
    matchup_periods = {int(k): [int(x) for x in v] for k, v in mp_raw.items()}
    final_week = int(status.get("finalScoringPeriod") or (max(
        (w for ws in matchup_periods.values() for w in ws), default=17)))

    snap = Snapshot(
        config=cfg, year=int(year), week=week,
        matchup_period=int(status.get("currentMatchupPeriod") or week),
        final_week=final_week,
        reg_season_periods=int(sched_set.get("matchupPeriodCount") or 14),
        matchup_periods=matchup_periods,
        playoff_teams=int(sched_set.get("playoffTeamCount") or 4),
        faab=bool(acq.get("isUsingAcquisitionBudget")),
        budget=int(acq.get("acquisitionBudget") or 0),
        lineup_slots=lineup_slots,
        teams=teams, rosters=rosters, free_agents=fas, schedule=schedule,
        transactions=_parse_transactions(transactions),
        games=_parse_games(pro_sched or {}),
        my_team_id=my_team_id,
    )
    now_ms = now_ms if now_ms is not None else time.time() * 1000
    for rows in list(rosters.values()) + [fas]:
        for r in rows:
            enrich(r, snap, now_ms)
            snap.names[r["player_id"]] = r["name"]
    return snap


def build_snapshot(client, fa_limit: int = 200) -> Snapshot:
    """Fetch everything the in-season tools need from ESPN."""
    league = client.raw_league()
    week = int(league.get("scoringPeriodId") or 1)
    ids = [e.get("playerId") for t in league.get("teams", [])
           for e in (t.get("roster", {}) or {}).get("entries", []) or []]
    try:
        cards = client.raw_players_by_id(ids, week)
    except Exception:  # noqa: BLE001 -- roster stats still parse without cards
        cards = {}
    fas = client.raw_free_agents(week, limit=fa_limit)
    try:
        pro = client.raw_pro_schedule()
    except Exception:  # noqa: BLE001 -- byes then come from "no game" stats
        pro = {}
    txns = []
    for per in {week, max(week - 1, 1)}:
        try:
            txns.append(client.raw_transactions(per))
        except Exception:  # noqa: BLE001
            pass
    my_tid = client.my_team_id(league)
    return snapshot_from_raw(league, cards, fas, pro, txns, client.year, my_tid)


# --------------------------------------------------------------------------
# Derived per-player values
# --------------------------------------------------------------------------
def enrich(row: dict, snap: Snapshot, now_ms: float) -> None:
    """Fill rate / std_week / locked on a parsed player row."""
    prior = []
    if row.get("season_proj") and row["season_proj"] > 0:
        prior.append(row["season_proj"] / 17.0)
    if row.get("week_proj") and row["week_proj"] > 0:
        prior.append(row["week_proj"])
    prior_rate = sum(prior) / len(prior) if prior else (row.get("ppg") or 0.0)
    g = row.get("games_played", 0)
    if g and row.get("ppg") is not None:
        w = g / (g + _PRIOR_GAMES)
        rate = (1 - w) * prior_rate + w * row["ppg"]
    else:
        rate = prior_rate
    row["rate"] = max(float(rate), 0.0)

    cv = WEEKLY_CV.get(row["pos"], 0.6)
    prior_sd = cv * max(row["rate"], 3.0)
    hist = list(row.get("weekly", {}).values())
    if len(hist) >= 2:
        m = sum(hist) / len(hist)
        emp_var = sum((x - m) ** 2 for x in hist) / (len(hist) - 1)
        n = len(hist)
        var = (n * emp_var + _SHRINK_GAMES * prior_sd ** 2) / (n + _SHRINK_GAMES)
        row["std_week"] = max(math.sqrt(var), 1.5)
    else:
        row["std_week"] = max(prior_sd, 1.5)

    kickoff = snap.games.get(row["pro_team"], {}).get(snap.week)
    row["locked"] = bool(kickoff and kickoff <= now_ms)
    row["game_over"] = bool(kickoff and kickoff + _GAME_LENGTH_MS <= now_ms)


def availability(row: dict, week: int, snap: Snapshot) -> float:
    """Expected share of a normal game the player plays in `week` (0..1)."""
    if not snap.has_game(row["pro_team"], week):
        return 0.0
    inj = row.get("injury", "ACTIVE")
    out_for = _OUT_WEEKS.get(inj, 0)
    if week < snap.week + out_for:
        return 0.0
    if week == snap.week:
        return _WEEK_FACTOR.get(inj, 1.0)
    return 1.0


def week_mean(row: dict, week: int, snap: Snapshot) -> float:
    """Expected fantasy points for `week`."""
    if week == snap.week:
        if row.get("game_over") and row.get("week_actual") is not None:
            return row["week_actual"]
        avail = availability(row, week, snap)
        if avail == 0.0:
            return 0.0
        base = row["week_proj"] if row.get("week_proj") is not None else row["rate"]
        return base * avail
    return row["rate"] * availability(row, week, snap)


def week_std(row: dict, week: int, snap: Snapshot) -> float:
    if week == snap.week and row.get("game_over"):
        return 0.0
    if week_mean(row, week, snap) <= 0:
        return 0.0
    s = row["std_week"]
    return s * 0.6 if (week == snap.week and row.get("locked")) else s


def ros_points(row: dict, snap: Snapshot) -> float:
    """Raw expected points the player scores over the rest of the season
    (what managers 'see' -- not lineup-aware)."""
    return sum(week_mean(row, w, snap) for w in snap.remaining_weeks)


# --------------------------------------------------------------------------
# Roster value over the rest of the season
# --------------------------------------------------------------------------
def roster_week_values(rows: List[dict], snap: Snapshot, weeks: Optional[List[int]] = None,
                       means: Optional[Dict[Tuple[str, int], float]] = None) -> Dict[int, float]:
    """Optimal starting-lineup points for each week. Weeks where the same set
    of players is available share one computation (byes repeat a lot)."""
    weeks = weeks if weeks is not None else snap.remaining_weeks
    cfg = snap.config
    out: Dict[int, float] = {}
    cache: Dict[tuple, float] = {}
    for w in weeks:
        vals = []
        key = []
        for r in rows:
            m = means[(r["player_id"], w)] if means is not None else week_mean(r, w, snap)
            vals.append({"pos": r["pos"], "proj": m})
            key.append(round(m, 2))
        k = tuple(key)
        if k not in cache:
            cache[k] = optimal_lineup_value(vals, cfg)
        out[w] = cache[k]
    return out


def roster_ros_value(rows: List[dict], snap: Snapshot) -> Tuple[float, float]:
    """(total ROS lineup points, this week's lineup points)."""
    wv = roster_week_values(rows, snap)
    return sum(wv.values()), wv.get(snap.week, 0.0)


def team_rows(snap: Snapshot, tid: int) -> List[dict]:
    return list(snap.rosters.get(int(tid), []))
