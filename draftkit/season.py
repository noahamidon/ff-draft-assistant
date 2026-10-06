"""In-season data: one Snapshot of the league that every in-season tool reads.

build_snapshot(client) makes ~5 ESPN calls (league, player cards, free agents,
NFL schedule, transactions) and parses them into plain dicts, so lineup,
waiver, trade and outlook logic never touch raw ESPN JSON.

Player row (dict) fields:
    player_id, name, pos, pro_team, slot (ESPN lineupSlotId or None), elig
    (eligible slot ids), injury, status (ROSTER/FREEAGENT/WAIVERS), pct_owned,
    week_proj, week_actual, season_proj, season_pts, games_played, ppg,
    weekly {period: pts}, src (other sources' raw numbers, see sources.py),
    mu {week: projected pts if he plays}, play {week: P(plays)},
    rate (blended ROS pts per game), est / est_ros (per-source numbers),
    std_week (weekly scoring std), locked (game already kicked off this week)

The projection model (apply_model):
  1. Blend sources with equal weight -- ESPN, Sleeper/RotoWire (rescored to
     your league), FantasyPros consensus (rank-matched to points) -- plus this
     season's production, shrunk by how predictive it is at the position.
  2. K and D/ST projections are pulled toward the position average: kicker
     scoring has close to no week-to-week or year-to-year predictability (PFF)
     and K/DST are the least accurate projections of any position (Fantasy
     Football Analytics); matchup/implied totals carry *some* weekly signal
     (4for4), so this week is shrunk less than future weeks.
  3. Availability: P(plays) = 0 on byes/OUT/IR, 0.71 if Questionable and 0.06
     if Doubtful (Footballguys injury index), and for future weeks a position
     injury hazard (QB ~1.0, RB/TE ~2.8, WR ~2.0 games missed per season).
  4. Replacement level comes from the actual waiver pool: for every position
     and week, a realistically obtainable free agent (the 3rd best; 2nd for
     the deep K/DST pools), less a small cost for using a roster move. An
     empty slot -- bye, injury -- is filled at that level instead of scoring 0.

Rest-of-season (ROS) value of a roster = sum over the remaining weeks of its
best possible starting lineup that week, where any slot can always be filled
at replacement level. So a pickup is only worth what it adds over streaming.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .config import LeagueConfig
from .espn_client import player_pos, pro_team_abbrev
from .sources import SOURCES, attach, fetch_all, rank_to_points
from .valuation import optimal_lineup_value

BENCH_SLOTS = {20, 21, 24, 25}       # bench, IR, (unused), rookie
IR_SLOT = 21

# weekly scoring volatility (std / mean) when a player has too little history
WEEKLY_CV = {"QB": 0.40, "RB": 0.55, "WR": 0.60, "TE": 0.65,
             "K": 0.50, "DST": 0.70, "IDP": 0.60}
_SHRINK_GAMES = 4                    # prior strength, in games, for weekly std
_GAME_LENGTH_MS = 3.5 * 3600 * 1000

# Weight on this season's points-per-game vs the projection sources is
# games / (games + PRIOR). K/DST production is mostly noise, so it barely counts.
PRIOR_GAMES = {"QB": 6, "RB": 6, "WR": 6, "TE": 6, "K": 30, "DST": 15, "IDP": 8}

# Share of a projection's gap from the position average that we believe.
RELIABILITY_WEEK = {"K": 0.6, "DST": 0.75}
RELIABILITY_ROS = {"K": 0.3, "DST": 0.5}

# Availability. Injury designations this week (Footballguys injury index:
# 71% of Questionable players play, 5.9% of Doubtful).
PLAY_PROB_WEEK = {"QUESTIONABLE": 0.71, "DOUBTFUL": 0.06, "OUT": 0.0,
                  "INJURY_RESERVE": 0.0, "SUSPENSION": 0.0}
# weeks an absence lasts (NFL IR is a 4-game minimum)
_OUT_WEEKS = {"INJURY_RESERVE": 4, "OUT": 1, "SUSPENSION": 1}
# future-week injury hazard: avg games missed per 17 (FantasySquawk, 2015-25)
MISS_RATE = {"QB": 1.0 / 17, "RB": 2.8 / 17, "WR": 2.0 / 17, "TE": 2.8 / 17,
             "K": 0.3 / 17, "DST": 0.0, "IDP": 2.0 / 17}

# Replacement level: the Nth best free agent that week, minus a cost for the
# roster move (claims compete; K/DST pools are deep and churned freely).
REPLACEMENT_RANK = {"K": 2, "DST": 2}
_DEFAULT_REPL_RANK = 3
STREAM_COST = {"K": 0.05, "DST": 0.05, "QB": 0.15}
_DEFAULT_STREAM_COST = 0.25


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
    scoring_items: Dict[int, float] = field(default_factory=dict)
    sources: Tuple[str, ...] = SOURCES
    source_coverage: Dict[str, int] = field(default_factory=dict)
    source_errors: Dict[str, str] = field(default_factory=dict)
    replacement: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def all_rows(self) -> List[dict]:
        return [r for rows in self.rosters.values() for r in rows] + list(self.free_agents)

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
    raw_sources: Optional[dict] = None,
    sources: Iterable[str] = SOURCES,
) -> Snapshot:
    """Pure parse of raw payloads -> Snapshot (no network). `raw_sources` is
    sources.fetch_all() output; without it only ESPN is used."""
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
    snap.scoring_items = {int(i["statId"]): float(i.get("points", 0) or 0)
                          for i in (settings.get("scoringSettings", {}) or {}).get("scoringItems", []) or []
                          if "statId" in i}
    now_ms = now_ms if now_ms is not None else time.time() * 1000
    for r in snap.all_rows():
        _game_state(r, snap, now_ms)
        snap.names[r["player_id"]] = r["name"]
    apply_model(snap, sources, raw_sources or {})
    return snap


def build_snapshot(client, fa_limit: int = 200, sources: Iterable[str] = SOURCES) -> Snapshot:
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
    final = int((league.get("status") or {}).get("finalScoringPeriod") or 17)
    raw_sources = fetch_all(client.year, range(week, final + 1), sources)
    return snapshot_from_raw(league, cards, fas, pro, txns, client.year, my_tid,
                             raw_sources=raw_sources, sources=sources)


# --------------------------------------------------------------------------
# The projection model
# --------------------------------------------------------------------------
def _game_state(row: dict, snap: Snapshot, now_ms: float) -> None:
    kickoff = snap.games.get(row["pro_team"], {}).get(snap.week)
    row["locked"] = bool(kickoff and kickoff <= now_ms)
    row["game_over"] = bool(kickoff and kickoff + _GAME_LENGTH_MS <= now_ms)


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sleeper_scale(rows: List[dict], snap: Snapshot) -> Dict[str, float]:
    """Per-position factor mapping Sleeper's points onto ESPN's league-scored
    points (catches scoring rules we couldn't map stat-by-stat)."""
    out = {}
    for pos in {r["pos"] for r in rows}:
        pairs = [(r["week_proj"], r["src"]["sleeper"].get(snap.week)) for r in rows
                 if r["pos"] == pos and r.get("week_proj") and r["src"]["sleeper"].get(snap.week)]
        pairs = sorted(pairs, key=lambda p: -p[0])[:40]
        if len(pairs) < 5:
            out[pos] = 1.0
            continue
        e = sorted(p[0] for p in pairs)[len(pairs) // 2]
        sl = sorted(p[1] for p in pairs)[len(pairs) // 2]
        out[pos] = min(max(e / sl, 0.7), 1.4) if sl > 0 else 1.0
    return out


def _play_prob(row: dict, week: int, snap: Snapshot) -> float:
    if not snap.has_game(row["pro_team"], week):
        return 0.0
    inj = row.get("injury", "ACTIVE")
    if week < snap.week + _OUT_WEEKS.get(inj, 0):
        return 0.0
    if week == snap.week:
        return PLAY_PROB_WEEK.get(inj, 1.0)
    ahead = week - snap.week
    return 1.0 - MISS_RATE.get(row["pos"], 0.1) * min(1.0, ahead / 4.0)


def _starter_demand(snap: Snapshot, pos: str) -> int:
    cfg = snap.config
    n = cfg.starters.get(pos, 0) + sum(c for c, e in cfg.flex_slots if pos in e) * 0.5
    return max(1, int(round(cfg.team_count * n)))


def apply_model(snap: Snapshot, sources: Optional[Iterable[str]] = None,
                raw_sources: Optional[dict] = None) -> None:
    """(Re)compute every player's projection from the enabled sources. Cheap to
    call again when the user toggles sources (raw numbers stay on the rows)."""
    if sources is not None:
        snap.sources = tuple(s for s in SOURCES if s in set(sources)) or ("espn",)
    rows = snap.all_rows()
    if raw_sources is not None:
        snap.source_coverage = attach(rows, raw_sources, snap.scoring_items, snap.config.ppr)
        snap.source_errors = dict(raw_sources.get("errors", {}))
    for r in rows:
        r.setdefault("src", {"sleeper": {}, "fp_week": None, "fp_ros": None})
    use = set(snap.sources)
    wk, weeks = snap.week, snap.remaining_weeks
    scale = _sleeper_scale(rows, snap)

    def sl(r, w):
        v = r["src"]["sleeper"].get(w)
        return None if v is None or "sleeper" not in use else v * scale.get(r["pos"], 1.0)

    def espn_ros(r):
        if "espn" not in use:
            return None
        if r.get("season_proj") and r["season_proj"] > 0:
            return r["season_proj"] / 17.0
        return r["week_proj"] if r.get("week_proj") else None

    def sleeper_ros(r):
        vals = [sl(r, w) for w in weeks if w > wk and snap.has_game(r["pro_team"], w)]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if len(vals) >= 2 else None

    # 1) per-source numbers for this week and for the rest of the season
    for r in rows:
        r["est"] = {"ESPN": r.get("week_proj") if "espn" in use else None,
                    "Sleeper": sl(r, wk)}
        r["est_ros"] = {"ESPN": espn_ros(r), "Sleeper": sleeper_ros(r)}
    if "fantasypros" in use:
        fpw = rank_to_points(rows, "fp_week", lambda r: _mean(r["est"].values()))
        fpr = rank_to_points(rows, "fp_ros", lambda r: _mean(r["est_ros"].values()))
    else:
        fpw, fpr = {}, {}
    for r in rows:
        r["est"]["FantasyPros"] = fpw.get(r["player_id"])
        r["est_ros"]["FantasyPros"] = fpr.get(r["player_id"])

    # 2) blend (equal weights) + this season's production
    for r in rows:
        src_rate = _mean(r["est_ros"].values())
        g, ppg = r.get("games_played", 0), r.get("ppg")
        if src_rate is None:
            rate = ppg or 0.0
        elif g and ppg is not None:
            w = g / (g + PRIOR_GAMES.get(r["pos"], 6))
            rate = (1 - w) * src_rate + w * ppg
        else:
            rate = src_rate
        rate = max(float(rate), 0.0)
        sl_avg = r["est_ros"]["Sleeper"]
        mu = {}
        for w in weeks:
            if w == wk:
                if r.get("game_over") and r.get("week_actual") is not None:
                    mu[w] = r["week_actual"]
                else:
                    this = _mean(r["est"].values())
                    mu[w] = this if this is not None else rate
            else:
                shape = 1.0                      # Sleeper's matchup/schedule shape
                v = sl(r, w)
                if v is not None and sl_avg:
                    shape = min(max(v / sl_avg, 0.6), 1.5)
                mu[w] = rate * shape
        r["rate"], r["mu"] = rate, mu

    # 3) shrink K / D/ST toward the position average
    for pos, rel_ros in RELIABILITY_ROS.items():
        prs = [r for r in rows if r["pos"] == pos]
        if not prs:
            continue
        n = _starter_demand(snap, pos)
        rel_wk = RELIABILITY_WEEK[pos]
        bases = []
        for w in weeks:
            top = sorted((r["mu"][w] for r in prs if snap.has_game(r["pro_team"], w)), reverse=True)[:n]
            if not top:
                continue
            base = sum(top) / len(top)
            bases.append(base)
            rel = rel_wk if w == wk else rel_ros
            for r in prs:
                if w == wk and r.get("game_over"):
                    continue
                r["mu"][w] = base + rel * (r["mu"][w] - base)
        if bases:
            b = sum(bases) / len(bases)
            for r in prs:
                r["rate"] = b + rel_ros * (r["rate"] - b)

    # 4) availability + weekly volatility
    for r in rows:
        r["play"] = {w: _play_prob(r, w, snap) for w in weeks}
        cv = WEEKLY_CV.get(r["pos"], 0.6)
        prior_sd = cv * max(r["rate"], 3.0)
        hist = list(r.get("weekly", {}).values())
        if len(hist) >= 2:
            m = sum(hist) / len(hist)
            emp_var = sum((x - m) ** 2 for x in hist) / (len(hist) - 1)
            n = len(hist)
            var = (n * emp_var + _SHRINK_GAMES * prior_sd ** 2) / (n + _SHRINK_GAMES)
            r["std_week"] = max(math.sqrt(var), 1.5)
        else:
            r["std_week"] = max(prior_sd, 1.5)

    # 5) replacement level from the real free-agent pool
    snap.replacement = {}
    for pos in {r["pos"] for r in rows}:
        fas = [r for r in snap.free_agents if r["pos"] == pos]
        k = REPLACEMENT_RANK.get(pos, _DEFAULT_REPL_RANK)
        cost = STREAM_COST.get(pos, _DEFAULT_STREAM_COST)
        per = {}
        for w in weeks:
            vals = sorted((r["play"][w] * r["mu"][w] for r in fas), reverse=True)
            per[w] = (vals[k - 1] if len(vals) >= k else (vals[-1] if vals else 0.0)) * (1 - cost)
        snap.replacement[pos] = per


def replacement(snap: Snapshot, pos: str, week: int) -> float:
    return snap.replacement.get(pos, {}).get(week, 0.0)


def play_prob(row: dict, week: int, snap: Snapshot) -> float:
    return row.get("play", {}).get(week, 0.0)


def week_mean(row: dict, week: int, snap: Snapshot) -> float:
    """Expected points the player himself scores in `week` (0 if he sits)."""
    if week == snap.week and row.get("game_over") and row.get("week_actual") is not None:
        return row["week_actual"]
    return play_prob(row, week, snap) * row.get("mu", {}).get(week, 0.0)


def slot_value(row: dict, week: int, snap: Snapshot) -> float:
    """Expected points from holding this player in a lineup slot: when he
    doesn't play, the slot is streamed at replacement level."""
    if week == snap.week and row.get("game_over") and row.get("week_actual") is not None:
        return row["week_actual"]
    p = play_prob(row, week, snap)
    return p * row.get("mu", {}).get(week, 0.0) + (1 - p) * replacement(snap, row["pos"], week)


def week_std(row: dict, week: int, snap: Snapshot) -> float:
    """Spread of the player's own score, including the chance he doesn't play."""
    if week == snap.week and row.get("game_over"):
        return 0.0
    p = play_prob(row, week, snap)
    mu = row.get("mu", {}).get(week, 0.0)
    if p <= 0 or mu <= 0:
        return 0.0
    s = row["std_week"] * (0.6 if (week == snap.week and row.get("locked")) else 1.0)
    return math.sqrt(p * s * s + p * (1 - p) * mu * mu)


def ros_points(row: dict, snap: Snapshot) -> float:
    """Raw expected points the player scores over the rest of the season
    (what managers 'see' -- not lineup-aware)."""
    return sum(week_mean(row, w, snap) for w in snap.remaining_weeks)


# --------------------------------------------------------------------------
# Roster value over the rest of the season
# --------------------------------------------------------------------------
def _phantom_slots(snap: Snapshot) -> Dict[str, int]:
    """How many lineup slots each position could fill (for replacement fill-ins)."""
    cfg = snap.config
    out = {}
    for pos in snap.replacement:
        n = cfg.starters.get(pos, 0) + sum(c for c, e in cfg.flex_slots if pos in e)
        if n:
            out[pos] = n
    return out


def roster_week_values(rows: List[dict], snap: Snapshot, weeks: Optional[List[int]] = None,
                       means: Optional[Dict[Tuple[str, int], float]] = None) -> Dict[int, float]:
    """Optimal starting-lineup points for each week, where every slot can also
    be streamed at replacement level. Weeks with the same inputs share one
    computation (byes repeat a lot)."""
    weeks = weeks if weeks is not None else snap.remaining_weeks
    cfg = snap.config
    phantoms = _phantom_slots(snap)
    out: Dict[int, float] = {}
    cache: Dict[tuple, float] = {}
    for w in weeks:
        vals, key = [], []
        for r in rows:
            m = means[(r["player_id"], w)] if means is not None else slot_value(r, w, snap)
            vals.append({"pos": r["pos"], "proj": m})
            key.append(round(m, 2))
        for pos, n in phantoms.items():
            rv = replacement(snap, pos, w)
            vals += [{"pos": pos, "proj": rv}] * n
            key.append(round(rv, 2))
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
