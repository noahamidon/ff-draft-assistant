"""Talk to ESPN.

Two jobs:
  1. Pull the STATIC config (league settings) -- the reliable, valuable half.
  2. Pull the draft board when available (post-draft recap; during a live
     draft ESPN's standard API is unreliable, so the app's primary live-input
     path is manual entry -- this is a best-effort convenience).

Auth for a private league uses two browser cookies, SWID and espn_s2, read
from environment variables (never hard-code them, never commit them).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests

# Current ESPN read host for the v3 fantasy API.
_READS_HOST = "https://lm-api-reads.fantasy.espn.com"
_V3 = "/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{league_id}"

# ESPN proTeamId -> abbreviation (0 = free agent / none).
_PRO_TEAM = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL",
    7: "DEN", 8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV",
    14: "LAR", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB",
    28: "WSH", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}


def _season_projection(player: dict, season: int):
    """Projected season fantasy points (league-scored) from a player's stats."""
    stats = player.get("stats", []) or []
    # preferred: projection (statSourceId 1), full-season split (0)
    for s in stats:
        if (s.get("statSourceId") == 1
                and s.get("seasonId") == season
                and s.get("statSplitTypeId", 0) == 0
                and s.get("appliedTotal") is not None):
            return s["appliedTotal"]
    # fallback: any projection at scoringPeriodId 0
    for s in stats:
        if (s.get("statSourceId") == 1
                and s.get("scoringPeriodId") == 0
                and s.get("appliedTotal") is not None):
            return s["appliedTotal"]
    return None


class ESPNClient:
    def __init__(
        self,
        league_id: int,
        year: int,
        swid: Optional[str] = None,
        espn_s2: Optional[str] = None,
        timeout: int = 20,
    ):
        self.league_id = int(league_id)
        self.year = int(year)
        self.swid = swid or os.environ.get("SWID")
        self.espn_s2 = espn_s2 or os.environ.get("ESPN_S2")
        self.timeout = timeout

    # -- low level ----------------------------------------------------------
    @property
    def _cookies(self) -> dict:
        c = {}
        if self.swid:
            c["SWID"] = self.swid
        if self.espn_s2:
            c["espn_s2"] = self.espn_s2
        return c

    def _get(self, views: List[str]) -> dict:
        url = _READS_HOST + _V3.format(year=self.year, league_id=self.league_id)
        params = [("view", v) for v in views]
        resp = requests.get(
            url,
            params=params,
            cookies=self._cookies,
            headers={"User-Agent": "Mozilla/5.0 (draftkit)"},
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise PermissionError(
                "ESPN returned 401 Unauthorized. For a private league this "
                "almost always means SWID / espn_s2 are missing, expired, or "
                "wrong. Re-copy them from your browser (see README) and check "
                "your .env."
            )
        resp.raise_for_status()
        return resp.json()

    # -- public -------------------------------------------------------------
    def raw_settings(self) -> dict:
        """Full league object including settings (for parsing + inspection)."""
        return self._get(["mSettings"])

    def raw_draft(self) -> dict:
        """Draft detail. Populated after picks are made."""
        return self._get(["mDraftDetail"])

    def raw_players(self, limit: int = 400) -> dict:
        """Player pool with rankings / ownership. Uses the players endpoint."""
        url = (
            _READS_HOST
            + f"/apis/v3/games/ffl/seasons/{self.year}/players?view=players_wl"
        )
        resp = requests.get(
            url,
            params={"scoringPeriodId": 0, "view": "kona_player_info"},
            cookies=self._cookies,
            headers={
                "User-Agent": "Mozilla/5.0 (draftkit)",
                "x-fantasy-filter": (
                    '{"players":{"limit":%d,'
                    '"sortDraftRanks":{"sortPriority":1,"sortAsc":true,'
                    '"value":"STANDARD"}}}' % int(limit)
                ),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def drafted_player_ids(self) -> List[int]:
        """ESPN player ids already selected, best-effort, from draft detail."""
        try:
            data = self.raw_draft()
        except Exception:
            return []
        picks = (
            data.get("draftDetail", {}).get("picks", [])
            if isinstance(data, dict)
            else []
        )
        return [int(p["playerId"]) for p in picks if p.get("playerId")]

    # -- rosters + draft order ---------------------------------------------
    def raw_rosters(self) -> dict:
        """All teams with their current rosters (view=mRoster + mTeam)."""
        return self._get(["mRoster", "mTeam"])

    def teams(self) -> Dict[int, str]:
        """{team_id: display name} for all teams."""
        data = self.raw_rosters()
        out: Dict[int, str] = {}
        for team in data.get("teams", []):
            tid = int(team.get("id"))
            name = (team.get("name") or "").strip()
            if not name:
                loc = (team.get("location") or "").strip()
                nick = (team.get("nickname") or "").strip()
                name = (loc + " " + nick).strip()
            if not name:
                name = team.get("abbrev") or f"Team {tid}"
            out[tid] = name
        return out

    def rosters(self) -> Dict[int, List[dict]]:
        """{team_id: [ {player_id, name, pos, pro_team}, ... ]} for all teams.

        Position is derived from ESPN's defaultPositionId. This is what feeds
        the keeper board: auto-pull every roster, then mark keepers by hand.
        """
        _POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
        _IDP_SLOTS = {8, 9, 10, 11, 12, 13, 14, 15}
        data = self.raw_rosters()
        out: Dict[int, List[dict]] = {}
        for team in data.get("teams", []):
            tid = int(team.get("id"))
            players = []
            entries = (team.get("roster", {}) or {}).get("entries", []) or []
            for e in entries:
                info = (e.get("playerPoolEntry", {}) or {}).get("player", {}) or {}
                pid = info.get("id", e.get("playerId"))
                elig = set(info.get("eligibleSlots", []) or [])
                if elig & _IDP_SLOTS:
                    pos = "IDP"
                else:
                    pos = _POS.get(info.get("defaultPositionId"), str(info.get("defaultPositionId")))
                players.append({
                    "player_id": str(pid),
                    "name": info.get("fullName", "Unknown"),
                    "pos": pos,
                    "pro_team": info.get("proTeamId"),
                })
            out[tid] = players
        return out

    def draft_order(self) -> List[int]:
        """Team ids in draft-slot order (index 0 == slot 1)."""
        data = self.raw_settings()
        return list(
            data.get("settings", {}).get("draftSettings", {}).get("pickOrder", [])
        )

    def my_slot(self, team_id: int) -> Optional[int]:
        """1-indexed draft slot for a team id, or None if not found."""
        order = self.draft_order()
        return order.index(int(team_id)) + 1 if int(team_id) in order else None

    # -- projections + ADP (free, league-scored) ---------------------------
    def projections(self, limit: int = 400, season: Optional[int] = None):
        """Season projections + ADP for the player pool, as a model-ready
        DataFrame [player_id, name, pos, team, proj, adp].

        `proj` is ESPN's projected season fantasy points computed in YOUR
        league's scoring (appliedTotal of the projection stat split), so your
        6-pt passing TDs and PPR are already reflected. `adp` is ESPN's average
        draft position.
        """
        import pandas as pd

        season = int(season or self.year)
        url = _READS_HOST + _V3.format(year=season, league_id=self.league_id)
        xff = {
            "players": {
                "limit": int(limit),
                "offset": 0,
                "sortDraftRanks": {
                    "sortPriority": 100, "sortAsc": True, "value": "PPR"
                },
            }
        }
        import json as _json
        resp = requests.get(
            url,
            params={"view": "kona_player_info"},
            cookies=self._cookies,
            headers={
                "User-Agent": "Mozilla/5.0 (draftkit)",
                "X-Fantasy-Filter": _json.dumps(xff),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("players", []) if isinstance(data, dict) else []

        _POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
        _IDP_SLOTS = {8, 9, 10, 11, 12, 13, 14, 15}
        rows = []
        for e in entries:
            p = e.get("player", e) or {}
            elig = set(p.get("eligibleSlots", []) or [])
            if elig & _IDP_SLOTS:            # any individual-defense eligibility
                pos = "IDP"
            else:
                pos = _POS.get(p.get("defaultPositionId"))
            if pos is None:
                continue
            proj = _season_projection(p, season)
            if proj is None:
                continue
            adp = (p.get("ownership") or {}).get("averageDraftPosition")
            rows.append({
                "player_id": str(p.get("id")),
                "name": p.get("fullName", "Unknown"),
                "pos": pos,
                "team": _PRO_TEAM.get(p.get("proTeamId"), str(p.get("proTeamId"))),
                "proj": round(float(proj), 1),
                "adp": float(adp) if adp and adp > 0 else None,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # fill missing ADP with projection rank so the sim still runs
        df["adp"] = df["adp"].fillna(df["proj"].rank(ascending=False, method="first"))
        return df.sort_values("proj", ascending=False).reset_index(drop=True)
