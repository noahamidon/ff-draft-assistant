"""LeagueConfig -- your exact league rules, which drive every valuation.

This is the "ROI based on league rules" layer. Scoring format and roster
requirements set the replacement baselines that make a point of production
worth more or less. Get this right and half the model is done.

There are two ways to build a LeagueConfig:
    1. LeagueConfig.from_espn_settings(raw)  -- parse ESPN's settings JSON
    2. LeagueConfig.from_yaml(path)          -- a hand-written override file

Step 1 of the workstream is to pull the raw ESPN settings and confirm the
parse matches your "weird" league. Until then, config/league.example.yaml is
a safe fallback so the app runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# ESPN lineup-slot id -> meaning. These ids are stable across ESPN's API.
# We only care about offense + K + D/ST for standard fantasy.
# ---------------------------------------------------------------------------
ESPN_SLOT_LABELS: Dict[int, str] = {
    0: "QB",
    1: "TQB",       # team QB (rare)
    2: "RB",
    3: "RB/WR",     # flex (2 positions)
    4: "WR",
    5: "WR/TE",     # flex (2 positions)
    6: "TE",
    7: "OP",        # offensive player utility == superflex (QB eligible)
    16: "DST",
    17: "K",
    20: "BE",       # bench
    21: "IR",
    23: "FLEX",     # RB/WR/TE
}

# Which real positions each flex-type slot accepts.
ESPN_FLEX_ELIGIBILITY: Dict[int, Set[str]] = {
    3: {"RB", "WR"},
    5: {"WR", "TE"},
    7: {"QB", "RB", "WR", "TE"},   # superflex
    23: {"RB", "WR", "TE"},
}

# Dedicated (single-position) starting slots.
ESPN_DEDICATED_SLOT: Dict[int, str] = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DST", 17: "K",
}

# Individual-defensive-player slots. Slot 15 (DP) is a generic IDP utility;
# 8-14 are specific defensive positions. A league using these instead of slot
# 16 (D/ST) starts real defenders, which needs IDP projections.
ESPN_IDP_SLOTS = {8, 9, 10, 11, 12, 13, 14, 15}

# A few scoring statIds worth surfacing. ESPN stores scoring as a list of
# {statId, points}. These are best-effort labels -- CONFIRM against your raw
# settings dump, since ESPN occasionally reuses ids across contexts.
SCORING_STAT_LABELS: Dict[int, str] = {
    3: "pass_yds",
    4: "pass_td",
    20: "interceptions",
    24: "rush_yds",
    25: "rush_td",
    42: "rec_yds",
    43: "rec_td",
    53: "receptions",   # the PPR knob
}

POSITIONS: Tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST", "IDP")


@dataclass
class LeagueConfig:
    name: str = "My League"
    team_count: int = 12
    # dedicated starting slots per real position
    starters: Dict[str, int] = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    )
    # each flex slot: (count, eligible-positions)
    flex_slots: List[Tuple[int, Set[str]]] = field(
        default_factory=lambda: [(1, {"RB", "WR", "TE"})]
    )
    bench: int = 6
    # scoring points keyed by our stat labels (best-effort from ESPN)
    scoring: Dict[str, float] = field(default_factory=dict)
    superflex: bool = False

    # -- derived ------------------------------------------------------------
    @property
    def positions(self) -> Tuple[str, ...]:
        return POSITIONS

    @property
    def idp_starters(self) -> int:
        return self.starters.get("IDP", 0)

    @property
    def flex_count(self) -> int:
        return sum(c for c, _ in self.flex_slots)

    @property
    def roster_size(self) -> int:
        return sum(self.starters.values()) + self.flex_count + self.bench

    @property
    def ppr(self) -> float:
        """Points per reception (0, 0.5, or 1 typically)."""
        return float(self.scoring.get("receptions", 0.0))

    def summary(self) -> str:
        starters = ", ".join(f"{v}{k}" for k, v in self.starters.items() if v)
        flex = ", ".join(
            f"{c}x FLEX({'/'.join(sorted(e))})" for c, e in self.flex_slots
        )
        pass_td = self.scoring.get("pass_td")
        lines = [
            f"League: {self.name}",
            f"Teams: {self.team_count}",
            f"Starters: {starters}" + (f"; {flex}" if flex else ""),
            f"Bench: {self.bench}  ->  roster size {self.roster_size}",
            f"PPR: {self.ppr}   Superflex: {self.superflex}"
            + (f"   Pass TD: {pass_td}" if pass_td else ""),
        ]
        return "\n".join(lines)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "LeagueConfig":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
        flex = [
            (int(f["count"]), set(f["eligible"]))
            for f in raw.get("flex_slots", [])
        ]
        return cls(
            name=raw.get("name", "My League"),
            team_count=int(raw["team_count"]),
            starters={k: int(v) for k, v in raw["starters"].items()},
            flex_slots=flex or [(0, set())],
            bench=int(raw.get("bench", 6)),
            scoring={k: float(v) for k, v in raw.get("scoring", {}).items()},
            superflex=bool(raw.get("superflex", False)),
        )

    @classmethod
    def from_espn_settings(cls, raw: dict) -> "LeagueConfig":
        """Parse the JSON returned by the ?view=mSettings ESPN endpoint.

        `raw` is the top-level league object (has a "settings" key).
        """
        settings = raw.get("settings", raw)
        name = settings.get("name", "My League")

        # team count
        team_count = int(
            settings.get("size")
            or raw.get("status", {}).get("teamsJoined")
            or 12
        )

        roster = settings.get("rosterSettings", {})
        slot_counts = roster.get("lineupSlotCounts", {}) or {}

        starters: Dict[str, int] = {p: 0 for p in POSITIONS}
        flex_slots: List[Tuple[int, Set[str]]] = []
        bench = 0
        superflex = False
        idp_starters = 0

        for slot_id_str, count in slot_counts.items():
            slot_id = int(slot_id_str)
            count = int(count)
            if count <= 0:
                continue
            if slot_id in (20, 21):        # bench / IR
                if slot_id == 20:
                    bench = count
                continue
            if slot_id in ESPN_IDP_SLOTS:  # individual defensive players
                idp_starters += count
                continue
            if slot_id in ESPN_DEDICATED_SLOT:
                starters[ESPN_DEDICATED_SLOT[slot_id]] += count
            elif slot_id in ESPN_FLEX_ELIGIBILITY:
                elig = ESPN_FLEX_ELIGIBILITY[slot_id]
                flex_slots.append((count, set(elig)))
                if slot_id == 7:           # OP == superflex
                    superflex = True

        starters["IDP"] = idp_starters

        # scoring: list of {statId, points}
        scoring: Dict[str, float] = {}
        for item in settings.get("scoringSettings", {}).get("scoringItems", []):
            sid = int(item.get("statId", -1))
            pts = float(item.get("points", 0.0))
            if sid in SCORING_STAT_LABELS:
                scoring[SCORING_STAT_LABELS[sid]] = pts

        if not flex_slots:
            flex_slots = [(0, set())]

        return cls(
            name=name,
            team_count=team_count,
            starters=starters,
            flex_slots=flex_slots,
            bench=bench or 6,
            scoring=scoring,
            superflex=superflex,
        )
