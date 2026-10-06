"""DraftState -- who has been picked, by whom, and whose turn it is.

Teams are 1..team_count. Draft position (your seat) is 1..team_count. Snake
order reverses every round. This object is the source of truth the UI mutates
as picks come in, and that the simulator reads to project the rest of the draft.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import LeagueConfig


@dataclass
class Pick:
    overall: int
    round: int
    team: int
    player_id: str
    name: str
    pos: str
    keeper: bool = False      # auto-filled from a keeper reservation


@dataclass
class DraftState:
    config: LeagueConfig
    my_team: int = 1                     # your draft seat, 1-indexed
    picks: List[Pick] = field(default_factory=list)
    drafted_ids: set = field(default_factory=set)
    # keepers not yet reached in the pick order. They are OFF THE BOARD from the
    # start (unavailable, and on their team's roster), and each is converted to a
    # normal pick when the snake reaches its slot (see auto_fill_keepers).
    reserved_picks: List[Pick] = field(default_factory=list)
    keeper_total: int = 0

    @property
    def team_count(self) -> int:
        return self.config.team_count

    @property
    def total_picks(self) -> int:
        return self.team_count * self.config.roster_size

    @property
    def linear_rounds(self) -> int:
        """Leading non-snaking rounds (the keeper rounds). ESPN runs these in
        set order; the snake begins the round after."""
        return int(getattr(self.config, "keeper_rounds", 0) or 0)

    # -- snake order --------------------------------------------------------
    def team_on_clock(self, overall: int) -> int:
        """1-indexed seat for a 1-indexed overall pick. Keeper rounds run in set
        (forward) order; snaking starts forward the round after them."""
        rnd = (overall - 1) // self.team_count          # 0-indexed round
        idx = (overall - 1) % self.team_count           # 0-indexed slot
        if rnd < self.linear_rounds:
            return idx + 1                               # linear keeper round
        if (rnd - self.linear_rounds) % 2 == 0:
            return idx + 1                               # forward
        return self.team_count - idx                     # reverse

    def overall_for(self, seat: int, rnd: int) -> int:
        """Overall pick number for a seat's round (both 1-indexed)."""
        rr = rnd - 1
        base = rr * self.team_count
        if rr < self.linear_rounds or (rr - self.linear_rounds) % 2 == 0:
            return base + seat                           # forward
        return base + (self.team_count - seat + 1)       # reverse

    @property
    def next_overall(self) -> int:
        return len(self.picks) + 1

    @property
    def current_team(self) -> int:
        return self.team_on_clock(self.next_overall)

    def is_my_turn(self) -> bool:
        return self.current_team == self.my_team and self.next_overall <= self.total_picks

    @property
    def reserved_overalls(self) -> set:
        """Overall slots already spoken for by a keeper (no live pick there)."""
        return {p.overall for p in self.reserved_picks}

    def my_pick_numbers(self) -> List[int]:
        res = self.reserved_overalls
        return [o for o in range(1, self.total_picks + 1)
                if self.team_on_clock(o) == self.my_team and o not in res]

    def _my_live_picks_from_now(self):
        res = self.reserved_overalls
        for o in range(self.next_overall, self.total_picks + 1):
            if self.team_on_clock(o) == self.my_team and o not in res:
                yield o

    def my_next_pick(self) -> Optional[int]:
        return next(self._my_live_picks_from_now(), None)

    def my_pick_after_next(self) -> Optional[int]:
        it = self._my_live_picks_from_now()
        next(it, None)
        return next(it, None)

    def picks_until_my_turn(self) -> int:
        nxt = self.my_next_pick()
        return 0 if nxt is None else nxt - self.next_overall

    # -- keepers ------------------------------------------------------------
    def set_keepers(self, keeper_slots: Dict[int, List[dict]]) -> None:
        """Seed keepers. keeper_slots maps a draft seat -> list of player dicts
        (index 0 = the round-1 keeper, index 1 = round-2, ...). Each keeper is
        reserved off the board and placed at that seat's pick in that round.
        """
        self.reserved_picks = []
        self.keeper_total = sum(len(v) for v in keeper_slots.values())
        for seat, plist in keeper_slots.items():
            for i, p in enumerate(plist):
                rnd = i + 1
                overall = self.overall_for(int(seat), rnd)
                self.reserved_picks.append(Pick(
                    overall=overall, round=rnd, team=int(seat),
                    player_id=str(p["player_id"]), name=p["name"], pos=p["pos"],
                ))

    @property
    def reserved_ids(self) -> set:
        return {p.player_id for p in self.reserved_picks}

    def auto_fill_keepers(self) -> None:
        """Convert any reserved keepers at the current front of the draft into
        real picks, so keeper slots are consumed automatically as reached."""
        moved = True
        while moved:
            moved = False
            o = self.next_overall
            if o > self.total_picks:
                break
            for i, pk in enumerate(self.reserved_picks):
                if pk.overall == o:
                    pk.keeper = True
                    self.picks.append(pk)
                    self.drafted_ids.add(pk.player_id)
                    self.reserved_picks.pop(i)
                    moved = True
                    break

    def next_is_keeper(self) -> bool:
        return any(pk.overall == self.next_overall for pk in self.reserved_picks)

    # -- mutate -------------------------------------------------------------
    def record_pick(self, player_row: dict, team: Optional[int] = None) -> Pick:
        overall = self.next_overall
        rnd = (overall - 1) // self.team_count + 1
        team = team if team is not None else self.team_on_clock(overall)
        pk = Pick(
            overall=overall,
            round=rnd,
            team=team,
            player_id=str(player_row["player_id"]),
            name=player_row["name"],
            pos=player_row["pos"],
        )
        self.picks.append(pk)
        self.drafted_ids.add(str(player_row["player_id"]))
        self.auto_fill_keepers()
        return pk

    def undo(self) -> Optional[Pick]:
        """Undo the last LIVE pick. Keeper picks auto-filled after it go back
        into reservation (still off the board) instead of returning to the pool."""
        def _restore_trailing_keepers():
            while self.picks and self.picks[-1].keeper:
                kp = self.picks.pop()
                self.drafted_ids.discard(kp.player_id)
                self.reserved_picks.append(kp)

        _restore_trailing_keepers()
        if not self.picks:
            return None
        pk = self.picks.pop()
        self.drafted_ids.discard(pk.player_id)
        _restore_trailing_keepers()
        return pk

    # -- rosters ------------------------------------------------------------
    def rosters(self) -> Dict[int, List[Pick]]:
        out: Dict[int, List[Pick]] = defaultdict(list)
        for p in self.picks:
            out[p.team].append(p)
        for p in self.reserved_picks:          # keepers not yet reached
            out[p.team].append(p)
        return out

    def roster_of(self, team: int) -> List[Pick]:
        return [p for p in self.picks + self.reserved_picks if p.team == team]

    def my_roster(self) -> List[Pick]:
        return self.roster_of(self.my_team)

    def available(self, players):
        """Undrafted rows, excluding kept players still reserved off the board."""
        gone = self.drafted_ids | self.reserved_ids
        return players[~players["player_id"].astype(str).isin(gone)].copy()
