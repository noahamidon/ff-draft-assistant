"""Dump the raw in-season ESPN payloads and a parsed summary.

    python scripts/dump_season.py ["Saved league name"]

Writes data/raw_season/*.json (gitignored) and prints what the in-season tools
see: current week, your team, your lineup with projections, and the free-agent
count. If projections show as None or rosters look wrong, these files show
exactly what ESPN returned.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from draftkit.espn_client import ESPNClient
from draftkit.profiles import cli_credentials
from draftkit.season import snapshot_from_raw, week_mean

load_dotenv()
OUT = os.path.join("data", "raw_season")


def main() -> int:
    creds = cli_credentials(sys.argv[1] if len(sys.argv) > 1 else None)
    if not creds["LEAGUE_ID"]:
        print("No league found. Save one in the app's Settings tab, or set LEAGUE_ID in .env.")
        return 1
    client = ESPNClient(int(creds["LEAGUE_ID"]), int(creds["SEASON"]),
                        swid=creds["SWID"] or None, espn_s2=creds["ESPN_S2"] or None)
    os.makedirs(OUT, exist_ok=True)

    def dump(name, fn):
        try:
            data = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: FAILED ({exc})")
            return {}
        with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"  {name}: ok")
        return data

    print(f"Pulling league {creds['LEAGUE_ID']} ...")
    league = dump("league", client.raw_league)
    if not league:
        return 2
    week = int(league.get("scoringPeriodId") or 1)
    ids = [e.get("playerId") for t in league.get("teams", [])
           for e in (t.get("roster", {}) or {}).get("entries", []) or []]
    cards = dump("player_cards", lambda: client.raw_players_by_id(ids, week))
    fas = dump("free_agents", lambda: client.raw_free_agents(week))
    pro = dump("pro_schedule", client.raw_pro_schedule)
    txns = [dump(f"transactions_wk{week}", lambda: client.raw_transactions(week))]

    snap = snapshot_from_raw(league, cards, fas, pro, txns, client.year, client.my_team_id(league))
    print(f"\nWeek {snap.week}/{snap.final_week} · your team: "
          f"{snap.team_name(snap.my_team_id) if snap.my_team_id else 'not detected'}")
    for r in snap.rosters.get(snap.my_team_id or -1, []):
        print(f"  {r['name']:<24} {r['pos']:<4} slot {str(r['slot']):>3}  "
              f"wk proj {'—' if r['week_proj'] is None else round(r['week_proj'], 1):>5}  -> {week_mean(r, snap.week, snap):5.1f}  "
              f"ros/g {r['rate']:5.1f}  sd {r['std_week']:4.1f}  games {r['games_played']}")
    print(f"Free agents parsed: {len(snap.free_agents)} · transactions: {len(snap.transactions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
