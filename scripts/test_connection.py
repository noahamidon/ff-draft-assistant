"""Step 1 of the workstream: prove auth works and capture your real league.

    python scripts/test_connection.py

Reads LEAGUE_ID / SEASON / SWID / ESPN_S2 from your .env, connects to ESPN,
prints a parsed summary of your league rules, and writes the raw settings JSON
to data/raw_settings.json so we can confirm the "weird" rules parse correctly
before building the model on top of them.
"""

import json
import os
import sys

# allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from draftkit.config import LeagueConfig
from draftkit.espn_client import ESPNClient

load_dotenv()


def main() -> int:
    league_id = os.environ.get("LEAGUE_ID")
    season = os.environ.get("SEASON", "2026")
    swid = os.environ.get("SWID")
    espn_s2 = os.environ.get("ESPN_S2")

    if not league_id:
        print("ERROR: set LEAGUE_ID in your .env (copy .env.example to .env).")
        return 1
    if not (swid and espn_s2):
        print("WARNING: SWID / ESPN_S2 not set. Private leagues will 401.\n")

    client = ESPNClient(int(league_id), int(season), swid=swid, espn_s2=espn_s2)

    print(f"Connecting to league {league_id}, season {season} ...")
    try:
        raw = client.raw_settings()
    except PermissionError as exc:
        print(f"\nAUTH FAILED:\n{exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\nRequest failed: {exc}")
        print("If this is a host/endpoint error, tell me the message and I'll adjust the client.")
        return 3

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "raw_settings.json")
    with open(out_path, "w") as fh:
        json.dump(raw, fh, indent=2)
    print(f"Raw settings written to {out_path}\n")

    try:
        cfg = LeagueConfig.from_espn_settings(raw)
        print("Parsed league configuration")
        print("-" * 40)
        print(cfg.summary())
        print("-" * 40)
        print("\nIf anything above looks wrong for your league, send me "
              "data/raw_settings.json and I'll calibrate the parser.")
    except Exception as exc:  # noqa: BLE001
        print(f"Parsed settings but could not build LeagueConfig cleanly: {exc}")
        print("Send me data/raw_settings.json and I'll fix the parser.")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
