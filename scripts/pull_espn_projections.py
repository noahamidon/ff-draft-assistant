"""Pull ESPN's own projections + ADP (free, league-scored) into a CSV.

    python scripts/pull_espn_projections.py

Uses LEAGUE_ID / SEASON / SWID / ESPN_S2 from your .env. Writes
data/projections.csv, already scored to your league (6-pt pass TDs, PPR, etc.).
If the top players look wrong, tell me what you see and I'll adjust the pull.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from draftkit.espn_client import ESPNClient

load_dotenv()

OUT = os.path.join("data", "projections.csv")


def main() -> int:
    league_id = os.environ.get("LEAGUE_ID")
    season = os.environ.get("SEASON", "2026")
    if not league_id:
        print("Set LEAGUE_ID in your .env first.")
        return 1

    client = ESPNClient(
        int(league_id), int(season),
        swid=os.environ.get("SWID"), espn_s2=os.environ.get("ESPN_S2"),
    )
    print(f"Pulling ESPN projections for league {league_id}, season {season} ...")
    try:
        df = client.projections(limit=500)
    except Exception as exc:  # noqa: BLE001
        print(f"Pull failed: {exc}")
        print("Send me this message and I'll adjust the endpoint/filter.")
        return 2

    if df.empty:
        print("No players returned. Send me this so I can adjust the filter.")
        return 3

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} players -> {OUT}\n")
    print("Top 15 by projected points (league-scored):")
    print(df.head(15)[["name", "pos", "team", "proj", "adp"]].to_string(index=False))
    counts = ", ".join(f"{p}:{(df['pos'] == p).sum()}" for p in df["pos"].unique())
    print(f"\nPositions parsed: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
