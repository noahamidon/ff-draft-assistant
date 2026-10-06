"""Saved league logins -- several ESPN leagues, one active at a time.

Stored as JSON in config/leagues.local.json, which is gitignored and written
with owner-only permissions (0600). The SWID / espn_s2 cookies are effectively
your ESPN login, so this file must never be committed.

    {
      "active": "Work league",
      "leagues": {
        "Work league": {"league_id": "123", "season": "2026",
                        "swid": "{...}", "espn_s2": "...", "team_id": 4},
        ...
      }
    }

If the file doesn't exist yet but a .env has LEAGUE_ID/SWID/ESPN_S2, that is
offered as a single profile so the old setup keeps working.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

PROFILE_PATH = os.path.join("config", "leagues.local.json")
FIELDS = ("league_id", "season", "swid", "espn_s2", "team_id")


def _empty() -> dict:
    return {"active": None, "leagues": {}}


def load(path: Optional[str] = None) -> dict:
    path = path or PROFILE_PATH
    if not os.path.exists(path):
        return _from_env()
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return _empty()
    data.setdefault("leagues", {})
    data.setdefault("active", None)
    if data["active"] not in data["leagues"]:
        data["active"] = next(iter(data["leagues"]), None)
    return data


def _from_env() -> dict:
    lid = os.environ.get("LEAGUE_ID")
    if not lid:
        return _empty()
    name = f"League {lid}"
    return {"active": name, "leagues": {name: {
        "league_id": lid,
        "season": os.environ.get("SEASON", "2026"),
        "swid": os.environ.get("SWID", ""),
        "espn_s2": os.environ.get("ESPN_S2", ""),
        "team_id": None,
    }}}


def save(data: dict, path: Optional[str] = None) -> None:
    path = path or PROFILE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # create with 0600 so the cookies are never world-readable, even briefly
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def upsert(name: str, profile: dict, make_active: bool = True, path: Optional[str] = None) -> dict:
    path = path or PROFILE_PATH
    data = load(path)
    data["leagues"][name] = {k: profile.get(k) for k in FIELDS}
    if make_active:
        data["active"] = name
    save(data, path)
    return data


def set_active(name: str, path: Optional[str] = None) -> dict:
    path = path or PROFILE_PATH
    data = load(path)
    if name in data["leagues"]:
        data["active"] = name
        save(data, path)
    return data


def remove(name: str, path: Optional[str] = None) -> dict:
    path = path or PROFILE_PATH
    data = load(path)
    data["leagues"].pop(name, None)
    if data["active"] == name:
        data["active"] = next(iter(data["leagues"]), None)
    save(data, path)
    return data


def active_profile(path: Optional[str] = None) -> Optional[Dict]:
    path = path or PROFILE_PATH
    data = load(path)
    name = data.get("active")
    return dict(data["leagues"][name], name=name) if name else None


def cli_credentials(name: Optional[str] = None) -> Dict[str, str]:
    """Credentials for the CLI scripts: a named saved league, else the active
    saved league, else .env / environment variables."""
    data = load()
    prof = data["leagues"].get(name) if name else None
    if prof is None and not name and not os.environ.get("LEAGUE_ID"):
        prof = data["leagues"].get(data.get("active"))
    if prof:
        return {"LEAGUE_ID": str(prof.get("league_id") or ""),
                "SEASON": str(prof.get("season") or "2026"),
                "SWID": prof.get("swid") or "", "ESPN_S2": prof.get("espn_s2") or ""}
    return {k: os.environ.get(k, d) for k, d in
            (("LEAGUE_ID", ""), ("SEASON", "2026"), ("SWID", ""), ("ESPN_S2", ""))}
