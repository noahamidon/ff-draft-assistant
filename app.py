"""Fantasy draft assistant -- Streamlit front end.

Run from the repo root:
    streamlit run app.py

Primary live-input path is fast manual entry (one click per pick). Load your
league config (scraped from ESPN or the YAML fallback) and your projections in
the sidebar, set your draft seat, and go.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from draftkit import profiles
from draftkit.config import LeagueConfig
from draftkit.draft_state import DraftState
from draftkit.projections import load_projections
from draftkit.recommender import build_board, reason_for
from draftkit.simulation import simulate_candidates

load_dotenv()

st.set_page_config(page_title="Draft Assistant", layout="wide", page_icon="🏈")

DATA_DIR = "data"

# --------------------------------------------------------------------------
# Theme -- warm editorial palette (Oswald, cream paper, earthy accents)
# --------------------------------------------------------------------------
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@200;300;400;500;600;700&family=EB+Garamond:ital@0;1&display=swap');

:root {
    --paper:   #f7f5f2;
    --panel:   #edeae5;
    --panel-2: #f2ede6;
    --ink:     #2c2c2c;
    --muted:   #888;
    --line:    #d4cec4;
    --line-2:  #b5aea5;
    --maroon:  #943c3c;
    --green:   #3d6e4e;
    --gold:    #9e862f;
    --clay:    #b07348;
}

html, body, [class*="css"], .stApp, .block-container {
    font-family: 'Oswald', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}

.block-container { padding-top: 2.2rem; max-width: 1250px; }

/* headings */
h1, h2, h3, h4 { font-family: 'Oswald', sans-serif; letter-spacing: 1px; color: var(--ink); }
h2 { font-weight: 600; }
h3 { font-weight: 600; font-size: 1.25rem; }

/* hero */
.hero { padding: 8px 0 22px; border-bottom: 1px solid var(--line); margin-bottom: 22px; }
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 400; letter-spacing: 3px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}
.hero-title {
    font-size: 2.9rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; line-height: 1; margin: 0; color: var(--ink);
}
.hero-title .accent { color: var(--maroon); }
.hero-sub { font-size: 0.95rem; font-weight: 300; color: #555; margin-top: 10px; }

/* tabs -> elegant, letter-spaced, maroon active underline */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; border-bottom: 1px solid var(--line); background: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 46px; padding: 0 22px; background: transparent;
    font-family: 'Oswald', sans-serif; font-weight: 500; font-size: 0.82rem;
    letter-spacing: 2px; text-transform: uppercase; color: var(--muted);
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: var(--maroon) !important; border-bottom: 2px solid var(--maroon) !important; }

/* metrics -> warm cards */
[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted);
}
[data-testid="stMetricValue"] { font-weight: 600; color: var(--ink); }

/* buttons */
.stButton > button {
    font-family: 'Oswald', sans-serif; font-weight: 500; letter-spacing: 1.5px;
    text-transform: uppercase; font-size: 0.78rem; border-radius: 5px;
    border: 1.5px solid var(--line-2); background: var(--panel-2); color: var(--ink);
    transition: all 0.15s;
}
.stButton > button:hover { border-color: var(--maroon); color: var(--maroon); }
.stButton > button[kind="primary"] { background: var(--maroon); border-color: var(--maroon); color: #fdf4e8; }
.stButton > button[kind="primary"]:hover { background: #7d3232; color: #fdf4e8; }

/* section subheaders get a small maroon rule */
.section-label {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--maroon); border-bottom: 1px solid var(--line); padding-bottom: 6px; margin: 6px 0 14px;
}

/* dataframes, inputs, expanders -> warm borders */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
.stSelectbox div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    border-radius: 5px; border-color: var(--line-2);
}
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
[data-testid="stExpander"] summary { font-weight: 500; letter-spacing: 1px; }

/* sidebar */
[data-testid="stSidebar"] { background: #efe9e1; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase; color: var(--maroon);
}

/* alerts -> softer, on-theme */
[data-testid="stNotificationContentSuccess"] { color: var(--green); }

/* hide chrome for a cleaner, less "developer" feel */
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)


def hero(subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">Fantasy Draft Intelligence</div>
          <div class="hero-title">Draft <span class="accent">War Room</span></div>
          {f'<div class="hero-sub">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Config + projections loading (cached)
# --------------------------------------------------------------------------
def load_config() -> LeagueConfig:
    """Startup fallback config. Real config comes from the Settings tab
    'Connect' action, which pulls your league live and overwrites this.
    """
    yaml_path = os.path.join("config", "league.example.yaml")
    return LeagueConfig.from_yaml(yaml_path)


@st.cache_data(show_spinner=False)
def load_players(path: str, cfg_key: str, _cfg: LeagueConfig, mtime: float) -> pd.DataFrame:
    return load_projections(path, _cfg)


def cfg_signature(c: LeagueConfig) -> str:
    return f"{c.summary()}|{sorted(c.scoring.items())}|{c.keeper_rounds}"


# --------------------------------------------------------------------------
# Credentials: saved league profiles (config/leagues.local.json, gitignored)
# loaded into the session; .env is the fallback.
# --------------------------------------------------------------------------
CRED_KEYS = {"LEAGUE_ID": "league_id", "SEASON": "season", "SWID": "swid", "ESPN_S2": "espn_s2"}
# session keys that belong to one league and must be cleared when switching
_LEAGUE_KEYS = ("state", "rosters", "keeper_result", "snapshot", "snapshot_error",
                "my_team_id", "pick_order", "team_names", "apply_my_keepers",
                "my_keepers_sel", "trade_partner", "trade_give", "trade_get", "_trade_loaded",
                "trade_load", "ss_opp")


def get_cred(key: str, default: str = "") -> str:
    """Credential value: session (Settings tab / saved league) first, then .env."""
    return st.session_state.get(f"cred_{key}") or os.environ.get(key, default)


def reset_league_state() -> None:
    for k in list(st.session_state.keys()):
        if k in _LEAGUE_KEYS or str(k).startswith("keep_team_"):
            del st.session_state[k]


def connect(profile: dict, save_as: str = "") -> None:
    """Connect to a league (raises on failure). Optionally save it as a profile."""
    from draftkit.season import build_snapshot
    reset_league_state()
    for env_key, pkey in CRED_KEYS.items():
        st.session_state[f"cred_{env_key}"] = str(profile.get(pkey) or "").strip()
    if not st.session_state["cred_SEASON"]:
        st.session_state["cred_SEASON"] = "2026"
    client = espn_client()
    raw = client.raw_settings()
    new_cfg = LeagueConfig.from_espn_settings(raw)
    st.session_state.config = new_cfg
    order = list(raw.get("settings", {}).get("draftSettings", {}).get("pickOrder", []))
    st.session_state.pick_order = order
    st.session_state.team_names = client.teams()
    tid = profile.get("team_id") or client.my_team_id()
    st.session_state.my_team_id = int(tid) if tid else None
    if tid and int(tid) in order:
        # applied before the seat widget renders on the next run
        st.session_state._pending_my_slot = order.index(int(tid)) + 1
    st.session_state.connected = True
    st.session_state.active_league = save_as or profile.get("name") or f"League {profile.get('league_id')}"
    if save_as:
        prof = {k: st.session_state[f"cred_{e}"] for e, k in CRED_KEYS.items()}
        prof["team_id"] = st.session_state.my_team_id
        profiles.upsert(save_as, prof)
    try:
        st.session_state.snapshot = build_snapshot(client)
    except Exception as exc:  # noqa: BLE001 -- draft tools still work
        st.session_state.snapshot = None
        st.session_state.snapshot_error = str(exc)


def refresh_snapshot() -> None:
    from draftkit.season import build_snapshot
    try:
        snap = build_snapshot(espn_client())
        if st.session_state.get("my_team_id") and snap.my_team_id is None:
            snap.my_team_id = st.session_state.my_team_id
        st.session_state.snapshot = snap
        st.session_state.pop("snapshot_error", None)
    except Exception as exc:  # noqa: BLE001
        st.session_state.snapshot_error = str(exc)


def is_connected() -> bool:
    return bool(st.session_state.get("connected"))


def espn_client():
    """Build an ESPN client from the current session credentials."""
    from draftkit.espn_client import ESPNClient
    return ESPNClient(
        int(get_cred("LEAGUE_ID", "0") or 0),
        int(get_cred("SEASON", "2026") or 2026),
        swid=get_cred("SWID") or None,
        espn_s2=get_cred("ESPN_S2") or None,
    )


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "config" not in st.session_state:
    st.session_state.config = load_config()

cfg: LeagueConfig = st.session_state.config

if not st.session_state.get("_autoconnect_tried"):
    st.session_state._autoconnect_tried = True
    _prof = profiles.active_profile()
    if _prof and _prof.get("league_id") and _prof.get("swid") and _prof.get("espn_s2"):
        try:
            with st.spinner(f"Connecting to {_prof['name']}..."):
                connect(_prof)
        except Exception as exc:  # noqa: BLE001
            st.session_state.connected = False
            st.session_state.autoconnect_error = f"{_prof['name']}: {exc}"
    cfg = st.session_state.config

if "pick_order" not in st.session_state:
    st.session_state.pick_order = []
if "team_names" not in st.session_state:
    st.session_state.team_names = {}


def seat_name(seat: int) -> str:
    """Map a draft seat (1-indexed) to its ESPN team name, if known."""
    order = st.session_state.get("pick_order", [])
    names = st.session_state.get("team_names", {})
    if 1 <= seat <= len(order):
        tid = order[seat - 1]
        return names.get(tid, names.get(int(tid), f"Seat {seat}"))
    return f"Seat {seat}"

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("League")
_saved = profiles.load()
if _saved["leagues"]:
    _names = list(_saved["leagues"])
    _cur = st.session_state.get("active_league")
    _choice = st.sidebar.selectbox(
        "Saved leagues", _names,
        index=_names.index(_cur) if _cur in _names else 0,
        key=f"league_switch_{_cur}",
    )
    if _choice != _cur and st.sidebar.button(f"Switch to {_choice}", type="primary"):
        try:
            with st.spinner(f"Connecting to {_choice}..."):
                connect(dict(_saved["leagues"][_choice], name=_choice))
            profiles.set_active(_choice)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Couldn't connect to {_choice}: {exc}")

snap = st.session_state.get("snapshot")
if snap is not None and snap.teams:
    _tids = sorted(snap.teams)
    _mine = st.session_state.get("my_team_id") or snap.my_team_id
    _sel = st.sidebar.selectbox(
        "Your team", _tids, index=_tids.index(_mine) if _mine in _tids else 0,
        format_func=snap.team_name,
        help="Auto-detected from your SWID when you own a team in this league.",
    )
    if _sel != _mine:
        st.session_state.my_team_id = int(_sel)
        _al = st.session_state.get("active_league")
        if _al in _saved["leagues"]:
            profiles.upsert(_al, dict(_saved["leagues"][_al], team_id=int(_sel)), make_active=False)
    snap.my_team_id = int(_sel)
    if st.sidebar.button("Refresh league data", help="Re-pull rosters, projections, waivers."):
        with st.spinner("Refreshing from ESPN..."):
            refresh_snapshot()
        st.rerun()
    st.sidebar.caption(f"Week {snap.week} · updated "
                       f"{pd.Timestamp(snap.fetched_at, unit='s').tz_localize('UTC').tz_convert('America/New_York'):%a %-I:%M %p}")
my_tid = snap.my_team_id if snap is not None else None

with st.sidebar.expander("League rules"):
    st.code(cfg.summary(), language=None)

st.sidebar.header("Projections")
proj_source = st.sidebar.radio(
    "Source",
    ["CSV file", "ESPN (pull live)"],
    help="ESPN pulls league-scored projections + ADP for free using your login.",
)

proj_default = os.path.join(DATA_DIR, "projections.csv")
if not os.path.exists(proj_default):
    proj_default = os.path.join(DATA_DIR, "projections.sample.csv")
proj_path = st.sidebar.text_input("Projections CSV", value=proj_default)

if proj_source == "ESPN (pull live)":
    if st.sidebar.button("Pull ESPN projections now"):
        if not is_connected():
            st.sidebar.error("Connect first in the Settings tab.")
        else:
            try:
                dfp = espn_client().projections(limit=500)
                if dfp.empty:
                    st.sidebar.error("ESPN returned no players. Tell me and I'll adjust.")
                else:
                    dfp.to_csv(os.path.join(DATA_DIR, "projections.csv"), index=False)
                    st.sidebar.success(f"Pulled {len(dfp)} players. Saved to projections.csv.")
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Pull failed: {exc}")

if "_pending_my_slot" in st.session_state:
    st.session_state.my_slot = st.session_state.pop("_pending_my_slot")
st.session_state.my_slot = min(max(int(st.session_state.get("my_slot", 1)), 1), cfg.team_count)
my_seat = st.sidebar.number_input(
    "Your draft seat", min_value=1, max_value=cfg.team_count, step=1, key="my_slot",
)

st.sidebar.header("Simulation")
use_sim = st.sidebar.checkbox("Use Monte Carlo (stronger)", value=True)
n_sims = st.sidebar.slider("Sims per candidate", 30, 300, 80, step=10)
n_cands = st.sidebar.slider("Candidates simulated", 4, 12, 6, step=1)
sim_lookahead = st.sidebar.slider(
    "Only simulate within N picks of your turn", 1, 40, 10, step=1,
    help="When you're further away than this, the fast VORP/VONA board is shown "
         "instead — no need to simulate a pick that's 30 slots off.",
)
defense = st.sidebar.slider(
    "Draft defense (depth / blocking)", 0.0, 0.5, 0.15, step=0.05,
    help="Small bonus for scarce, high-value players even if they'd sit your "
         "bench — insurance value and denying rivals. Kept a fraction so it "
         "never outweighs filling your own starters. 0 = pure lineup optimization.",
)

if st.sidebar.button("Reset draft"):
    st.session_state.pop("state", None)
    st.rerun()

# players
try:
    players = load_players(proj_path, cfg_signature(cfg), cfg,
                           os.path.getmtime(proj_path) if os.path.exists(proj_path) else 0.0)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load projections from {proj_path}: {exc}")
    st.stop()

# draft state -- rebuilt when the league changes; changing your seat keeps
# picks and keepers (they're recorded by seat, not by "you")
if "state" not in st.session_state or st.session_state.state.config is not cfg:
    st.session_state.state = DraftState(config=cfg, my_team=int(my_seat))
state: DraftState = st.session_state.state
state.my_team = int(my_seat)


# --------------------------------------------------------------------------
# Cached simulation -- only recomputes when the draft actually changes.
# Editing keepers / switching tabs hits the cache and returns instantly.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_sim(sig: str, _state, _players, _cfg, n_sims: int, n_cands: int):
    from draftkit.recommender import candidate_pool
    cands = candidate_pool(_state, _players, _cfg, n_cands)
    if cands.empty:
        return cands
    return simulate_candidates(
        _state, _players, _cfg, candidates=cands,
        n_sims=n_sims, max_candidates=len(cands), seed=1,
    )


def sim_signature() -> str:
    from draftkit.recommender import suppressed_positions
    drafted = ",".join(sorted(state.drafted_ids))
    reserved = ",".join(sorted(f"{p.overall}:{p.player_id}" for p in state.reserved_picks))
    supp = ",".join(sorted(suppressed_positions(state, cfg)))
    return f"{cfg_signature(cfg)}|{drafted}|{reserved}|{state.my_team}|{n_sims}|{n_cands}|{len(players)}|{round(float(players['proj'].sum()),1)}|{supp}"

# --------------------------------------------------------------------------
# Hero + tabs
# --------------------------------------------------------------------------
hero(f"{cfg.name} · {cfg.team_count}-team {'superflex ' if cfg.superflex else ''}"
     f"league · you draft at seat {int(my_seat)}")

if not is_connected():
    st.warning(
        "**Not connected to ESPN.** Open the **Settings** tab to paste your "
        "league ID and cookies — then your real league, rosters, and team "
        "names load. Until then you're seeing placeholder settings."
    )

if st.session_state.get("autoconnect_error") and not is_connected():
    st.error(f"Couldn't reconnect to your saved league — {st.session_state.autoconnect_error}. "
             f"Cookies may have expired; update them in Settings.")
if st.session_state.get("snapshot_error"):
    st.warning(f"League connected, but in-season data didn't load: "
               f"{st.session_state.snapshot_error}. Try **Refresh league data** in the sidebar.")

(tab_lineup, tab_waivers, tab_trades, tab_outlook,
 tab_draft, tab_keepers, tab_league, tab_settings) = st.tabs(
    ["Start / Sit", "Waivers", "Trades", "Outlook",
     "Draft board", "Keepers", "League & rosters", "Settings"]
)

import season_ui  # noqa: E402

with tab_lineup:
    season_ui.render_start_sit(snap, my_tid)
with tab_waivers:
    season_ui.render_waivers(snap, my_tid)
with tab_trades:
    season_ui.render_trades(snap, my_tid)
with tab_outlook:
    season_ui.render_outlook(snap, my_tid)

# ==========================================================================
# TAB 4 -- SETTINGS (credentials entered here, held only in memory)
# ==========================================================================
with tab_settings:
    section("Connect to ESPN")
    st.markdown(
        "Paste your league ID and cookies below. Tick **Remember** to save the league "
        "to `config/leagues.local.json` — a gitignored, owner-only file on this computer "
        "— so it reconnects automatically next time. Save as many leagues as you like "
        "and switch between them from the sidebar."
    )
    if is_connected():
        st.success(f"Connected to **{cfg.name}** "
                   f"({cfg.team_count} teams, {len(st.session_state.team_names)} names loaded).")

    saved = profiles.load()
    edit_names = ["➕ New league"] + list(saved["leagues"])
    active = st.session_state.get("active_league")
    edit = st.selectbox("League to edit", edit_names,
                        index=edit_names.index(active) if active in edit_names else 0)
    base = saved["leagues"].get(edit, {}) if edit != "➕ New league" else {}
    last = next(iter(saved["leagues"].values()), {})   # reuse cookies for a new league

    def _pref(key, env_key, fallback=""):
        return str(base.get(key) or (last.get(key) if key in ("swid", "espn_s2") else "")
                   or os.environ.get(env_key, fallback) or "")

    with st.form("connect_form"):
        sc1, sc2 = st.columns(2)
        with sc1:
            v_name = st.text_input("Name for this league",
                                   value="" if edit == "➕ New league" else edit,
                                   placeholder="e.g. Work league")
            v_league = st.text_input("League ID", value=_pref("league_id", "LEAGUE_ID"),
                                     placeholder="198442399")
            v_season = st.text_input("Season", value=_pref("season", "SEASON", "2026"))
        with sc2:
            v_swid = st.text_input("SWID cookie", value=_pref("swid", "SWID"),
                                   type="password", placeholder="{XXXXXXXX-....}")
            v_s2 = st.text_input("espn_s2 cookie", value=_pref("espn_s2", "ESPN_S2"),
                                 type="password", placeholder="long string with %2F, %3D ...")
            v_remember = st.checkbox("Remember this league on this computer", value=True)
        go = st.form_submit_button("Connect", type="primary")

    if go:
        prof = {"league_id": v_league.strip(), "season": v_season.strip() or "2026",
                "swid": v_swid.strip(), "espn_s2": v_s2.strip(),
                "team_id": base.get("team_id"),
                "name": v_name.strip() or f"League {v_league.strip()}"}
        try:
            with st.spinner("Connecting..."):
                connect(prof, save_as=prof["name"] if v_remember else "")
            if v_remember and edit not in ("➕ New league", prof["name"]):
                profiles.remove(edit)                     # renamed
            st.success("Connected. Your league, draft order, and team names loaded.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state.connected = False
            st.error(f"Could not connect: {exc}\n\nDouble-check the League ID and "
                     f"that both cookies are current (they expire — re-copy if needed).")

    with st.expander("How to get your SWID and espn_s2 cookies"):
        st.markdown(
            "1. Log into ESPN in your browser and open your league.\n"
            "2. Open DevTools (**F12** or right-click → Inspect).\n"
            "3. Go to **Application** (Chrome/Edge) or **Storage** (Firefox) → "
            "**Cookies** → `https://fantasy.espn.com`.\n"
            "4. Copy the **Value** of `SWID` (keep the curly braces) and "
            "`espn_s2` (a long string), and paste them above.\n\n"
            "Your League ID is the number in your league's URL "
            "(`.../leagueId=XXXXXXXXX`). The same two cookies work for every "
            "league on your ESPN account."
        )
        st.caption("These cookies act like your ESPN login — keep them private. Saved "
                   "leagues live only in config/leagues.local.json, which git ignores.")

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        if edit != "➕ New league" and st.button(f"Forget “{edit}”"):
            profiles.remove(edit)
            st.rerun()
    with d2:
        if st.button("Disconnect"):
            for k in ("cred_LEAGUE_ID", "cred_SEASON", "cred_SWID", "cred_ESPN_S2", "active_league"):
                st.session_state.pop(k, None)
            reset_league_state()
            st.session_state.connected = False
            st.rerun()

# ==========================================================================
# TAB 1 -- DRAFT BOARD
# ==========================================================================
with tab_draft:
    state.auto_fill_keepers()   # consume any leading keeper slots first
    if state.keeper_total:
        st.caption(f"🔒 {state.keeper_total} keepers locked in — off the board and "
                   f"holding each team's early picks.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall pick", state.next_overall)
    c2.metric("Round", (state.next_overall - 1) // cfg.team_count + 1)
    c3.metric("On the clock", seat_name(state.current_team))
    c4.metric("Until your pick", state.picks_until_my_turn())

    on_clock = state.current_team
    my_turn = state.is_my_turn()
    if my_turn:
        st.success("**You are on the clock.** Your best picks below.")
    else:
        st.markdown(f"On the clock: **{seat_name(on_clock)}** — showing the best "
                    f"pick *for them* (a read on what may go before your turn).")

    # Confidence sim runs only for YOUR decision (near your pick). For opponents
    # we show their roster-aware best-available board as a fast prediction.
    sim_results = None
    near_turn = my_turn or state.picks_until_my_turn() <= sim_lookahead
    if use_sim and my_turn and state.next_overall <= state.total_picks:
        with st.spinner("Simulating the rest of the draft..."):
            sim_results = cached_sim(sim_signature(), state, players, cfg, n_sims, n_cands)

    # main board = the ON-CLOCK team's perspective
    board = build_board(state, players, cfg, sim_results, team=on_clock, defense=defense)

    # confidence: P(this candidate is the best pick), from the simulation spread
    probs = {}
    if sim_results is not None and not sim_results.empty:
        from draftkit.simulation import pick_probabilities
        probs = pick_probabilities(sim_results)
        # rank by confidence, with a small draft-defense nudge (scarce/high-value
        # players get a minor bump even if bench-bound). Defense is capped small,
        # so a genuine starter's confidence still dominates the #1 slot.
        vmax = max(float(board["vorp"].clip(lower=0).max()), 1.0)
        board["_conf"] = board["player_id"].astype(str).map(probs)
        bench_only = board["bench_only"] if "bench_only" in board else 0.0
        board["_def"] = defense * (board["vorp"].clip(lower=0) / vmax) * bench_only
        board["_rank"] = board["_conf"].fillna(-1.0) + board["_def"].fillna(0.0)
        board = board.sort_values("_rank", ascending=False).drop(
            columns=["_conf", "_def", "_rank"]).reset_index(drop=True)

    left, right = st.columns([3, 2])

    _POS_COLORS = {
        "QB": "#3d6e4e", "RB": "#943c3c", "WR": "#9e862f",
        "TE": "#b07348", "K": "#6b6b6b", "IDP": "#4a6670", "DST": "#6b6b6b",
    }

    with left:
        section("Recommended picks")
        if board.empty:
            st.info("Draft complete.")
        else:
            top = board.head(8).copy()
            top["confidence"] = top["player_id"].astype(str).map(probs).astype(float) * 100

            # colored top-pick banner
            tr = board.iloc[0]
            conf = probs.get(str(tr["player_id"]))
            if conf is not None:
                pct = conf * 100
                color = "#3d6e4e" if pct >= 70 else ("#9e862f" if pct >= 45 else "#943c3c")
                verdict = ("clear pick" if pct >= 70 else
                           "lean" if pct >= 45 else "close call")
                st.markdown(
                    f"<div style='padding:12px 16px;border-left:5px solid {color};"
                    f"background:#f2ede6;border-radius:6px;margin-bottom:12px;'>"
                    f"<span style='font-size:0.7rem;letter-spacing:2px;text-transform:"
                    f"uppercase;color:#888;'>Top pick · {verdict}</span><br>"
                    f"<span style='font-size:1.4rem;font-weight:700;color:{color};'>"
                    f"{tr['name']}</span> <span style='color:#555;'>({tr['pos']}, "
                    f"{tr['team']}) — {pct:.0f}% confidence</span></div>",
                    unsafe_allow_html=True,
                )

            top["why"] = top.apply(lambda r: reason_for(r, cfg), axis=1)
            show_cols = ["name", "pos", "team", "proj", "vorp", "vona", "tier"]
            if top["sim_ev"].notna().any():
                show_cols.insert(3, "sim_ev")
            if top["confidence"].notna().any():
                show_cols.insert(0, "confidence")
            display = top[show_cols + ["why"]].copy()
            for col in ("proj", "vorp", "vona", "sim_ev"):
                if col in display:
                    display[col] = display[col].round(1)

            def _conf_bg(v):
                if pd.isna(v):
                    return ""
                frac = min(max(v / 100.0, 0), 1)
                return f"background-color: rgba(61,110,78,{0.12 + 0.55 * frac:.2f}); font-weight:600"

            try:
                sty = display.style
                if "confidence" in display:
                    sty = sty.map(_conf_bg, subset=["confidence"]).format({"confidence": "{:.0f}%"})
                sty = sty.map(
                    lambda v: f"color:{_POS_COLORS.get(v, '#2c2c2c')}; font-weight:600",
                    subset=["pos"],
                )
                st.dataframe(sty, use_container_width=True, hide_index=True)
            except Exception:  # styling is cosmetic; never let it break the board
                if "confidence" in display:
                    display["confidence"] = display["confidence"].map(
                        lambda v: "" if pd.isna(v) else f"{v:.0f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)

            st.markdown("**Draft a player** — defaults to the top pick; "
                        "type to search, Enter to select, then Record. "
                        "Applies to the team on the clock.")
            avail_df = state.available(players).copy()
            # order the list by the recommendation ranking so the #1 pick is the
            # default; gated/late players fall to the end but stay searchable.
            order_index = {str(pid): i for i, pid in enumerate(board["player_id"].tolist())}
            avail_df["_ord"] = avail_df["player_id"].astype(str).map(
                lambda p: order_index.get(p, 10 ** 9))
            pick_pool = avail_df.sort_values(["_ord", "adp"])
            options = {
                f"{r['name']} ({r['pos']}, {r['team']}) — ADP {r['adp']:.0f}": r["player_id"]
                for _, r in pick_pool.iterrows()
            }
            opt_labels = list(options.keys())
            with st.form("draft_form"):
                chosen = st.selectbox("Player", opt_labels, index=0)
                submitted = st.form_submit_button("Record pick", type="primary")
            if submitted and chosen:
                pid = options[chosen]
                row = players.loc[players["player_id"] == pid].iloc[0].to_dict()
                state.record_pick(row)
                st.rerun()
            if st.button("Undo last pick"):
                state.undo()
                st.rerun()

    with right:
        # Your recommendation, always from your roster's view. On your turn the
        # look-ahead target is your NEXT pick; otherwise it's your upcoming pick.
        my_board = board if my_turn else build_board(state, players, cfg, team=state.my_team, defense=defense)
        look_pick = state.my_pick_after_next() if my_turn else (state.my_next_pick() or state.next_overall)

        section("Around at your next pick?" if my_turn else "Your pick — quick view")
        if my_board.empty:
            st.caption("Nothing to suggest.")
        elif my_turn and look_pick is None:
            st.caption("This is your last pick — no next pick to look ahead to.")
        else:
            from draftkit.simulation import availability_until
            if my_turn:
                # exactly the main board's top 5, in the same order, with the
                # odds each falls back to your next pick appended.
                cand = my_board.head(5).copy().reset_index(drop=True)
                avail_map = availability_until(
                    state, players, cfg, look_pick, cand["player_id"].tolist(), n_sims=250,
                )
                cand["avail"] = cand["player_id"].astype(str).map(avail_map).fillna(1.0) * 100
            else:
                cand = my_board.head(10).copy().reset_index(drop=True)
                avail_map = availability_until(
                    state, players, cfg, look_pick, cand["player_id"].tolist(), n_sims=250,
                )
                cand["avail"] = cand["player_id"].astype(str).map(avail_map).fillna(1.0) * 100
                # fold availability into the ordering: a mild urgency boost so a
                # scarce, high-value target edges above one that'll come back.
                n = max(1, len(cand) - 1)
                cand["_value_rank"] = 1 - cand.index / n
                cand["priority"] = cand["_value_rank"] + 0.35 * (1 - cand["avail"] / 100)
                cand = cand.sort_values("priority", ascending=False).head(5).reset_index(drop=True)

            if my_turn:
                st.caption(f"Draft now (main panel), but here's how likely each "
                           f"is to fall back to your next pick (#{look_pick}). "
                           f"Grab the ones that won't.")
            else:
                st.caption(f"Your next pick is #{look_pick} "
                           f"({state.picks_until_my_turn()} away). "
                           f"“Avail” = chance they're still there, if rivals draft like this model.")

            t0 = cand.iloc[0]
            st.markdown(
                f"<div style='padding:10px 14px;border-left:4px solid #943c3c;"
                f"background:#f2ede6;border-radius:6px;margin-bottom:8px;'>"
                f"<span style='font-size:0.65rem;letter-spacing:2px;text-transform:"
                f"uppercase;color:#888;'>{'Grab now' if float(t0['avail'])<55 else 'Top target'}</span><br>"
                f"<span style='font-size:1.1rem;font-weight:700;color:#943c3c;'>"
                f"{t0['name']}</span> <span style='color:#555;'>({t0['pos']}) · "
                f"{float(t0['avail']):.0f}% avail next</span></div>",
                unsafe_allow_html=True,
            )

            mini = cand[["name", "pos", "avail"]].copy()
            try:
                msty = mini.style.map(
                    lambda v: f"color:{_POS_COLORS.get(v, '#2c2c2c')};font-weight:600",
                    subset=["pos"]).format({"avail": "{:.0f}%"})
                st.dataframe(msty, use_container_width=True, hide_index=True)
            except Exception:
                mini["avail"] = mini["avail"].map(lambda v: f"{v:.0f}%")
                st.dataframe(mini, use_container_width=True, hide_index=True)

        section("Your roster")
        mine = state.my_roster()
        proj_by_id = dict(zip(players["player_id"].astype(str), players["proj"].astype(float)))
        if mine:
            rdf = pd.DataFrame([{"round": p.round, "player": p.name, "pos": p.pos} for p in mine])
            try:
                rsty = rdf.style.map(
                    lambda v: f"color:{_POS_COLORS.get(v, '#2c2c2c')}; font-weight:600",
                    subset=["pos"])
                st.dataframe(rsty, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(rdf, use_container_width=True, hide_index=True)
            from draftkit.valuation import optimal_lineup_value
            roster_rows = [{"pos": p.pos, "proj": proj_by_id.get(str(p.player_id), 0.0)}
                           for p in mine]
            lineup_val = optimal_lineup_value(roster_rows, cfg)
            st.markdown(
                f"<div style='padding:10px 14px;background:#3d6e4e;border-radius:6px;"
                f"color:#fdf4e8;'><span style='font-size:0.7rem;letter-spacing:2px;"
                f"text-transform:uppercase;opacity:0.85;'>Starting lineup value</span>"
                f"<br><span style='font-size:1.6rem;font-weight:700;'>{lineup_val:.0f}"
                f" pts</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No picks yet.")

        section("Positional scarcity")
        avail = state.available(players)
        scarce = (
            avail.groupby("pos")["proj"]
            .agg(["count", "max"])
            .rename(columns={"count": "left", "max": "best_left"})
            .round(1)
        )
        st.dataframe(scarce, use_container_width=True)

    with st.expander("Full available board"):
        if not board.empty:
            cols = ["name", "pos", "team", "proj", "vorp", "vona", "tier", "adp"]
            st.dataframe(board[cols].round(1), use_container_width=True, hide_index=True)

# ==========================================================================
# TAB 3 (part A) -- LEAGUE & ROSTERS : draft log
# ==========================================================================
with tab_league:
    section("Draft log")
    if state.picks:
        log = pd.DataFrame(
            [{"overall": p.overall, "round": p.round, "team": seat_name(p.team),
              "player": p.name, "pos": p.pos} for p in state.picks]
        )
        st.dataframe(log, use_container_width=True, hide_index=True)
    else:
        st.caption("No picks recorded yet.")

# ==========================================================================
# TAB 2 -- KEEPERS
# ==========================================================================
with tab_keepers:
    section("Keeper board")
    st.caption(
        "Keeping costs your round-1/2/3 picks (1st/2nd/3rd keeper). Load every "
        "team's roster from ESPN; your whole roster is considered automatically, "
        "and you mark each rival's keepers under their team name."
    )

    if "rosters" not in st.session_state:
        st.session_state.rosters = {}

    kc1, kc2 = st.columns([1, 2])
    with kc1:
        my_team_id = st.number_input(
            "Your ESPN team id", min_value=1, step=1,
            value=int(st.session_state.get("my_team_id") or 1),
            help="Auto-detected from your SWID when you connect.",
        )
        if st.button("Load all rosters from ESPN"):
            if not is_connected():
                st.error("Connect first in the Settings tab.")
            else:
                try:
                    client = espn_client()
                    st.session_state.rosters = client.rosters()
                    st.session_state.team_names = client.teams()
                    slot = client.my_slot(int(my_team_id))
                    if slot:
                        st.session_state._pending_my_slot = slot
                    st.success(f"Loaded {len(st.session_state.rosters)} rosters. "
                               + (f"Your draft slot: {slot}." if slot else ""))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Roster pull failed: {exc}. You can still mark keepers "
                             f"manually below.")

    with kc2:
        my_slot_val = int(my_seat)
        st.metric("Your draft slot", my_slot_val, help="Set in the sidebar (auto-filled from ESPN).")

    rosters = st.session_state.rosters
    team_names = st.session_state.team_names

    def _tname(tid: int) -> str:
        return team_names.get(tid, f"Team {tid}")

    def _norm_name(s: str) -> str:
        return "".join(c for c in str(s).lower() if c.isalnum())

    name_to_pid = {r["name"]: str(r["player_id"]) for _, r in players.iterrows()}
    pid_set = set(name_to_pid.values())
    projname_to_pid = {_norm_name(r["name"]): str(r["player_id"]) for _, r in players.iterrows()}

    def _resolve_pid(player: dict):
        pid = str(player.get("player_id"))
        if pid in pid_set:
            return pid
        return projname_to_pid.get(_norm_name(player.get("name", "")))

    _looks_sample = bool(players["name"].str.match(r"^(QB|RB|WR|TE|K|DST)\d+$").all())
    st.info(
        f"Projections loaded: **{len(players)}** players "
        f"({'⚠️ SAMPLE data — pull ESPN projections in the sidebar' if _looks_sample else 'real names ✓'}).  "
        f"Rosters loaded: **{len(rosters)}**."
    )

    pool_names = sorted(name_to_pid.keys())
    name_to_pos = {r["name"]: r["pos"] for _, r in players.iterrows()}
    team_ids = (sorted(rosters) if rosters
                else sorted(team_names) if team_names
                else list(range(1, cfg.team_count + 1)))
    rosters_populated = bool(rosters) and any(len(v) > 0 for v in rosters.values())

    # ---- your keeper candidates: pick from the full pool (auto-fill if the
    #      roster is already populated, e.g. mid-draft) ---------------------
    section("Your keeper candidates")
    default_my = []
    if rosters_populated and int(my_team_id) in rosters:
        for p in rosters[int(my_team_id)]:
            rid = _resolve_pid(p)
            nm = next((n for n, pid in name_to_pid.items() if pid == rid), None) if rid else None
            if nm:
                default_my.append(nm)
    st.caption("Add the players you're considering keeping. Rosters are empty until "
               "the draft fills them, so just search the full list — this works live.")
    picked_my = st.multiselect("Your keeper candidates", options=pool_names,
                               default=default_my, key="my_keepers_sel")
    my_candidate_ids = [name_to_pid[n] for n in picked_my]

    # ---- rival keepers in a form (full pool, per team) --------------------
    section("Other teams' keepers")
    st.caption("For each rival, search and add the player(s) they've kept. "
               "Nothing recalculates until you press Compute.")
    if not st.session_state.get("pick_order"):
        st.warning("Pull league settings in the Settings tab so teams map to draft "
                   "seats — needed to place rival keepers on the board correctly.")
    others = [t for t in team_ids if t != int(my_team_id)]
    with st.form("keepers_form"):
        fcols = st.columns(2)
        for i, tid in enumerate(others):
            with fcols[i % 2]:
                st.multiselect(f"{_tname(tid)} kept:", options=pool_names,
                               key=f"keep_team_{tid}",
                               max_selections=cfg.keeper_rounds or 4)
        submitted = st.form_submit_button("Compute keeper recommendation", type="primary")

    if submitted:
        other_keeper_ids: set = set()
        other_keeper_overalls: list = []
        labels = []
        order = st.session_state.get("pick_order", [])
        from draftkit.keepers import pick_overall_numbers
        for tid in others:
            names_kept = st.session_state.get(f"keep_team_{tid}", [])
            for nm in names_kept:
                if nm in name_to_pid:
                    other_keeper_ids.add(name_to_pid[nm])
                    labels.append(f"{_tname(tid)} kept {nm}")
            c = len(names_kept)
            if c and tid in order:
                seat = order.index(tid) + 1
                other_keeper_overalls += pick_overall_numbers(
                    seat, cfg.team_count, cfg.roster_size, cfg.keeper_rounds)[:c]

        if my_candidate_ids:
            from draftkit.keepers import evaluate_keepers
            res = evaluate_keepers(
                [], players, cfg, int(my_slot_val),
                other_keeper_ids=other_keeper_ids, candidate_ids=my_candidate_ids,
                other_keeper_overalls=other_keeper_overalls or None,
            )
            st.session_state.keeper_result = {
                "per": res.per_keeper, "by": res.by_count,
                "keep": res.recommended_keep, "labels": labels,
            }
        else:
            st.session_state.keeper_result = None

    # ---- render the last committed result (persists across reruns) --------
    kr = st.session_state.get("keeper_result")
    if kr:
        if kr["labels"]:
            st.caption(" · ".join(kr["labels"]))
        st.markdown(f"### Recommendation: keep **{kr['keep']}** players")
        st.markdown("**Each candidate vs. the pick it would cost:**")
        st.dataframe(kr["per"], use_container_width=True, hide_index=True)
        st.markdown("**Net surplus by number kept** (pick the peak):")
        st.dataframe(kr["by"], use_container_width=True, hide_index=True)
        st.caption("Positive surplus = the player is worth more than the pick you "
                   "give up. Values are VORP, so superflex + your scoring are baked in.")

    # ---- apply keepers to the draft board ---------------------------------
    section("Lock keepers into the draft")
    st.caption("Applying keepers removes every kept player from the board and "
               "assigns each team's keepers to its first picks (1st→R1, 2nd→R2, "
               "3rd→R3). Do this once, before live drafting.")

    default_keep = []
    if kr and not kr["per"].empty:
        default_keep = [n for n in kr["per"].head(kr["keep"])["keeper"] if n in name_to_pid]
    my_keepers = st.multiselect(
        "Your keepers (these go off the board and use your first picks)",
        options=pool_names, default=default_keep or picked_my,
        key="apply_my_keepers",
        help="Defaults to the recommended set; adjust to whatever you'll actually keep.",
    )

    order = st.session_state.get("pick_order", [])
    if st.button("Apply keepers to draft board", type="primary"):
        keeper_slots: dict = {}
        applied = 0
        missing_seat = []
        for tid in team_ids:
            if tid == int(my_team_id):
                names, seat = my_keepers, int(my_slot_val)
            else:
                names = st.session_state.get(f"keep_team_{tid}", [])
                seat = order.index(tid) + 1 if tid in order else None
            if not names:
                continue
            if seat is None:
                missing_seat.append(_tname(tid))
                continue
            plist = [{"player_id": name_to_pid[nm], "name": nm, "pos": name_to_pos.get(nm, "")}
                     for nm in names if nm in name_to_pid]
            keeper_slots[seat] = plist
            applied += len(plist)

        new_state = DraftState(config=cfg, my_team=int(my_slot_val))
        new_state.set_keepers(keeper_slots)
        new_state.auto_fill_keepers()
        st.session_state.state = new_state
        msg = (f"Locked in {applied} keepers across {len(keeper_slots)} teams. "
               f"They're off the board and hold each team's early picks. "
               f"Head to the Draft board tab.")
        if missing_seat:
            msg += (f"  ⚠️ Couldn't place: {', '.join(missing_seat)} "
                    f"(no draft seat — pull league settings).")
        st.success(msg)
        st.rerun()

# ==========================================================================
# TAB 3 (part B) -- LEAGUE & ROSTERS : loaded rosters
# ==========================================================================
with tab_league:
    if rosters:
        section("Loaded rosters")
        for tid in sorted(rosters):
            tag = "  ·  (you)" if tid == int(my_team_id) else ""
            st.markdown(f"**{_tname(tid)}**{tag}")
            rdf = pd.DataFrame(rosters.get(tid) or [])
            if rdf.empty:
                st.caption("No players.")
                continue
            show = [c for c in ("name", "pos", "pro_team") if c in rdf.columns]
            st.dataframe(
                rdf[show] if show else rdf,
                use_container_width=True, hide_index=True,
            )
