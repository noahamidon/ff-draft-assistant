"""In-season Streamlit tabs: Start/Sit, Waivers, Trades, Outlook.

Kept out of app.py so the draft UI and the in-season UI stay readable. Every
function takes the current Snapshot (draftkit.season) and renders into
whatever container it's called inside.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from draftkit.lineup import start_sit
from draftkit.outlook import season_outlook
from draftkit.season import IR_SLOT, Snapshot, ros_points, week_mean
from draftkit.trades import evaluate_trade, suggest_trades
from draftkit.waivers import rank_pickups, recent_activity, waiver_table

_POS_COLORS = {
    "QB": "#3d6e4e", "RB": "#943c3c", "WR": "#9e862f",
    "TE": "#b07348", "K": "#6b6b6b", "IDP": "#4a6670", "DST": "#6b6b6b",
}


def _section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def _show(df: pd.DataFrame, pos_cols=("pos",), **kw) -> None:
    try:
        sty = df.style
        for c in pos_cols:
            if c in df:
                sty = sty.map(lambda v: f"color:{_POS_COLORS.get(v, '#2c2c2c')};font-weight:600",
                              subset=[c])
        st.dataframe(sty, use_container_width=True, hide_index=True, **kw)
    except Exception:  # styling is cosmetic
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def _banner(label: str, value: str, sub: str, color: str = "#3d6e4e") -> None:
    st.markdown(
        f"<div style='padding:12px 16px;border-left:5px solid {color};background:#f2ede6;"
        f"border-radius:6px;margin-bottom:12px;'><span style='font-size:0.7rem;letter-spacing:2px;"
        f"text-transform:uppercase;color:#888;'>{label}</span><br><span style='font-size:1.4rem;"
        f"font-weight:700;color:{color};'>{value}</span> <span style='color:#555;'>{sub}</span></div>",
        unsafe_allow_html=True,
    )


def _snap_key(snap: Snapshot) -> str:
    return f"{snap.config.name}|{snap.year}|{snap.fetched_at}|{snap.my_team_id}"


@st.cache_data(show_spinner=False)
def _pickups(key: str, _snap: Snapshot, tid: int) -> pd.DataFrame:
    return rank_pickups(_snap, tid)


@st.cache_data(show_spinner=False)
def _suggestions(key: str, _snap: Snapshot, tid: int) -> pd.DataFrame:
    return suggest_trades(_snap, tid)


@st.cache_data(show_spinner=False)
def _outlook(key: str, _snap: Snapshot) -> pd.DataFrame:
    return season_outlook(_snap)


def _need(snap, tid) -> bool:
    if snap is None:
        st.info("Connect to ESPN in **Settings** to load your league for the season.")
        return True
    if tid is None:
        st.warning("Pick **your team** in the sidebar (we couldn't match your SWID to a team owner).")
        return True
    return False


# --------------------------------------------------------------------------
def render_start_sit(snap: Snapshot, tid) -> None:
    if _need(snap, tid):
        return
    tid = int(tid)
    sched_opp = snap.opponent_of(tid)
    others = [t for t in snap.teams if t != tid]
    labels = {t: snap.team_name(t) for t in others}
    default = others.index(sched_opp) if sched_opp in others else 0
    c1, c2 = st.columns([2, 1])
    with c1:
        opp = st.selectbox(
            f"Opponent — week {snap.week}", others, index=default,
            format_func=lambda t: labels[t] + ("  (scheduled)" if t == sched_opp else ""),
            key="ss_opp",
        )
    res = start_sit(snap, tid, opp)
    rec, cur = res["recommended"], res["current"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win chance (recommended)", f"{100 * rec['win_prob']:.0f}%",
              delta=f"{100 * (rec['win_prob'] - cur['win_prob']):+.0f} pts vs current"
              if cur["lineup"] else None)
    m2.metric("Your projection", f"{rec['mean']:.1f}", help=f"± {rec['std']:.1f} (1 sd)")
    m3.metric(f"{snap.team_name(opp)}", f"{res['opp_mean']:.1f}",
              help=f"± {res['opp_std']:.1f} (1 sd), from their {res['opp_source']}")
    m4.metric("Opponent volatility", f"± {res['opp_std']:.0f}",
              help="Std. dev. of the lineup they've set. A risky opponent means your own "
                   "variance matters less; a safe one means it matters more.")

    color = "#3d6e4e" if rec["win_prob"] >= 0.6 else ("#9e862f" if rec["win_prob"] >= 0.45 else "#943c3c")
    ch = res["changes"]
    if ch["start"] or ch["bench"]:
        _banner("Lineup changes", "Start " + ", ".join(ch["start"]),
                "· bench " + ", ".join(ch["bench"]), color)
    else:
        _banner("Lineup", "No changes needed", "— your set lineup is already the best call.", color)
    st.caption(res["stance"])

    left, right = st.columns([3, 2])
    with left:
        _section("Recommended lineup")
        lu = pd.DataFrame([{
            "slot": e["slot_label"], "player": e["name"], "pos": e["pos"], "team": e["pro_team"],
            "proj": round(e["mean"], 1), "± sd": round(e["std"], 1),
            "status": ("🔒 locked" if e["locked"] else "") +
                      ("" if e["injury"] == "ACTIVE" else f" {e['injury'].title().replace('_', ' ')}"),
        } for e in rec["lineup"]])
        _show(lu)
        _section("Bench")
        starters = {e["player_id"] for e in rec["lineup"]}
        bench = [r for r in snap.rosters.get(tid, []) if r["player_id"] not in starters]
        bdf = pd.DataFrame([{
            "player": r["name"], "pos": r["pos"], "team": r["pro_team"],
            "proj": round(week_mean(r, snap.week, snap), 1),
            "where": "IR" if r.get("slot") == IR_SLOT else "bench",
            "inj": "" if r["injury"] == "ACTIVE" else r["injury"].title().replace("_", " "),
        } for r in bench])
        if not bdf.empty:
            _show(bdf.sort_values("proj", ascending=False))
    with right:
        _section("Lineup options")
        comp = pd.DataFrame([
            {"lineup": "Currently set", "proj": cur["mean"], "± sd": cur["std"], "win %": cur["win_prob"]},
            {"lineup": "Max projected pts", "proj": res["max_points"]["mean"],
             "± sd": res["max_points"]["std"], "win %": res["max_points"]["win_prob"]},
            {"lineup": "Recommended", "proj": rec["mean"], "± sd": rec["std"], "win %": rec["win_prob"]},
        ])
        comp["win %"] = (comp["win %"].fillna(0) * 100).round(0)
        st.dataframe(comp.round(1), use_container_width=True, hide_index=True)
        _section(f"{snap.team_name(opp)} — {res['opp_source']}")
        odf = pd.DataFrame([{"slot": e["slot_label"], "player": e["name"], "pos": e["pos"],
                             "proj": round(e["mean"], 1), "± sd": round(e["std"], 1)}
                            for e in res["opp_lineup"]])
        if not odf.empty:
            _show(odf)
    st.caption("Projections are ESPN's weekly numbers; ± sd comes from each player's weekly "
               "scoring history this season, shrunk toward a position norm. Players whose "
               "games have kicked off are locked.")


# --------------------------------------------------------------------------
def render_waivers(snap: Snapshot, tid) -> None:
    if _need(snap, tid):
        return
    _section("Best pickups for your roster")
    st.caption("Ranked by rest-of-season lineup points gained (this week counted with ESPN's "
               "weekly projection, so bye/injury fill-ins get credit), after dropping whoever "
               "costs you least.")
    with st.spinner("Scoring free agents against your roster..."):
        pk = _pickups(_snap_key(snap), snap, int(tid))
    if pk.empty:
        st.caption("No free agent improves your lineup right now.")
    else:
        _show(pk.drop(columns=["player_id"]), pos_cols=("pos", "drop_pos"),
              column_config={
                  "gain_ros": st.column_config.NumberColumn("gain ROS", help="Rest-of-season lineup points added"),
                  "gain_this_wk": st.column_config.NumberColumn("gain this wk"),
              })

    _section("League waiver activity")
    wt = waiver_table(snap)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Waiver order" + (" & FAAB" if snap.faab else "") + "**")
        if not wt.empty:
            st.dataframe(wt, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Recent adds & drops**")
        ra = recent_activity(snap)
        if ra.empty:
            st.caption("No waiver or free-agent moves in the last two weeks.")
        else:
            st.dataframe(ra, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
def _trade_result(snap: Snapshot, ev: dict, partner: int) -> None:
    color = "#3d6e4e" if ev["p_my_gain"] >= 0.7 else ("#9e862f" if ev["p_my_gain"] >= 0.55 else "#943c3c")
    _banner("Verdict", ev["verdict"].split(" — ")[0], "— " + ev["verdict"].split(" — ")[-1], color)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Your ROS lineup", f"{ev['my_ros']:+.1f} pts",
              help=f"80% range {ev['my_ros_p10']:+.0f} to {ev['my_ros_p90']:+.0f}")
    m2.metric("Chance it helps you", f"{100 * ev['p_my_gain']:.0f}%")
    m3.metric(f"{snap.team_name(partner)} ROS lineup", f"{ev['their_ros']:+.1f} pts")
    m4.metric("Accept odds (heuristic)", f"{100 * ev['accept_odds']:.0f}%")
    st.caption(f"This week: you {ev['my_week']:+.1f}, them {ev['their_week']:+.1f}.  "
               f"Raw ROS points changing hands: you send {ev['raw_given']:.0f}, "
               f"get {ev['raw_received']:.0f}.")
    if ev["my_drop"] is not None:
        st.caption(f"⚠️ You'd be over the roster limit and drop **{ev['my_drop']['name']}** (counted).")
    if ev["their_drop"] is not None:
        st.caption(f"They'd need to drop **{ev['their_drop']['name']}** (counted).")


def render_trades(snap: Snapshot, tid) -> None:
    if _need(snap, tid):
        return
    tid = int(tid)
    _section("Trades worth proposing")
    st.caption("Searched 1-for-1, 2-for-1 and 1-for-2 deals with every team. Shown only when the "
               "trade helps you with ≥60% confidence and has a realistic shot of being accepted.")
    with st.spinner("Searching trades across the league (a few seconds)..."):
        sug = _suggestions(_snap_key(snap), snap, tid)
    if sug.empty:
        st.caption("No confident win-win trades found right now.")
    else:
        st.dataframe(
            sug.drop(columns=["partner_id", "give_ids", "get_ids"]),
            use_container_width=True, hide_index=True,
            column_config={
                "your_win_%": st.column_config.ProgressColumn("helps you", format="%d%%", min_value=0, max_value=100),
                "accept_%": st.column_config.ProgressColumn("accept odds", format="%d%%", min_value=0, max_value=100),
            },
        )
        pick = st.selectbox("Load a suggestion into the evaluator", ["—"] + list(range(len(sug))),
                            format_func=lambda i: "—" if i == "—" else
                            f"{sug.iloc[i]['partner']}: give {sug.iloc[i]['you_give']} for {sug.iloc[i]['you_get']}",
                            key="trade_load")
        if pick != "—" and st.session_state.get("_trade_loaded") != pick:
            row = sug.iloc[pick]
            st.session_state["trade_partner"] = int(row["partner_id"])
            st.session_state["trade_give"] = list(row["give_ids"])
            st.session_state["trade_get"] = list(row["get_ids"])
            st.session_state["_trade_loaded"] = pick

    _section("Trade evaluator")
    st.caption("Evaluate any offer — one you received, or one you're thinking of sending.")
    others = [t for t in snap.teams if t != tid]
    if st.session_state.get("trade_partner") not in others:
        st.session_state["trade_partner"] = others[0]
    partner = st.selectbox("Trade partner", others, format_func=snap.team_name, key="trade_partner")

    def _opts(t):
        rows = sorted(snap.rosters.get(int(t), []), key=lambda r: ros_points(r, snap), reverse=True)
        return {r["player_id"]: f"{r['name']} ({r['pos']}, {r['pro_team']})" for r in rows}

    mine, theirs = _opts(tid), _opts(partner)
    for k, pool in (("trade_give", mine), ("trade_get", theirs)):
        st.session_state[k] = [p for p in st.session_state.get(k, []) if p in pool]
    c1, c2 = st.columns(2)
    with c1:
        give = st.multiselect("You give", list(mine), format_func=mine.get, key="trade_give")
    with c2:
        get = st.multiselect("You get", list(theirs), format_func=theirs.get, key="trade_get")
    if give or get:
        with st.spinner("Evaluating..."):
            ev = evaluate_trade(snap, tid, partner, give, get)
        _trade_result(snap, ev, partner)


# --------------------------------------------------------------------------
def render_outlook(snap: Snapshot, tid) -> None:
    if snap is None:
        st.info("Connect to ESPN in **Settings** to load your league for the season.")
        return
    _section("Rest-of-season outlook")
    st.caption(f"Week {snap.week} of {snap.final_week}. Remaining regular-season games simulated "
               f"4,000 times from each roster's bye- and injury-aware weekly lineup strength. "
               f"Top {snap.playoff_teams} make the playoffs (wins, then points; divisions ignored).")
    with st.spinner("Simulating the rest of the season..."):
        ol = _outlook(_snap_key(snap), snap)
    if not ol.empty:
        st.dataframe(
            ol, use_container_width=True, hide_index=True,
            column_config={
                "playoff_%": st.column_config.ProgressColumn("playoff odds", format="%d%%", min_value=0, max_value=100),
                "proj_weekly": st.column_config.NumberColumn("proj pts/wk"),
                "espn_playoff_%": st.column_config.NumberColumn("ESPN's odds", format="%d%%"),
            },
        )
    if tid is None:
        return
    _section("Your roster, rest of season")
    rows = snap.rosters.get(int(tid), [])
    df = pd.DataFrame([{
        "player": r["name"], "pos": r["pos"], "team": r["pro_team"],
        "ros_ppg": round(r["rate"], 1), "ros_pts": round(ros_points(r, snap), 0),
        "ppg_so_far": round(r["ppg"], 1) if r.get("ppg") is not None else None,
        "byes_left": ", ".join(str(w) for w in snap.remaining_weeks
                               if not snap.has_game(r["pro_team"], w)),
        "inj": "" if r["injury"] == "ACTIVE" else r["injury"].title().replace("_", " "),
    } for r in rows])
    if not df.empty:
        _show(df.sort_values("ros_pts", ascending=False))
