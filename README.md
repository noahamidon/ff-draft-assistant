# 🏈 Fantasy Draft Assistant

A mathematically-grounded live draft assistant for ESPN fantasy football. Input
picks as they happen and get recommendations based on **value over replacement
(VORP)**, **value over next available (VONA)**, and a **Monte Carlo simulation**
of the rest of the draft — all calibrated to *your* exact league rules.

The core idea: don't rank by raw projected points. Rank by the expected value
of your final **optimal starting lineup**, integrated over how the draft is
likely to unfold. Points you can't start are worthless; points you could get
anyway off waivers aren't really yours.

---

## Hosting on GitHub / entering credentials

This repo contains **no secrets**. Your ESPN cookies are entered at runtime in
the app's **Settings tab** and held only in memory for that session — nothing
is written to disk or committed. To use the app:

1. `streamlit run app.py`
2. Open the **Settings** tab, paste your **League ID**, **SWID**, and
   **espn_s2** (the tab explains how to grab the cookies), and click
   **Connect to ESPN**. Your real league, draft order, and team names load.
3. Pull projections (sidebar) and go.

You re-enter the cookies each time you start the app — they expire anyway.

**Publishing:** push with `git` so `.gitignore` keeps `.env` and any pulled
league data out of the repo. If you upload through GitHub's web UI instead
(which ignores `.gitignore`), delete any local `.env` first. The `.env` file is
optional — only useful if you'd rather pre-fill credentials for local dev, in
which case the CLI scripts in `scripts/` can read them.

---

## Quick start

```bash
# 1. clone / unzip, then from the repo root:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. configure secrets
cp .env.example .env               # then edit .env (see "Grabbing cookies")

# 3. make sample data so the app runs immediately

# 4. confirm ESPN auth + pull your real league settings  <-- do this first
python scripts/test_connection.py

# 5. run the app
streamlit run app.py
```

If `streamlit run app.py` complains it can't find `draftkit`, run it from the
repo root (the folder containing `app.py`).

---

## The workstream

1. **Connect + confirm league rules.** `scripts/test_connection.py` logs into
   ESPN with your cookies, writes `data/raw_settings.json`, and prints a parsed
   summary of your league (teams, starters, flex, PPR, superflex). If the "weird"
   rules parse wrong, send me `data/raw_settings.json` and I'll calibrate the
   parser in `draftkit/config.py`.
2. **Projections.** Drop real projections at `data/projections.csv` (see
   [Projections](#projections)). This is the single biggest quality lever.
3. **Draft night.** `streamlit run app.py`, set your seat, click picks as they
   come in, read the recommendation panel.

---

## Grabbing cookies (private leagues)

`espn_s2` is HttpOnly, so the `document.cookie` console trick won't show it —
use the cookies panel in dev tools:

1. Log into ESPN, open your league.
2. DevTools (`F12`) → **Application** tab → **Storage → Cookies →**
   `https://fantasy.espn.com`.
3. Copy the **Value** of:
   - `SWID` — a GUID in curly braces `{....}` (keep the braces)
   - `espn_s2` — a long percent-encoded string
4. Paste both into `.env`.

Firefox: same under the **Storage** tab. Safari: enable the Develop menu first.

**Security:** these two cookies are effectively your ESPN login. They live only
in your local `.env`, which is gitignored. Never commit them. If one leaks, log
out of ESPN and back in to rotate `espn_s2`.

---

## Projections

The model reads `data/projections.csv` (falls back to
`data/projections.sample.csv`). Required columns: `name, pos, team`, plus **one**
of:

- **Pre-scored:** a `proj` (or `fpts`) column of season point totals. Easiest.
  Export consensus projections from FantasyPros and save as CSV.
- **Raw stats:** columns like `pass_yds, pass_td, rush_yds, rush_td, rec_yds,
  rec_td, receptions`. The model scores them with *your* league's scoring
  settings — the most accurate path for unusual scoring.

Optional: `adp` (used by the simulator; if absent, projection rank is used) and
`std` (risk; defaulted per position if absent). Blend multiple sources with
`draftkit.projections.blend_projections`.

---

## Architecture

```
draftkit/
  config.py       LeagueConfig + ESPN settings parser (the ROI inputs)
  espn_client.py  auth + pull settings / draft / players
  projections.py  load, (re)score to league rules, blend
  valuation.py    replacement levels, VORP, VONA, optimal-lineup value
  simulation.py   Monte Carlo over the remaining draft
  draft_state.py  snake order, picks, rosters, whose turn
  recommender.py  fold every signal into one ranked board + reasoning
app.py            Streamlit front end
scripts/          test_connection.py, pull_espn_projections.py
```

**How the recommendation is formed.** For each candidate you could take now, the
simulator plays out the rest of the draft many times — opponents pick from ADP
(softmax + noise), and future-you fills picks greedily by marginal starting-lineup
value — then scores each sim by your final optimal-lineup points. The candidate
with the best mean wins; the spread across sims is a risk read. VORP and VONA are
shown alongside as fast, interpretable cross-checks.

---

## Notes on ESPN live drafts

ESPN's standard API exposes the draft as a *recap* that populates after the fact,
not a clean live pick-by-pick feed — the live draft room is a separate real-time
system. So the app's primary live-input path is **fast manual entry** (one click
per pick), which is platform-agnostic and reliable under the clock. Pulling your
**settings** from ESPN, on the other hand, is fully reliable and automated.

---

## Roadmap / stretch goals

- Auto-ingest picks from the ESPN draft room via a browser extension (v2).
- Custom re-scoring from a full raw-stat projection source.
- Auction-draft value model (`$` bids from VORP).
- Injury/bye-aware bench valuation and boom/bust (Sharpe-style) risk toggles.
```
