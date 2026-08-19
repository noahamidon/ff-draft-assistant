"""draftkit: a mathematically-grounded fantasy football draft assistant.

Layers:
    config        -> LeagueConfig: your exact league rules (the ROI engine's inputs)
    espn_client   -> connect to ESPN, pull settings + live-ish draft board
    projections   -> load / score / blend player projections
    valuation     -> replacement levels, VORP, VONA, optimal-lineup value
    simulation    -> Monte Carlo over the rest of the draft
    draft_state   -> snake order, picks made, rosters, whose turn
    recommender   -> ties it all together into a ranked recommendation table
"""

__version__ = "0.1.0"
