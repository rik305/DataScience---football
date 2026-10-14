"""
Structured tools the FPL agent can call.

Each tool has a Pydantic input schema (for validation + for generating the
JSON schema passed to the LLM as a function-calling "tool" definition) and a
plain Python function that executes against live data via `data_loader`.

These wrap the repository's existing rating pipeline (`fpl.py`) - there is no
separate/duplicate scoring logic here, and no mock data.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

import fpl
from fpl_agent.data_loader import (
    POSITION_NAMES,
    ManagerNotFoundError,
    PlayerNotFoundError,
    injury_status,
    resolve_position,
    store,
)


def _match_note(match) -> Optional[str]:
    """Build a human-readable note when a player name was fuzzy-matched rather than exact,
    so the agent can flag ambiguity instead of silently assuming the guess is correct."""
    if match.exact:
        return None
    note = f"'{match.name}' was the closest match found (not an exact name match)."
    if match.alternatives:
        note += f" Other possibilities: {', '.join(match.alternatives)}."
    return note


class ToolError(BaseModel):
    """Structured error payload returned to the agent instead of raising, so the
    LLM can see what went wrong and recover (e.g. retry with a suggested name)."""

    error: str
    suggestions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# get_player_stats
# ---------------------------------------------------------------------------

class GetPlayerStatsInput(BaseModel):
    player_name: str = Field(..., description="Full or partial player name, e.g. 'Watkins' or 'Ollie Watkins'.")


class FixtureDifficulty(BaseModel):
    gameweek: Optional[int]
    opponent: str
    venue: Literal["H", "A"]
    fdr: float


class PlayerStats(BaseModel):
    player_name: str
    match_note: Optional[str] = Field(
        None,
        description="Set when the name was fuzzy-matched (not exact); flags other plausible candidates.",
    )
    team: str
    position: str
    price_m: float
    form: float
    points_per_game: float
    total_points: int
    expected_goals: float
    expected_assists: float
    expected_goals_per_90: float
    expected_assists_per_90: float
    selected_by_percent: float
    injury_status: str
    upcoming_fixtures: list[FixtureDifficulty]
    rank_score: Optional[int] = Field(
        None,
        description=(
            "Lower is better; the repo's existing multi-criteria rank score. "
            "Null if the player currently falls outside the ranked pool (ppg<=2 or form<=1)."
        ),
    )


def get_player_stats(player_name: str) -> PlayerStats | ToolError:
    """Return current price, form, xG/xA, next-3-fixture FDR, and injury status for a player."""
    try:
        match = store.find_player(player_name)
    except PlayerNotFoundError as exc:
        return ToolError(error=str(exc), suggestions=exc.suggestions)

    matched_name, row = match.name, match.row

    universe = store.get_universe()
    fixtures_df = universe["fixtures"]
    team_names = universe["team_names"]

    breakdown = fpl.player_fixture_breakdown(row, fixtures_df, team_names, lookahead=3)

    return PlayerStats(
        player_name=matched_name,
        match_note=_match_note(match),
        team=team_names[row["team"]] if row["team"] < len(team_names) else str(row["team"]),
        position=POSITION_NAMES.get(row["element_type"], "Unknown"),
        price_m=round(row["now_cost"] / 10, 1),
        form=float(row["form"]),
        points_per_game=float(row["ppg"]),
        total_points=int(row["total_points"]),
        expected_goals=float(row.get("expected_goals", 0.0)),
        expected_assists=float(row.get("expected_assists", 0.0)),
        expected_goals_per_90=float(row.get("expected_goals_per_90", 0.0)),
        expected_assists_per_90=float(row.get("expected_assists_per_90", 0.0)),
        selected_by_percent=float(row.get("selected_by_percent", 0.0)),
        injury_status=injury_status(row),
        upcoming_fixtures=[FixtureDifficulty(**f) for f in breakdown],
        rank_score=store.rank_score(matched_name),
    )


# ---------------------------------------------------------------------------
# recommend_transfers
# ---------------------------------------------------------------------------

class RecommendTransfersInput(BaseModel):
    current_player_name: str = Field(..., description="The player being replaced/sold.")
    bank_budget: float = Field(..., ge=0, description="Money in the bank (ITB), in millions, e.g. 1.2.")
    position: str = Field(..., description="Position of the replacement: GK, DEF, MID, or FWD.")

    @field_validator("position")
    @classmethod
    def _validate_position(cls, v: str) -> str:
        resolve_position(v)  # raises ValueError if invalid, surfaced to the agent
        return v


class TransferCandidate(BaseModel):
    player_name: str
    team: str
    price_m: float
    form: float
    points_per_game: float
    fdr_next_5: float
    rank_score: int
    reason: str


class TransferRecommendation(BaseModel):
    current_player: str
    match_note: Optional[str] = None
    current_player_price_m: float
    available_budget_m: float
    position: str
    candidates: list[TransferCandidate]


def recommend_transfers(
    current_player_name: str, bank_budget: float, position: str
) -> TransferRecommendation | ToolError:
    """
    Use the repo's existing rank-score pipeline to find the top 3 replacements
    for `current_player_name` affordable within (selling price + bank_budget).
    """
    try:
        match = store.find_player(current_player_name)
    except PlayerNotFoundError as exc:
        return ToolError(error=str(exc), suggestions=exc.suggestions)

    matched_current, current_row = match.name, match.row

    try:
        element_type = resolve_position(position)
    except ValueError as exc:
        return ToolError(error=str(exc))

    selling_price_m = round(current_row["now_cost"] / 10, 1)
    max_budget_m = selling_price_m + bank_budget
    max_cost = round(max_budget_m * 10)

    pool = store.by_position(element_type)
    affordable = pool[(pool["now_cost"] <= max_cost) & (pool.index != matched_current)]

    if affordable.empty:
        return ToolError(
            error=(
                f"No affordable {POSITION_NAMES.get(element_type)} found within "
                f"£{max_budget_m:.1f}m (selling {matched_current} for £{selling_price_m:.1f}m "
                f"+ £{bank_budget:.1f}m ITB)."
            )
        )

    top3 = affordable.sort_values(by="count", ascending=True).head(3)

    candidates = []
    for name, row in top3.iterrows():
        candidates.append(
            TransferCandidate(
                player_name=name,
                team=store.team_names()[row["team"]],
                price_m=round(row["now_cost"] / 10, 1),
                form=float(row["form"]),
                points_per_game=float(row["ppg"]),
                fdr_next_5=round(float(row["fdr"]), 2),
                rank_score=int(row["count"]),
                reason=(
                    f"Form {row['form']:.1f}, {row['ppg']:.1f} pts/game, "
                    f"xG90 {row.get('expected_goals_per_90', 0):.2f}, "
                    f"xA90 {row.get('expected_assists_per_90', 0):.2f}, "
                    f"5-fixture FDR {row['fdr']:.1f} - fits within £{max_budget_m:.1f}m budget."
                ),
            )
        )

    return TransferRecommendation(
        current_player=matched_current,
        match_note=_match_note(match),
        current_player_price_m=selling_price_m,
        available_budget_m=round(max_budget_m, 1),
        position=POSITION_NAMES.get(element_type, position),
        candidates=candidates,
    )


# ---------------------------------------------------------------------------
# recommend_captain
# ---------------------------------------------------------------------------

class RecommendCaptainInput(BaseModel):
    candidate_names: list[str] = Field(..., min_length=1, description="Candidate player names to compare.")


class CaptainCandidateScore(BaseModel):
    player_name: str
    match_note: Optional[str] = None
    team: str
    expected_points_proxy: float = Field(
        ..., description="ppg + (xG90 + xA90) adjusted by next-fixture difficulty; higher is better."
    )
    next_fixture_opponent: Optional[str]
    next_fixture_fdr: Optional[float]
    form: float


class CaptainRecommendation(BaseModel):
    captain: str
    vice_captain: str
    ranked_candidates: list[CaptainCandidateScore]
    unresolved: list[ToolError] = Field(default_factory=list)


def recommend_captain(candidate_names: list[str]) -> CaptainRecommendation | ToolError:
    """
    Compare candidates by an expected-points proxy (form + xG/xA, adjusted for
    next-fixture difficulty) to rank Captain and Vice-Captain picks.
    """
    if len(candidate_names) < 1:
        return ToolError(error="Provide at least one candidate name.")

    universe = store.get_universe()
    fixtures_df = universe["fixtures"]
    team_names = universe["team_names"]

    scored: list[CaptainCandidateScore] = []
    unresolved: list[ToolError] = []

    for name in candidate_names:
        try:
            match = store.find_player(name)
        except PlayerNotFoundError as exc:
            unresolved.append(ToolError(error=str(exc), suggestions=exc.suggestions))
            continue

        matched_name, row = match.name, match.row
        breakdown = fpl.player_fixture_breakdown(row, fixtures_df, team_names, lookahead=1)
        next_fixture = breakdown[0] if breakdown else None

        xgi90 = float(row.get("expected_goals_per_90", 0.0)) + float(row.get("expected_assists_per_90", 0.0))
        ppg = float(row["ppg"])
        # Difficulty adjustment: positive fdr (favourable fixture per this repo's convention,
        # since attacking_fdr = own_attack - opp_defense) nudges the proxy up slightly.
        fdr_adjustment = (next_fixture["fdr"] / 50.0) if next_fixture else 0.0
        expected_points_proxy = round(ppg + xgi90 * 2 + fdr_adjustment, 3)

        scored.append(
            CaptainCandidateScore(
                player_name=matched_name,
                match_note=_match_note(match),
                team=team_names[row["team"]],
                expected_points_proxy=expected_points_proxy,
                next_fixture_opponent=next_fixture["opponent"] if next_fixture else None,
                next_fixture_fdr=next_fixture["fdr"] if next_fixture else None,
                form=float(row["form"]),
            )
        )

    if not scored:
        return ToolError(
            error="None of the candidate names could be resolved.",
            suggestions=[s for u in unresolved for s in u.suggestions],
        )

    scored.sort(key=lambda c: c.expected_points_proxy, reverse=True)
    captain = scored[0].player_name
    vice_captain = scored[1].player_name if len(scored) > 1 else scored[0].player_name

    return CaptainRecommendation(
        captain=captain,
        vice_captain=vice_captain,
        ranked_candidates=scored,
        unresolved=unresolved,
    )


# ---------------------------------------------------------------------------
# set_fpl_team / get_my_team
# ---------------------------------------------------------------------------

class SetFPLTeamInput(BaseModel):
    fpl_id: int = Field(
        ...,
        gt=0,
        description=(
            "The user's public FPL manager/team ID (the number in the URL when viewing "
            "their team on the official FPL site, e.g. fantasy.premierleague.com/entry/1234567/...)."
        ),
    )


class SquadPlayer(BaseModel):
    player_name: str
    position: str
    team: str
    price_m: float
    squad_slot: int = Field(..., description="1-11 = starting XI (ordered GK/DEF/MID/FWD), 12-15 = bench.")
    on_bench: bool
    is_captain: bool
    is_vice_captain: bool
    form: Optional[float] = None
    points_per_game: Optional[float] = None
    next_fixture_opponent: Optional[str] = None
    next_fixture_fdr: Optional[float] = None
    rank_score: Optional[int] = Field(
        None, description="The repo's existing multi-criteria rank score, if the player is in the ranked pool."
    )


class ManagerSquad(BaseModel):
    fpl_id: int
    team_name: Optional[str]
    manager_name: Optional[str]
    gameweek: int
    bank_m: float
    team_value_m: float
    squad: list[SquadPlayer] = Field(..., description="All 15 squad players, list ordered by squad slot.")


def _build_squad_player(player: dict, players_all: pd.DataFrame, fixtures_df, team_names: list) -> SquadPlayer:
    """
    Build a SquadPlayer, enriching each pick with form/points-per-game and their
    single next-fixture FDR. This is deliberately included inline (rather than
    requiring a separate get_player_stats call per squad member) so a single
    set_fpl_team/get_my_team call carries enough signal for common questions
    ("who should I bench", "who's out of form") - keeping tool round-trips (and
    therefore LLM token/request usage against the local rate limiter) low.
    """
    form = ppg = None
    opponent = fdr = None
    name = player["player_name"]
    if name in players_all.index:
        row = players_all.loc[name]
        form = float(row.get("form", 0.0))
        ppg = float(row.get("ppg", 0.0))
        breakdown = fpl.player_fixture_breakdown(row, fixtures_df, team_names, lookahead=1)
        if breakdown:
            opponent = breakdown[0]["opponent"]
            fdr = breakdown[0]["fdr"]

    return SquadPlayer(
        player_name=name,
        position=POSITION_NAMES.get(player["element_type"], "Unknown"),
        team=player["team"],
        price_m=float(player["price_m"]),
        squad_slot=int(player["squad_slot"]),
        on_bench=bool(player["on_bench"]),
        is_captain=bool(player["is_captain"]),
        is_vice_captain=bool(player["is_vice_captain"]),
        form=form,
        points_per_game=ppg,
        next_fixture_opponent=opponent,
        next_fixture_fdr=fdr,
        rank_score=store.rank_score(name),
    )


def _build_manager_squad_response(data: dict) -> ManagerSquad:
    universe = store.get_universe()
    players_all = universe["players_all"]
    fixtures_df = universe["fixtures"]
    team_names = universe["team_names"]

    return ManagerSquad(
        fpl_id=data["manager_id"],
        team_name=data.get("team_name"),
        manager_name=data.get("manager_name") or None,
        gameweek=data["gameweek"],
        bank_m=data["bank_m"],
        team_value_m=data["team_value_m"],
        squad=[_build_squad_player(p, players_all, fixtures_df, team_names) for p in data["squad"]],
    )


def set_fpl_team(fpl_id: int) -> ManagerSquad | ToolError:
    """
    Look up a manager's live FPL squad by their public team ID and store it as the
    active team for this session (as a list of 15 player records), so later
    questions can reference "my team" without re-supplying the ID.
    """
    try:
        data = store.get_manager_squad(fpl_id)
    except ManagerNotFoundError as exc:
        return ToolError(error=str(exc))

    return _build_manager_squad_response(data)


class GetMyTeamInput(BaseModel):
    """No arguments - recalls whichever FPL team was most recently set via set_fpl_team."""


def get_my_team() -> ManagerSquad | ToolError:
    """Return the currently active manager's squad (set earlier via set_fpl_team this session)."""
    try:
        data = store.active_squad()
    except ManagerNotFoundError as exc:
        return ToolError(error=str(exc))

    return _build_manager_squad_response(data)


# ---------------------------------------------------------------------------
# Tool registry used by the agent loop (name -> (input_schema, fn))
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_player_stats": {
        "schema": GetPlayerStatsInput,
        "fn": get_player_stats,
        "description": (
            "Get a player's current price, form, points per game, xG/xA, "
            "upcoming 3-fixture difficulty (FDR), and injury/availability status."
        ),
    },
    "recommend_transfers": {
        "schema": RecommendTransfersInput,
        "fn": recommend_transfers,
        "description": (
            "Recommend the top 3 replacement players for a given position, affordable "
            "within the player's selling price plus remaining bank budget (ITB)."
        ),
    },
    "recommend_captain": {
        "schema": RecommendCaptainInput,
        "fn": recommend_captain,
        "description": (
            "Compare a list of candidate players by expected points and upcoming fixture "
            "difficulty to recommend a Captain and Vice-Captain."
        ),
    },
    "set_fpl_team": {
        "schema": SetFPLTeamInput,
        "fn": set_fpl_team,
        "description": (
            "Load a manager's live current 15-player FPL squad by their public FPL team ID "
            "(from their team URL on the official site) and set it as the active team for "
            "this conversation, so later questions can refer to 'my team' / 'my squad'."
        ),
    },
    "get_my_team": {
        "schema": GetMyTeamInput,
        "fn": get_my_team,
        "description": (
            "Recall the squad most recently loaded via set_fpl_team for this conversation "
            "(the user's current 15 players, prices, bank, captain/vice-captain, bench)."
        ),
    },
}
