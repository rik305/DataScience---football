"""
Data access layer for the FPL agent.

This module is the *only* place the agent touches real data. It wraps the
repository's existing rating pipeline in `fpl.py` (`fpl.load_rated_players`),
adds a small TTL cache so repeated tool calls within one conversation don't
re-hit the live FPL API, and provides fuzzy player-name resolution so the
agent can gracefully handle typos / partial names.

No mock or synthetic data is introduced here - everything is sourced from
the live FPL API via `fpl.py`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from rapidfuzz import fuzz

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dependency
    import difflib

    _HAVE_RAPIDFUZZ = False

import fpl

CACHE_TTL_SECONDS = int(os.getenv("FPL_AGENT_CACHE_TTL_SECONDS", "600"))

POSITION_NAMES = {
    fpl.GOALKEEPER: "Goalkeeper",
    fpl.DEFENDER: "Defender",
    fpl.MIDFIELDER: "Midfielder",
    fpl.FORWARD: "Forward",
}

POSITION_ALIASES = {
    "gk": fpl.GOALKEEPER,
    "goalkeeper": fpl.GOALKEEPER,
    "goalkeepers": fpl.GOALKEEPER,
    "def": fpl.DEFENDER,
    "defender": fpl.DEFENDER,
    "defenders": fpl.DEFENDER,
    "mid": fpl.MIDFIELDER,
    "midfielder": fpl.MIDFIELDER,
    "midfielders": fpl.MIDFIELDER,
    "fwd": fpl.FORWARD,
    "forward": fpl.FORWARD,
    "forwards": fpl.FORWARD,
    "striker": fpl.FORWARD,
    "strikers": fpl.FORWARD,
}

# FPL `status` field codes -> human readable injury/availability status.
STATUS_LABELS = {
    "a": "Available",
    "d": "Doubtful",
    "i": "Injured",
    "s": "Suspended",
    "u": "Unavailable",
    "n": "Not available (on loan/left club)",
}


class PlayerNotFoundError(Exception):
    """Raised when a player name cannot be resolved, even with fuzzy matching."""

    def __init__(self, query: str, suggestions: list[str]):
        self.query = query
        self.suggestions = suggestions
        msg = f"Could not find a player matching '{query}'."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        super().__init__(msg)


def resolve_position(position: str) -> int:
    """Map a free-text position string ('DEF', 'midfielder', ...) to the FPL element_type int."""
    key = position.strip().lower()
    if key in POSITION_ALIASES:
        return POSITION_ALIASES[key]
    raise ValueError(
        f"Unknown position '{position}'. Expected one of: "
        f"GK, DEF, MID, FWD (or their full names)."
    )


def injury_status(row: pd.Series) -> str:
    """Derive a human-readable availability/injury status from FPL `status`/`news` fields."""
    status_code = str(row.get("status", "a") or "a").lower()
    label = STATUS_LABELS.get(status_code, "Unknown")
    chance = row.get("chance_of_playing_this_round")
    news = row.get("news") or ""

    parts = [label]
    if chance is not None and not pd.isna(chance) and status_code != "a":
        parts.append(f"{int(chance)}% chance of playing this round")
    if news:
        parts.append(news)
    return " - ".join(parts)


@dataclass
class PlayerMatch:
    """Result of `FPLDataStore.find_player`: the resolved row plus match confidence info."""

    name: str
    row: pd.Series
    exact: bool
    alternatives: list[str]


@dataclass
class _CacheEntry:
    data: dict
    fetched_at: float


class FPLDataStore:
    """TTL-cached accessor around `fpl.load_rated_players()`."""

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._cache: Optional[_CacheEntry] = None

    def get_universe(self, force_refresh: bool = False) -> dict:
        now = time.time()
        if (
            force_refresh
            or self._cache is None
            or (now - self._cache.fetched_at) > self.ttl_seconds
        ):
            data = fpl.load_rated_players()
            self._cache = _CacheEntry(data=data, fetched_at=now)
        return self._cache.data

    def players(self, force_refresh: bool = False) -> pd.DataFrame:
        """The filtered, ranked pool (ppg>2, form>1) used for squad-building/transfers."""
        return self.get_universe(force_refresh=force_refresh)["players"]

    def players_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """The full, unfiltered player universe - used for name lookups so out-of-form
        or currently-benched players can still be found (no silent mismatches)."""
        return self.get_universe(force_refresh=force_refresh)["players_all"]

    def rank_score(self, player_name: str) -> Optional[int]:
        """Return the existing multi-criteria rank score ('count') if the player is in
        the ranked pool (ppg>2, form>1), else None."""
        ranked = self.players()
        if player_name in ranked.index:
            return int(ranked.loc[player_name]["count"])
        return None

    def teams(self, force_refresh: bool = False) -> pd.DataFrame:
        return self.get_universe(force_refresh=force_refresh)["teams"]

    def team_names(self, force_refresh: bool = False) -> list:
        return self.get_universe(force_refresh=force_refresh)["team_names"]

    def fixtures(self, force_refresh: bool = False) -> pd.DataFrame:
        return self.get_universe(force_refresh=force_refresh)["fixtures"]

    def by_position(self, element_type: int, force_refresh: bool = False) -> pd.DataFrame:
        return self.get_universe(force_refresh=force_refresh)["by_position"][element_type]

    def find_player(self, query: str, limit: int = 5) -> "PlayerMatch":
        """
        Fuzzy-resolve a free-text player name to the closest matching row in
        the *full* player universe (not just the currently in-form/ranked pool,
        so out-of-form or injured players are still resolvable). Matches against
        both full names and surnames (since users often say "Watkins" rather
        than "Ollie Watkins"). Raises PlayerNotFoundError (with suggestions) if
        no good match is found.

        Returns a `PlayerMatch` so callers (and ultimately the LLM) can tell
        when a match was exact vs. a fuzzy best-guess, plus see runner-up
        alternatives - important because a confidently-wrong fuzzy match
        (e.g. a short/common surname) can otherwise look identical to a
        correct one.
        """
        players = self.players_all()
        names = list(players.index)

        query_norm = query.strip()
        lower_map = {n.lower(): n for n in names}
        if query_norm.lower() in lower_map:
            matched = lower_map[query_norm.lower()]
            return PlayerMatch(name=matched, row=players.loc[matched], exact=True, alternatives=[])

        # Exact, unambiguous surname match (e.g. "Haaland") counts as exact too.
        surname_map: dict[str, list[str]] = {}
        for n in names:
            parts = n.split()
            if parts:
                surname_map.setdefault(parts[-1].lower(), []).append(n)
        surname_hits = surname_map.get(query_norm.lower())
        if surname_hits and len(surname_hits) == 1:
            matched = surname_hits[0]
            return PlayerMatch(name=matched, row=players.loc[matched], exact=True, alternatives=[])

        candidates = self._rank_candidates(query_norm, names, limit=limit)
        if not candidates or candidates[0][1] < 70:
            suggestions = [name for name, score in candidates if score >= 45]
            raise PlayerNotFoundError(query, suggestions)

        best_name, best_score = candidates[0]
        # Only surface runner-ups that are genuinely close to the best match (not just
        # anything clearing the floor threshold), so a clear winner like "Haaland" -> a
        # single real player doesn't get cluttered with unrelated low-relevance names.
        alternatives = [
            name for name, score in candidates[1:]
            if score >= 60 and (best_score - score) <= 10
        ]
        return PlayerMatch(name=best_name, row=players.loc[best_name], exact=False, alternatives=alternatives)

    @staticmethod
    def _score(query: str, candidate: str) -> float:
        # Plain Levenshtein-style ratio (not partial/token-based) deliberately: partial-ratio
        # scorers like WRatio can give deceptively high scores for short candidate strings that
        # happen to share a few characters with a much longer, unrelated query (e.g. a 4-letter
        # surname "hiding" inside a nonsense/garbled query), which would silently misidentify a
        # player instead of reporting "not found".
        if _HAVE_RAPIDFUZZ:
            return fuzz.ratio(query, candidate)
        return difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio() * 100

    def _rank_candidates(self, query: str, names: list[str], limit: int) -> list[tuple[str, float]]:
        scored = []
        for name in names:
            surname = name.split()[-1] if name.split() else name
            score = max(self._score(query, name), self._score(query, surname))
            scored.append((name, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


# Module-level singleton used by the tools layer. Kept as a simple default
# instance so `fpl_agent.tools` can import one shared, cached data store.
store = FPLDataStore()
