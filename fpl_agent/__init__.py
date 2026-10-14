"""fpl_agent: a lightweight, modular AI agent for Fantasy Premier League strategy.

This package sits directly on top of the repository's existing FPL rating
pipeline (`fpl.py`) and exposes it to an LLM via structured, Pydantic-validated
tools (`fpl_agent.tools`), an agent loop with function calling
(`fpl_agent.agent`), and an interactive CLI (`fpl_agent.cli`).

No dummy/mock data is used anywhere in this package: every tool call reads
through `fpl_agent.data_loader`, which wraps `fpl.load_rated_players()` -
the same live FPL API + rating/FDR pipeline used by the rest of the repo.
"""
