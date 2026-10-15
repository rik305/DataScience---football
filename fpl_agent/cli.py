"""
Interactive terminal loop for the FPL agent.

Usage:
    python -m fpl_agent.cli

Example questions:
    "I have 1.2 ITB and want to replace Watkins. Who are the top 3 picks for GW 28-30?"
    "Should I captain Haaland or Salah this week?"
    "What's Palmer's form and fixture difficulty looking like?"
    "My FPL ID is 1234567, who should I bench this week?"
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Optional

from fpl_agent.agent import FPLAgent
from fpl_agent.tools import (
    CaptainRecommendation,
    ManagerSquad,
    PlayerStats,
    ToolError,
    TransferRecommendation,
    get_my_team,
    get_player_stats,
    recommend_captain,
    recommend_transfers,
    set_fpl_team,
)

BANNER = """\
⚽ FPL Agent - data-driven Fantasy Premier League assistant
Type your question (transfers, captaincy, chip timing, player stats).

Direct commands (call the underlying algorithm/data directly - no LLM, no
rate-limit usage, instant, fully deterministic):
  myteam <fpl_id>                        load/display your squad by FPL team ID
  myteam                                 show the currently loaded squad
  stats <player name>                    price/form/xG/xA/FDR/injury for one player
  transfers "<player>" <budget> <pos>    top 3 replacements, e.g. transfers "Watkins" 1.2 FWD
  captain <player1>, <player2>, ...      rank captain/vice-captain candidates

Other commands:
  reset            clear conversation history
  limits           show rate-limit usage
  exit / quit      leave
"""


def _print_squad(squad: ManagerSquad) -> None:
    label = squad.team_name or f"Manager {squad.fpl_id}"
    print(f"\n📋 {label} ({squad.manager_name or 'unknown manager'}) - GW{squad.gameweek}")
    print(f"   Bank: £{squad.bank_m}m | Squad value: £{squad.team_value_m}m")
    for p in squad.squad:
        tag = " (C)" if p.is_captain else (" (VC)" if p.is_vice_captain else "")
        bench = " [bench]" if p.on_bench else ""
        rank = f", rank {p.rank_score}" if p.rank_score is not None else ""
        form = f", form {p.form}" if p.form is not None else ""
        fixture = f", next: vs {p.next_fixture_opponent} (FDR {p.next_fixture_fdr})" if p.next_fixture_opponent else ""
        print(f"   {p.squad_slot:>2}. {p.position:<10} {p.player_name}{tag}{bench} - £{p.price_m}m{rank}{form}{fixture}")


def _handle_myteam_command(arg: str) -> None:
    if arg:
        try:
            fpl_id = int(arg)
        except ValueError:
            print(f"❌ '{arg}' isn't a valid FPL team ID (should be a number from your team's URL).")
            return
        result = set_fpl_team(fpl_id)
    else:
        result = get_my_team()

    if isinstance(result, ToolError):
        print(f"❌ {result.error}")
        if result.suggestions:
            print(f"   Suggestions: {', '.join(result.suggestions)}")
        return
    _print_squad(result)


def _print_tool_error(result: ToolError) -> None:
    print(f"❌ {result.error}")
    if result.suggestions:
        print(f"   Suggestions: {', '.join(result.suggestions)}")


def _print_stats(stats: PlayerStats) -> None:
    note = f" ({stats.match_note})" if stats.match_note else ""
    print(f"\n📊 {stats.player_name}{note} - {stats.position}, {stats.team}")
    print(f"   Price £{stats.price_m}m | Form {stats.form} | PPG {stats.points_per_game} | Total pts {stats.total_points}")
    print(f"   xG {stats.expected_goals} (xG90 {stats.expected_goals_per_90}) | xA {stats.expected_assists} (xA90 {stats.expected_assists_per_90})")
    print(f"   Selected by {stats.selected_by_percent}% | Status: {stats.injury_status}")
    rank = stats.rank_score if stats.rank_score is not None else "not in ranked pool (ppg<=2 or form<=1)"
    print(f"   Rank score: {rank}")
    print("   Next fixtures:")
    for f in stats.upcoming_fixtures:
        print(f"     GW{f.gameweek}: {'vs' if f.venue == 'H' else '@'} {f.opponent} (FDR {f.fdr})")


def _handle_stats_command(arg: str) -> None:
    if not arg:
        print("❌ Usage: stats <player name>  (e.g. stats Haaland)")
        return
    result = get_player_stats(arg)
    if isinstance(result, ToolError):
        _print_tool_error(result)
        return
    _print_stats(result)


def _print_transfers(rec: TransferRecommendation) -> None:
    note = f" ({rec.match_note})" if rec.match_note else ""
    print(
        f"\n🔄 Replacing {rec.current_player}{note} (£{rec.current_player_price_m}m) - "
        f"budget £{rec.available_budget_m}m, position {rec.position}"
    )
    for i, c in enumerate(rec.candidates, 1):
        print(f"   {i}. {c.player_name} ({c.team}) - £{c.price_m}m, form {c.form}, {c.points_per_game} ppg, "
              f"5-fixture FDR {c.fdr_next_5}, rank {c.rank_score}")
        print(f"      {c.reason}")


def _handle_transfers_command(arg: str, parts: Optional[list[str]] = None) -> None:
    if parts is None:
        try:
            parts = shlex.split(arg)
        except ValueError as exc:
            print(f"❌ Couldn't parse arguments: {exc}")
            return
    if len(parts) != 3:
        print('❌ Usage: transfers "<player name>" <bank_budget> <position>  (e.g. transfers "Watkins" 1.2 FWD)')
        return
    player_name, budget_str, position = parts
    try:
        budget = float(budget_str)
    except ValueError:
        print(f"❌ '{budget_str}' isn't a valid budget number (e.g. 1.2).")
        return
    try:
        result = recommend_transfers(player_name, budget, position)
    except ValueError as exc:  # invalid position, raised by the Pydantic-adjacent validator
        print(f"❌ {exc}")
        return
    if isinstance(result, ToolError):
        _print_tool_error(result)
        return
    _print_transfers(result)


def _print_captain(rec: CaptainRecommendation) -> None:
    print(f"\n🧢 Captain: {rec.captain}  |  Vice-Captain: {rec.vice_captain}")
    for c in rec.ranked_candidates:
        note = f" ({c.match_note})" if c.match_note else ""
        fixture = f", next: vs {c.next_fixture_opponent} (FDR {c.next_fixture_fdr})" if c.next_fixture_opponent else ""
        print(f"   {c.player_name}{note} ({c.team}) - score {c.expected_points_proxy}, form {c.form}{fixture}")
    for err in rec.unresolved:
        _print_tool_error(err)


def _handle_captain_command(arg: str) -> None:
    names = [n.strip() for n in arg.split(",") if n.strip()]
    if not names:
        print('❌ Usage: captain <player1>, <player2>, ...  (e.g. captain Haaland, Salah, Palmer)')
        return
    result = recommend_captain(names)
    if isinstance(result, ToolError):
        _print_tool_error(result)
        return
    _print_captain(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive FPL AI assistant.")
    parser.add_argument("--model", default=None, help="Override the LLM model (LiteLLM model string).")
    parser.add_argument("--verbose", action="store_true", help="Print tool calls as they happen.")
    parser.add_argument("--fpl-id", type=int, default=None, help="Load this FPL team ID as the active team at startup.")
    parser.add_argument(
        "question",
        nargs="*",
        help=(
            "Ask a single question and exit (non-interactive). If the first word is a direct "
            "command (myteam/stats/transfers/captain), it's run directly against the algorithm "
            "with no LLM call - e.g. `transfers Watkins 1.2 FWD`."
        ),
    )
    args = parser.parse_args()

    if args.question and args.question[0].lower() in {"myteam", "stats", "transfers", "captain"}:
        command, rest = args.question[0].lower(), args.question[1:]
        if command == "myteam":
            _handle_myteam_command(rest[0] if rest else "")
        elif command == "stats":
            _handle_stats_command(" ".join(rest))
        elif command == "transfers":
            _handle_transfers_command("", parts=rest)
        elif command == "captain":
            # Each shell arg is one candidate name (quote multi-word names), so join with commas.
            _handle_captain_command(",".join(rest))
        return

    try:
        agent = FPLAgent(model=args.model, verbose=args.verbose)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if args.fpl_id is not None:
        _handle_myteam_command(str(args.fpl_id))

    if args.question:
        print(agent.ask(" ".join(args.question)))
        return

    print(BANNER)
    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "reset":
            agent.reset()
            print("🔄 Conversation history cleared.")
            continue
        if question.lower() == "myteam" or question.lower().startswith("myteam "):
            _handle_myteam_command(question[len("myteam"):].strip())
            continue
        if question.lower() == "stats" or question.lower().startswith("stats "):
            _handle_stats_command(question[len("stats"):].strip())
            continue
        if question.lower() == "transfers" or question.lower().startswith("transfers "):
            _handle_transfers_command(question[len("transfers"):].strip())
            continue
        if question.lower() == "captain" or question.lower().startswith("captain "):
            _handle_captain_command(question[len("captain"):].strip())
            continue
        if question.lower() == "limits":
            status = agent.rate_limiter.status()
            print(
                f"📊 Rate limit status: {status['requests_last_minute']}/"
                f"{status['max_requests_per_minute']} req/min, "
                f"{status['tokens_last_minute']}/{status['max_tokens_per_minute']} tokens/min, "
                f"{status['requests_today']}/{status['max_requests_per_day']} req/day "
                f"({status['remaining_today']} remaining today)."
            )
            continue

        answer = agent.ask(question)
        print(f"\n🤖 Agent: {answer}")


if __name__ == "__main__":
    main()
