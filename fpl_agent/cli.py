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
import sys

from fpl_agent.agent import FPLAgent
from fpl_agent.tools import ManagerSquad, ToolError, get_my_team, set_fpl_team

BANNER = """\
⚽ FPL Agent - data-driven Fantasy Premier League assistant
Type your question (transfers, captaincy, chip timing, player stats).
Commands:
  myteam <fpl_id>  load/display your squad by FPL team ID (no LLM call, instant)
  myteam           show the currently loaded squad
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive FPL AI assistant.")
    parser.add_argument("--model", default=None, help="Override the LLM model (LiteLLM model string).")
    parser.add_argument("--verbose", action="store_true", help="Print tool calls as they happen.")
    parser.add_argument("--fpl-id", type=int, default=None, help="Load this FPL team ID as the active team at startup.")
    parser.add_argument("question", nargs="*", help="Ask a single question and exit (non-interactive).")
    args = parser.parse_args()

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
