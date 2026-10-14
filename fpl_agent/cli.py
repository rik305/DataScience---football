"""
Interactive terminal loop for the FPL agent.

Usage:
    python -m fpl_agent.cli

Example questions:
    "I have 1.2 ITB and want to replace Watkins. Who are the top 3 picks for GW 28-30?"
    "Should I captain Haaland or Salah this week?"
    "What's Palmer's form and fixture difficulty looking like?"
"""

from __future__ import annotations

import argparse
import sys

from fpl_agent.agent import FPLAgent

BANNER = """\
⚽ FPL Agent - data-driven Fantasy Premier League assistant
Type your question (transfers, captaincy, chip timing, player stats).
Commands: 'reset' to clear history, 'limits' to show rate-limit usage, 'exit'/'quit' to leave.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive FPL AI assistant.")
    parser.add_argument("--model", default=None, help="Override the LLM model (LiteLLM model string).")
    parser.add_argument("--verbose", action="store_true", help="Print tool calls as they happen.")
    parser.add_argument("question", nargs="*", help="Ask a single question and exit (non-interactive).")
    args = parser.parse_args()

    try:
        agent = FPLAgent(model=args.model, verbose=args.verbose)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

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
        if question.lower() == "limits":
            status = agent.rate_limiter.status()
            print(
                f"📊 Rate limit status: {status['requests_last_minute']}/"
                f"{status['max_requests_per_minute']} req/min, "
                f"{status['requests_today']}/{status['max_requests_per_day']} req/day "
                f"({status['remaining_today']} remaining today)."
            )
            continue

        answer = agent.ask(question)
        print(f"\n🤖 Agent: {answer}")


if __name__ == "__main__":
    main()
