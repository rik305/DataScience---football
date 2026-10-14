"""
Agent loop for the FPL assistant.

Uses LiteLLM so the same code path supports OpenAI (`OPENAI_API_KEY`),
Google Gemini (`GEMINI_API_KEY`), and Groq (`GROQ_API_KEY`) function calling
without branching logic per-provider. API keys are loaded via python-dotenv
from a local `.env` file (see `.env.example`).

The agent is given a concise system prompt establishing it as a data-driven
FPL strategist, then runs a standard "tool calling" loop: the LLM picks a
tool + arguments, we validate the arguments with the tool's Pydantic schema,
execute it against live data (via `fpl_agent.tools`), and feed the structured
result back to the LLM until it produces a final natural-language answer.

Tool execution errors (including player-not-found / invalid input) are
caught and returned to the model as structured error payloads rather than
raised, so the LLM can recover (e.g. retry with a suggested player name)
instead of crashing the conversation.

Every LLM call passes through `fpl_agent.rate_limiter.LocalRateLimiter`
first - a hard, client-side RPM/RPD/TPM ceiling enforced *before* any network
call, so the agent can never run away and exceed a free tier (Groq's
included) regardless of what the provider's own rate limits are set to.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from fpl_agent.rate_limiter import LocalRateLimiter, RateLimitExceeded
from fpl_agent.tools import TOOL_REGISTRY

load_dotenv()

SYSTEM_PROMPT = """\
You are an elite, data-driven Fantasy Premier League (FPL) strategist.

You have access to tools that read live FPL data: player prices, form, \
points-per-game, expected goals/assists (xG/xA), fixture difficulty ratings \
(FDR) for upcoming gameweeks, and injury/availability status.

Rules:
- Always call a tool to fetch real numbers before giving transfer, captaincy, \
  or chip-timing advice. Never invent stats.
- Cite the specific numbers you used (xG, xA, FDR, form, price, rank score) \
  in your final answer so the manager can see your reasoning.
- If a tool reports a player could not be found, try again using one of the \
  suggested names before giving up, or ask the user to clarify.
- Be concise and decisive: give a clear recommendation, then 2-4 bullet \
  points of supporting evidence.
- If asked about chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain), \
  reason using the fixture difficulty (FDR) data available via the tools - \
  favour chips around fixture swings/double gameweeks evident in the FDR data, \
  and say so explicitly when the data is insufficient to be certain.
"""

MAX_TOOL_ITERATIONS = 6


def _default_model() -> str:
    override = os.getenv("FPL_AGENT_MODEL")
    if override:
        return override
    if os.getenv("GROQ_API_KEY"):
        return "groq/openai/gpt-oss-20b"
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-1.5-flash"
    raise RuntimeError(
        "No API key found. Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in your "
        "environment (or a .env file) before running the agent."
    )


def _tool_definitions() -> list[dict[str, Any]]:
    """Build OpenAI/LiteLLM-style function-calling tool definitions from TOOL_REGISTRY."""
    defs = []
    for name, spec in TOOL_REGISTRY.items():
        schema: type[BaseModel] = spec["schema"]
        parameters = schema.model_json_schema()
        parameters.pop("title", None)
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": parameters,
                },
            }
        )
    return defs


def _execute_tool(name: str, raw_arguments: str) -> str:
    """Validate arguments against the tool's Pydantic schema, run it, and return a JSON string."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})

    try:
        kwargs = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Malformed arguments for '{name}': {exc}"})

    try:
        validated = spec["schema"].model_validate(kwargs)
    except ValidationError as exc:
        return json.dumps({"error": f"Invalid arguments for '{name}': {exc.errors()}"})

    try:
        result = spec["fn"](**validated.model_dump())
    except Exception as exc:  # noqa: BLE001 - surface any unexpected tool failure to the LLM, don't crash
        return json.dumps({"error": f"Tool '{name}' failed: {exc}"})

    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result)


class FPLAgent:
    """A minimal, provider-agnostic tool-calling agent for FPL questions."""

    def __init__(
        self,
        model: Optional[str] = None,
        verbose: bool = False,
        rate_limiter: Optional[LocalRateLimiter] = None,
    ):
        self.model = model or _default_model()
        self.verbose = verbose
        self.tools = _tool_definitions()
        self.history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Hard local ceiling on LLM calls (RPM/RPD/TPM) - enforced before every request,
        # regardless of provider, so usage can never run away past free-tier limits.
        self.rate_limiter = rate_limiter or LocalRateLimiter()

    def reset(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, question: str) -> str:
        """Send a user question through the tool-calling loop and return the final answer."""
        import litellm  # imported lazily so importing this module doesn't require network/key setup

        self.history.append({"role": "user", "content": question})

        for _ in range(MAX_TOOL_ITERATIONS):
            estimated_tokens = sum(
                self.rate_limiter.estimate_tokens(str(m.get("content") or "")) for m in self.history
            )
            try:
                self.rate_limiter.check_and_record(estimated_tokens=estimated_tokens)
            except RateLimitExceeded as exc:
                return f"🛑 {exc}"

            try:
                response = litellm.completion(
                    model=self.model,
                    messages=self.history,
                    tools=self.tools,
                    tool_choice="auto",
                )
            except Exception as exc:  # noqa: BLE001
                return f"⚠️ Agent error while calling the LLM ({self.model}): {exc}"

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            self.history.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    **({"tool_calls": [tc.model_dump() for tc in tool_calls]} if tool_calls else {}),
                }
            )

            if not tool_calls:
                return message.content or ""

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments
                if self.verbose:
                    print(f"  🔧 calling {fn_name}({fn_args})")
                tool_result = _execute_tool(fn_name, fn_args)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": tool_result,
                    }
                )

        return (
            "⚠️ Reached the maximum number of tool calls without a final answer. "
            "Try rephrasing your question."
        )
