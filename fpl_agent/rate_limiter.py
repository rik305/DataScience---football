"""
Local, provider-independent rate limiting for LLM calls.

This is a client-side safety net that enforces conservative request-per-minute
(RPM), request-per-day (RPD), and token-per-minute (TPM) ceilings *before* any
call reaches the LLM provider. It fails closed: if a call would exceed a
configured limit, it raises `RateLimitExceeded` instead of making the network
call at all.

Why this exists for Groq specifically: Groq's free tier does not require a
card on file and does not do overage billing - it simply rejects requests
past the free-tier limits with a 429. The real risk this guards against is:
(a) the account later being upgraded to a paid/on-demand plan (at which point
exceeding limits could mean real charges), and (b) simply hammering the API
hard enough to get temporarily blocked. Enforcing a hard ceiling *locally*,
well below Groq's published free-tier numbers, means the agent can never
accidentally cross that line even if the account's billing status changes
later or Groq's published limits change.

Defaults are intentionally conservative. Check current limits for your model
at https://console.groq.com/settings/limits and tune via the environment
variables below if needed - these defaults are deliberately cautious, not a
promise that they exactly mirror Groq's current published limits.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

# Where the persistent daily counter is stored, so the daily cap survives
# process restarts (e.g. re-running `python -m fpl_agent.cli`). Kept inside
# fpl_agent/ and git-ignored - it's local runtime state, not source.
STATE_FILE = Path(os.getenv("FPL_AGENT_STATE_FILE", str(Path(__file__).parent / ".rate_limit_state.json")))


class RateLimitExceeded(Exception):
    """Raised when a call would exceed a configured local rate limit (request refused, not sent)."""


@dataclass
class RateLimitConfig:
    max_requests_per_minute: int
    max_requests_per_day: int
    max_tokens_per_minute: int

    @classmethod
    def from_env(cls, prefix: str = "FPL_AGENT") -> "RateLimitConfig":
        """
        Conservative defaults, comfortably below Groq's free-tier limits for
        typical models (which are commonly in the ~30 RPM / ~1,000-14,400 RPD
        / several-thousand-to-tens-of-thousands TPM range depending on model).
        Override via env vars if you've confirmed higher headroom for your
        model at https://console.groq.com/settings/limits.
        """
        return cls(
            max_requests_per_minute=int(os.getenv(f"{prefix}_MAX_RPM", "15")),
            max_requests_per_day=int(os.getenv(f"{prefix}_MAX_RPD", "250")),
            max_tokens_per_minute=int(os.getenv(f"{prefix}_MAX_TPM", "4000")),
        )


def _now() -> float:
    return time.time()


def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


class LocalRateLimiter:
    """
    Enforces, per process (RPM/TPM via in-memory sliding windows) and across
    restarts (RPD via a small on-disk counter), a hard ceiling on LLM calls.

    This does NOT rely on the provider's own rate-limit responses/headers -
    it blocks locally, before the network call is made.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None, state_file: Path = STATE_FILE):
        self.config = config or RateLimitConfig.from_env()
        self.state_file = state_file
        self._lock = Lock()
        self._request_timestamps: list[float] = []
        self._token_events: list[tuple[float, int]] = []
        self._daily_state = self._load_daily_state()

    # -- persistence ---------------------------------------------------------

    def _load_daily_state(self) -> dict:
        data: dict = {}
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        if data.get("date") != _today_key():
            data = {"date": _today_key(), "count": 0}
        return data

    def _save_daily_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self._daily_state))
        except OSError:
            pass  # best-effort; the in-process RPM/TPM checks still apply either way

    # -- estimation ------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars/token). Deliberately conservative (rounds up)."""
        return max(1, (len(text) + 3) // 4)

    # -- core check --------------------------------------------------------------

    def check_and_record(self, estimated_tokens: int = 0) -> None:
        """
        Call immediately before making an LLM request. Raises
        `RateLimitExceeded` (without recording/consuming budget) if any
        configured ceiling would be breached; otherwise records the request
        and lets it proceed.
        """
        with self._lock:
            now = _now()

            if self._daily_state.get("date") != _today_key():
                self._daily_state = {"date": _today_key(), "count": 0}

            self._request_timestamps = [t for t in self._request_timestamps if now - t < 60]
            self._token_events = [(t, n) for t, n in self._token_events if now - t < 60]

            if self._daily_state["count"] >= self.config.max_requests_per_day:
                raise RateLimitExceeded(
                    f"Daily request cap reached ({self.config.max_requests_per_day} requests/day, "
                    "local safety limit - not a provider error). Try again tomorrow, or raise "
                    "FPL_AGENT_MAX_RPD if you've confirmed it's safe for your account/tier."
                )

            if len(self._request_timestamps) >= self.config.max_requests_per_minute:
                raise RateLimitExceeded(
                    f"Per-minute request cap reached ({self.config.max_requests_per_minute} "
                    "requests/min, local safety limit). Please wait a moment and retry."
                )

            tokens_in_window = sum(n for _, n in self._token_events)
            if tokens_in_window + estimated_tokens > self.config.max_tokens_per_minute:
                raise RateLimitExceeded(
                    f"Per-minute token budget reached (~{self.config.max_tokens_per_minute} "
                    "tokens/min, local safety limit). Please wait a moment and retry, or shorten "
                    "the conversation (try 'reset')."
                )

            self._request_timestamps.append(now)
            self._token_events.append((now, estimated_tokens))
            self._daily_state["count"] += 1
            self._save_daily_state()

    def remaining_today(self) -> int:
        return max(0, self.config.max_requests_per_day - self._daily_state.get("count", 0))

    def status(self) -> dict:
        now = _now()
        rpm_used = len([t for t in self._request_timestamps if now - t < 60])
        return {
            "requests_last_minute": rpm_used,
            "max_requests_per_minute": self.config.max_requests_per_minute,
            "requests_today": self._daily_state.get("count", 0),
            "max_requests_per_day": self.config.max_requests_per_day,
            "remaining_today": self.remaining_today(),
        }
