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

IMPORTANT: state is persisted to disk (not just kept in memory), because Groq
enforces its limits at the organization/API-key level, not per-process. A
naive in-memory-only limiter would reset every time `python -m fpl_agent.cli`
is invoked, so several quick one-shot CLI calls (or multiple terminals) could
each think they were starting from zero and collectively blow past the real
limit before any single process's in-memory counter noticed. Persisting RPM/
TPM/RPD state to a shared file closes that gap.

Defaults are calibrated against Groq's published free-tier limits for this
project's default model (see RateLimitConfig.from_env). Check current limits
for your model at https://console.groq.com/settings/limits and tune via the
environment variables below if you switch models/plans.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator, Optional

try:
    import fcntl  # POSIX-only; used for best-effort cross-process locking

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - e.g. native Windows
    _HAVE_FCNTL = False

# Where rate-limit state (RPM/TPM sliding windows + the daily counter) is
# persisted, so it's shared across process invocations, not just within one.
# Kept inside fpl_agent/ and git-ignored - it's local runtime state, not source.
STATE_FILE = Path(os.getenv("FPL_AGENT_STATE_FILE", str(Path(__file__).parent / ".rate_limit_state.json")))
LOCK_FILE = STATE_FILE.with_suffix(".lock")

WINDOW_SECONDS = 60


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
        Defaults calibrated against Groq's published free-tier limits for
        `openai/gpt-oss-20b` (the default model here): 30 RPM / 1,000 RPD /
        8,000 TPM / 200,000 TPD (see https://console.groq.com/docs/rate-limits,
        or https://console.groq.com/settings/limits for your account's exact
        current numbers). We stay at roughly 50-75% of each published ceiling
        so this local check consistently trips *before* Groq's own limit ever
        could, even accounting for our token estimate being approximate.
        Override via env vars if you switch models/plans with different limits.
        """
        return cls(
            max_requests_per_minute=int(os.getenv(f"{prefix}_MAX_RPM", "15")),  # Groq free: 30 RPM
            max_requests_per_day=int(os.getenv(f"{prefix}_MAX_RPD", "250")),  # Groq free: 1,000 RPD
            max_tokens_per_minute=int(os.getenv(f"{prefix}_MAX_TPM", "6000")),  # Groq free: 8,000 TPM
        )


def _now() -> float:
    return time.time()


def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _empty_state() -> dict:
    return {"date": _today_key(), "count": 0, "request_timestamps": [], "token_events": []}


class LocalRateLimiter:
    """
    Enforces a hard ceiling on LLM calls - RPM and TPM via sliding 60s windows,
    RPD via a daily counter - all persisted to `state_file` so the limits are
    shared across process invocations (CLI runs, scripts, multiple terminals),
    not reset every time a new process starts.

    This does NOT rely on the provider's own rate-limit responses/headers -
    it blocks locally, before the network call is made.

    Locking is best-effort: an in-process `threading.Lock` plus (on POSIX) an
    `flock`-based file lock around the read-check-write sequence. This is
    sufficient for a local, single-user tool; it is not a guarantee of perfect
    atomicity under heavy concurrent multi-process load.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None, state_file: Path = STATE_FILE):
        self.config = config or RateLimitConfig.from_env()
        self.state_file = state_file
        self.lock_file = state_file.with_suffix(".lock")
        self._thread_lock = Lock()

    # -- locking -----------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            if not _HAVE_FCNTL:
                yield
                return
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_file, "w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    # -- persistence ---------------------------------------------------------

    def _load_state(self) -> dict:
        if not self.state_file.exists():
            return _empty_state()
        try:
            data = json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return _empty_state()
        if data.get("date") != _today_key():
            return _empty_state()
        data.setdefault("request_timestamps", [])
        data.setdefault("token_events", [])
        data.setdefault("count", 0)
        return data

    def _save_state(self, state: dict) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state))
        except OSError:
            pass  # best-effort persistence; still safe within this process via the in-memory check

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
        and lets it proceed. Shared across processes via the persisted state
        file, so rapid successive CLI invocations can't each start from zero.
        """
        with self._locked():
            now = _now()
            state = self._load_state()

            request_timestamps = [t for t in state["request_timestamps"] if now - t < WINDOW_SECONDS]
            token_events = [e for e in state["token_events"] if now - e[0] < WINDOW_SECONDS]
            daily_count = state["count"]

            if daily_count >= self.config.max_requests_per_day:
                raise RateLimitExceeded(
                    f"Daily request cap reached ({self.config.max_requests_per_day} requests/day, "
                    "local safety limit - not a provider error). Try again tomorrow, or raise "
                    "FPL_AGENT_MAX_RPD if you've confirmed it's safe for your account/tier."
                )

            if len(request_timestamps) >= self.config.max_requests_per_minute:
                raise RateLimitExceeded(
                    f"Per-minute request cap reached ({self.config.max_requests_per_minute} "
                    "requests/min, local safety limit). Please wait a moment and retry."
                )

            tokens_in_window = sum(n for _, n in token_events)
            if tokens_in_window + estimated_tokens > self.config.max_tokens_per_minute:
                raise RateLimitExceeded(
                    f"Per-minute token budget reached (~{self.config.max_tokens_per_minute} "
                    "tokens/min, local safety limit). Please wait a moment and retry, or shorten "
                    "the conversation (try 'reset')."
                )

            request_timestamps.append(now)
            token_events.append([now, estimated_tokens])
            self._save_state({
                "date": _today_key(),
                "count": daily_count + 1,
                "request_timestamps": request_timestamps,
                "token_events": token_events,
            })

    def remaining_today(self) -> int:
        state = self._load_state()
        return max(0, self.config.max_requests_per_day - state["count"])

    def status(self) -> dict:
        state = self._load_state()
        now = _now()
        rpm_used = len([t for t in state["request_timestamps"] if now - t < WINDOW_SECONDS])
        tpm_used = sum(n for t, n in state["token_events"] if now - t < WINDOW_SECONDS)
        return {
            "requests_last_minute": rpm_used,
            "max_requests_per_minute": self.config.max_requests_per_minute,
            "tokens_last_minute": tpm_used,
            "max_tokens_per_minute": self.config.max_tokens_per_minute,
            "requests_today": state["count"],
            "max_requests_per_day": self.config.max_requests_per_day,
            "remaining_today": max(0, self.config.max_requests_per_day - state["count"]),
        }
