"""Concurrency control, rate limiting, and retry with backoff."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from ..models.base import FatalError, TransientError

T = TypeVar("T")


class TokenBucket:
    """Async token bucket limiting requests per minute.

    Refills continuously rather than in discrete windows, so a burst at a window
    boundary cannot exceed the configured rate.
    """

    def __init__(self, requests_per_minute: float | None) -> None:
        self.rate = (requests_per_minute / 60.0) if requests_per_minute else None
        self.capacity = max(1.0, (requests_per_minute or 0) / 10.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.rate is None:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self.rate
            await asyncio.sleep(min(deficit, 5.0))


@dataclass
class RetryPolicy:
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.3

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Exponential backoff with proportional jitter, to avoid thundering herds."""
        raw = min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))
        return raw * (1.0 + rng.uniform(-self.jitter, self.jitter))


@dataclass
class AttemptResult:
    value: Any = None
    attempts: int = 0
    error: str | None = None


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    timeout: float | None = None,
    rng: random.Random | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> AttemptResult:
    """Run ``fn``, retrying transient failures.

    Never raises for an expected failure: the error is returned on the result so
    a single bad cell cannot abort a long sweep. ``FatalError`` short-circuits
    the retries, since retrying an auth or bad-request error only wastes budget.
    """
    rng = rng or random.Random()
    last_err: str | None = None

    for attempt in range(1, policy.max_retries + 2):
        try:
            coro = fn()
            value = await (asyncio.wait_for(coro, timeout) if timeout else coro)
            return AttemptResult(value=value, attempts=attempt)
        except FatalError as exc:
            return AttemptResult(attempts=attempt, error=f"fatal: {exc}")
        except asyncio.CancelledError:
            raise
        except (TransientError, asyncio.TimeoutError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - unknown errors are treated as transient once
            last_err = f"{type(exc).__name__}: {exc}"

        if attempt > policy.max_retries:
            break
        delay = policy.delay_for(attempt, rng)
        if on_retry:
            on_retry(attempt, Exception(last_err or ""), delay)
        await asyncio.sleep(delay)

    return AttemptResult(attempts=policy.max_retries + 1, error=last_err or "unknown error")
