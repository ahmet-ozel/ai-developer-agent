"""Shared retry mechanism with exponential backoff and jitter.

Provides a canonical ``retry_with_backoff`` helper used across the pipeline
to handle transient failures (MCP connection errors, LLM rate limits, etc.).

Requirements: 2.5, 3.5, 11.1, 11.6
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retryable classification
# ---------------------------------------------------------------------------

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

RETRYABLE_STATUS_CODES: set[int] = {429, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Check whether *exc* should trigger a retry.

    An exception is considered retryable when **any** of the following hold:

    1. It is an instance of one of :data:`RETRYABLE_EXCEPTIONS`.
    2. It carries a ``status_code`` or ``status`` attribute whose value is in
       :data:`RETRYABLE_STATUS_CODES`.
    3. Its string representation contains rate-limit related keywords
       (``"rate limit"``, ``"too many requests"``).
    """
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True

    # HTTP response status code (works with httpx, aiohttp, requests, etc.)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status and status in RETRYABLE_STATUS_CODES:
        return True

    # Keyword-based detection for non-standard exceptions
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg:
        return True

    return False


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------


async def retry_with_backoff(
    func: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Call *func* with exponential backoff on retryable failures.

    Parameters
    ----------
    func:
        An async callable (no arguments) to invoke.
    max_retries:
        Total number of attempts (including the first call).
    base_delay:
        Initial delay in seconds before the first retry.
    max_delay:
        Upper bound for the computed delay.

    Delay formula::

        delay = min(base_delay * 2^attempt, max_delay)
        jitter = delay * 0.25 * uniform(-1, 1)   # ±25 %
        sleep(delay + jitter)

    Raises the last exception when all attempts are exhausted or when the
    exception is not retryable.
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.25 * (random.random() * 2 - 1)
            actual_delay = delay + jitter
            logger.warning(
                "Retry attempt %d/%d after error: %s (delay=%.2fs)",
                attempt + 1,
                max_retries,
                exc,
                actual_delay,
            )
            await asyncio.sleep(actual_delay)

    # Unreachable in practice, but satisfies the type checker.
    raise RuntimeError("Retry loop exhausted unexpectedly")  # pragma: no cover
