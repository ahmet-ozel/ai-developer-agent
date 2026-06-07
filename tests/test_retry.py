"""Unit tests for src/pipeline/retry.py - shared retry mechanism.

Covers:
- Retryable exception  retries up to max_retries
- Non-retryable exception  raises immediately (1 attempt)
- Successful call  returns result (1 attempt)
- Max retries exhausted  raises last exception
- Rate limit message detection
- _is_retryable classification for various exception types
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.retry import (
    RETRYABLE_EXCEPTIONS,
    RETRYABLE_STATUS_CODES,
    _is_retryable,
    retry_with_backoff,
)


# ---------------------------------------------------------------------------
# _is_retryable tests
# ---------------------------------------------------------------------------


class TestIsRetryable:
    """Tests for the _is_retryable helper."""

    @pytest.mark.parametrize("exc_cls", RETRYABLE_EXCEPTIONS)
    def test_retryable_exception_types(self, exc_cls: type[Exception]) -> None:
        assert _is_retryable(exc_cls("boom")) is True

    @pytest.mark.parametrize("code", sorted(RETRYABLE_STATUS_CODES))
    def test_retryable_status_code_attribute(self, code: int) -> None:
        exc = Exception("http error")
        exc.status_code = code  # type: ignore[attr-defined]
        assert _is_retryable(exc) is True

    @pytest.mark.parametrize("code", sorted(RETRYABLE_STATUS_CODES))
    def test_retryable_status_attribute(self, code: int) -> None:
        exc = Exception("http error")
        exc.status = code  # type: ignore[attr-defined]
        assert _is_retryable(exc) is True

    def test_non_retryable_status_code(self) -> None:
        exc = Exception("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        assert _is_retryable(exc) is False

    @pytest.mark.parametrize(
        "msg",
        ["Rate limit exceeded", "Too Many Requests", "rate limit hit"],
    )
    def test_rate_limit_message_detection(self, msg: str) -> None:
        assert _is_retryable(Exception(msg)) is True

    def test_non_retryable_exception(self) -> None:
        assert _is_retryable(ValueError("bad value")) is False

    def test_non_retryable_generic_message(self) -> None:
        assert _is_retryable(Exception("something went wrong")) is False


# ---------------------------------------------------------------------------
# retry_with_backoff tests
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Tests for the async retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_successful_call_returns_result(self) -> None:
        func = AsyncMock(return_value="ok")
        result = await retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        func = AsyncMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            await retry_with_backoff(func, max_retries=3)
        assert func.call_count == 1

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retryable_exception_retries_up_to_max(
        self, mock_sleep: AsyncMock
    ) -> None:
        func = AsyncMock(side_effect=ConnectionError("conn refused"))
        with pytest.raises(ConnectionError, match="conn refused"):
            await retry_with_backoff(func, max_retries=3, base_delay=1.0)
        assert func.call_count == 3
        # Should have slept between attempts (2 sleeps for 3 attempts)
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retryable_then_success(self, mock_sleep: AsyncMock) -> None:
        func = AsyncMock(
            side_effect=[TimeoutError("timeout"), TimeoutError("timeout"), "done"]
        )
        result = await retry_with_backoff(func, max_retries=4, base_delay=0.5)
        assert result == "done"
        assert func.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_max_retries_exhausted_raises_last_exception(
        self, mock_sleep: AsyncMock
    ) -> None:
        errors = [OSError("err1"), OSError("err2"), OSError("err3")]
        func = AsyncMock(side_effect=errors)
        with pytest.raises(OSError, match="err3"):
            await retry_with_backoff(func, max_retries=3, base_delay=1.0)
        assert func.call_count == 3

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retryable_status_code_triggers_retry(
        self, mock_sleep: AsyncMock
    ) -> None:
        exc = Exception("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        func = AsyncMock(side_effect=[exc, "ok"])
        result = await retry_with_backoff(func, max_retries=3, base_delay=1.0)
        assert result == "ok"
        assert func.call_count == 2

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_rate_limit_message_triggers_retry(
        self, mock_sleep: AsyncMock
    ) -> None:
        func = AsyncMock(
            side_effect=[Exception("rate limit exceeded"), "success"]
        )
        result = await retry_with_backoff(func, max_retries=3, base_delay=1.0)
        assert result == "success"
        assert func.call_count == 2

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_backoff_delay_increases_exponentially(
        self, mock_sleep: AsyncMock
    ) -> None:
        func = AsyncMock(side_effect=ConnectionError("fail"))
        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                func, max_retries=4, base_delay=1.0, max_delay=100.0
            )
        # 3 sleeps for 4 attempts
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # base_delay * 2^attempt ± 25% jitter
        # attempt 0: 1.0 ± 0.25  [0.75, 1.25]
        # attempt 1: 2.0 ± 0.50  [1.50, 2.50]
        # attempt 2: 4.0 ± 1.00  [3.00, 5.00]
        assert 0.75 <= delays[0] <= 1.25
        assert 1.50 <= delays[1] <= 2.50
        assert 3.00 <= delays[2] <= 5.00

    @pytest.mark.asyncio
    @patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_max_delay_caps_backoff(self, mock_sleep: AsyncMock) -> None:
        func = AsyncMock(side_effect=ConnectionError("fail"))
        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                func, max_retries=3, base_delay=10.0, max_delay=5.0
            )
        # All delays should be capped at max_delay (5.0) ± 25% jitter  [3.75, 6.25]
        for call in mock_sleep.call_args_list:
            assert 3.75 <= call.args[0] <= 6.25

    @pytest.mark.asyncio
    async def test_single_retry_non_retryable_no_sleep(self) -> None:
        """Non-retryable exception with max_retries=1 raises immediately."""
        func = AsyncMock(side_effect=KeyError("missing"))
        with pytest.raises(KeyError):
            await retry_with_backoff(func, max_retries=1)
        assert func.call_count == 1


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


from hypothesis import given, settings
from hypothesis import strategies as st


class TestRetryProperties:
    """Property-based tests for retry_with_backoff and _is_retryable.

    Validates: Requirements 2.5, 3.5, 11.1
    """

    # ------------------------------------------------------------------
    # Property 1: Retryable exception always retries up to max_retries
    # **Validates: Requirements 2.5, 3.5, 11.1**
    # ------------------------------------------------------------------

    @given(
        exc_cls=st.sampled_from(RETRYABLE_EXCEPTIONS),
        max_retries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_retryable_exception_retries_exactly_max_retries(
        self, exc_cls: type[Exception], max_retries: int
    ) -> None:
        """Retryable exception  function is called exactly max_retries times.

        **Validates: Requirements 2.5, 3.5, 11.1**
        """
        import asyncio

        async def _run() -> None:
            with patch("src.pipeline.retry.asyncio.sleep", new_callable=AsyncMock):
                func = AsyncMock(side_effect=exc_cls("transient error"))
                with pytest.raises(exc_cls):
                    await retry_with_backoff(func, max_retries=max_retries, base_delay=0.0)
                assert func.call_count == max_retries

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # Property 2: Non-retryable exception raises immediately (1 attempt)
    # **Validates: Requirements 2.5, 3.5, 11.1**
    # ------------------------------------------------------------------

    @given(
        msg=st.text(min_size=0, max_size=100),
        max_retries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_non_retryable_exception_raises_immediately(
        self, msg: str, max_retries: int
    ) -> None:
        """Non-retryable exception  function is called exactly once.

        **Validates: Requirements 2.5, 3.5, 11.1**
        """
        import asyncio

        safe_msg = msg.replace("rate limit", "").replace("too many requests", "")

        async def _run() -> None:
            func = AsyncMock(side_effect=ValueError(safe_msg))
            with pytest.raises(ValueError):
                await retry_with_backoff(func, max_retries=max_retries, base_delay=0.0)
            assert func.call_count == 1

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # Property 3: Successful call returns correct result (1 attempt)
    # **Validates: Requirements 2.5, 3.5, 11.1**
    # ------------------------------------------------------------------

    @given(
        return_value=st.one_of(st.integers(), st.text()),
        max_retries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_successful_call_single_attempt_correct_result(
        self, return_value: int | str, max_retries: int
    ) -> None:
        """Successful call  called exactly once and returns the correct value.

        **Validates: Requirements 2.5, 3.5, 11.1**
        """
        import asyncio

        async def _run() -> None:
            func = AsyncMock(return_value=return_value)
            result = await retry_with_backoff(func, max_retries=max_retries, base_delay=0.0)
            assert result == return_value
            assert func.call_count == 1

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # Property 4: _is_retryable is consistent - same exception always
    # gives same result
    # **Validates: Requirements 2.5, 3.5, 11.1**
    # ------------------------------------------------------------------

    @given(msg=st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_is_retryable_is_consistent(self, msg: str) -> None:
        """_is_retryable called twice on the same exception returns same value.

        **Validates: Requirements 2.5, 3.5, 11.1**
        """
        exc = Exception(msg)
        first = _is_retryable(exc)
        second = _is_retryable(exc)
        assert first == second
