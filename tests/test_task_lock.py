"""Tests for TaskLock mechanism."""

import asyncio

import pytest

from src.webhook.task_lock import TaskLock


@pytest.fixture
def task_lock() -> TaskLock:
    return TaskLock()


async def test_acquire_returns_true_for_new_key(task_lock: TaskLock) -> None:
    result = await task_lock.acquire("PROJ-123")
    assert result is True


async def test_acquire_returns_false_for_already_locked_key(task_lock: TaskLock) -> None:
    await task_lock.acquire("PROJ-123")
    result = await task_lock.acquire("PROJ-123")
    assert result is False


async def test_release_allows_reacquire(task_lock: TaskLock) -> None:
    await task_lock.acquire("PROJ-123")
    await task_lock.release("PROJ-123")
    result = await task_lock.acquire("PROJ-123")
    assert result is True


async def test_release_nonexistent_key_is_noop(task_lock: TaskLock) -> None:
    # Should not raise any error
    await task_lock.release("PROJ-999")


async def test_different_keys_are_independent(task_lock: TaskLock) -> None:
    await task_lock.acquire("PROJ-1")
    result = await task_lock.acquire("PROJ-2")
    assert result is True


async def test_concurrent_acquire_only_one_wins(task_lock: TaskLock) -> None:
    """Two concurrent acquires for the same key — only one should succeed."""
    results = await asyncio.gather(
        task_lock.acquire("PROJ-100"),
        task_lock.acquire("PROJ-100"),
    )
    assert sorted(results) == [False, True]


async def test_release_then_acquire_cycle(task_lock: TaskLock) -> None:
    """Multiple acquire/release cycles should all work."""
    for _ in range(3):
        assert await task_lock.acquire("PROJ-5") is True
        await task_lock.release("PROJ-5")


# =========================================================================
# Property Tests (Hypothesis)
# =========================================================================

import asyncio

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st
from tests.conftest import jira_issue_keys


class TestTaskLockMutualExclusionProperty:
    """Property 4: Task Lock Mutual Exclusion.

    Validates: Requirements 1.9
    """

    @given(issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_first_acquire_always_succeeds(self, issue_key: str) -> None:
        """First acquire for any key always returns True."""
        lock = TaskLock()

        async def _run() -> bool:
            return await lock.acquire(issue_key)

        result = asyncio.run(_run())
        assert result is True

    @given(issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_second_acquire_always_fails(self, issue_key: str) -> None:
        """Second acquire for same key (without release) always returns False."""
        lock = TaskLock()

        async def _run() -> bool:
            await lock.acquire(issue_key)
            return await lock.acquire(issue_key)

        result = asyncio.run(_run())
        assert result is False

    @given(issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_acquire_after_release_always_succeeds(self, issue_key: str) -> None:
        """Acquire after release always returns True."""
        lock = TaskLock()

        async def _run() -> bool:
            await lock.acquire(issue_key)
            await lock.release(issue_key)
            return await lock.acquire(issue_key)

        result = asyncio.run(_run())
        assert result is True

    @given(key1=jira_issue_keys, key2=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_different_keys_are_independent(self, key1: str, key2: str) -> None:
        """Locking one key does not affect a different key."""
        from hypothesis import assume
        assume(key1 != key2)
        lock = TaskLock()

        async def _run() -> bool:
            await lock.acquire(key1)
            return await lock.acquire(key2)

        result = asyncio.run(_run())
        assert result is True
