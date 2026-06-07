"""In-memory task-level lock mechanism.

Prevents duplicate pipeline executions for the same Jira issue.

⚠️ LIMITATION: TaskLock is in-memory and only works in single-process deployments.
For multi-replica deployments, use Redis-based distributed lock or PostgreSQL advisory lock.
Docker must use --workers 1 for this reason.
"""

import asyncio


class TaskLock:
    """In-memory task-level lock keyed by issue_key.

    Uses a simple dict to track locked issue keys. This is NOT thread-safe
    across workers - Docker deployment must use --workers 1.
    """

    def __init__(self) -> None:
        self._locked: dict[str, bool] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, issue_key: str) -> bool:
        """Acquire lock for the given issue_key.

        Returns True if the lock was successfully acquired (key was not locked).
        Returns False if the key is already locked.
        """
        async with self._lock:
            if issue_key in self._locked:
                return False
            self._locked[issue_key] = True
            return True

    async def release(self, issue_key: str) -> None:
        """Release lock for the given issue_key.

        No-op if the key is not currently locked.
        """
        async with self._lock:
            self._locked.pop(issue_key, None)
