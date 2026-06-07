"""TaskReader Agent - reads Jira issue details and produces TaskContext.

Uses mcp-agent Agent + AugmentedLLM pattern with server_names=["atlassian"]
to interact with Jira MCP server. Estimates task scope via fast tier LLM.
Includes exponential backoff retry (max 3 attempts).

Requirements: 2.1, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import httpx

from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import TaskContext, TaskScope

# ---------------------------------------------------------------------------
# Try importing mcp-agent Agent class. Falls back to placeholder when
# mcp-agent is not installed (e.g. in test environments).
# ---------------------------------------------------------------------------

try:
    from mcp_agent.agents.agent import Agent
except ImportError:  # pragma: no cover

    class Agent:  # type: ignore[no-redef]
        """Placeholder for mcp-agent Agent when not installed."""

        def __init__(self, **kwargs: Any) -> None:
            self.name = kwargs.get("name", "")
            self.instruction = kwargs.get("instruction", "")
            self.server_names = kwargs.get("server_names", [])

        async def __aenter__(self) -> "Agent":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def attach_llm(self, llm_class: type) -> Any:
            raise NotImplementedError("mcp-agent is not installed")


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic Jira field discovery - finds the "repository" custom field ID
# at runtime by querying /rest/api/3/field. Result is cached per Jira URL.
# ---------------------------------------------------------------------------

_repo_field_id_cache: dict[str, str | None] = {}


async def _discover_repository_field_id(settings: Settings) -> str | None:
    """Query Jira /rest/api/3/field and return the customfield_* ID whose
    name matches 'repository' (case-insensitive). Result is cached."""
    cache_key = settings.jira_url
    if cache_key in _repo_field_id_cache:
        return _repo_field_id_cache[cache_key]

    try:
        auth = (settings.jira_username, settings.jira_api_token.get_secret_value())
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.jira_url.rstrip('/')}/rest/api/3/field",
                auth=auth,
            )
        if resp.status_code == 200:
            for field in resp.json():
                name = (field.get("name") or "").lower()
                fid = field.get("id", "")
                if name == "repository" and fid.startswith("customfield_"):
                    _repo_field_id_cache[cache_key] = fid
                    logger.info("Discovered repository field ID: %s", fid)
                    return fid
    except Exception:
        logger.warning("Could not discover repository field ID from Jira", exc_info=True)

    _repo_field_id_cache[cache_key] = None
    return None

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class RepositoryFieldMissingError(Exception):
    """Raised when the Jira issue lacks a 'repository' custom field value."""

    def __init__(self, issue_key: str) -> None:
        self.issue_key = issue_key
        super().__init__(
            f"Issue {issue_key} is missing the 'repository' custom field. "
            "Pipeline cannot proceed without a target repository."
        )


# ---------------------------------------------------------------------------
# Retry helper (inline until Task 10.3 implements src/pipeline/retry.py)
# ---------------------------------------------------------------------------

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def _is_retryable(exc: Exception) -> bool:
    """Check whether an exception is retryable."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status and status in {429, 502, 503, 504}:
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "too many requests", "temporarily unavailable")):
        return True
    return False


async def _retry_with_backoff(
    func: Any,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Exponential backoff retry with ±25% jitter."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.25 * (random.random() * 2 - 1)
            logger.warning(
                "Retry attempt %d/%d after error: %s (delay=%.2fs)",
                attempt + 1,
                max_retries,
                exc,
                delay + jitter,
            )
            await asyncio.sleep(delay + jitter)
    # Should not reach here, but satisfy type checker
    raise RuntimeError("Retry loop exhausted unexpectedly")  # pragma: no cover


# ---------------------------------------------------------------------------
# Instruction prompt for the TaskReader agent
# ---------------------------------------------------------------------------

TASK_READER_INSTRUCTION = """You are a task reader agent. Your job is to read Jira issue details
using the available Jira MCP tools and return structured information.

When asked to read an issue:
1. Use jira_get_issue to get the issue details
2. Use jira_get_comments to get the latest comments
3. Look at linked issues for additional context
4. Search Confluence for related documentation

When returning issue data, normalize the JSON so that:
- The "repository_name" field contains the value of any custom field named "repository",
  "Repository", "repo", or any customfield_XXXXX whose display name contains "repository".
  Look through ALL customfield_* keys and their string values.
- Return the full fields object so all customfield_* keys are visible.

Return the information as structured JSON."""

SCOPE_ESTIMATION_PROMPT = """Analyze the following Jira task and estimate its scope.

Task Summary: {summary}
Task Description: {description}
Acceptance Criteria: {acceptance_criteria}

Based on the complexity of the task, estimate the scope as one of:
- "small": Simple change, single file, clear fix (e.g., typo fix, config change, simple bug fix)
- "medium": Moderate change, 2-5 files, some logic (e.g., new endpoint, refactor, feature addition)
- "large": Complex change, 5+ files, significant logic (e.g., new module, architecture change, major feature)

Respond with ONLY one word: small, medium, or large"""


# ---------------------------------------------------------------------------
# TaskReaderAgent
# ---------------------------------------------------------------------------


class TaskReaderAgent:
    """Reads Jira issue details via mcp-atlassian MCP and produces TaskContext.

    Uses mcp-agent Agent pattern with server_names=["atlassian"] and fast
    tier LLM for scope estimation.
    """

    def __init__(self, settings: Settings, llm_router: LLMRouter) -> None:
        self._settings = settings
        self._llm_router = llm_router
        self._repo_field_id: str | None = None  # discovered at first use

    async def read_task(self, issue_key: str) -> TaskContext:
        """Read a Jira issue and build a TaskContext.

        Retries with exponential backoff (max 3 attempts) on transient errors.
        """
        return await _retry_with_backoff(
            lambda: self._read_task_impl(issue_key),
            max_retries=3,
        )

    async def _read_task_impl(self, issue_key: str) -> TaskContext:
        """Core implementation: read issue via Jira REST API + MCP agent."""
        llm_class = self._llm_router.get_llm_class("fast")

        # Discover repository field ID once (cached after first call)
        if self._repo_field_id is None:
            self._repo_field_id = await _discover_repository_field_id(self._settings)

        # 1. Fetch raw issue directly from Jira REST API (reliable, no LLM parsing)
        auth = (self._settings.jira_username, self._settings.jira_api_token.get_secret_value())
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self._settings.jira_url.rstrip('/')}/rest/api/3/issue/{issue_key}",
                auth=auth,
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Jira API returned {resp.status_code} for {issue_key}: {resp.text[:200]}"
            )
        raw_issue = resp.json()
        issue = self._parse_issue_data(
            json.dumps(raw_issue), issue_key, repo_field_id=self._repo_field_id
        )

        agent = Agent(
            name="task_reader",
            instruction=TASK_READER_INSTRUCTION,
            server_names=["atlassian"],
        )

        async with agent:
            llm = await agent.attach_llm(llm_class)

            # 2. Get last 5 comments
            comments_data = await llm.generate_str(
                f"Use jira_get_comments for issue {issue_key}, limit to 5 most recent. "
                "Return the raw JSON response."
            )
            comments = self._parse_comments(comments_data)

            # 3. Get linked issues (already in raw_issue)
            linked_summaries = self._extract_linked_issues(issue)

            # 4. Search Confluence for related docs
            confluence_data = await llm.generate_str(
                f"Search Confluence for documentation related to: {issue.get('summary', '')}. "
                "Return the raw JSON response."
            )
            confluence_docs = self._parse_confluence_results(confluence_data)

            # 5. Estimate scope via LLM
            scope = await self._estimate_scope(
                llm=llm,
                summary=issue.get("summary", ""),
                description=issue.get("description", ""),
                acceptance_criteria=issue.get("acceptance_criteria"),
            )

        # 6. Extract repository name
        repository_name = (issue.get("repository_name", "") or "").strip()
        if not repository_name:
            raise RepositoryFieldMissingError(issue_key)

        # 7. Build TaskContext
        return TaskContext(
            issue_key=issue_key,
            summary=issue.get("summary", ""),
            description=issue.get("description", ""),
            acceptance_criteria=issue.get("acceptance_criteria"),
            repository_name=repository_name,
            estimated_scope=scope,
            comments=comments[:5],
            confluence_docs=confluence_docs[:3],
            labels=issue.get("labels", []),
            linked_issue_summaries=linked_summaries,
            issue_type=issue.get("issue_type"),
            reporter=issue.get("reporter"),
            base_branch=self._settings.git_base_branch,
            priority=issue.get("priority"),
        )

    async def _estimate_scope(
        self,
        llm: Any,
        summary: str,
        description: str,
        acceptance_criteria: str | None,
    ) -> TaskScope:
        """Use fast tier LLM to estimate task scope."""
        prompt = SCOPE_ESTIMATION_PROMPT.format(
            summary=summary,
            description=description or "No description provided",
            acceptance_criteria=acceptance_criteria or "None specified",
        )
        try:
            response = await llm.generate_str(prompt)
            return self._parse_scope(response)
        except Exception:
            logger.warning("Scope estimation failed, defaulting to MEDIUM")
            return TaskScope.MEDIUM

    @staticmethod
    def _parse_scope(response: str) -> TaskScope:
        """Parse LLM scope response into TaskScope enum."""
        cleaned = response.strip().lower()
        for scope in TaskScope:
            if scope.value in cleaned:
                return scope
        return TaskScope.MEDIUM

    @staticmethod
    def _parse_issue_data(raw: str, issue_key: str, repo_field_id: str | None = None) -> dict[str, Any]:
        """Parse raw issue data (JSON or structured text) into a normalized dict."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = {"raw_text": raw}

        # Handle nested Jira API format
        fields = data.get("fields", data)

        # Extract repository name.
        # Priority order:
        # 1. Dynamically discovered field ID (e.g. customfield_10039)
        # 2. Top-level "repository_name" key (set by LLM normalization)
        # 3. Legacy key name fallback
        # 4. Scan ALL customfield_* keys for any string value whose key contains "repo"
        repository_name = ""

        if repo_field_id:
            val = fields.get(repo_field_id, "")
            if isinstance(val, dict):
                val = val.get("value") or val.get("name") or ""
            repository_name = str(val or "").strip()

        if not repository_name:
            repository_name = str(
                fields.get("repository_name")
                or fields.get("customfield_repository")
                or ""
            ).strip()

        if not repository_name:
            for key, val in fields.items():
                if not key.startswith("customfield_"):
                    continue
                if isinstance(val, str) and val.strip() and "repo" in key.lower():
                    repository_name = val.strip()
                    break
                elif isinstance(val, dict):
                    inner = str(val.get("value") or val.get("name") or "").strip()
                    if inner and "repo" in key.lower():
                        repository_name = inner
                        break

        return {
            "summary": fields.get("summary", ""),
            "description": _adf_to_text(fields.get("description")),
            "acceptance_criteria": _adf_to_text(fields.get("acceptance_criteria")),
            "repository_name": repository_name,
            "issue_type": _nested_get(fields, "issuetype", "name"),
            "reporter": _nested_get(fields, "reporter", "name"),
            "labels": _extract_labels(fields.get("labels", [])),
            "priority": _nested_get(fields, "priority", "name"),
            "linked_issues": fields.get("issuelinks", []),
        }

    @staticmethod
    def _parse_comments(raw: str) -> list[str]:
        """Parse raw comments data into a list of comment body strings."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

        if isinstance(data, list):
            return [c.get("body", "") if isinstance(c, dict) else str(c) for c in data]

        # Handle nested format: {"comments": [...]}
        comments = data.get("comments", [])
        if isinstance(comments, list):
            return [c.get("body", "") if isinstance(c, dict) else str(c) for c in comments]

        return []

    @staticmethod
    def _parse_confluence_results(raw: str) -> list[str]:
        """Parse Confluence search results into a list of document excerpts."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

        results = data.get("results", [])
        if isinstance(results, list):
            docs = []
            for r in results[:3]:
                if isinstance(r, dict):
                    title = r.get("title", "")
                    excerpt = r.get("excerpt", r.get("body", ""))
                    docs.append(f"{title}: {excerpt}" if title else str(excerpt))
                else:
                    docs.append(str(r))
            return docs
        return []

    @staticmethod
    def _extract_linked_issues(issue: dict[str, Any]) -> list[str]:
        """Extract summaries from linked issues."""
        linked = issue.get("linked_issues", [])
        if not isinstance(linked, list):
            return []
        summaries = []
        for link in linked:
            if isinstance(link, dict):
                # Jira format: {"outwardIssue": {"fields": {"summary": "..."}}}
                for direction in ("outwardIssue", "inwardIssue"):
                    linked_issue = link.get(direction, {})
                    if isinstance(linked_issue, dict):
                        fields = linked_issue.get("fields", {})
                        if isinstance(fields, dict) and fields.get("summary"):
                            summaries.append(fields["summary"])
        return summaries


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _adf_to_text(node: Any, depth: int = 0) -> str:
    """Recursively extract plain text from Atlassian Document Format (ADF) nodes."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return str(node)

    node_type = node.get("type", "")
    content = node.get("content", [])
    text = node.get("text", "")

    if text:
        return text

    parts: list[str] = []
    for child in content:
        parts.append(_adf_to_text(child, depth + 1))

    separator = "\n" if node_type in ("paragraph", "heading", "bulletList", "orderedList", "listItem", "blockquote", "codeBlock") else ""
    return separator.join(p for p in parts if p)


def _nested_get(data: dict[str, Any], *keys: str) -> str | None:
    """Safely get a nested value from a dict."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current if isinstance(current, str) else None


def _extract_labels(labels: Any) -> list[str]:
    """Extract label strings from Jira label format (list of strings or dicts)."""
    if not isinstance(labels, list):
        return []
    result = []
    for label in labels:
        if isinstance(label, str):
            result.append(label)
        elif isinstance(label, dict):
            name = label.get("name", label.get("label", ""))
            if name:
                result.append(name)
    return result

# ---------------------------------------------------------------------------
# Task type filtering (Requirement 2.10)
# ---------------------------------------------------------------------------


def should_skip_task(task_ctx: TaskContext, settings: Settings) -> tuple[bool, str]:
    """Check whether a task should be skipped based on task type filtering.

    Rules:
    - If task_ctx.issue_type is None, no filtering can be applied  don't skip.
    - If issue_type is in settings.skip_task_types  skip.
    - If settings.allowed_task_types is non-empty and issue_type is not in it  skip.
    - Otherwise  don't skip.

    Returns:
        (should_skip, reason) - reason is empty string when not skipping.
    """
    issue_type = task_ctx.issue_type
    if issue_type is None:
        return (False, "")

    if issue_type in settings.skip_task_types:
        return (True, "skipped: task type in skip list")

    if settings.allowed_task_types and issue_type not in settings.allowed_task_types:
        return (True, "skipped: task type not in allowed list")

    return (False, "")

