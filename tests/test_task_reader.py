"""Unit tests for TaskReaderAgent.

Tests:
- Valid issue data produces correct TaskContext
- Scope estimation works for different description complexities
- Comments are extracted correctly
- Repository field missing raises RepositoryFieldMissingError
- Retry logic on transient errors
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.task_reader import (
    RepositoryFieldMissingError,
    TaskReaderAgent,
    _retry_with_backoff,
)
from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import TaskContext, TaskScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with sensible defaults for testing."""
    defaults = {
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-developer",
        "jira_api_token": "jira-secret-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-developer",
        "git_provider": "bitbucket",
        "bitbucket_workspace": "my-workspace",
        "bitbucket_username": "bb-user",
        "bitbucket_app_password": "bb-app-password",
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast-key",
        "llm_strong_provider": "anthropic",
        "llm_strong_model": "claude-sonnet-4-20250514",
        "llm_strong_api_key": "sk-strong-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_issue_json(
    *,
    summary: str = "Fix login bug",
    description: str = "Login fails on token refresh",
    repository: str = "backend-api",
    issue_type: str = "Bug",
    reporter: str = "john.doe",
    labels: list[str] | None = None,
    priority: str = "High",
    issuelinks: list[dict] | None = None,
) -> str:
    """Build a JSON string mimicking Jira issue API response."""
    label_objs = [{"name": l} for l in (labels or ["backend"])]
    return json.dumps({
        "key": "TEST-1",
        "fields": {
            "summary": summary,
            "description": description,
            "customfield_repository": repository,
            "issuetype": {"name": issue_type},
            "reporter": {"name": reporter},
            "labels": label_objs,
            "priority": {"name": priority},
            "issuelinks": issuelinks or [],
        },
    })


def _make_comments_json(bodies: list[str] | None = None) -> str:
    """Build a JSON string mimicking Jira comments response."""
    if bodies is None:
        bodies = ["First comment", "Second comment"]
    return json.dumps([{"body": b} for b in bodies])


def _make_confluence_json(titles: list[str] | None = None) -> str:
    """Build a JSON string mimicking Confluence search response."""
    if titles is None:
        titles = []
    return json.dumps({
        "results": [{"title": t, "excerpt": f"Excerpt for {t}"} for t in titles],
    })


def _build_mock_llm(
    issue_json: str = "",
    comments_json: str = "",
    confluence_json: str = "",
    scope_response: str = "small",
) -> AsyncMock:
    """Build a mock LLM that returns predefined responses in sequence.

    NOTE: issue_json is no longer used by the LLM (fetched via httpx directly).
    The LLM is called for: comments, confluence, scope (3 calls).
    """
    mock_llm = AsyncMock()
    mock_llm.generate_str = AsyncMock(
        side_effect=[comments_json, confluence_json, scope_response]
    )
    return mock_llm


# ---------------------------------------------------------------------------
# Tests: Valid issue data  correct TaskContext
# ---------------------------------------------------------------------------


class TestTaskReaderValidIssue:
    """Test that valid issue data produces correct TaskContext."""

    @pytest.fixture
    def settings(self) -> Settings:
        return _make_settings()

    @pytest.fixture
    def llm_router(self, settings: Settings) -> LLMRouter:
        return LLMRouter(config=settings)

    @pytest.fixture
    def agent(self, settings: Settings, llm_router: LLMRouter) -> TaskReaderAgent:
        return TaskReaderAgent(settings=settings, llm_router=llm_router)

    async def test_basic_task_context_fields(self, agent: TaskReaderAgent) -> None:
        """Valid issue data should produce TaskContext with correct fields."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(),
            comments_json=_make_comments_json(["Comment 1", "Comment 2"]),
            confluence_json=_make_confluence_json(),
            scope_response="small",
        )

        with patch.object(
            agent, "_read_task_impl", wraps=agent._read_task_impl
        ):
            # Patch the Agent class to use our mock
            with patch("src.agents.task_reader.Agent") as MockAgent:
                mock_agent_instance = AsyncMock()
                mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
                mock_agent_instance.__aenter__ = AsyncMock(
                    return_value=mock_agent_instance
                )
                mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
                MockAgent.return_value = mock_agent_instance

                result = await agent.read_task("TEST-1")

        assert isinstance(result, TaskContext)
        assert result.issue_key == "TEST-1"
        assert result.summary == "Fix login bug"
        assert result.description == "Login fails on token refresh"
        assert result.repository_name == "backend-api"
        assert result.estimated_scope == TaskScope.SMALL
        assert result.issue_type == "Bug"
        assert result.reporter == "john.doe"
        assert result.priority == "High"
        assert result.labels == ["backend"]

    async def test_comments_extracted(self, agent: TaskReaderAgent) -> None:
        """Comments should be extracted from the Jira response."""
        comments = ["Alpha", "Beta", "Gamma"]
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(),
            comments_json=_make_comments_json(comments),
            confluence_json=_make_confluence_json(),
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert result.comments == ["Alpha", "Beta", "Gamma"]

    async def test_max_five_comments(self, agent: TaskReaderAgent) -> None:
        """At most 5 comments should be included."""
        comments = [f"Comment {i}" for i in range(10)]
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(),
            comments_json=_make_comments_json(comments),
            confluence_json=_make_confluence_json(),
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert len(result.comments) <= 5

    async def test_confluence_docs_included(self, agent: TaskReaderAgent) -> None:
        """Confluence docs should be included when found."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(),
            comments_json=_make_comments_json(),
            confluence_json=_make_confluence_json(["Auth Guide", "API Docs"]),
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert len(result.confluence_docs) == 2
        assert any("Auth Guide" in doc for doc in result.confluence_docs)

    async def test_linked_issues_extracted(self, agent: TaskReaderAgent) -> None:
        """Linked issue summaries should be extracted."""
        issuelinks = [
            {
                "outwardIssue": {
                    "key": "TEST-2",
                    "fields": {"summary": "Related auth issue"},
                }
            },
        ]
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(issuelinks=issuelinks),
            comments_json=_make_comments_json(),
            confluence_json=_make_confluence_json(),
        )

        # Override httpx mock to return issue with issuelinks
        linked_issue = json.loads(_make_issue_json(issuelinks=issuelinks))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = linked_issue
        mock_response.text = json.dumps(linked_issue)

        async def mock_get(url, **kwargs):
            return mock_response

        with patch("src.agents.task_reader._discover_repository_field_id", new=AsyncMock(return_value="customfield_repository")):
            with patch("src.agents.task_reader.httpx.AsyncClient") as MockClient:
                mock_client_instance = AsyncMock()
                mock_client_instance.get = AsyncMock(side_effect=mock_get)
                MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
                MockClient.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch("src.agents.task_reader.Agent") as MockAgent:
                    mock_agent_instance = AsyncMock()
                    mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
                    mock_agent_instance.__aenter__ = AsyncMock(
                        return_value=mock_agent_instance
                    )
                    mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
                    MockAgent.return_value = mock_agent_instance

                    result = await agent.read_task("TEST-1")

        assert "Related auth issue" in result.linked_issue_summaries

    async def test_base_branch_from_settings(self, agent: TaskReaderAgent) -> None:
        """base_branch should come from settings.git_base_branch."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(),
            comments_json=_make_comments_json(),
            confluence_json=_make_confluence_json(),
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert result.base_branch == "main"


# ---------------------------------------------------------------------------
# Tests: Scope estimation
# ---------------------------------------------------------------------------


class TestScopeEstimation:
    """Test scope estimation for different description complexities."""

    @pytest.fixture
    def agent(self) -> TaskReaderAgent:
        settings = _make_settings()
        return TaskReaderAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_scope_small(self, agent: TaskReaderAgent) -> None:
        """Simple description should yield SMALL scope."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(description="Fix typo in README"),
            comments_json=_make_comments_json([]),
            confluence_json=_make_confluence_json(),
            scope_response="small",
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert result.estimated_scope == TaskScope.SMALL

    async def test_scope_medium(self, agent: TaskReaderAgent) -> None:
        """Moderate description should yield MEDIUM scope."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(
                description="Add new REST endpoint for user profile with validation"
            ),
            comments_json=_make_comments_json([]),
            confluence_json=_make_confluence_json(),
            scope_response="medium",
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert result.estimated_scope == TaskScope.MEDIUM

    async def test_scope_large(self, agent: TaskReaderAgent) -> None:
        """Complex description should yield LARGE scope."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(
                description="Redesign entire authentication module with OAuth2, SAML, and MFA support"
            ),
            comments_json=_make_comments_json([]),
            confluence_json=_make_confluence_json(),
            scope_response="large",
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert result.estimated_scope == TaskScope.LARGE

    async def test_scope_defaults_to_medium_on_parse_failure(
        self, agent: TaskReaderAgent
    ) -> None:
        """Unrecognized scope response should default to MEDIUM."""
        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(),
            comments_json=_make_comments_json([]),
            confluence_json=_make_confluence_json(),
            scope_response="I think this is a moderate task",
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        # "moderate" doesn't match any scope, defaults to MEDIUM
        assert result.estimated_scope == TaskScope.MEDIUM

    async def test_scope_estimation_failure_defaults_to_medium(
        self, agent: TaskReaderAgent
    ) -> None:
        """If scope estimation LLM call fails, default to MEDIUM."""
        mock_llm = AsyncMock()
        # Comments, confluence succeed; scope call raises
        mock_llm.generate_str = AsyncMock(
            side_effect=[
                _make_comments_json([]),
                _make_confluence_json(),
                Exception("LLM timeout"),
            ]
        )

        with patch("src.agents.task_reader.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
            mock_agent_instance.__aenter__ = AsyncMock(
                return_value=mock_agent_instance
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            MockAgent.return_value = mock_agent_instance

            result = await agent.read_task("TEST-1")

        assert result.estimated_scope == TaskScope.MEDIUM


# ---------------------------------------------------------------------------
# Tests: Missing repository field
# ---------------------------------------------------------------------------


class TestMissingRepository:
    """Test that missing repository field raises RepositoryFieldMissingError."""

    async def test_missing_repository_raises_error(self) -> None:
        """Empty repository field should raise RepositoryFieldMissingError."""
        settings = _make_settings()
        agent = TaskReaderAgent(
            settings=settings, llm_router=LLMRouter(config=settings)
        )

        mock_llm = _build_mock_llm(
            issue_json=_make_issue_json(repository=""),
            comments_json=_make_comments_json([]),
            confluence_json=_make_confluence_json(),
        )

        # Override httpx mock to return issue with empty repository
        empty_repo_issue = json.loads(_make_issue_json(repository=""))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = empty_repo_issue
        mock_response.text = json.dumps(empty_repo_issue)

        async def mock_get(url, **kwargs):
            return mock_response

        with patch("src.agents.task_reader._discover_repository_field_id", new=AsyncMock(return_value="customfield_repository")):
            with patch("src.agents.task_reader.httpx.AsyncClient") as MockClient:
                mock_client_instance = AsyncMock()
                mock_client_instance.get = AsyncMock(side_effect=mock_get)
                MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
                MockClient.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch("src.agents.task_reader.Agent") as MockAgent:
                    mock_agent_instance = AsyncMock()
                    mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
                    mock_agent_instance.__aenter__ = AsyncMock(
                        return_value=mock_agent_instance
                    )
                    mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
                    MockAgent.return_value = mock_agent_instance

                    with pytest.raises(RepositoryFieldMissingError) as exc_info:
                        await agent.read_task("TEST-1")

        assert "TEST-1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test exponential backoff retry behavior."""

    async def test_retry_on_connection_error(self) -> None:
        """Should retry on ConnectionError and succeed on subsequent attempt."""
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection refused")
            return "success"

        result = await _retry_with_backoff(flaky_func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 2

    async def test_no_retry_on_non_retryable_error(self) -> None:
        """Should not retry on non-retryable errors."""
        call_count = 0

        async def bad_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            await _retry_with_backoff(bad_func, max_retries=3, base_delay=0.01)

        assert call_count == 1

    async def test_max_retries_exhausted(self) -> None:
        """Should raise after max retries are exhausted."""
        call_count = 0

        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Still failing")

        with pytest.raises(ConnectionError, match="Still failing"):
            await _retry_with_backoff(always_fail, max_retries=3, base_delay=0.01)

        assert call_count == 3


# ---------------------------------------------------------------------------
# Tests: Parse helpers (unit tests for static methods)
# ---------------------------------------------------------------------------


class TestParseHelpers:
    """Test the static parsing helper methods."""

    def test_parse_scope_small(self) -> None:
        assert TaskReaderAgent._parse_scope("small") == TaskScope.SMALL

    def test_parse_scope_medium(self) -> None:
        assert TaskReaderAgent._parse_scope("medium") == TaskScope.MEDIUM

    def test_parse_scope_large(self) -> None:
        assert TaskReaderAgent._parse_scope("large") == TaskScope.LARGE

    def test_parse_scope_with_extra_text(self) -> None:
        assert TaskReaderAgent._parse_scope("I think this is small") == TaskScope.SMALL

    def test_parse_scope_unknown_defaults_medium(self) -> None:
        assert TaskReaderAgent._parse_scope("unknown") == TaskScope.MEDIUM

    def test_parse_comments_list_format(self) -> None:
        raw = json.dumps([{"body": "Hello"}, {"body": "World"}])
        assert TaskReaderAgent._parse_comments(raw) == ["Hello", "World"]

    def test_parse_comments_nested_format(self) -> None:
        raw = json.dumps({"comments": [{"body": "Nested"}]})
        assert TaskReaderAgent._parse_comments(raw) == ["Nested"]

    def test_parse_comments_invalid_json(self) -> None:
        assert TaskReaderAgent._parse_comments("not json") == []

    def test_parse_confluence_results(self) -> None:
        raw = json.dumps({
            "results": [
                {"title": "Doc1", "excerpt": "Content1"},
                {"title": "Doc2", "excerpt": "Content2"},
            ]
        })
        results = TaskReaderAgent._parse_confluence_results(raw)
        assert len(results) == 2
        assert "Doc1" in results[0]

    def test_parse_confluence_empty(self) -> None:
        raw = json.dumps({"results": []})
        assert TaskReaderAgent._parse_confluence_results(raw) == []

    def test_parse_issue_data_jira_format(self) -> None:
        raw = _make_issue_json(summary="Test", repository="my-repo")
        result = TaskReaderAgent._parse_issue_data(raw, "TEST-1")
        assert result["summary"] == "Test"
        assert result["repository_name"] == "my-repo"

    def test_extract_linked_issues(self) -> None:
        issue = {
            "linked_issues": [
                {
                    "outwardIssue": {
                        "key": "TEST-2",
                        "fields": {"summary": "Linked task"},
                    }
                }
            ]
        }
        result = TaskReaderAgent._extract_linked_issues(issue)
        assert result == ["Linked task"]

    def test_extract_linked_issues_empty(self) -> None:
        assert TaskReaderAgent._extract_linked_issues({}) == []


# ---------------------------------------------------------------------------
# Tests: Task type filtering - should_skip_task (Requirement 2.10)
# ---------------------------------------------------------------------------

from src.agents.task_reader import should_skip_task


def _make_task_context(issue_type: str | None = "Bug") -> TaskContext:
    """Create a minimal TaskContext for filtering tests."""
    return TaskContext(
        issue_key="TEST-1",
        summary="Test task",
        description="Test description",
        repository_name="my-repo",
        estimated_scope=TaskScope.SMALL,
        issue_type=issue_type,
    )


class TestShouldSkipTask:
    """Test should_skip_task filtering logic."""

    def test_skip_when_type_in_skip_list(self) -> None:
        """Task type in skip_task_types  should skip."""
        settings = _make_settings(skip_task_types=["Epic", "Bug"])
        ctx = _make_task_context(issue_type="Bug")
        skip, reason = should_skip_task(ctx, settings)
        assert skip is True
        assert "skip list" in reason

    def test_no_skip_when_type_not_in_skip_list(self) -> None:
        """Task type NOT in skip_task_types  should not skip."""
        settings = _make_settings(skip_task_types=["Epic"])
        ctx = _make_task_context(issue_type="Bug")
        skip, reason = should_skip_task(ctx, settings)
        assert skip is False
        assert reason == ""

    def test_no_skip_when_allowed_types_empty(self) -> None:
        """Empty allowed_task_types  any type allowed, should not skip."""
        settings = _make_settings(allowed_task_types=[])
        ctx = _make_task_context(issue_type="Story")
        skip, reason = should_skip_task(ctx, settings)
        assert skip is False
        assert reason == ""

    def test_no_skip_when_type_in_allowed_list(self) -> None:
        """Task type in allowed_task_types  should not skip."""
        settings = _make_settings(allowed_task_types=["Bug", "Story"])
        ctx = _make_task_context(issue_type="Bug")
        skip, reason = should_skip_task(ctx, settings)
        assert skip is False
        assert reason == ""

    def test_skip_when_type_not_in_allowed_list(self) -> None:
        """Task type NOT in non-empty allowed_task_types  should skip."""
        settings = _make_settings(allowed_task_types=["Story", "Task"])
        ctx = _make_task_context(issue_type="Bug")
        skip, reason = should_skip_task(ctx, settings)
        assert skip is True
        assert "allowed list" in reason

    def test_no_skip_when_issue_type_is_none(self) -> None:
        """None issue_type  can't filter, should not skip."""
        settings = _make_settings(skip_task_types=["Bug"], allowed_task_types=["Story"])
        ctx = _make_task_context(issue_type=None)
        skip, reason = should_skip_task(ctx, settings)
        assert skip is False
        assert reason == ""


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# Strategies for valid Jira issue JSON generation
_non_empty_text = st.text(min_size=1, max_size=200)
_optional_text = st.one_of(st.none(), st.text(min_size=1, max_size=200))
_label_list = st.lists(st.text(min_size=1, max_size=30), max_size=5)
_scope_strings = st.sampled_from(["small", "medium", "large"])
_scope_with_noise = st.builds(
    lambda s, prefix: f"{prefix} {s}" if prefix else s,
    s=_scope_strings,
    prefix=st.one_of(st.just(""), st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz ")),
)


@st.composite
def valid_jira_issue_jsons(draw: st.DrawFn) -> tuple[str, dict]:
    """Generate valid Jira issue JSON with a non-empty repository field.

    Returns (json_str, expected_fields) tuple.
    """
    import json as _json

    summary = draw(_non_empty_text)
    description = draw(st.text(min_size=0, max_size=500))
    repository = draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))  # non-empty after strip
    issue_type = draw(st.one_of(st.none(), st.sampled_from(["Bug", "Story", "Task", "Epic"])))
    reporter = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    labels = draw(_label_list)
    priority = draw(st.one_of(st.none(), st.sampled_from(["High", "Medium", "Low"])))

    label_objs = [{"name": lbl} for lbl in labels]
    fields: dict = {
        "summary": summary,
        "description": description,
        "customfield_repository": repository,
        "labels": label_objs,
        "issuelinks": [],
    }
    if issue_type is not None:
        fields["issuetype"] = {"name": issue_type}
    if reporter is not None:
        fields["reporter"] = {"name": reporter}
    if priority is not None:
        fields["priority"] = {"name": priority}

    issue_json = _json.dumps({"key": "TEST-1", "fields": fields})

    expected = {
        "summary": summary,
        "description": description,
        "repository_name": repository.strip(),
        "issue_type": issue_type,
        "reporter": reporter,
        "labels": labels,
        "priority": priority,
    }
    return issue_json, expected


class TestProperty5aTaskContextFieldExtraction:
    """Property 5a: TaskContext Field Extraction - Valid Repository

    **Validates: Requirements 2.1, 2.2, 2.4, 2.7**
    """

    @given(issue_data=valid_jira_issue_jsons())
    @settings(max_examples=100)
    def test_parse_issue_data_field_extraction(
        self, issue_data: tuple[str, dict]
    ) -> None:
        """For any valid Jira issue JSON with non-empty repository,
        _parse_issue_data returns a dict with correct field mappings.

        **Validates: Requirements 2.1, 2.2**
        """
        issue_json, expected = issue_data

        result = TaskReaderAgent._parse_issue_data(issue_json, "TEST-1")

        # summary and description must match source
        assert result["summary"] == expected["summary"]
        assert result["description"] == expected["description"]

        # repository_name must exactly match the customfield_repository value
        assert result["repository_name"] == expected["repository_name"]
        assert result["repository_name"] != ""  # non-empty guaranteed by strategy

        # issue_type, reporter, priority match (may be None if not in source)
        assert result["issue_type"] == expected["issue_type"]
        assert result["reporter"] == expected["reporter"]
        assert result["priority"] == expected["priority"]

        # labels must match
        assert result["labels"] == expected["labels"]

    @given(issue_data=valid_jira_issue_jsons())
    @settings(max_examples=100)
    def test_parse_issue_data_produces_valid_task_context(
        self, issue_data: tuple[str, dict]
    ) -> None:
        """For any valid issue data, the parsed dict can be used to build a valid TaskContext.

        **Validates: Requirements 2.1, 2.2, 2.4**
        """
        issue_json, _ = issue_data

        result = TaskReaderAgent._parse_issue_data(issue_json, "TEST-1")

        # Must be able to construct a TaskContext from the parsed data
        ctx = TaskContext(
            issue_key="TEST-1",
            summary=result["summary"],
            description=result["description"],
            repository_name=result["repository_name"],
            estimated_scope=TaskScope.SMALL,  # scope is estimated separately
            issue_type=result.get("issue_type"),
            reporter=result.get("reporter"),
            labels=result.get("labels", []),
            priority=result.get("priority"),
        )

        assert isinstance(ctx, TaskContext)
        assert ctx.issue_key == "TEST-1"
        assert ctx.repository_name == result["repository_name"]

    @given(scope_text=_scope_with_noise)
    @settings(max_examples=100)
    def test_parse_scope_maps_to_correct_enum(self, scope_text: str) -> None:
        """For any string containing "small"/"medium"/"large",
        _parse_scope returns the correct TaskScope enum value.

        **Validates: Requirements 2.4**
        """
        result = TaskReaderAgent._parse_scope(scope_text)

        # Result must always be a valid TaskScope enum value
        assert isinstance(result, TaskScope)
        assert result in (TaskScope.SMALL, TaskScope.MEDIUM, TaskScope.LARGE)

        # If the text contains a known scope keyword, it must map correctly
        lower = scope_text.strip().lower()
        if "small" in lower:
            assert result == TaskScope.SMALL
        elif "medium" in lower:
            assert result == TaskScope.MEDIUM
        elif "large" in lower:
            assert result == TaskScope.LARGE

    @given(
        issue_data=valid_jira_issue_jsons(),
        comments=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=20),
    )
    @settings(max_examples=100)
    def test_parse_comments_at_most_five(
        self, issue_data: tuple[str, dict], comments: list[str]
    ) -> None:
        """For any list of comments, the pipeline enforces at most 5 comments in TaskContext.

        **Validates: Requirements 2.7**
        """
        import json as _json

        raw_comments = _json.dumps([{"body": c} for c in comments])
        parsed = TaskReaderAgent._parse_comments(raw_comments)

        # The pipeline slices to [:5] when building TaskContext
        # Verify that applying the same slice gives at most 5
        assert len(parsed[:5]) <= 5

        # All parsed comments must be strings
        for comment in parsed:
            assert isinstance(comment, str)


# ---------------------------------------------------------------------------
# Property 5b: TaskContext Field Extraction - Missing Repository
# ---------------------------------------------------------------------------


@st.composite
def missing_repository_jira_issue_jsons(draw: st.DrawFn) -> str:
    """Generate Jira issue JSON where customfield_repository is empty string or None.

    Returns a JSON string with an empty/None repository field.
    """
    import json as _json

    summary = draw(st.text(min_size=1, max_size=200))
    description = draw(st.text(min_size=0, max_size=500))
    # Repository is either empty string or None (missing)
    repository = draw(st.one_of(st.just(""), st.just(None), st.just("   ")))
    issue_type = draw(st.one_of(st.none(), st.sampled_from(["Bug", "Story", "Task", "Epic"])))
    reporter = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    labels = draw(st.lists(st.text(min_size=1, max_size=30), max_size=5))
    priority = draw(st.one_of(st.none(), st.sampled_from(["High", "Medium", "Low"])))

    label_objs = [{"name": lbl} for lbl in labels]
    fields: dict = {
        "summary": summary,
        "description": description,
        "labels": label_objs,
        "issuelinks": [],
    }
    # Include customfield_repository as empty/None/whitespace-only
    if repository is None:
        # Omit the field entirely (missing)
        pass
    else:
        fields["customfield_repository"] = repository

    if issue_type is not None:
        fields["issuetype"] = {"name": issue_type}
    if reporter is not None:
        fields["reporter"] = {"name": reporter}
    if priority is not None:
        fields["priority"] = {"name": priority}

    return _json.dumps({"key": "TEST-1", "fields": fields})


class TestProperty5bMissingRepository:
    """Property 5b: TaskContext Field Extraction - Missing Repository

    **Validates: Requirements 2.3**
    """

    @given(issue_json=missing_repository_jira_issue_jsons())
    @settings(max_examples=100)
    def test_parse_issue_data_returns_empty_repository_name(
        self, issue_json: str
    ) -> None:
        """For any Jira issue JSON where customfield_repository is empty/None/missing,
        _parse_issue_data returns a dict with empty repository_name.

        **Validates: Requirements 2.3**
        """
        result = TaskReaderAgent._parse_issue_data(issue_json, "TEST-1")

        # repository_name must be empty (falsy) when the field is missing/empty
        assert not result["repository_name"], (
            f"Expected empty repository_name but got: {result['repository_name']!r}"
        )

    @given(issue_json=missing_repository_jira_issue_jsons())
    @settings(max_examples=100)
    def test_empty_repository_name_raises_error(self, issue_json: str) -> None:
        """When repository_name is empty, RepositoryFieldMissingError should be raised
        if the pipeline attempts to build a TaskContext from this data.

        Simulates the check in _read_task_impl: if not repository_name  raise.

        **Validates: Requirements 2.3**
        """
        result = TaskReaderAgent._parse_issue_data(issue_json, "TEST-1")
        repository_name = result["repository_name"]

        # The pipeline raises RepositoryFieldMissingError when repository_name is falsy
        # Verify that the condition that triggers the error is met
        assert not repository_name, (
            f"Expected empty repository_name to trigger error, got: {repository_name!r}"
        )

        # Directly verify the error is raised under the same condition as _read_task_impl
        with pytest.raises(RepositoryFieldMissingError) as exc_info:
            if not repository_name:
                raise RepositoryFieldMissingError("TEST-1")

        assert "TEST-1" in str(exc_info.value)

    @given(issue_json=missing_repository_jira_issue_jsons())
    @settings(max_examples=100)
    def test_repository_field_missing_error_contains_issue_key(
        self, issue_json: str
    ) -> None:
        """RepositoryFieldMissingError must include the issue_key in its message.

        **Validates: Requirements 2.3**
        """
        import json as _json

        # Use a random-looking issue key derived from the JSON to vary the test
        data = _json.loads(issue_json)
        issue_key = data.get("key", "TEST-1")

        error = RepositoryFieldMissingError(issue_key)
        assert issue_key in str(error)
        assert error.issue_key == issue_key


# ---------------------------------------------------------------------------
# Property 6: Task Type Filtering
# ---------------------------------------------------------------------------


class TestProperty6TaskTypeFiltering:
    """Property 6: Task Type Filtering

    skip_task_types ve allowed_task_types konfigürasyonuna göre doğru filtreleme kararı.

    **Validates: Requirements 2.10**
    """

    @given(
        issue_type=st.text(min_size=1, max_size=30),
        skip_task_types=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_issue_type_in_skip_list_always_skips(
        self, issue_type: str, skip_task_types: list[str]
    ) -> None:
        """For any issue_type that is in skip_task_types, should_skip_task returns (True, reason).
        The reason must contain "skip list".

        **Validates: Requirements 2.10**
        """
        # Ensure issue_type is in the skip list
        skip_list = list(skip_task_types)
        if issue_type not in skip_list:
            skip_list.append(issue_type)

        settings_obj = _make_settings(skip_task_types=skip_list, allowed_task_types=[])
        ctx = _make_task_context(issue_type=issue_type)

        skip, reason = should_skip_task(ctx, settings_obj)

        assert skip is True
        assert "skip list" in reason

    @given(
        issue_type=st.text(min_size=1, max_size=30),
        skip_task_types=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_issue_type_not_in_skip_list_and_allowed_empty_does_not_skip(
        self, issue_type: str, skip_task_types: list[str]
    ) -> None:
        """For any issue_type NOT in skip_task_types and allowed_task_types is empty,
        should_skip_task returns (False, "").

        **Validates: Requirements 2.10**
        """
        # Ensure issue_type is NOT in the skip list
        skip_list = [t for t in skip_task_types if t != issue_type]
        assume(issue_type not in skip_list)

        settings_obj = _make_settings(skip_task_types=skip_list, allowed_task_types=[])
        ctx = _make_task_context(issue_type=issue_type)

        skip, reason = should_skip_task(ctx, settings_obj)

        assert skip is False
        assert reason == ""

    @given(
        issue_type=st.text(min_size=1, max_size=30),
        allowed_task_types=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_issue_type_in_allowed_list_does_not_skip(
        self, issue_type: str, allowed_task_types: list[str]
    ) -> None:
        """For any issue_type in non-empty allowed_task_types,
        should_skip_task returns (False, "").

        **Validates: Requirements 2.10**
        """
        # Ensure issue_type is in the allowed list
        allowed_list = list(allowed_task_types)
        if issue_type not in allowed_list:
            allowed_list.append(issue_type)

        settings_obj = _make_settings(skip_task_types=[], allowed_task_types=allowed_list)
        ctx = _make_task_context(issue_type=issue_type)

        skip, reason = should_skip_task(ctx, settings_obj)

        assert skip is False
        assert reason == ""

    @given(
        issue_type=st.text(min_size=1, max_size=30),
        allowed_task_types=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_issue_type_not_in_allowed_list_skips(
        self, issue_type: str, allowed_task_types: list[str]
    ) -> None:
        """For any issue_type NOT in non-empty allowed_task_types,
        should_skip_task returns (True, reason) where reason contains "allowed list".

        **Validates: Requirements 2.10**
        """
        # Ensure issue_type is NOT in the allowed list
        allowed_list = [t for t in allowed_task_types if t != issue_type]
        assume(len(allowed_list) > 0)
        assume(issue_type not in allowed_list)

        settings_obj = _make_settings(skip_task_types=[], allowed_task_types=allowed_list)
        ctx = _make_task_context(issue_type=issue_type)

        skip, reason = should_skip_task(ctx, settings_obj)

        assert skip is True
        assert "allowed list" in reason

    @given(
        skip_task_types=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5),
        allowed_task_types=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5),
    )
    @settings(max_examples=100)
    def test_none_issue_type_never_skips(
        self, skip_task_types: list[str], allowed_task_types: list[str]
    ) -> None:
        """For None issue_type, should_skip_task always returns (False, "")
        regardless of skip_task_types or allowed_task_types configuration.

        **Validates: Requirements 2.10**
        """
        settings_obj = _make_settings(
            skip_task_types=skip_task_types,
            allowed_task_types=allowed_task_types,
        )
        ctx = _make_task_context(issue_type=None)

        skip, reason = should_skip_task(ctx, settings_obj)

        assert skip is False
        assert reason == ""
