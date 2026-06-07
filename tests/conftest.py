"""Shared fixtures, Hypothesis strategies, and MCP mock fixtures for tests.

Provides:
- Hypothesis strategies for generating valid model instances
- MCP server mock fixtures (Atlassian, Git)
- Shared test fixtures (sample Settings, TaskContext, CodeContext)
"""

from __future__ import annotations

# Exclude e2e tests from default collection - run them separately with:
#   pytest tests/e2e/ -v
collect_ignore = ["e2e"]

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import strategies as st

from src.pipeline.models import (
    ChangeType,
    CodeChange,
    CodeContext,
    CodeFile,
    FileChange,
    FindingSeverity,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    SkippedFile,
    TaskContext,
    TaskScope,
)
from src.webhook.models import WebhookEvent


# ---------------------------------------------------------------------------
# Global mock: prevent real mcp-agent Agent from polluting global state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_mcp_agent_in_orchestrator():
    """Prevent real mcp-agent Agent from creating global Context in unit tests.

    When mcp-agent is installed, Agent.__aenter__ creates a global Context
    and Settings singleton that accumulates across hypothesis examples.
    We patch the Agent class in the orchestrator module to use a lightweight
    mock that doesn't create global state.

    Tests that explicitly test Agent behavior should patch Agent themselves
    (their local patch will override this one).
    """
    mock_agent = MagicMock()
    mock_agent.return_value.__aenter__ = AsyncMock(return_value=mock_agent.return_value)
    mock_agent.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_agent.return_value.attach_llm = AsyncMock(side_effect=NotImplementedError("mocked"))

    # Also patch ConfluencePublisher.publish so it never calls real Confluence in unit tests
    mock_confluence = MagicMock()
    mock_confluence.return_value.publish = AsyncMock(return_value=None)

    with (
        patch("src.pipeline.orchestrator.Agent", mock_agent),
        patch("src.pipeline.orchestrator.ConfluencePublisher", mock_confluence),
    ):
        yield


@pytest.fixture(autouse=True)
def _clear_repo_field_cache():
    """Clear the _repo_field_id_cache between tests to prevent cross-test leaks."""
    try:
        from src.agents.task_reader import _repo_field_id_cache
        _repo_field_id_cache.clear()
    except ImportError:
        pass
    yield
    try:
        from src.agents.task_reader import _repo_field_id_cache
        _repo_field_id_cache.clear()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _mock_jira_http_in_unit_tests(request):
    """Mock httpx calls to Jira REST API in unit tests to prevent real network calls.

    This patches _discover_repository_field_id to return None (no custom field)
    and patches httpx.AsyncClient to return a mock Jira issue response.
    Only applies to non-e2e, non-integration tests.
    """
    # Skip for e2e and integration tests
    test_path = str(request.fspath)
    if "e2e" in test_path or "integration" in test_path:
        yield
        return

    import json

    # Default mock issue response
    mock_issue = {
        "key": "TEST-1",
        "fields": {
            "summary": "Fix login bug",
            "description": "Login fails on token refresh",
            "customfield_repository": "backend-api",
            "issuetype": {"name": "Bug"},
            "reporter": {"name": "john.doe"},
            "labels": [{"name": "backend"}],
            "priority": {"name": "High"},
            "issuelinks": [],
        },
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_issue
    mock_response.text = json.dumps(mock_issue)

    async def mock_get(url, **kwargs):
        if "/rest/api/3/field" in str(url):
            field_resp = MagicMock()
            field_resp.status_code = 200
            field_resp.json.return_value = [
                {"id": "customfield_10100", "name": "Repository"},
            ]
            return field_resp
        return mock_response

    with patch("src.agents.task_reader._discover_repository_field_id", new=AsyncMock(return_value="customfield_repository")):
        with patch("src.agents.task_reader.httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=mock_get)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=None)
            yield


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Jira issue key: 2-10 uppercase letters, dash, 1-5 digits
jira_issue_keys = st.from_regex(r"[A-Z]{2,10}-\d{1,5}", fullmatch=True)

# Webhook event types (valid + some noise for routing tests)
webhook_event_types = st.sampled_from(
    [
        "jira:issue_created",
        "jira:issue_updated",
        "jira:issue_deleted",
        "jira:worklog_updated",
        "comment_created",
        "sprint_started",
    ]
)

# Git provider strategy
git_providers = st.sampled_from(["bitbucket", "github", "gitlab"])

# LLM tier strategy
llm_tiers = st.sampled_from(["fast", "strong"])

# Conventional Commits building blocks
_commit_types = st.sampled_from(
    [
        "feat",
        "fix",
        "docs",
        "style",
        "refactor",
        "perf",
        "test",
        "build",
        "ci",
        "chore",
        "revert",
    ]
)
_commit_scopes = st.from_regex(r"[a-z][a-z0-9_-]{0,20}", fullmatch=True)
_commit_descs = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)
conventional_commit_messages = st.builds(
    lambda t, s, m: f"{t}({s}): {m}",
    t=_commit_types,
    s=_commit_scopes,
    m=_commit_descs,
)


# --- Composite strategies for Pydantic models ---

_file_paths = st.from_regex(
    r"src/[a-z_]+/[a-z_]+\.(py|ts|go|rs|java)", fullmatch=True
)
_languages = st.sampled_from(["python", "typescript", "go", "rust", "java"])
_non_empty_text = st.text(min_size=1, max_size=200)


@st.composite
def task_contexts(draw: st.DrawFn) -> TaskContext:
    """Generate valid TaskContext instances."""
    return TaskContext(
        issue_key=draw(jira_issue_keys),
        summary=draw(st.text(min_size=1, max_size=200)),
        description=draw(st.text(min_size=1, max_size=2000)),
        repository_name=draw(st.text(min_size=1, max_size=100)),
        estimated_scope=draw(st.sampled_from(TaskScope)),
        comments=draw(st.lists(st.text(min_size=1, max_size=100), max_size=5)),
        confluence_docs=draw(st.lists(st.text(min_size=1, max_size=200), max_size=3)),
        labels=draw(st.lists(st.text(min_size=1, max_size=30), max_size=5)),
        linked_issue_summaries=draw(
            st.lists(st.text(min_size=1, max_size=200), max_size=3)
        ),
        previous_review_feedback=draw(
            st.one_of(st.none(), st.text(min_size=1, max_size=500))
        ),
        issue_type=draw(
            st.one_of(st.none(), st.sampled_from(["Story", "Bug", "Task"]))
        ),
        reporter=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        base_branch=draw(st.sampled_from(["main", "master", "develop"])),
        priority=draw(
            st.one_of(st.none(), st.sampled_from(["High", "Medium", "Low"]))
        ),
    )


@st.composite
def code_files(draw: st.DrawFn) -> CodeFile:
    """Generate valid CodeFile instances with auto-computed line_count."""
    content = draw(st.text(min_size=1, max_size=5000))
    return CodeFile(
        path=draw(_file_paths),
        content=content,
        language=draw(_languages),
        is_test=draw(st.just(False)),
    )


@st.composite
def file_changes(draw: st.DrawFn) -> FileChange:
    """Generate valid FileChange instances respecting model validators.

    DELETE → new_content=None, CREATE/MODIFY → new_content required.
    """
    change_type = draw(st.sampled_from(ChangeType))
    if change_type == ChangeType.DELETE:
        new_content = None
    else:
        new_content = draw(st.text(min_size=1, max_size=5000))
    return FileChange(
        path=draw(_file_paths),
        new_content=new_content,
        change_type=change_type,
        explanation=draw(st.text(min_size=1, max_size=500)),
    )


@st.composite
def code_changes(draw: st.DrawFn) -> CodeChange:
    """Generate valid CodeChange instances with conventional commit messages."""
    return CodeChange(
        changes=draw(st.lists(file_changes(), min_size=1, max_size=5)),
        test_changes=draw(st.lists(file_changes(), min_size=0, max_size=3)),
        commit_message=draw(conventional_commit_messages),
        pr_title=draw(st.text(min_size=1, max_size=200)),
        pr_description=draw(st.text(min_size=1, max_size=2000)),
    )


# ---------------------------------------------------------------------------
# MCP Server Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_atlassian_mcp() -> AsyncMock:
    """mcp-atlassian MCP server mock.

    Simulates Jira issue reading, comment adding, transitions,
    and Confluence search operations.
    """
    mock = AsyncMock()

    def _side_effect(tool_name: str, **kwargs: Any) -> Any:
        responses: dict[str, Any] = {
            "jira_get_issue": {
                "key": kwargs.get("issue_key", "TEST-1"),
                "fields": {
                    "summary": "Test task",
                    "description": "Test description",
                    "customfield_repository": "my-repo",
                    "assignee": {"name": "ai-developer"},
                    "issuetype": {"name": "Story"},
                    "reporter": {"name": "john.doe"},
                    "labels": [{"name": "backend"}],
                    "priority": {"name": "Medium"},
                },
            },
            "jira_get_comments": [{"body": "Previous comment"}],
            "jira_add_comment": {"id": "12345"},
            "jira_transition_issue": None,
            "confluence_search": {"results": []},
        }
        return responses.get(tool_name)

    mock.call_tool = AsyncMock(side_effect=_side_effect)
    return mock


@pytest.fixture
def mock_git_mcp() -> AsyncMock:
    """Git MCP server mock (provider-agnostic).

    Simulates file tree, file content, branch, commit, and PR operations.
    """
    mock = AsyncMock()

    def _side_effect(tool_name: str, **kwargs: Any) -> Any:
        responses: dict[str, Any] = {
            "get_file_tree": "src/\n  main.py\n  utils.py\ntests/\n  test_main.py",
            "get_file_content": "# sample file content\nprint('hello')\n",
            "create_branch": {"name": "feature/TEST-1-ai"},
            "commit_files": {"sha": "abc123"},
            "create_pull_request": {
                "url": "https://example.com/pr/1",
                "id": 1,
            },
        }
        return responses.get(tool_name)

    mock.call_tool = AsyncMock(side_effect=_side_effect)
    return mock


# ---------------------------------------------------------------------------
# Shared Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_settings() -> dict[str, Any]:
    """Sample Settings as a dict (Settings class is implemented in Task 2.1).

    Contains all expected fields with sensible defaults for testing.
    """
    return {
        # Jira
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-developer",
        "jira_api_token": "jira-secret-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-developer",
        "jira_transition_in_progress": "21",
        "jira_transition_in_review": "31",
        # Git Provider
        "git_provider": "bitbucket",
        "bitbucket_workspace": "my-workspace",
        "bitbucket_username": "bb-user",
        "bitbucket_app_password": "bb-app-password",
        "github_token": None,
        "github_owner": None,
        "gitlab_url": "https://gitlab.com",
        "gitlab_token": None,
        "gitlab_group": None,
        # LLM
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast-key",
        "llm_fast_endpoint": None,
        "llm_strong_provider": "anthropic",
        "llm_strong_model": "claude-sonnet-4-20250514",
        "llm_strong_api_key": "sk-strong-key",
        "llm_strong_endpoint": None,
        "llm_fallback_chain": ["openai", "anthropic"],
        # Pipeline
        "max_review_retries": 2,
        "max_files_per_task": 10,
        "max_file_changes": 15,
        "max_context_tokens": 100000,
        "branch_pattern": "feature/{issue_key}-ai",
        "auto_create_pr": True,
        "pr_auto_assign_reviewer": False,
        "dry_run": False,
        # Task Filtering
        "skip_task_types": [],
        "allowed_task_types": [],
        # LLM Tier Overrides
        "task_reader_llm_tier": "fast",
        "code_finder_llm_tier": "fast",
        "code_writer_llm_tier": "strong",
        "code_reviewer_llm_tier": "strong",
    }


@pytest.fixture
def sample_task_context() -> TaskContext:
    """A ready-to-use TaskContext for tests that don't need randomised data."""
    return TaskContext(
        issue_key="TEST-42",
        summary="Fix authentication bug",
        description="Login fails when session expires during OAuth refresh",
        acceptance_criteria="Users should stay logged in after token refresh",
        repository_name="backend-api",
        estimated_scope=TaskScope.SMALL,
        comments=["Happens on mobile too"],
        labels=["backend", "auth"],
        issue_type="Bug",
        reporter="john.doe",
        base_branch="main",
        priority="High",
    )


@pytest.fixture
def sample_code_context() -> CodeContext:
    """A ready-to-use CodeContext for tests that don't need randomised data."""
    return CodeContext(
        files=[
            CodeFile(
                path="src/auth/oauth.py",
                content="class OAuthHandler:\n    def refresh_token(self):\n        pass\n",
                language="python",
                is_test=False,
            ),
            CodeFile(
                path="src/auth/session.py",
                content="class SessionManager:\n    def validate(self):\n        pass\n",
                language="python",
                is_test=False,
            ),
        ],
        test_files=[
            CodeFile(
                path="tests/test_oauth.py",
                content="def test_refresh():\n    assert True\n",
                language="python",
                is_test=True,
            ),
        ],
        tech_stack=["python"],
        repository_name="backend-api",
        file_tree="src/\n  auth/\n    oauth.py\n    session.py\ntests/\n  test_oauth.py",
    )
