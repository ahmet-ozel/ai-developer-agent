"""Unit tests for ConfluencePublisher.

Tests page content generation, disabled/error scenarios, and publish flow.

Requirements: FR-5, FR-6, FR-7, FR-8, CP-3, CP-4, CP-5, CP-6
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from src.config.settings import Settings
from src.pipeline.confluence_publisher import ConfluencePublisher, _MAX_DIFF_CHARS
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    ChangeType,
    CodeChange,
    FileChange,
    FindingSeverity,
    PipelineResult,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    TaskContext,
    TaskScope,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON = {
    "jira_url": "https://jira.example.com",
    "jira_username": "ai-developer",
    "jira_api_token": "jira-secret-token",
    "jira_webhook_secret": "webhook-secret",
    "jira_bot_username": "ai-developer",
    "llm_fast_provider": "openai",
    "llm_fast_model": "gpt-4o-mini",
    "llm_fast_api_key": "sk-fast-key",
    "llm_strong_provider": "anthropic",
    "llm_strong_model": "claude-sonnet-4-20250514",
    "llm_strong_api_key": "sk-strong-key",
    "git_provider": "bitbucket",
    "bitbucket_workspace": "my-workspace",
    "bitbucket_username": "bb-user",
    "bitbucket_app_password": "bb-app-password",
}

_CONFLUENCE_CREDS = {
    "confluence_enabled": True,
    "confluence_url": "https://wiki.example.com",
    "confluence_username": "wiki-user",
    "confluence_api_token": "wiki-token",
    "confluence_space_key": "DEV",
}


def _make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{**_COMMON, **overrides})


def _make_publisher(enabled: bool = True, **overrides) -> ConfluencePublisher:
    if enabled:
        settings = _make_settings(**_CONFLUENCE_CREDS, **overrides)
    else:
        settings = _make_settings(**overrides)
    llm_router = LLMRouter(config=settings)
    return ConfluencePublisher(config=settings, llm_router=llm_router)


def _make_task_ctx(issue_key: str = "TEST-1") -> TaskContext:
    return TaskContext(
        issue_key=issue_key,
        summary="Fix login bug",
        description="Login fails on mobile",
        repository_name="backend-api",
        estimated_scope=TaskScope.SMALL,
    )


def _make_result(success: bool = True, failure_stage: str | None = None) -> PipelineResult:
    return PipelineResult(
        issue_key="TEST-1",
        success=success,
        failure_stage=failure_stage,
        failure_reason="Something went wrong" if not success else None,
    )


def _make_code_change() -> CodeChange:
    return CodeChange(
        changes=[
            FileChange(
                path="src/auth.py",
                new_content="def login(): pass",
                change_type=ChangeType.MODIFY,
                explanation="Fixed login",
            )
        ],
        test_changes=[
            FileChange(
                path="tests/test_auth.py",
                new_content="def test_login(): assert True",
                change_type=ChangeType.CREATE,
                explanation="Added test",
            )
        ],
        commit_message="fix(auth): fix login bug",
        pr_title="Fix login",
        pr_description="Fixes login on mobile",
    )


def _make_review() -> ReviewResult:
    return ReviewResult(
        verdict=ReviewVerdict.APPROVE,
        score=8,
        findings=[
            ReviewFinding(
                file_path="src/auth.py",
                severity=FindingSeverity.SUGGESTION,
                category="style",
                message="Consider adding docstring",
            )
        ],
        acceptance_criteria_met=True,
    )


# ---------------------------------------------------------------------------
# Tests: enabled/disabled state
# ---------------------------------------------------------------------------


class TestConfluencePublisherEnabled:
    """CP-3: Confluence disabled by default."""

    def test_disabled_by_default(self) -> None:
        publisher = _make_publisher(enabled=False)
        assert publisher._enabled is False

    def test_enabled_when_configured(self) -> None:
        publisher = _make_publisher(enabled=True)
        assert publisher._enabled is True

    def test_disabled_when_url_missing(self) -> None:
        settings = _make_settings(
            confluence_enabled=False,
            confluence_url="",
        )
        publisher = ConfluencePublisher(config=settings)
        assert publisher._enabled is False

    def test_disabled_when_token_missing(self) -> None:
        settings = _make_settings(
            confluence_enabled=False,
            confluence_api_token=None,
        )
        publisher = ConfluencePublisher(config=settings)
        assert publisher._enabled is False


# ---------------------------------------------------------------------------
# Tests: publish() returns None when disabled
# ---------------------------------------------------------------------------


class TestPublishDisabled:
    """FR-7: Confluence integration gracefully disabled when not configured."""

    def test_publish_returns_none_when_disabled(self) -> None:
        publisher = _make_publisher(enabled=False)
        task_ctx = _make_task_ctx()
        result = _make_result()

        async def _run():
            return await publisher.publish(result, task_ctx, None, None, None)

        url = asyncio.run(_run())
        assert url is None

    def test_publish_returns_none_when_no_llm_router(self) -> None:
        settings = _make_settings(**_CONFLUENCE_CREDS)
        publisher = ConfluencePublisher(config=settings, llm_router=None)
        task_ctx = _make_task_ctx()
        result = _make_result()

        async def _run():
            return await publisher.publish(result, task_ctx, None, None, None)

        url = asyncio.run(_run())
        assert url is None


# ---------------------------------------------------------------------------
# Tests: publish() error handling
# ---------------------------------------------------------------------------


class TestPublishErrorHandling:
    """FR-8, CP-4: Pipeline never fails due to Confluence errors."""

    def test_publish_returns_none_on_agent_error(self) -> None:
        publisher = _make_publisher(enabled=True)
        task_ctx = _make_task_ctx()
        result = _make_result()

        mock_agent = MagicMock()
        mock_agent.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("Confluence unavailable")
        )
        mock_agent.return_value.__aexit__ = AsyncMock(return_value=None)

        async def _run():
            with patch("src.pipeline.confluence_publisher.Agent", mock_agent):
                return await publisher.publish(result, task_ctx, None, None, None)

        url = asyncio.run(_run())
        assert url is None  # Never raises, returns None

    def test_publish_never_raises(self) -> None:
        publisher = _make_publisher(enabled=True)
        task_ctx = _make_task_ctx()
        result = _make_result()

        async def _run():
            with patch.object(
                publisher, "_create_page", side_effect=Exception("boom")
            ):
                try:
                    return await publisher.publish(result, task_ctx, None, None, None)
                except Exception as e:
                    return e

        outcome = asyncio.run(_run())
        assert not isinstance(outcome, Exception), f"publish() raised: {outcome!r}"
        assert outcome is None


# ---------------------------------------------------------------------------
# Tests: _build_page_content()
# ---------------------------------------------------------------------------


class TestBuildPageContent:
    """CP-5: Page content always contains required sections."""

    def test_contains_issue_key(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx("PROJ-42")
        result = _make_result()
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert "PROJ-42" in content

    def test_contains_success_status(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result(success=True)
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert "SUCCESS" in content

    def test_contains_failure_status(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result(success=False, failure_stage="review")
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert "FAILED" in content
        assert "review" in content

    def test_contains_changed_files(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        code_change = _make_code_change()
        content = publisher._build_page_content(result, task_ctx, code_change, None, None, 0.0)
        assert "src/auth.py" in content
        assert "tests/test_auth.py" in content

    def test_contains_review_verdict(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        review = _make_review()
        content = publisher._build_page_content(result, task_ctx, None, review, None, 0.0)
        assert "APPROVE" in content
        assert "8/10" in content

    def test_contains_pr_link(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        pr_url = "https://github.com/org/repo/pull/42"
        content = publisher._build_page_content(result, task_ctx, None, None, pr_url, 0.0)
        assert pr_url in content

    def test_no_pr_section_when_no_pr(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert "Pull Request" not in content

    def test_diff_truncated_to_max_chars(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        long_content = "x" * (_MAX_DIFF_CHARS + 1000)
        code_change = CodeChange(
            changes=[
                FileChange(
                    path="src/big.py",
                    new_content=long_content,
                    change_type=ChangeType.MODIFY,
                    explanation="big change",
                )
            ],
            test_changes=[],
            commit_message="feat: big",
            pr_title="big",
            pr_description="big",
        )
        content = publisher._build_page_content(result, task_ctx, code_change, None, None, 0.0)
        assert "[truncated]" in content

    def test_diff_not_truncated_when_within_limit(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        short_content = "x" * 100
        code_change = CodeChange(
            changes=[
                FileChange(
                    path="src/small.py",
                    new_content=short_content,
                    change_type=ChangeType.MODIFY,
                    explanation="small change",
                )
            ],
            test_changes=[],
            commit_message="fix: small",
            pr_title="small",
            pr_description="small",
        )
        content = publisher._build_page_content(result, task_ctx, code_change, None, None, 0.0)
        assert "[truncated]" not in content

    def test_no_code_change_no_files_section(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        result = _make_result()
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert "Changed Files" not in content

    def test_contains_jira_link(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx("TEST-99")
        result = _make_result()
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert "jira.example.com" in content
        assert "TEST-99" in content


# ---------------------------------------------------------------------------
# Tests: _build_page_title()
# ---------------------------------------------------------------------------


class TestBuildPageTitle:
    def test_title_contains_issue_key(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx("PROJ-7")
        title = publisher._build_page_title(task_ctx)
        assert "PROJ-7" in title

    def test_title_contains_ai_agent_prefix(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        title = publisher._build_page_title(task_ctx)
        assert "[AI-Agent]" in title

    def test_title_contains_summary(self) -> None:
        publisher = _make_publisher()
        task_ctx = _make_task_ctx()
        title = publisher._build_page_title(task_ctx)
        assert task_ctx.summary in title


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestConfluencePublisherProperties:
    """Property-based tests for ConfluencePublisher correctness properties."""

    @given(
        issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True),
        success=st.booleans(),
    )
    @h_settings(max_examples=50)
    def test_page_content_always_contains_issue_key(
        self, issue_key: str, success: bool
    ) -> None:
        """CP-5: Page content always contains the issue key."""
        publisher = _make_publisher()
        task_ctx = _make_task_ctx(issue_key)
        result = _make_result(success=success)
        content = publisher._build_page_content(result, task_ctx, None, None, None, 0.0)
        assert issue_key in content

    @given(success=st.booleans())
    @h_settings(max_examples=20)
    def test_publish_disabled_always_returns_none(self, success: bool) -> None:
        """CP-3: Disabled publisher always returns None."""
        publisher = _make_publisher(enabled=False)
        task_ctx = _make_task_ctx()
        result = _make_result(success=success)

        async def _run():
            return await publisher.publish(result, task_ctx, None, None, None)

        url = asyncio.run(_run())
        assert url is None

    @given(
        error_msg=st.text(min_size=1, max_size=100),
    )
    @h_settings(max_examples=20)
    def test_publish_never_raises_on_error(self, error_msg: str) -> None:
        """CP-4: Confluence errors never fail the pipeline."""
        publisher = _make_publisher(enabled=True)
        task_ctx = _make_task_ctx()
        result = _make_result()

        async def _run():
            with patch.object(
                publisher, "_create_page", side_effect=Exception(error_msg)
            ):
                try:
                    return await publisher.publish(result, task_ctx, None, None, None)
                except Exception as e:
                    return e

        outcome = asyncio.run(_run())
        assert not isinstance(outcome, Exception)
        assert outcome is None
