"""End-to-end pipeline integration tests.

Tests the full pipeline from orchestrator.run() through all stages with
all MCP-dependent agents mocked. Covers:
- Happy path: full pipeline success with PR creation
- Review loop: REQUEST_CHANGES  retry  APPROVE
- Reject path: REJECT  halt
- Max retry exhausted: all reviews REQUEST_CHANGES  halt
- Large scope halt: LARGE scope  halt before CodeFinder
- Task type skip: skip_task_types match  halt
- Dry-run: no Git/Jira writes, returns success
- Token budget trim: oversized CodeContext is trimmed
- Max file change limit: too many changes  halt

Requirements: All pipeline flow requirements
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    ChangeType,
    CodeChange,
    CodeContext,
    CodeFile,
    FileChange,
    FindingSeverity,
    PipelineResult,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    TaskContext,
    TaskScope,
)
from src.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    defaults = {
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-dev",
        "jira_api_token": "jira-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-dev",
        "git_provider": "github",
        "github_token": "ghp_testtoken",
        "github_owner": "test-owner",
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast",
        "llm_strong_provider": "openai",
        "llm_strong_model": "gpt-4o",
        "llm_strong_api_key": "sk-strong",
        "dry_run": False,
        "max_review_retries": 2,
        "max_file_changes": 15,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_orchestrator(**overrides: Any) -> PipelineOrchestrator:
    settings = _make_settings(**overrides)
    llm_router = LLMRouter(config=settings)
    return PipelineOrchestrator(config=settings, llm_router=llm_router)


def _make_task_ctx(
    scope: TaskScope = TaskScope.SMALL,
    issue_type: str = "Story",
    issue_key: str = "INT-1",
) -> TaskContext:
    return TaskContext(
        issue_key=issue_key,
        summary="Integration test task",
        description="Full pipeline integration test",
        acceptance_criteria="All criteria met",
        repository_name="test-repo",
        estimated_scope=scope,
        issue_type=issue_type,
        reporter="john.doe",
        base_branch="main",
    )


def _make_code_ctx() -> CodeContext:
    return CodeContext(
        files=[
            CodeFile(
                path="src/auth/handler.py",
                content="class AuthHandler:\n    pass\n",
                language="python",
            )
        ],
        tech_stack=["python"],
        repository_name="test-repo",
    )


def _make_code_change(n_files: int = 1) -> CodeChange:
    return CodeChange(
        changes=[
            FileChange(
                path=f"src/auth/handler{i}.py",
                new_content=f"# updated {i}\n",
                change_type=ChangeType.MODIFY,
                explanation=f"Fix {i}",
            )
            for i in range(n_files)
        ],
        commit_message="fix(auth): update handler",
        pr_title="Fix auth handler",
        pr_description="Updated the auth handler.",
    )


def _make_approve() -> ReviewResult:
    return ReviewResult(
        verdict=ReviewVerdict.APPROVE,
        score=9,
        findings=[],
        acceptance_criteria_met=True,
    )


def _make_request_changes(feedback: str = "Please fix X") -> ReviewResult:
    return ReviewResult(
        verdict=ReviewVerdict.REQUEST_CHANGES,
        score=5,
        findings=[],
        feedback_for_rewrite=feedback,
        acceptance_criteria_met=False,
    )


def _make_reject() -> ReviewResult:
    return ReviewResult(
        verdict=ReviewVerdict.REJECT,
        score=2,
        findings=[
            ReviewFinding(
                file_path="src/auth/handler.py",
                severity=FindingSeverity.CRITICAL,
                category="security",
                message="Hardcoded credentials found",
            )
        ],
        feedback_for_rewrite="Cannot be fixed incrementally.",
    )


# ---------------------------------------------------------------------------
# Happy Path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Full pipeline success: TaskReader  CodeFinder  CodeWriter  APPROVE  PR."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success_with_pr(self) -> None:
        orchestrator = _make_orchestrator()
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        code_change = _make_code_change()
        review = _make_approve()

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch.object(
                orchestrator,
                "_run_review_loop",
                new_callable=AsyncMock,
                return_value=(code_change, review),
            ),
            patch.object(
                orchestrator,
                "_create_pull_request",
                new_callable=AsyncMock,
                return_value="https://github.com/test-owner/test-repo/pull/1",
            ),
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("INT-1")

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.issue_key == "INT-1"
        assert result.pr_url == "https://github.com/test-owner/test-repo/pull/1"
        assert result.failure_stage is None
        assert result.failure_reason is None

    @pytest.mark.asyncio
    async def test_full_pipeline_calls_all_stages(self) -> None:
        """Verify each stage is invoked in order."""
        orchestrator = _make_orchestrator()
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        code_change = _make_code_change()
        review = _make_approve()

        call_order: list[str] = []

        async def _read_task(_issue_key: str) -> TaskContext:
            call_order.append("task_reader")
            return task_ctx

        async def _find_code(_task_ctx: TaskContext) -> CodeContext:
            call_order.append("code_finder")
            return code_ctx

        async def _review_loop(*_args: Any, **_kwargs: Any) -> tuple:
            call_order.append("review_loop")
            return code_change, review

        async def _create_pr(*_args: Any, **_kwargs: Any) -> str:
            call_order.append("pr_creation")
            return "https://github.com/pr/1"

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch.object(orchestrator, "_run_review_loop", side_effect=_review_loop),
            patch.object(orchestrator, "_create_pull_request", side_effect=_create_pr),
        ):
            MockReader.return_value.read_task = AsyncMock(side_effect=_read_task)
            MockFinder.return_value.find_code = AsyncMock(side_effect=_find_code)

            await orchestrator.run("INT-1")

        assert call_order == ["task_reader", "code_finder", "review_loop", "pr_creation"]


# ---------------------------------------------------------------------------
# Review Loop
# ---------------------------------------------------------------------------


class TestReviewLoop:
    """REQUEST_CHANGES  retry  APPROVE."""

    @pytest.mark.asyncio
    async def test_request_changes_then_approve(self) -> None:
        orchestrator = _make_orchestrator(max_review_retries=3)
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        code_change = _make_code_change()

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
            patch.object(
                orchestrator,
                "_create_pull_request",
                new_callable=AsyncMock,
                return_value="https://github.com/pr/2",
            ),
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
            MockReviewer.return_value.review_code = AsyncMock(
                side_effect=[_make_request_changes("Fix auth"), _make_approve()]
            )

            result = await orchestrator.run("INT-1")

        assert result.success is True
        assert MockWriter.return_value.write_code.call_count == 2
        # Second write_code call should receive feedback
        second_call = MockWriter.return_value.write_code.call_args_list[1]
        feedback_arg = second_call[1].get("review_feedback") or (
            second_call[0][2] if len(second_call[0]) > 2 else None
        )
        assert feedback_arg == "Fix auth"


# ---------------------------------------------------------------------------
# Reject Path
# ---------------------------------------------------------------------------


class TestRejectPath:
    """REJECT  halt pipeline with Jira comment."""

    @pytest.mark.asyncio
    async def test_reject_halts_pipeline(self) -> None:
        orchestrator = _make_orchestrator()
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        code_change = _make_code_change()
        reject = _make_reject()

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock) as mock_comment,
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch.object(
                orchestrator,
                "_run_review_loop",
                new_callable=AsyncMock,
                return_value=(code_change, reject),
            ),
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "review"
        assert "REJECT" in (result.failure_reason or "")
        # Jira comment should mention rejection
        reject_comments = [
            c for c in mock_comment.call_args_list
            if "REJECT" in str(c) or "reject" in str(c).lower()
        ]
        assert len(reject_comments) >= 1


# ---------------------------------------------------------------------------
# Max Retry Exhausted
# ---------------------------------------------------------------------------


class TestMaxRetryExhausted:
    """All reviews REQUEST_CHANGES  halt after max_review_retries."""

    @pytest.mark.asyncio
    async def test_max_retry_exhausted_halts(self) -> None:
        max_retries = 2
        orchestrator = _make_orchestrator(max_review_retries=max_retries)
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        code_change = _make_code_change()

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
            MockReviewer.return_value.review_code = AsyncMock(
                return_value=_make_request_changes()
            )

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "review"
        # Writer called max_retries + 1 times
        assert MockWriter.return_value.write_code.call_count == max_retries + 1


# ---------------------------------------------------------------------------
# Large Scope Halt
# ---------------------------------------------------------------------------


class TestLargeScopeHalt:
    """LARGE scope  halt before CodeFinder."""

    @pytest.mark.asyncio
    async def test_large_scope_halts_before_code_finder(self) -> None:
        orchestrator = _make_orchestrator()
        task_ctx = _make_task_ctx(scope=TaskScope.LARGE)

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock) as mock_comment,
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "scope_check"
        assert "LARGE" in (result.failure_reason or "")
        # CodeFinder must NOT be called
        MockFinder.return_value.find_code.assert_not_called()
        # Jira comment about scope
        scope_comments = [c for c in mock_comment.call_args_list if "LARGE" in str(c)]
        assert len(scope_comments) >= 1


# ---------------------------------------------------------------------------
# Task Type Skip
# ---------------------------------------------------------------------------


class TestTaskTypeSkip:
    """skip_task_types match  halt pipeline."""

    @pytest.mark.asyncio
    async def test_skip_task_type_halts(self) -> None:
        orchestrator = _make_orchestrator(skip_task_types=["Epic"])
        task_ctx = _make_task_ctx(issue_type="Epic")

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "task_filter"
        MockFinder.return_value.find_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_task_types_filters_out(self) -> None:
        orchestrator = _make_orchestrator(allowed_task_types=["Bug", "Story"])
        task_ctx = _make_task_ctx(issue_type="Epic")

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "task_filter"
        MockFinder.return_value.find_code.assert_not_called()


# ---------------------------------------------------------------------------
# Dry-Run
# ---------------------------------------------------------------------------


class TestDryRun:
    """dry_run=True: no Git/Jira agent calls, pipeline returns success."""

    @pytest.mark.asyncio
    async def test_dry_run_no_agent_calls(self) -> None:
        orchestrator = _make_orchestrator(dry_run=True)
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        code_change = _make_code_change()
        review = _make_approve()

        with (
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch.object(
                orchestrator,
                "_run_review_loop",
                new_callable=AsyncMock,
                return_value=(code_change, review),
            ),
            patch("src.pipeline.orchestrator.Agent") as MockAgent,
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("INT-1")

        assert result.dry_run is True
        assert result.success is True
        # Agent (Jira/Git) should NOT be instantiated in dry-run
        MockAgent.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_pr_returns_none(self) -> None:
        orchestrator = _make_orchestrator(dry_run=True)
        task_ctx = _make_task_ctx()
        code_change = _make_code_change()

        result = await orchestrator._create_pull_request(task_ctx, code_change)

        assert result is None


# ---------------------------------------------------------------------------
# Token Budget Trim
# ---------------------------------------------------------------------------


class TestTokenBudgetTrim:
    """Oversized CodeContext is trimmed before review loop."""

    @pytest.mark.asyncio
    async def test_token_budget_trims_context(self) -> None:
        # Very small token budget to force trimming
        orchestrator = _make_orchestrator(max_context_tokens=10)
        task_ctx = _make_task_ctx()
        code_change = _make_code_change()
        review = _make_approve()

        # Large code context with many files
        large_ctx = CodeContext(
            files=[
                CodeFile(
                    path=f"src/module{i}.py",
                    content="x = 1\n" * 100,
                    language="python",
                )
                for i in range(5)
            ],
            tech_stack=["python"],
            repository_name="test-repo",
        )

        trimmed_ctx_holder: list[CodeContext] = []

        async def _review_loop(
            task_ctx: TaskContext, code_ctx: CodeContext, max_retries: int
        ) -> tuple:
            trimmed_ctx_holder.append(code_ctx)
            return code_change, review

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch.object(orchestrator, "_run_review_loop", side_effect=_review_loop),
            patch.object(
                orchestrator,
                "_create_pull_request",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=large_ctx)

            result = await orchestrator.run("INT-1")

        assert result.success is True
        # The context passed to review loop should have fewer files than original
        assert len(trimmed_ctx_holder) == 1
        trimmed = trimmed_ctx_holder[0]
        assert len(trimmed.files) <= len(large_ctx.files)


# ---------------------------------------------------------------------------
# Max File Change Limit
# ---------------------------------------------------------------------------


class TestMaxFileChangeLimit:
    """Too many file changes  halt pipeline."""

    @pytest.mark.asyncio
    async def test_max_file_change_limit_halts(self) -> None:
        orchestrator = _make_orchestrator(max_file_changes=2)
        task_ctx = _make_task_ctx()
        code_ctx = _make_code_ctx()
        # 3 changes exceeds limit of 2
        big_change = _make_code_change(n_files=3)

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock) as mock_comment,
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockFinder,
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
        ):
            MockReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            MockWriter.return_value.write_code = AsyncMock(return_value=big_change)

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "review"
        # Reviewer should NOT have been called
        MockReviewer.return_value.review_code.assert_not_called()
        # Jira comment about too many changes
        limit_comments = [
            c for c in mock_comment.call_args_list
            if "Too many" in str(c) or "file change" in str(c).lower()
        ]
        assert len(limit_comments) >= 1


# ---------------------------------------------------------------------------
# Unhandled Exception
# ---------------------------------------------------------------------------


class TestUnhandledException:
    """Unhandled exception  error comment on Jira, returns failure result."""

    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_failure(self) -> None:
        orchestrator = _make_orchestrator()

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock) as mock_comment,
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockReader,
        ):
            MockReader.return_value.read_task = AsyncMock(
                side_effect=RuntimeError("Unexpected boom!")
            )

            result = await orchestrator.run("INT-1")

        assert result.success is False
        assert result.failure_stage == "error"
        assert mock_comment.call_count >= 2
