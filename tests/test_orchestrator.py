"""Unit tests for PipelineOrchestrator - basic flow and Jira communication.

Covers:
- Basic pipeline flow (happy path with mocked agents)
- Large scope halts pipeline
- Task type filtering skips pipeline
- Jira comment is sent on start
- Unhandled exception produces error comment
- Dry-run mode skips Jira writes
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Create a Settings instance with sensible defaults for testing."""
    defaults = {
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-dev",
        "jira_api_token": "jira-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-dev",
        "jira_transition_in_progress": "21",
        "jira_transition_in_review": "31",
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
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_task_context(
    issue_key: str = "TEST-1",
    scope: TaskScope = TaskScope.SMALL,
    issue_type: str | None = "Story",
) -> TaskContext:
    return TaskContext(
        issue_key=issue_key,
        summary="Test task",
        description="Test description",
        repository_name="my-repo",
        estimated_scope=scope,
        issue_type=issue_type,
        base_branch="main",
    )


def _make_code_context() -> CodeContext:
    return CodeContext(
        files=[
            CodeFile(
                path="src/main.py",
                content="print('hello')\n",
                language="python",
            )
        ],
        tech_stack=["python"],
        repository_name="my-repo",
    )


def _make_code_change() -> CodeChange:
    return CodeChange(
        changes=[
            FileChange(
                path="src/main.py",
                new_content="print('updated')\n",
                change_type=ChangeType.MODIFY,
                explanation="Updated greeting",
            )
        ],
        commit_message="fix(main): update greeting",
        pr_title="Fix greeting",
        pr_description="Updated the greeting message.",
    )


def _make_review_approve() -> ReviewResult:
    return ReviewResult(
        verdict=ReviewVerdict.APPROVE,
        score=8,
        findings=[],
        acceptance_criteria_met=True,
    )


def _make_review_reject() -> ReviewResult:
    return ReviewResult(
        verdict=ReviewVerdict.REJECT,
        score=3,
        findings=[
            ReviewFinding(
                file_path="src/main.py",
                severity=FindingSeverity.CRITICAL,
                category="logic",
                message="Fundamental design flaw",
            )
        ],
        feedback_for_rewrite="Cannot be fixed with simple changes.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineOrchestratorHappyPath:
    """Basic pipeline flow - happy path with mocked agents."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_success(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ) as mock_comment,
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ) as mock_transition,
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
            patch(
                "src.pipeline.orchestrator.CodeFinderAgent"
            ) as MockCodeFinder,
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
                return_value="https://github.com/pr/1",
            ),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.success is True
        assert result.issue_key == "TEST-1"
        assert result.pr_url == "https://github.com/pr/1"
        assert result.failure_stage is None

        # Jira comment called at least for start and completion
        assert mock_comment.call_count >= 2
        # Transition called for In Progress and In Review
        assert mock_transition.call_count == 2


class TestLargeScopeHalt:
    """Large scope halts pipeline before CodeFinder."""

    @pytest.mark.asyncio
    async def test_large_scope_halts_pipeline(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(scope=TaskScope.LARGE)

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ) as mock_comment,
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
            patch(
                "src.pipeline.orchestrator.CodeFinderAgent"
            ) as MockCodeFinder,
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.success is False
        assert result.failure_stage == "scope_check"
        assert "LARGE" in (result.failure_reason or "")
        # CodeFinder should NOT have been called
        MockCodeFinder.return_value.find_code.assert_not_called()
        # Jira comment for start + scope halt
        assert mock_comment.call_count >= 2


class TestTaskTypeFiltering:
    """Task type filtering skips pipeline."""

    @pytest.mark.asyncio
    async def test_skip_task_type_halts_pipeline(self) -> None:
        settings = _make_settings(skip_task_types=["Epic"])
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_type="Epic")

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ),
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
            patch(
                "src.pipeline.orchestrator.CodeFinderAgent"
            ) as MockCodeFinder,
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.success is False
        assert result.failure_stage == "task_filter"
        assert "skip" in (result.failure_reason or "").lower()
        MockCodeFinder.return_value.find_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_task_types_filters_out(self) -> None:
        settings = _make_settings(allowed_task_types=["Bug", "Story"])
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_type="Epic")

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ),
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.success is False
        assert result.failure_stage == "task_filter"


class TestJiraCommentOnStart:
    """Jira comment is sent at pipeline start."""

    @pytest.mark.asyncio
    async def test_start_comment_sent(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ) as mock_comment,
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
            patch(
                "src.pipeline.orchestrator.CodeFinderAgent"
            ) as MockCodeFinder,
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
                return_value=None,
            ),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            await orchestrator.run("TEST-1")

        # First call should be the "started" comment
        first_call_args = mock_comment.call_args_list[0]
        assert first_call_args[0][0] == "TEST-1"
        assert "started" in first_call_args[0][1].lower()


class TestUnhandledException:
    """Unhandled exception produces error comment on Jira."""

    @pytest.mark.asyncio
    async def test_unhandled_exception_comments_on_jira(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ) as mock_comment,
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
        ):
            # TaskReader raises an unexpected error
            MockTaskReader.return_value.read_task = AsyncMock(
                side_effect=RuntimeError("Unexpected boom!")
            )

            result = await orchestrator.run("TEST-1")

        assert result.success is False
        assert result.failure_stage == "error"
        assert "Unhandled exception" in (result.failure_reason or "")

        # Should have at least 2 comments: start + error
        assert mock_comment.call_count >= 2
        # Last comment should be the error comment
        last_call_args = mock_comment.call_args_list[-1]
        assert "error" in last_call_args[0][1].lower() or "unexpected" in last_call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_pipeline_result(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ),
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
        ):
            MockTaskReader.return_value.read_task = AsyncMock(
                side_effect=ValueError("Bad data")
            )

            result = await orchestrator.run("TEST-1")

        assert isinstance(result, PipelineResult)
        assert result.success is False
        assert result.issue_key == "TEST-1"


class TestDryRunMode:
    """Dry-run mode skips Jira writes."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_jira_comment(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = _make_settings(dry_run=True)
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        # Call _comment_on_jira directly
        with caplog.at_level(logging.INFO):
            await orchestrator._comment_on_jira("TEST-1", "Hello")

        assert any("[DRY-RUN]" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_dry_run_skips_jira_transition(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = _make_settings(dry_run=True)
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        with caplog.at_level(logging.INFO):
            await orchestrator._transition_jira("TEST-1", "21")

        assert any("[DRY-RUN]" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_dry_run_full_pipeline_no_jira_agent_calls(self) -> None:
        """Full pipeline in dry-run: no actual Jira agent calls are made."""
        settings = _make_settings(dry_run=True)
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        with (
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
            patch(
                "src.pipeline.orchestrator.CodeFinderAgent"
            ) as MockCodeFinder,
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
                return_value=None,
            ),
            patch(
                "src.pipeline.orchestrator.Agent"
            ) as MockAgent,
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.dry_run is True
        assert result.success is True
        # Agent (for Jira) should NOT have been instantiated
        MockAgent.assert_not_called()


class TestRejectVerdict:
    """Review REJECT halts pipeline with Jira comment."""

    @pytest.mark.asyncio
    async def test_reject_halts_pipeline(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_reject()

        with (
            patch.object(
                orchestrator, "_comment_on_jira", new_callable=AsyncMock
            ) as mock_comment,
            patch.object(
                orchestrator, "_transition_jira", new_callable=AsyncMock
            ),
            patch(
                "src.pipeline.orchestrator.TaskReaderAgent"
            ) as MockTaskReader,
            patch(
                "src.pipeline.orchestrator.CodeFinderAgent"
            ) as MockCodeFinder,
            patch.object(
                orchestrator,
                "_run_review_loop",
                new_callable=AsyncMock,
                return_value=(code_change, review),
            ),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.success is False
        assert result.failure_stage == "review"
        assert "REJECT" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# TestReviewLoop
# ---------------------------------------------------------------------------


class TestReviewLoop:
    """_run_review_loop behavior."""

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def _make_request_changes_review(self, feedback: str = "Please fix X") -> ReviewResult:
        return ReviewResult(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            score=5,
            findings=[],
            feedback_for_rewrite=feedback,
            acceptance_criteria_met=False,
        )

    @pytest.mark.asyncio
    async def test_approve_on_first_try(self) -> None:
        """CodeWriter returns change, CodeReviewer returns APPROVE  returns immediately."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        with (
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
        ):
            MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
            MockReviewer.return_value.review_code = AsyncMock(return_value=review)

            result_change, result_review = await orchestrator._run_review_loop(
                task_ctx, code_ctx, max_retries=3
            )

        assert result_review.verdict == ReviewVerdict.APPROVE
        assert result_change is code_change
        # Writer called exactly once
        MockWriter.return_value.write_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_changes_retries(self) -> None:
        """First review: REQUEST_CHANGES, second: APPROVE  2 CodeWriter calls."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        request_changes = self._make_request_changes_review("Fix the bug")
        approve = _make_review_approve()

        with (
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
        ):
            MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
            MockReviewer.return_value.review_code = AsyncMock(
                side_effect=[request_changes, approve]
            )

            result_change, result_review = await orchestrator._run_review_loop(
                task_ctx, code_ctx, max_retries=3
            )

        assert result_review.verdict == ReviewVerdict.APPROVE
        assert MockWriter.return_value.write_code.call_count == 2
        # Second call should pass feedback
        second_call_kwargs = MockWriter.return_value.write_code.call_args_list[1]
        assert second_call_kwargs[1].get("review_feedback") == "Fix the bug" or \
               second_call_kwargs[0][2] == "Fix the bug"

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        """All reviews REQUEST_CHANGES  returns last (change, review) after max_retries+1 iterations."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        request_changes = self._make_request_changes_review()

        with (
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
        ):
            MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
            MockReviewer.return_value.review_code = AsyncMock(return_value=request_changes)

            max_retries = 2
            result_change, result_review = await orchestrator._run_review_loop(
                task_ctx, code_ctx, max_retries=max_retries
            )

        assert result_review.verdict == ReviewVerdict.REQUEST_CHANGES
        # max_retries + 1 total iterations
        assert MockWriter.return_value.write_code.call_count == max_retries + 1

    @pytest.mark.asyncio
    async def test_reject_returns_immediately(self) -> None:
        """CodeReviewer returns REJECT  returns immediately without retry."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        reject = _make_review_reject()

        with (
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
        ):
            MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
            MockReviewer.return_value.review_code = AsyncMock(return_value=reject)

            result_change, result_review = await orchestrator._run_review_loop(
                task_ctx, code_ctx, max_retries=5
            )

        assert result_review.verdict == ReviewVerdict.REJECT
        # Only one iteration - no retry on REJECT
        MockWriter.return_value.write_code.assert_called_once()
        MockReviewer.return_value.review_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_file_changes_halts(self) -> None:
        """CodeWriter returns change with too many files  returns REJECT ReviewResult."""
        orchestrator = self._make_orchestrator(max_file_changes=2)
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()

        # Build a code_change with 3 file changes (exceeds limit of 2)
        big_change = CodeChange(
            changes=[
                FileChange(path=f"src/file{i}.py", new_content="x", change_type=ChangeType.MODIFY, explanation="x")
                for i in range(3)
            ],
            commit_message="fix: many files",
            pr_title="Many files",
            pr_description="desc",
        )

        with (
            patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
            patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock) as mock_comment,
        ):
            MockWriter.return_value.write_code = AsyncMock(return_value=big_change)

            result_change, result_review = await orchestrator._run_review_loop(
                task_ctx, code_ctx, max_retries=3
            )

        assert result_review.verdict == ReviewVerdict.REJECT
        assert any(
            f.severity == FindingSeverity.CRITICAL for f in result_review.findings
        )
        assert "Too many file changes" in (result_review.feedback_for_rewrite or "")
        # Reviewer should NOT have been called
        MockReviewer.return_value.review_code.assert_not_called()
        # Jira comment should have been made
        mock_comment.assert_called_once()


# ---------------------------------------------------------------------------
# TestCreatePullRequest
# ---------------------------------------------------------------------------


class TestCreatePullRequest:
    """_create_pull_request behavior."""

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    @pytest.mark.asyncio
    async def test_dry_run_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        orchestrator = self._make_orchestrator(dry_run=True)
        task_ctx = _make_task_context()
        code_change = _make_code_change()

        with caplog.at_level(logging.INFO):
            result = await orchestrator._create_pull_request(task_ctx, code_change)

        assert result is None
        assert any("[DRY-RUN]" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_auto_create_pr_false_comments_jira(self) -> None:
        orchestrator = self._make_orchestrator(auto_create_pr=False)
        task_ctx = _make_task_context()
        code_change = _make_code_change()

        with patch.object(
            orchestrator, "_comment_on_jira", new_callable=AsyncMock
        ) as mock_comment:
            result = await orchestrator._create_pull_request(task_ctx, code_change)

        assert result is None
        mock_comment.assert_called_once()
        # Comment should contain changes summary
        comment_text = mock_comment.call_args[0][1]
        assert "src/main.py" in comment_text

    @pytest.mark.asyncio
    async def test_creates_pr_via_git_agent(self) -> None:
        orchestrator = self._make_orchestrator(auto_create_pr=True)
        task_ctx = _make_task_context()
        code_change = _make_code_change()

        mock_llm = MagicMock()
        mock_llm.generate_str = AsyncMock(
            return_value="PR created: https://github.com/owner/repo/pull/42"
        )

        mock_agent_instance = MagicMock()
        mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
        mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
        mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)

        with patch("src.pipeline.orchestrator.Agent", return_value=mock_agent_instance):
            result = await orchestrator._create_pull_request(task_ctx, code_change)

        assert result == "https://github.com/owner/repo/pull/42"

    @pytest.mark.asyncio
    async def test_branch_collision_retries_with_suffix(self) -> None:
        orchestrator = self._make_orchestrator(auto_create_pr=True)
        task_ctx = _make_task_context()
        code_change = _make_code_change()

        mock_llm = MagicMock()
        mock_llm.generate_str = AsyncMock(
            return_value="PR created: https://github.com/owner/repo/pull/99"
        )

        call_count = 0

        class FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> "FakeAgent":
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("branch already exists")
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def attach_llm(self, llm_class: type) -> Any:
                return mock_llm

        with patch("src.pipeline.orchestrator.Agent", FakeAgent):
            result = await orchestrator._create_pull_request(task_ctx, code_change)

        assert result == "https://github.com/owner/repo/pull/99"
        # Should have been called twice (first collision, then retry)
        assert call_count == 2


# ---------------------------------------------------------------------------
# TestBuildPrDescription
# ---------------------------------------------------------------------------


class TestBuildPrDescription:
    """Unit tests for _build_pr_description()."""

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def test_basic_description_with_review(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context(issue_key="PROJ-42")
        code_change = _make_code_change()
        review = _make_review_approve()

        result = orchestrator._build_pr_description(task_ctx, code_change, review)

        assert "## Summary" in result
        assert "Test task" in result
        assert "## Changes" in result
        assert "src/main.py" in result
        assert "modify" in result
        assert "## Review" in result
        assert "APPROVE" in result
        assert "8/10" in result
        assert "## Jira" in result
        assert "[PROJ-42]" in result
        assert "jira.example.com/browse/PROJ-42" in result

    def test_description_without_review(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()

        result = orchestrator._build_pr_description(task_ctx, code_change, review=None)

        assert "## Summary" in result
        assert "## Changes" in result
        assert "## Review" not in result
        assert "## Jira" in result

    def test_description_with_review_findings(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        review = ReviewResult(
            verdict=ReviewVerdict.APPROVE,
            score=8,
            findings=[
                ReviewFinding(
                    file_path="src/main.py",
                    severity=FindingSeverity.WARNING,
                    category="style",
                    message="Consider adding docstring",
                ),
                ReviewFinding(
                    file_path="src/main.py",
                    severity=FindingSeverity.SUGGESTION,
                    category="style",
                    message="Use type hints",
                ),
            ],
            acceptance_criteria_met=True,
        )

        result = orchestrator._build_pr_description(task_ctx, code_change, review)

        assert "[warning] Consider adding docstring" in result
        assert "[suggestion] Use type hints" in result

    def test_description_with_empty_changes(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = CodeChange(
            changes=[],
            test_changes=[],
            commit_message="empty",
            pr_title="Empty",
            pr_description="No changes",
        )

        result = orchestrator._build_pr_description(task_ctx, code_change)

        assert "## Summary" in result
        assert "## Changes" not in result
        assert "## Jira" in result

    def test_description_includes_test_changes(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = CodeChange(
            changes=[
                FileChange(
                    path="src/main.py",
                    new_content="print('updated')\n",
                    change_type=ChangeType.MODIFY,
                    explanation="Updated",
                )
            ],
            test_changes=[
                FileChange(
                    path="tests/test_main.py",
                    new_content="def test(): pass\n",
                    change_type=ChangeType.CREATE,
                    explanation="Added test",
                )
            ],
            commit_message="fix: update",
            pr_title="Fix",
            pr_description="desc",
        )

        result = orchestrator._build_pr_description(task_ctx, code_change)

        assert "src/main.py" in result
        assert "tests/test_main.py" in result
        assert "modify" in result
        assert "create" in result

    def test_jira_url_trailing_slash_stripped(self) -> None:
        orchestrator = self._make_orchestrator(jira_url="https://jira.example.com/")
        task_ctx = _make_task_context(issue_key="TEST-1")
        code_change = _make_code_change()

        result = orchestrator._build_pr_description(task_ctx, code_change)

        assert "https://jira.example.com/browse/TEST-1" in result
        assert "https://jira.example.com//browse" not in result


# ---------------------------------------------------------------------------
# TestBuildPrPrompt
# ---------------------------------------------------------------------------


class TestBuildPrPrompt:
    """Unit tests for updated _build_pr_prompt()."""

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def test_title_uses_ai_bot_prefix(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context(issue_key="PROJ-10")
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/PROJ-10-ai", all_changes
        )

        assert "[AI-BOT] PROJ-10: Test task" in result

    def test_draft_mode_instruction_when_enabled(self) -> None:
        orchestrator = self._make_orchestrator(pr_draft_mode=True)
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/TEST-1-ai", all_changes
        )

        assert "Create PR as draft" in result

    def test_no_draft_instruction_when_disabled(self) -> None:
        orchestrator = self._make_orchestrator(pr_draft_mode=False)
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/TEST-1-ai", all_changes
        )

        assert "draft" not in result.lower()

    def test_reviewer_instruction_from_config(self) -> None:
        orchestrator = self._make_orchestrator(pr_reviewer="alice, bob")
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/TEST-1-ai", all_changes
        )

        assert "Assign alice, bob as reviewers" in result

    def test_no_reviewer_instruction_when_empty(self) -> None:
        orchestrator = self._make_orchestrator(pr_reviewer="")
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/TEST-1-ai", all_changes
        )

        assert "reviewer" not in result.lower()

    def test_prompt_includes_pr_description_body(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context(issue_key="TEST-5")
        code_change = _make_code_change()
        all_changes = code_change.changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/TEST-5-ai", all_changes
        )

        # Should contain structured description sections
        assert "## Summary" in result
        assert "## Jira" in result

    def test_prompt_with_review_passed(self) -> None:
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        review = _make_review_approve()
        all_changes = code_change.changes

        result = orchestrator._build_pr_prompt(
            task_ctx, code_change, "feature/TEST-1-ai", all_changes, review=review
        )

        assert "## Review" in result
        assert "APPROVE" in result


# ---------------------------------------------------------------------------
# TestReassignment
# ---------------------------------------------------------------------------


class TestReassignment:
    """Dry-run and reassignment (Task 11.4) behavior."""

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    @pytest.mark.asyncio
    async def test_get_previous_feedback_dry_run_returns_none(self) -> None:
        orchestrator = self._make_orchestrator(dry_run=True)
        result = await orchestrator._get_previous_review_feedback("TEST-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_previous_feedback_populated_in_run(self) -> None:
        """When task_ctx.previous_review_feedback is None, run() populates it."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()  # previous_review_feedback is None by default
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        captured_task_ctx: list[TaskContext] = []

        async def fake_review_loop(tc: TaskContext, cc: CodeContext, max_retries: int):
            captured_task_ctx.append(tc)
            return code_change, review

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
            patch.object(orchestrator, "_run_review_loop", side_effect=fake_review_loop),
            patch.object(
                orchestrator,
                "_create_pull_request",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                orchestrator,
                "_get_previous_review_feedback",
                new_callable=AsyncMock,
                return_value="Previous AI feedback: fix the tests",
            ) as mock_get_feedback,
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

            result = await orchestrator.run("TEST-1")

        assert result.success is True
        # _get_previous_review_feedback should have been called
        mock_get_feedback.assert_called_once_with("TEST-1")
        # The task_ctx passed to review loop should have the feedback populated
        assert len(captured_task_ctx) == 1
        assert captured_task_ctx[0].previous_review_feedback == "Previous AI feedback: fix the tests"

    @pytest.mark.asyncio
    async def test_get_previous_feedback_returns_none_on_exception(self) -> None:
        """_get_previous_review_feedback returns None when agent raises."""
        orchestrator = self._make_orchestrator(dry_run=False)

        mock_agent_instance = MagicMock()
        mock_agent_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("Jira down"))
        mock_agent_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("src.pipeline.orchestrator.Agent", return_value=mock_agent_instance):
            result = await orchestrator._get_previous_review_feedback("TEST-1")

        assert result is None


# ---------------------------------------------------------------------------
# Property-Based Tests: Large Scope Pipeline Halt
# ---------------------------------------------------------------------------

import asyncio

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


class TestLargeScopePipelineHaltProperties:
    """Property 7: Large Scope Pipeline Halt

    **Validates: Requirements 2.9**

    Verifies that any task with estimated_scope == LARGE causes the pipeline
    to halt before CodeFinderAgent is invoked, and that non-LARGE scopes
    continue past the scope check.
    """

    # ------------------------------------------------------------------
    # Property 7a: Any LARGE scope task always halts before CodeFinder
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_large_scope_always_halts_before_code_finder(self, issue_key: str) -> None:
        """Property 7a: For any issue_key, LARGE scope always halts before CodeFinder.

        **Validates: Requirements 2.9**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_key=issue_key, scope=TaskScope.LARGE)

        async def _run() -> PipelineResult:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                result = await orchestrator.run(issue_key)
                # Assert CodeFinder was NOT called inside the async context
                MockCodeFinder.return_value.find_code.assert_not_called()
                return result

        result = asyncio.run(_run())

        assert result.success is False
        assert result.failure_stage == "scope_check"

    # ------------------------------------------------------------------
    # Property 7b: Non-LARGE scope tasks do NOT halt at scope_check
    # ------------------------------------------------------------------

    @given(scope=st.sampled_from([TaskScope.SMALL, TaskScope.MEDIUM]))
    @h_settings(max_examples=100)
    def test_non_large_scope_does_not_halt_at_scope_check(self, scope: TaskScope) -> None:
        """Property 7b: SMALL and MEDIUM scopes never halt at scope_check.

        **Validates: Requirements 2.9**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(scope=scope)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        async def _run() -> PipelineResult:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                    return_value=None,
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                return await orchestrator.run("TEST-1")

        result = asyncio.run(_run())

        assert result.failure_stage != "scope_check"

    # ------------------------------------------------------------------
    # Property 7c: LARGE scope always produces a Jira comment about scope
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_large_scope_always_comments_about_scope(self, issue_key: str) -> None:
        """Property 7c: LARGE scope always produces a Jira comment mentioning 'LARGE'.

        **Validates: Requirements 2.9**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_key=issue_key, scope=TaskScope.LARGE)
        captured_comments: list[str] = []

        async def fake_comment(ik: str, msg: str) -> None:
            captured_comments.append(msg)

        async def _run() -> None:
            with (
                patch.object(orchestrator, "_comment_on_jira", side_effect=fake_comment),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent"),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                await orchestrator.run(issue_key)

        asyncio.run(_run())

        assert any("large" in c.lower() for c in captured_comments), (
            f"Expected at least one comment mentioning 'LARGE', got: {captured_comments}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests: Review Verdict Pipeline Routing
# ---------------------------------------------------------------------------


class TestReviewVerdictRoutingProperties:
    """Property 15: Review Verdict Pipeline Routing

    **Validates: Requirements 5.3, 5.4, 5.6**

    Verifies that the pipeline routes correctly based on the review verdict:
    - APPROVE  PR creation attempt, success=True
    - REJECT  halt with failure_stage="review", "REJECT" in failure_reason
    - REJECT  at least one Jira comment after the start comment
    - APPROVE  failure_stage is None
    """

    # ------------------------------------------------------------------
    # Property 15a: APPROVE verdict always leads to PR creation attempt
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_approve_always_creates_pr(self, issue_key: str) -> None:
        """Property 15a: APPROVE verdict always leads to PR creation attempt.

        **Validates: Requirements 5.3**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        approve_review = _make_review_approve()

        async def _run() -> tuple[PipelineResult, int]:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
                patch.object(
                    orchestrator,
                    "_run_review_loop",
                    new_callable=AsyncMock,
                    return_value=(code_change, approve_review),
                ),
                patch.object(
                    orchestrator,
                    "_create_pull_request",
                    new_callable=AsyncMock,
                    return_value="https://github.com/pr/1",
                ) as mock_create_pr,
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                result = await orchestrator.run(issue_key)
                pr_call_count = mock_create_pr.call_count
            return result, pr_call_count

        result, pr_call_count = asyncio.run(_run())

        assert result.success is True
        assert pr_call_count == 1, (
            f"Expected _create_pull_request to be called once, got {pr_call_count}"
        )

    # ------------------------------------------------------------------
    # Property 15b: REJECT verdict always halts with failure_stage="review"
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_reject_always_halts_pipeline(self, issue_key: str) -> None:
        """Property 15b: REJECT verdict always halts pipeline with failure_stage='review'.

        **Validates: Requirements 5.4**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        reject_review = _make_review_reject()

        async def _run() -> PipelineResult:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
                patch.object(
                    orchestrator,
                    "_run_review_loop",
                    new_callable=AsyncMock,
                    return_value=(code_change, reject_review),
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                return await orchestrator.run(issue_key)

        result = asyncio.run(_run())

        assert result.success is False
        assert result.failure_stage == "review"
        assert "REJECT" in (result.failure_reason or ""), (
            f"Expected 'REJECT' in failure_reason, got: {result.failure_reason!r}"
        )

    # ------------------------------------------------------------------
    # Property 15c: REJECT verdict always produces a Jira comment
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_reject_always_produces_jira_comment(self, issue_key: str) -> None:
        """Property 15c: REJECT verdict always produces at least one Jira comment after start.

        **Validates: Requirements 5.6**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        reject_review = _make_review_reject()

        async def _run() -> int:
            with (
                patch.object(
                    orchestrator, "_comment_on_jira", new_callable=AsyncMock
                ) as mock_comment,
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
                patch.object(
                    orchestrator,
                    "_run_review_loop",
                    new_callable=AsyncMock,
                    return_value=(code_change, reject_review),
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                await orchestrator.run(issue_key)
                return mock_comment.call_count

        total_comments = asyncio.run(_run())

        # At minimum: start comment + reject comment = 2
        assert total_comments >= 2, (
            f"Expected at least 2 Jira comments (start + reject), got {total_comments}"
        )

    # ------------------------------------------------------------------
    # Property 15d: APPROVE verdict does NOT halt pipeline (failure_stage is None)
    # ------------------------------------------------------------------

    @given(
        pr_url=st.one_of(
            st.none(),
            st.text(min_size=5, max_size=50),
        )
    )
    @h_settings(max_examples=100)
    def test_approve_does_not_halt_pipeline(self, pr_url: str | None) -> None:
        """Property 15d: APPROVE verdict never halts pipeline (failure_stage is None).

        **Validates: Requirements 5.3**
        """
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)

        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        approve_review = _make_review_approve()

        async def _run() -> PipelineResult:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
                patch.object(
                    orchestrator,
                    "_run_review_loop",
                    new_callable=AsyncMock,
                    return_value=(code_change, approve_review),
                ),
                patch.object(
                    orchestrator,
                    "_create_pull_request",
                    new_callable=AsyncMock,
                    return_value=pr_url,
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                return await orchestrator.run("TEST-1")

        result = asyncio.run(_run())

        assert result.failure_stage is None, (
            f"Expected failure_stage to be None for APPROVE, got: {result.failure_stage!r}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests: Review Loop Retry Limit
# ---------------------------------------------------------------------------


class TestReviewLoopRetryLimitProperties:
    """Property 16: Review Loop Retry Limit

    **Validates: Requirements 5.5, 5.7**

    Verifies that _run_review_loop respects the max_retries parameter:
    - All REQUEST_CHANGES  CodeWriter called exactly max_retries+1 times
    - APPROVE on iteration K  CodeWriter called exactly K times
    - max_retries=0  exactly 1 iteration regardless of verdict
    """

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def _make_request_changes_review(self) -> ReviewResult:
        return ReviewResult(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            score=5,
            findings=[],
            feedback_for_rewrite="Fix it",
            acceptance_criteria_met=False,
        )

    # ------------------------------------------------------------------
    # Property 16a: All REQUEST_CHANGES  CodeWriter called max_retries+1 times
    # ------------------------------------------------------------------

    @given(max_retries=st.integers(min_value=1, max_value=4))
    @h_settings(max_examples=100)
    def test_all_request_changes_calls_writer_max_retries_plus_one(
        self, max_retries: int
    ) -> None:
        """Property 16a: When every review returns REQUEST_CHANGES, CodeWriter is called
        exactly max_retries+1 times and the final verdict is REQUEST_CHANGES.

        **Validates: Requirements 5.5, 5.7**
        """
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        request_changes = self._make_request_changes_review()

        async def _run() -> tuple[ReviewResult, int]:
            with (
                patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
                patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
            ):
                MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
                MockReviewer.return_value.review_code = AsyncMock(
                    return_value=request_changes
                )

                _result_change, result_review = await orchestrator._run_review_loop(
                    task_ctx, code_ctx, max_retries=max_retries
                )
                call_count = MockWriter.return_value.write_code.call_count
            return result_review, call_count

        result_review, call_count = asyncio.run(_run())

        assert result_review.verdict == ReviewVerdict.REQUEST_CHANGES, (
            f"Expected REQUEST_CHANGES verdict after exhausting retries, got {result_review.verdict}"
        )
        assert call_count == max_retries + 1, (
            f"Expected write_code called {max_retries + 1} times, got {call_count}"
        )

    # ------------------------------------------------------------------
    # Property 16b: APPROVE on iteration K  CodeWriter called exactly K times
    # ------------------------------------------------------------------

    @given(approve_on_iteration=st.integers(min_value=1, max_value=4))
    @h_settings(max_examples=100)
    def test_approve_on_iteration_k_calls_writer_exactly_k_times(
        self, approve_on_iteration: int
    ) -> None:
        """Property 16b: When reviewer approves on iteration K (after K-1 REQUEST_CHANGES),
        CodeWriter is called exactly K times and the final verdict is APPROVE.

        **Validates: Requirements 5.5**
        """
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        request_changes = self._make_request_changes_review()
        approve = _make_review_approve()

        # Build side_effect: (approve_on_iteration - 1) REQUEST_CHANGES, then APPROVE
        review_side_effects = [request_changes] * (approve_on_iteration - 1) + [approve]

        async def _run() -> tuple[ReviewResult, int]:
            with (
                patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
                patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
            ):
                MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
                MockReviewer.return_value.review_code = AsyncMock(
                    side_effect=review_side_effects
                )

                _result_change, result_review = await orchestrator._run_review_loop(
                    task_ctx, code_ctx, max_retries=approve_on_iteration + 1
                )
                call_count = MockWriter.return_value.write_code.call_count
            return result_review, call_count

        result_review, call_count = asyncio.run(_run())

        assert result_review.verdict == ReviewVerdict.APPROVE, (
            f"Expected APPROVE verdict on iteration {approve_on_iteration}, got {result_review.verdict}"
        )
        assert call_count == approve_on_iteration, (
            f"Expected write_code called {approve_on_iteration} times, got {call_count}"
        )

    # ------------------------------------------------------------------
    # Property 16c: max_retries=0  exactly 1 iteration regardless of verdict
    # ------------------------------------------------------------------

    @given(verdict=st.sampled_from([ReviewVerdict.REQUEST_CHANGES, ReviewVerdict.APPROVE, ReviewVerdict.REJECT]))
    @h_settings(max_examples=100)
    def test_zero_max_retries_always_one_iteration(self, verdict: ReviewVerdict) -> None:
        """Property 16c: With max_retries=0, CodeWriter is called exactly once
        regardless of the review verdict.

        **Validates: Requirements 5.5, 5.7**
        """
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()

        # Build the appropriate review result for the given verdict
        if verdict == ReviewVerdict.APPROVE:
            review = _make_review_approve()
        elif verdict == ReviewVerdict.REJECT:
            review = _make_review_reject()
        else:
            review = ReviewResult(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                score=5,
                findings=[],
                feedback_for_rewrite="Fix it",
                acceptance_criteria_met=False,
            )

        async def _run() -> tuple[ReviewResult, int]:
            with (
                patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
                patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
            ):
                MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
                MockReviewer.return_value.review_code = AsyncMock(return_value=review)

                _result_change, result_review = await orchestrator._run_review_loop(
                    task_ctx, code_ctx, max_retries=0
                )
                call_count = MockWriter.return_value.write_code.call_count
            return result_review, call_count

        result_review, call_count = asyncio.run(_run())

        assert call_count == 1, (
            f"Expected write_code called exactly once with max_retries=0 "
            f"(verdict={verdict}), got {call_count}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests: Max File Change Limit Enforcement
# ---------------------------------------------------------------------------


class TestMaxFileChangeLimitProperties:
    """Property 27: Max File Change Limit Enforcement

    **Validates: Requirements 12.4, 12.5**

    Verifies that when changes + test_changes > max_file_changes:
    - Pipeline halts with REJECT verdict and "Too many file changes" in feedback
    - CodeReviewer is NOT called
    - A Jira comment is always produced

    And when changes + test_changes <= max_file_changes:
    - CodeReviewer IS called
    """

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    # ------------------------------------------------------------------
    # Property 27a: changes + test_changes > max_file_changes  REJECT
    # ------------------------------------------------------------------

    @given(
        n_changes=st.integers(min_value=1, max_value=10),
        n_test_changes=st.integers(min_value=0, max_value=5),
        max_file_changes=st.integers(min_value=1, max_value=8),
    )
    @h_settings(max_examples=100)
    def test_exceeding_limit_always_rejects(
        self, n_changes: int, n_test_changes: int, max_file_changes: int
    ) -> None:
        """Property 27a: When changes + test_changes > max_file_changes, always REJECT
        with 'Too many file changes' in feedback and CodeReviewer NOT called.

        **Validates: Requirements 12.4, 12.5**
        """
        from hypothesis import assume

        assume(n_changes + n_test_changes > max_file_changes)

        orchestrator = self._make_orchestrator(max_file_changes=max_file_changes)
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()

        code_change = CodeChange(
            changes=[
                FileChange(
                    path=f"src/file{i}.py",
                    new_content="x",
                    change_type=ChangeType.MODIFY,
                    explanation="x",
                )
                for i in range(n_changes)
            ],
            test_changes=[
                FileChange(
                    path=f"tests/test_file{i}.py",
                    new_content="x",
                    change_type=ChangeType.MODIFY,
                    explanation="x",
                )
                for i in range(n_test_changes)
            ],
            commit_message="fix: changes",
            pr_title="Changes",
            pr_description="desc",
        )

        async def _run() -> tuple[ReviewResult, int]:
            with (
                patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
                patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            ):
                MockWriter.return_value.write_code = AsyncMock(return_value=code_change)

                _result_change, result_review = await orchestrator._run_review_loop(
                    task_ctx, code_ctx, max_retries=3
                )
                reviewer_call_count = MockReviewer.return_value.review_code.call_count
            return result_review, reviewer_call_count

        result_review, reviewer_call_count = asyncio.run(_run())

        assert result_review.verdict == ReviewVerdict.REJECT, (
            f"Expected REJECT when {n_changes}+{n_test_changes} > {max_file_changes}, "
            f"got {result_review.verdict}"
        )
        assert "Too many file changes" in (result_review.feedback_for_rewrite or ""), (
            f"Expected 'Too many file changes' in feedback, got: {result_review.feedback_for_rewrite!r}"
        )
        assert reviewer_call_count == 0, (
            f"Expected CodeReviewer NOT called, but it was called {reviewer_call_count} times"
        )

    # ------------------------------------------------------------------
    # Property 27b: changes + test_changes <= max_file_changes  Reviewer IS called
    # ------------------------------------------------------------------

    @given(
        n_changes=st.integers(min_value=1, max_value=5),
        n_test_changes=st.integers(min_value=0, max_value=3),
        max_file_changes=st.integers(min_value=5, max_value=15),
    )
    @h_settings(max_examples=100)
    def test_within_limit_calls_reviewer(
        self, n_changes: int, n_test_changes: int, max_file_changes: int
    ) -> None:
        """Property 27b: When changes + test_changes <= max_file_changes, CodeReviewer IS called.

        **Validates: Requirements 12.4**
        """
        from hypothesis import assume

        assume(n_changes + n_test_changes <= max_file_changes)

        orchestrator = self._make_orchestrator(max_file_changes=max_file_changes)
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()

        code_change = CodeChange(
            changes=[
                FileChange(
                    path=f"src/file{i}.py",
                    new_content="x",
                    change_type=ChangeType.MODIFY,
                    explanation="x",
                )
                for i in range(n_changes)
            ],
            test_changes=[
                FileChange(
                    path=f"tests/test_file{i}.py",
                    new_content="x",
                    change_type=ChangeType.MODIFY,
                    explanation="x",
                )
                for i in range(n_test_changes)
            ],
            commit_message="fix: changes",
            pr_title="Changes",
            pr_description="desc",
        )

        approve_review = _make_review_approve()

        async def _run() -> int:
            with (
                patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
                patch("src.pipeline.orchestrator.CodeReviewerAgent") as MockReviewer,
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            ):
                MockWriter.return_value.write_code = AsyncMock(return_value=code_change)
                MockReviewer.return_value.review_code = AsyncMock(return_value=approve_review)

                await orchestrator._run_review_loop(task_ctx, code_ctx, max_retries=3)
                return MockReviewer.return_value.review_code.call_count

        reviewer_call_count = asyncio.run(_run())

        assert reviewer_call_count >= 1, (
            f"Expected CodeReviewer to be called when {n_changes}+{n_test_changes} <= {max_file_changes}, "
            f"but it was called {reviewer_call_count} times"
        )

    # ------------------------------------------------------------------
    # Property 27c: Max file change violation always produces a Jira comment
    # ------------------------------------------------------------------

    @given(
        n_changes=st.integers(min_value=1, max_value=10),
        max_file_changes=st.integers(min_value=1, max_value=5),
    )
    @h_settings(max_examples=100)
    def test_violation_always_produces_jira_comment(
        self, n_changes: int, max_file_changes: int
    ) -> None:
        """Property 27c: Max file change violation always produces a Jira comment.

        **Validates: Requirements 12.5**
        """
        from hypothesis import assume

        assume(n_changes > max_file_changes)

        orchestrator = self._make_orchestrator(max_file_changes=max_file_changes)
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()

        code_change = CodeChange(
            changes=[
                FileChange(
                    path=f"src/file{i}.py",
                    new_content="x",
                    change_type=ChangeType.MODIFY,
                    explanation="x",
                )
                for i in range(n_changes)
            ],
            commit_message="fix: changes",
            pr_title="Changes",
            pr_description="desc",
        )

        async def _run() -> int:
            with (
                patch("src.pipeline.orchestrator.CodeWriterAgent") as MockWriter,
                patch("src.pipeline.orchestrator.CodeReviewerAgent"),
                patch.object(
                    orchestrator, "_comment_on_jira", new_callable=AsyncMock
                ) as mock_comment,
            ):
                MockWriter.return_value.write_code = AsyncMock(return_value=code_change)

                await orchestrator._run_review_loop(task_ctx, code_ctx, max_retries=3)
                return mock_comment.call_count

        comment_count = asyncio.run(_run())

        assert comment_count >= 1, (
            f"Expected _comment_on_jira called at least once when {n_changes} > {max_file_changes}, "
            f"but it was called {comment_count} times"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests: Dry-Run Mode No Side Effects
# ---------------------------------------------------------------------------


class TestDryRunModeProperties:
    """Property 19: Dry-Run Mode No Side Effects

    **Validates: Requirements 16.2, 16.3**

    Verifies that when dry_run == True:
    - Agent is never instantiated during the full pipeline
    - _create_pull_request always returns None
    - _comment_on_jira never calls Agent
    - _transition_jira never calls Agent
    """

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    # ------------------------------------------------------------------
    # Property 19a: In dry-run mode, Agent is never instantiated during full pipeline
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_dry_run_agent_never_instantiated_in_full_pipeline(
        self, issue_key: str
    ) -> None:
        """Property 19a: In dry-run mode, Agent is never instantiated during full pipeline.

        **Validates: Requirements 16.2, 16.3**
        """
        orchestrator = self._make_orchestrator(dry_run=True)

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        async def _run() -> tuple[PipelineResult, int]:
            with (
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                    return_value=None,
                ),
                patch("src.pipeline.orchestrator.Agent") as MockAgent,
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)

                result = await orchestrator.run(issue_key)
                agent_call_count = MockAgent.call_count
            return result, agent_call_count

        result, agent_call_count = asyncio.run(_run())

        assert result.dry_run is True, (
            f"Expected result.dry_run to be True, got {result.dry_run!r}"
        )
        assert agent_call_count == 0, (
            f"Expected Agent to never be instantiated in dry-run mode, "
            f"but it was called {agent_call_count} times"
        )

    # ------------------------------------------------------------------
    # Property 19b: In dry-run mode, _create_pull_request always returns None
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_dry_run_create_pull_request_always_returns_none(
        self, issue_key: str
    ) -> None:
        """Property 19b: In dry-run mode, _create_pull_request always returns None.

        **Validates: Requirements 16.2**
        """
        orchestrator = self._make_orchestrator(dry_run=True)
        task_ctx = _make_task_context(issue_key=issue_key)
        code_change = _make_code_change()

        async def _run() -> Any:
            return await orchestrator._create_pull_request(task_ctx, code_change)

        result = asyncio.run(_run())

        assert result is None, (
            f"Expected _create_pull_request to return None in dry-run mode, got {result!r}"
        )

    # ------------------------------------------------------------------
    # Property 19c: In dry-run mode, _comment_on_jira never calls Agent
    # ------------------------------------------------------------------

    @given(
        issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True),
        message=st.text(min_size=1, max_size=50),
    )
    @h_settings(max_examples=100)
    def test_dry_run_comment_on_jira_never_calls_agent(
        self, issue_key: str, message: str
    ) -> None:
        """Property 19c: In dry-run mode, _comment_on_jira never calls Agent.

        **Validates: Requirements 16.2, 16.3**
        """
        orchestrator = self._make_orchestrator(dry_run=True)

        async def _run() -> int:
            with patch("src.pipeline.orchestrator.Agent") as MockAgent:
                await orchestrator._comment_on_jira(issue_key, message)
                return MockAgent.call_count

        agent_call_count = asyncio.run(_run())

        assert agent_call_count == 0, (
            f"Expected Agent to never be called in _comment_on_jira dry-run mode, "
            f"but it was called {agent_call_count} times"
        )

    # ------------------------------------------------------------------
    # Property 19d: In dry-run mode, _transition_jira never calls Agent
    # ------------------------------------------------------------------

    @given(
        issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True),
        transition_id=st.text(min_size=1, max_size=10),
    )
    @h_settings(max_examples=100)
    def test_dry_run_transition_jira_never_calls_agent(
        self, issue_key: str, transition_id: str
    ) -> None:
        """Property 19d: In dry-run mode, _transition_jira never calls Agent.

        **Validates: Requirements 16.2, 16.3**
        """
        orchestrator = self._make_orchestrator(dry_run=True)

        async def _run() -> int:
            with patch("src.pipeline.orchestrator.Agent") as MockAgent:
                await orchestrator._transition_jira(issue_key, transition_id)
                return MockAgent.call_count

        agent_call_count = asyncio.run(_run())

        assert agent_call_count == 0, (
            f"Expected Agent to never be called in _transition_jira dry-run mode, "
            f"but it was called {agent_call_count} times"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests: Auto-Create PR Disabled
# ---------------------------------------------------------------------------


class TestAutoCreatePRDisabledProperties:
    """Property 20: Auto-Create PR Disabled

    **Validates: Requirements 6.6**

    Verifies that when auto_create_pr == False:
    - _create_pull_request always returns None (no PR URL)
    - Exactly one Jira comment is made containing at least one file path
    - Agent is never instantiated (no Git MCP calls)
    """

    def _make_orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def _make_code_change_with_n_files(self, n: int) -> CodeChange:
        """Build a CodeChange with n source file changes."""
        return CodeChange(
            changes=[
                FileChange(
                    path=f"src/module_{i}.py",
                    new_content=f"# module {i}\n",
                    change_type=ChangeType.MODIFY,
                    explanation=f"Updated module {i}",
                )
                for i in range(n)
            ],
            commit_message="fix: update modules",
            pr_title="Update modules",
            pr_description="Updated several modules.",
        )

    # ------------------------------------------------------------------
    # Property 20a: auto_create_pr=False always returns None
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_auto_create_pr_false_always_returns_none(self, issue_key: str) -> None:
        """Property 20a: When auto_create_pr=False, _create_pull_request always returns None.

        **Validates: Requirements 6.6**
        """
        orchestrator = self._make_orchestrator(auto_create_pr=False)
        task_ctx = _make_task_context(issue_key=issue_key)
        code_change = _make_code_change()

        async def _run() -> Any:
            with patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock):
                return await orchestrator._create_pull_request(task_ctx, code_change)

        result = asyncio.run(_run())

        assert result is None, (
            f"Expected _create_pull_request to return None when auto_create_pr=False, "
            f"got {result!r}"
        )

    # ------------------------------------------------------------------
    # Property 20b: auto_create_pr=False always produces a Jira comment with file paths
    # ------------------------------------------------------------------

    @given(n_changes=st.integers(min_value=1, max_value=5))
    @h_settings(max_examples=100)
    def test_auto_create_pr_false_always_comments_with_file_paths(
        self, n_changes: int
    ) -> None:
        """Property 20b: When auto_create_pr=False, exactly 1 Jira comment is made
        and it contains at least one file path from the code change.

        **Validates: Requirements 6.6**
        """
        orchestrator = self._make_orchestrator(auto_create_pr=False)
        task_ctx = _make_task_context()
        code_change = self._make_code_change_with_n_files(n_changes)

        captured_comments: list[tuple[str, str]] = []

        async def fake_comment(ik: str, msg: str) -> None:
            captured_comments.append((ik, msg))

        async def _run() -> None:
            with patch.object(orchestrator, "_comment_on_jira", side_effect=fake_comment):
                await orchestrator._create_pull_request(task_ctx, code_change)

        asyncio.run(_run())

        assert len(captured_comments) == 1, (
            f"Expected exactly 1 Jira comment when auto_create_pr=False, "
            f"got {len(captured_comments)}"
        )

        comment_text = captured_comments[0][1]
        expected_paths = [fc.path for fc in code_change.changes]
        assert any(path in comment_text for path in expected_paths), (
            f"Expected at least one file path in comment, "
            f"paths={expected_paths!r}, comment={comment_text!r}"
        )

    # ------------------------------------------------------------------
    # Property 20c: auto_create_pr=False never instantiates Agent
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_auto_create_pr_false_never_instantiates_agent(
        self, issue_key: str
    ) -> None:
        """Property 20c: When auto_create_pr=False, Agent is never instantiated
        (no Git MCP calls are made).

        **Validates: Requirements 6.6**
        """
        orchestrator = self._make_orchestrator(auto_create_pr=False)
        task_ctx = _make_task_context(issue_key=issue_key)
        code_change = _make_code_change()

        async def _run() -> int:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.Agent") as MockAgent,
            ):
                await orchestrator._create_pull_request(task_ctx, code_change)
                return MockAgent.call_count

        agent_call_count = asyncio.run(_run())

        assert agent_call_count == 0, (
            f"Expected Agent to never be instantiated when auto_create_pr=False, "
            f"but it was called {agent_call_count} times"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests: Jira Transition Non-Blocking
# ---------------------------------------------------------------------------


class TestJiraTransitionNonBlockingProperties:
    """Property 24: Jira Transition Non-Blocking

    **Validates: Requirements 9.1, 9.2**

    Verifies that when _transition_jira raises an exception:
    - The pipeline still completes successfully (success=True)
    - The error is logged (not re-raised)
    - The pipeline does NOT halt at the transition step
    """

    def _make_orchestrator(self, **overrides) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    # ------------------------------------------------------------------
    # Property 24a: Transition failure never halts the pipeline
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_transition_failure_does_not_halt_pipeline(self, issue_key: str) -> None:
        """Property 24a: When _transition_jira raises, pipeline still completes.

        **Validates: Requirements 9.1, 9.2**
        """
        orchestrator = self._make_orchestrator()

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        async def _run() -> PipelineResult:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(
                    orchestrator,
                    "_transition_jira",
                    new_callable=AsyncMock,
                    side_effect=Exception("Jira transition failed"),
                ),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                    return_value="https://github.com/pr/1",
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                return await orchestrator.run(issue_key)

        result = asyncio.run(_run())

        assert result.success is True, (
            f"Expected pipeline to succeed despite transition failure, "
            f"got success={result.success}, failure_stage={result.failure_stage!r}"
        )

    # ------------------------------------------------------------------
    # Property 24b: Transition failure is logged (not re-raised)
    # ------------------------------------------------------------------

    @given(
        issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True),
        transition_id=st.sampled_from(["21", "31", "41"]),
    )
    @h_settings(max_examples=100)
    def test_transition_failure_is_logged_not_raised(
        self, issue_key: str, transition_id: str
    ) -> None:
        """Property 24b: _transition_jira swallows exceptions and logs them.

        **Validates: Requirements 9.1, 9.2**
        """
        orchestrator = self._make_orchestrator()

        async def _run() -> bool:
            """Returns True if no exception was raised."""
            with patch("src.pipeline.orchestrator.Agent") as MockAgent:
                mock_agent_instance = MagicMock()
                mock_agent_instance.__aenter__ = AsyncMock(
                    side_effect=Exception("Jira down")
                )
                mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
                MockAgent.return_value = mock_agent_instance

                try:
                    await orchestrator._transition_jira(issue_key, transition_id)
                    return True  # No exception raised - correct behavior
                except Exception:
                    return False  # Exception propagated - incorrect behavior

        no_exception_raised = asyncio.run(_run())

        assert no_exception_raised, (
            f"Expected _transition_jira to swallow exceptions, "
            f"but it raised for issue_key={issue_key!r}, transition_id={transition_id!r}"
        )

    # ------------------------------------------------------------------
    # Property 24c: Pipeline result is always a valid PipelineResult even with transition failure
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_transition_failure_always_returns_pipeline_result(
        self, issue_key: str
    ) -> None:
        """Property 24c: Pipeline always returns a valid PipelineResult even when
        all Jira transitions fail.

        **Validates: Requirements 9.1**
        """
        orchestrator = self._make_orchestrator()

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        async def _run() -> PipelineResult:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(
                    orchestrator,
                    "_transition_jira",
                    new_callable=AsyncMock,
                    side_effect=ConnectionError("Jira unreachable"),
                ),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                    return_value=None,
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                return await orchestrator.run(issue_key)

        result = asyncio.run(_run())

        assert isinstance(result, PipelineResult), (
            f"Expected PipelineResult instance, got {type(result)!r}"
        )
        assert result.issue_key == issue_key


# ---------------------------------------------------------------------------
# Property-Based Tests: Reassignment Detection
# ---------------------------------------------------------------------------


class TestReassignmentDetectionProperties:
    """Property 34: Reassignment Detection

    **Validates: Requirements 17.1, 17.3, 17.4**

    Verifies that when a task is reassigned to the bot:
    - _get_previous_review_feedback is always called during run()
    - When previous feedback exists, it is propagated to task_ctx.previous_review_feedback
    - When previous feedback is None, task_ctx.previous_review_feedback remains None
    - _get_previous_review_feedback never raises (always returns str or None)
    """

    def _make_orchestrator(self, **overrides) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    # ------------------------------------------------------------------
    # Property 34a: _get_previous_review_feedback is always called during run()
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_get_previous_feedback_always_called_during_run(
        self, issue_key: str
    ) -> None:
        """Property 34a: _get_previous_review_feedback is always called during run().

        **Validates: Requirements 17.1, 17.3**
        """
        orchestrator = self._make_orchestrator()

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        async def _run() -> int:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                    return_value=None,
                ),
                patch.object(
                    orchestrator,
                    "_get_previous_review_feedback",
                    new_callable=AsyncMock,
                    return_value=None,
                ) as mock_get_feedback,
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                await orchestrator.run(issue_key)
                return mock_get_feedback.call_count

        call_count = asyncio.run(_run())

        assert call_count == 1, (
            f"Expected _get_previous_review_feedback to be called exactly once, "
            f"got {call_count} for issue_key={issue_key!r}"
        )

    # ------------------------------------------------------------------
    # Property 34b: Previous feedback is propagated to task_ctx when present
    # ------------------------------------------------------------------

    @given(
        issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True),
        feedback=st.text(min_size=1, max_size=200),
    )
    @h_settings(max_examples=100)
    def test_previous_feedback_propagated_to_task_ctx(
        self, issue_key: str, feedback: str
    ) -> None:
        """Property 34b: When previous feedback exists, it is propagated to
        task_ctx.previous_review_feedback before the review loop.

        **Validates: Requirements 17.3, 17.4**
        """
        orchestrator = self._make_orchestrator()

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        captured_task_ctxs: list[TaskContext] = []

        async def fake_review_loop(tc: TaskContext, cc: CodeContext, max_retries: int):
            captured_task_ctxs.append(tc)
            return code_change, review

        async def _run() -> None:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
                patch.object(
                    orchestrator, "_run_review_loop", side_effect=fake_review_loop
                ),
                patch.object(
                    orchestrator,
                    "_create_pull_request",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    orchestrator,
                    "_get_previous_review_feedback",
                    new_callable=AsyncMock,
                    return_value=feedback,
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                await orchestrator.run(issue_key)

        asyncio.run(_run())

        assert len(captured_task_ctxs) == 1, (
            f"Expected review loop to be called once, got {len(captured_task_ctxs)}"
        )
        assert captured_task_ctxs[0].previous_review_feedback == feedback, (
            f"Expected previous_review_feedback={feedback!r}, "
            f"got {captured_task_ctxs[0].previous_review_feedback!r}"
        )

    # ------------------------------------------------------------------
    # Property 34c: When previous feedback is None, task_ctx.previous_review_feedback is None
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_no_previous_feedback_leaves_task_ctx_none(
        self, issue_key: str
    ) -> None:
        """Property 34c: When _get_previous_review_feedback returns None,
        task_ctx.previous_review_feedback remains None.

        **Validates: Requirements 17.3**
        """
        orchestrator = self._make_orchestrator()

        task_ctx = _make_task_context(issue_key=issue_key)
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        captured_task_ctxs: list[TaskContext] = []

        async def fake_review_loop(tc: TaskContext, cc: CodeContext, max_retries: int):
            captured_task_ctxs.append(tc)
            return code_change, review

        async def _run() -> None:
            with (
                patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
                patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
                patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
                patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
                patch.object(
                    orchestrator, "_run_review_loop", side_effect=fake_review_loop
                ),
                patch.object(
                    orchestrator,
                    "_create_pull_request",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    orchestrator,
                    "_get_previous_review_feedback",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
                MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
                await orchestrator.run(issue_key)

        asyncio.run(_run())

        assert len(captured_task_ctxs) == 1
        assert captured_task_ctxs[0].previous_review_feedback is None, (
            f"Expected previous_review_feedback to be None when no feedback returned, "
            f"got {captured_task_ctxs[0].previous_review_feedback!r}"
        )

    # ------------------------------------------------------------------
    # Property 34d: _get_previous_review_feedback never raises
    # ------------------------------------------------------------------

    @given(issue_key=st.from_regex(r"[A-Z]{2,5}-\d{1,3}", fullmatch=True))
    @h_settings(max_examples=100)
    def test_get_previous_feedback_never_raises(self, issue_key: str) -> None:
        """Property 34d: _get_previous_review_feedback always returns str or None,
        never raises an exception.

        **Validates: Requirements 17.1**
        """
        orchestrator = self._make_orchestrator()

        async def _run() -> Any:
            """Call _get_previous_review_feedback with a failing Agent - should return None."""
            mock_agent_instance = MagicMock()
            mock_agent_instance.__aenter__ = AsyncMock(
                side_effect=RuntimeError("Jira unavailable")
            )
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)

            with patch("src.pipeline.orchestrator.Agent", return_value=mock_agent_instance):
                try:
                    result = await orchestrator._get_previous_review_feedback(issue_key)
                    return result  # Should be None
                except Exception as e:
                    return e  # Should NOT happen

        result = asyncio.run(_run())

        assert not isinstance(result, Exception), (
            f"Expected _get_previous_review_feedback to return None on error, "
            f"but it raised: {result!r}"
        )
        assert result is None, (
            f"Expected None when agent fails, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Task 2: Tests for _build_pr_description and _build_pr_prompt
# ---------------------------------------------------------------------------


class TestBuildPrDescription:
    """Tests for PipelineOrchestrator._build_pr_description()."""

    def _make_orchestrator(self, **overrides) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def test_contains_summary(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        desc = orch._build_pr_description(task_ctx, code_change)
        assert task_ctx.summary in desc

    def test_contains_file_changes(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        desc = orch._build_pr_description(task_ctx, code_change)
        for fc in code_change.changes + code_change.test_changes:
            assert fc.path in desc

    def test_contains_jira_link(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        desc = orch._build_pr_description(task_ctx, code_change)
        assert task_ctx.issue_key in desc
        assert "jira.example.com" in desc

    def test_no_review_section_when_review_is_none(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        desc = orch._build_pr_description(task_ctx, code_change, review=None)
        assert "## Review" not in desc

    def test_review_section_present_when_review_provided(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        review = _make_review_approve()
        desc = orch._build_pr_description(task_ctx, code_change, review=review)
        assert "## Review" in desc
        assert "APPROVE" in desc

    def test_review_findings_included(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        review = _make_review_approve()
        desc = orch._build_pr_description(task_ctx, code_change, review=review)
        for finding in review.findings:
            assert finding.message in desc

    def test_empty_changes_no_changes_table(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = CodeChange(
            changes=[],
            test_changes=[],
            commit_message="empty",
            pr_title="empty",
            pr_description="empty",
        )
        desc = orch._build_pr_description(task_ctx, code_change)
        assert "## Changes" not in desc


class TestBuildPrPrompt:
    """Tests for updated PipelineOrchestrator._build_pr_prompt()."""

    def _make_orchestrator(self, **overrides) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    def test_title_uses_ai_bot_prefix(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        prompt = orch._build_pr_prompt(task_ctx, code_change, "feature/TEST-1-ai", all_changes)
        assert "[AI-BOT]" in prompt
        assert task_ctx.issue_key in prompt

    def test_draft_mode_instruction_included_when_enabled(self) -> None:
        orch = self._make_orchestrator(pr_draft_mode=True)
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        prompt = orch._build_pr_prompt(task_ctx, code_change, "feature/TEST-1-ai", all_changes)
        assert "draft" in prompt.lower()

    def test_draft_mode_instruction_absent_when_disabled(self) -> None:
        orch = self._make_orchestrator(pr_draft_mode=False)
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        prompt = orch._build_pr_prompt(task_ctx, code_change, "feature/TEST-1-ai", all_changes)
        assert "draft" not in prompt.lower()

    def test_reviewer_instruction_included_when_configured(self) -> None:
        orch = self._make_orchestrator(pr_reviewer="alice,bob")
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        prompt = orch._build_pr_prompt(task_ctx, code_change, "feature/TEST-1-ai", all_changes)
        assert "alice,bob" in prompt

    def test_reviewer_instruction_absent_when_empty(self) -> None:
        orch = self._make_orchestrator(pr_reviewer="")
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        prompt = orch._build_pr_prompt(task_ctx, code_change, "feature/TEST-1-ai", all_changes)
        assert "reviewer" not in prompt.lower()

    def test_prompt_contains_branch_name(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        branch = "feature/TEST-42-ai"
        prompt = orch._build_pr_prompt(task_ctx, code_change, branch, all_changes)
        assert branch in prompt

    def test_prompt_contains_pr_url_instruction(self) -> None:
        orch = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_change = _make_code_change()
        all_changes = code_change.changes + code_change.test_changes
        prompt = orch._build_pr_prompt(task_ctx, code_change, "feature/TEST-1-ai", all_changes)
        assert "PR URL" in prompt or "url" in prompt.lower()


# ---------------------------------------------------------------------------
# Task 5: Tests for Confluence integration in pipeline run()
# ---------------------------------------------------------------------------


class TestConfluencePipelineIntegration:
    """Tests for ConfluencePublisher integration in PipelineOrchestrator.run()."""

    def _make_orchestrator(self, **overrides) -> PipelineOrchestrator:
        settings = _make_settings(**overrides)
        llm_router = LLMRouter(config=settings)
        return PipelineOrchestrator(config=settings, llm_router=llm_router)

    @pytest.mark.asyncio
    async def test_confluence_publish_called_on_success(self) -> None:
        """ConfluencePublisher.publish() is called on successful pipeline run."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        mock_confluence = MagicMock()
        mock_confluence.return_value.publish = AsyncMock(return_value=None)

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                return_value="https://github.com/org/repo/pull/1",
            ),
            patch.object(
                orchestrator,
                "_get_previous_review_feedback",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("src.pipeline.orchestrator.ConfluencePublisher", mock_confluence),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            result = await orchestrator.run(task_ctx.issue_key)

        assert result.success is True
        mock_confluence.return_value.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_confluence_publish_called_on_review_reject(self) -> None:
        """ConfluencePublisher.publish() is called even when review rejects."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_reject()

        mock_confluence = MagicMock()
        mock_confluence.return_value.publish = AsyncMock(return_value=None)

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
            patch.object(
                orchestrator,
                "_run_review_loop",
                new_callable=AsyncMock,
                return_value=(code_change, review),
            ),
            patch.object(
                orchestrator,
                "_get_previous_review_feedback",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("src.pipeline.orchestrator.ConfluencePublisher", mock_confluence),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            result = await orchestrator.run(task_ctx.issue_key)

        assert result.success is False
        mock_confluence.return_value.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_confluence_doc_url_commented_on_jira(self) -> None:
        """When Confluence returns a URL, it is commented on Jira."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()
        doc_url = "https://wiki.example.com/pages/123"

        mock_confluence = MagicMock()
        mock_confluence.return_value.publish = AsyncMock(return_value=doc_url)
        jira_comments: list[str] = []

        async def capture_comment(issue_key: str, message: str) -> None:
            jira_comments.append(message)

        with (
            patch.object(orchestrator, "_comment_on_jira", side_effect=capture_comment),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                return_value=None,
            ),
            patch.object(
                orchestrator,
                "_get_previous_review_feedback",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("src.pipeline.orchestrator.ConfluencePublisher", mock_confluence),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            await orchestrator.run(task_ctx.issue_key)

        assert any(doc_url in comment for comment in jira_comments), (
            f"Expected Confluence URL in Jira comments, got: {jira_comments}"
        )

    @pytest.mark.asyncio
    async def test_confluence_failure_does_not_fail_pipeline(self) -> None:
        """If Confluence publish raises, pipeline still returns success."""
        orchestrator = self._make_orchestrator()
        task_ctx = _make_task_context()
        code_ctx = _make_code_context()
        code_change = _make_code_change()
        review = _make_review_approve()

        mock_confluence = MagicMock()
        mock_confluence.return_value.publish = AsyncMock(
            side_effect=Exception("Confluence down")
        )

        with (
            patch.object(orchestrator, "_comment_on_jira", new_callable=AsyncMock),
            patch.object(orchestrator, "_transition_jira", new_callable=AsyncMock),
            patch("src.pipeline.orchestrator.TaskReaderAgent") as MockTaskReader,
            patch("src.pipeline.orchestrator.CodeFinderAgent") as MockCodeFinder,
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
                return_value=None,
            ),
            patch.object(
                orchestrator,
                "_get_previous_review_feedback",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("src.pipeline.orchestrator.ConfluencePublisher", mock_confluence),
        ):
            MockTaskReader.return_value.read_task = AsyncMock(return_value=task_ctx)
            MockCodeFinder.return_value.find_code = AsyncMock(return_value=code_ctx)
            result = await orchestrator.run(task_ctx.issue_key)

        # Pipeline should still succeed even if Confluence fails
        assert result.success is True
