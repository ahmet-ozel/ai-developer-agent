"""Unit tests for pipeline and webhook data models."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.pipeline.models import (
    ChangeType,
    CodeChange,
    CodeContext,
    CodeFile,
    FileChange,
    FindingSeverity,
    LLMResponse,
    PipelineResult,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    SkippedFile,
    TaskContext,
    TaskScope,
)
from src.webhook.models import WebhookEvent


# --- Enum Tests ---


class TestEnums:
    def test_task_scope_values(self):
        assert TaskScope.SMALL == "small"
        assert TaskScope.MEDIUM == "medium"
        assert TaskScope.LARGE == "large"

    def test_review_verdict_values(self):
        assert ReviewVerdict.APPROVE == "approve"
        assert ReviewVerdict.REQUEST_CHANGES == "request_changes"
        assert ReviewVerdict.REJECT == "reject"

    def test_finding_severity_values(self):
        assert FindingSeverity.CRITICAL == "critical"
        assert FindingSeverity.WARNING == "warning"
        assert FindingSeverity.SUGGESTION == "suggestion"
        assert FindingSeverity.GOOD == "good"

    def test_change_type_values(self):
        assert ChangeType.CREATE == "create"
        assert ChangeType.MODIFY == "modify"
        assert ChangeType.DELETE == "delete"


# --- TaskContext Tests ---


class TestTaskContext:
    def test_minimal_creation(self):
        ctx = TaskContext(
            issue_key="PROJ-123",
            summary="Fix login bug",
            description="Login fails on mobile",
            repository_name="my-repo",
            estimated_scope=TaskScope.SMALL,
        )
        assert ctx.issue_key == "PROJ-123"
        assert ctx.comments == []
        assert ctx.issue_type is None
        assert ctx.reporter is None
        assert ctx.base_branch == "main"
        assert ctx.priority is None

    def test_full_creation(self):
        ctx = TaskContext(
            issue_key="PROJ-456",
            summary="Add feature",
            description="New feature desc",
            acceptance_criteria="Must pass tests",
            repository_name="my-repo",
            estimated_scope=TaskScope.MEDIUM,
            comments=["comment1"],
            confluence_docs=["doc1"],
            labels=["backend"],
            linked_issue_summaries=["PROJ-100 summary"],
            previous_review_feedback="Fix naming",
            issue_type="Story",
            reporter="john.doe",
            base_branch="develop",
            priority="High",
        )
        assert ctx.acceptance_criteria == "Must pass tests"
        assert ctx.reporter == "john.doe"
        assert ctx.base_branch == "develop"
        assert ctx.priority == "High"


# --- CodeFile Tests ---


class TestCodeFile:
    def test_line_count_auto_computed(self):
        cf = CodeFile(path="src/main.py", content="line1\nline2\nline3")
        assert cf.line_count == 3

    def test_line_count_explicit(self):
        cf = CodeFile(path="src/main.py", content="line1\nline2", line_count=5)
        assert cf.line_count == 5

    def test_defaults(self):
        cf = CodeFile(path="src/main.py", content="hello")
        assert cf.language == ""
        assert cf.is_test is False

    def test_empty_content_line_count(self):
        cf = CodeFile(path="src/empty.py", content="")
        # Empty string splitlines returns [], so line_count = 0
        assert cf.line_count == 0


# --- FileChange Validator Tests ---


class TestFileChange:
    def test_create_requires_content(self):
        fc = FileChange(
            path="src/new.py",
            new_content="print('hello')",
            change_type=ChangeType.CREATE,
            explanation="New file",
        )
        assert fc.new_content == "print('hello')"

    def test_modify_requires_content(self):
        fc = FileChange(
            path="src/existing.py",
            new_content="updated content",
            change_type=ChangeType.MODIFY,
            explanation="Updated file",
        )
        assert fc.new_content == "updated content"

    def test_delete_content_must_be_none(self):
        fc = FileChange(
            path="src/old.py",
            change_type=ChangeType.DELETE,
            explanation="Remove file",
        )
        assert fc.new_content is None

    def test_create_without_content_raises(self):
        with pytest.raises(ValidationError, match="new_content is required"):
            FileChange(
                path="src/new.py",
                change_type=ChangeType.CREATE,
                explanation="New file",
            )

    def test_modify_without_content_raises(self):
        with pytest.raises(ValidationError, match="new_content is required"):
            FileChange(
                path="src/mod.py",
                change_type=ChangeType.MODIFY,
                explanation="Modified file",
            )

    def test_delete_with_content_raises(self):
        with pytest.raises(ValidationError, match="new_content must be None"):
            FileChange(
                path="src/old.py",
                new_content="should not be here",
                change_type=ChangeType.DELETE,
                explanation="Remove file",
            )


# --- ReviewResult Validator Tests ---


class TestReviewResult:
    def test_approve_valid(self):
        rr = ReviewResult(
            verdict=ReviewVerdict.APPROVE,
            score=8,
            findings=[],
            acceptance_criteria_met=True,
        )
        assert rr.verdict == ReviewVerdict.APPROVE

    def test_approve_low_score_raises(self):
        with pytest.raises(ValidationError, match="APPROVE requires score >= 7"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=6,
                findings=[],
                acceptance_criteria_met=True,
            )

    def test_approve_with_critical_finding_raises(self):
        finding = ReviewFinding(
            file_path="src/main.py",
            severity=FindingSeverity.CRITICAL,
            category="security",
            message="Hardcoded secret",
        )
        with pytest.raises(
            ValidationError, match="APPROVE requires zero CRITICAL findings"
        ):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=9,
                findings=[finding],
                acceptance_criteria_met=True,
            )

    def test_approve_without_acceptance_criteria_raises(self):
        with pytest.raises(
            ValidationError, match="APPROVE requires all acceptance criteria met"
        ):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=8,
                findings=[],
                acceptance_criteria_met=False,
            )

    def test_request_changes_no_constraints(self):
        rr = ReviewResult(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            score=4,
            findings=[
                ReviewFinding(
                    file_path="src/main.py",
                    severity=FindingSeverity.CRITICAL,
                    category="logic",
                    message="Bug found",
                )
            ],
            acceptance_criteria_met=False,
        )
        assert rr.verdict == ReviewVerdict.REQUEST_CHANGES

    def test_reject_no_constraints(self):
        rr = ReviewResult(
            verdict=ReviewVerdict.REJECT,
            score=2,
            acceptance_criteria_met=False,
        )
        assert rr.verdict == ReviewVerdict.REJECT

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            ReviewResult(
                verdict=ReviewVerdict.REJECT,
                score=0,
            )
        with pytest.raises(ValidationError):
            ReviewResult(
                verdict=ReviewVerdict.REJECT,
                score=11,
            )


# --- CodeChange Tests ---


class TestCodeChange:
    def test_creation(self):
        cc = CodeChange(
            changes=[
                FileChange(
                    path="src/main.py",
                    new_content="print('hi')",
                    change_type=ChangeType.MODIFY,
                    explanation="Updated",
                )
            ],
            commit_message="feat(auth): add login",
            pr_title="Add login feature",
            pr_description="This PR adds login",
        )
        assert cc.pr_title == "Add login feature"
        assert cc.test_changes == []
        assert cc.unfulfilled_criteria == []


# --- Other Models ---


class TestOtherModels:
    def test_skipped_file(self):
        sf = SkippedFile(path="dist/bundle.js", reason="generated")
        assert sf.reason == "generated"

    def test_code_context(self):
        ctx = CodeContext(repository_name="my-repo")
        assert ctx.files == []
        assert ctx.test_files == []
        assert ctx.tech_stack == []
        assert ctx.file_tree is None
        assert ctx.skipped_files == []

    def test_pipeline_result(self):
        pr = PipelineResult(issue_key="PROJ-1", success=True, pr_url="https://example.com/pr/1")
        assert pr.dry_run is False

    def test_llm_response(self):
        lr = LLMResponse(
            content="Generated code",
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
        )
        assert lr.success is True
        assert lr.error_message is None

    def test_webhook_event(self):
        we = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key="PROJ-1",
            assignee="ai-developer",
        )
        assert we.previous_assignee is None
        assert we.project_key is None
        assert we.raw_payload is None


# --- Serialization Round-Trip ---


class TestSerializationRoundTrip:
    def test_task_context_round_trip(self):
        ctx = TaskContext(
            issue_key="PROJ-1",
            summary="Test",
            description="Desc",
            repository_name="repo",
            estimated_scope=TaskScope.SMALL,
        )
        json_str = ctx.model_dump_json()
        restored = TaskContext.model_validate_json(json_str)
        assert restored == ctx

    def test_review_result_round_trip(self):
        rr = ReviewResult(
            verdict=ReviewVerdict.APPROVE,
            score=8,
            findings=[],
            acceptance_criteria_met=True,
        )
        json_str = rr.model_dump_json()
        restored = ReviewResult.model_validate_json(json_str)
        assert restored == rr

    def test_webhook_event_round_trip(self):
        we = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key="PROJ-1",
            assignee="bot",
            project_key="PROJ",
        )
        json_str = we.model_dump_json()
        restored = WebhookEvent.model_validate_json(json_str)
        assert restored == we


# --- Property Tests: Serialization Round-Trip (Property 28) ---

from tests.conftest import code_changes, code_files, task_contexts


class TestSerializationRoundTripProperty:
    """Property 28: Data Model Serialization Round-Trip.

    Validates: Requirements 14.3, 14.4, 14.5
    """

    @given(task_contexts())
    @settings(max_examples=100)
    def test_task_context_round_trip_property(self, ctx: TaskContext) -> None:
        """TaskContext survives JSON round-trip."""
        json_str = ctx.model_dump_json()
        restored = TaskContext.model_validate_json(json_str)
        assert restored == ctx

    @given(code_changes())
    @settings(max_examples=100)
    def test_code_change_round_trip_property(self, change: CodeChange) -> None:
        """CodeChange survives JSON round-trip."""
        json_str = change.model_dump_json()
        restored = CodeChange.model_validate_json(json_str)
        assert restored == change

    @given(code_files())
    @settings(max_examples=100)
    def test_code_file_round_trip_property(self, cf: CodeFile) -> None:
        """CodeFile survives JSON round-trip."""
        json_str = cf.model_dump_json()
        restored = CodeFile.model_validate_json(json_str)
        assert restored.path == cf.path
        assert restored.content == cf.content
        assert restored.language == cf.language
        assert restored.is_test == cf.is_test

    @given(
        st.builds(
            ReviewResult,
            verdict=st.just(ReviewVerdict.APPROVE),
            score=st.integers(min_value=7, max_value=10),
            findings=st.just([]),
            acceptance_criteria_met=st.just(True),
        )
    )
    @settings(max_examples=50)
    def test_review_result_approve_round_trip_property(self, rr: ReviewResult) -> None:
        """ReviewResult (APPROVE) survives JSON round-trip."""
        json_str = rr.model_dump_json()
        restored = ReviewResult.model_validate_json(json_str)
        assert restored == rr


# --- Property Tests: ReviewResult APPROVE Constraints (Property 17) ---


class TestReviewResultApproveConstraintsProperty:
    """Property 17: ReviewResult APPROVE Constraints.

    Validates: Requirements 5.8, 5.9
    """

    @given(
        score=st.integers(min_value=7, max_value=10),
    )
    @settings(max_examples=100)
    def test_approve_valid_score_range(self, score: int) -> None:
        """APPROVE with score in [7,10] and no CRITICAL findings is always valid."""
        rr = ReviewResult(
            verdict=ReviewVerdict.APPROVE,
            score=score,
            findings=[],
            acceptance_criteria_met=True,
        )
        assert rr.verdict == ReviewVerdict.APPROVE
        assert rr.score == score

    @given(
        score=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=100)
    def test_approve_low_score_always_raises(self, score: int) -> None:
        """APPROVE with score < 7 always raises ValidationError."""
        with pytest.raises(ValidationError, match="APPROVE requires score >= 7"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=score,
                findings=[],
                acceptance_criteria_met=True,
            )

    @given(
        score=st.integers(min_value=7, max_value=10),
        category=st.sampled_from(["security", "logic", "style", "performance", "test"]),
        message=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_approve_with_critical_finding_always_raises(
        self, score: int, category: str, message: str
    ) -> None:
        """APPROVE with any CRITICAL finding always raises ValidationError."""
        finding = ReviewFinding(
            file_path="src/main.py",
            severity=FindingSeverity.CRITICAL,
            category=category,
            message=message,
        )
        with pytest.raises(ValidationError, match="APPROVE requires zero CRITICAL findings"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=score,
                findings=[finding],
                acceptance_criteria_met=True,
            )

    @given(
        score=st.integers(min_value=7, max_value=10),
    )
    @settings(max_examples=50)
    def test_approve_criteria_not_met_always_raises(self, score: int) -> None:
        """APPROVE with acceptance_criteria_met=False always raises ValidationError."""
        with pytest.raises(ValidationError, match="APPROVE requires all acceptance criteria met"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=score,
                findings=[],
                acceptance_criteria_met=False,
            )

    @given(
        verdict=st.sampled_from([ReviewVerdict.REQUEST_CHANGES, ReviewVerdict.REJECT]),
        score=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_non_approve_verdicts_have_no_score_constraint(
        self, verdict: ReviewVerdict, score: int
    ) -> None:
        """REQUEST_CHANGES and REJECT verdicts have no score or criteria constraints."""
        rr = ReviewResult(
            verdict=verdict,
            score=score,
            findings=[],
            acceptance_criteria_met=False,
        )
        assert rr.verdict == verdict
        assert rr.score == score
