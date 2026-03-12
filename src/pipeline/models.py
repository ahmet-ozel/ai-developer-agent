"""Pipeline data models for the AI Developer Agent.

All models use Pydantic BaseModel for validation and JSON serialization.
Data flows between pipeline stages via these models.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Enums ---


class TaskScope(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class FindingSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    GOOD = "good"


class ChangeType(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


# --- Task Context ---


class TaskContext(BaseModel):
    issue_key: str
    summary: str
    description: str
    acceptance_criteria: Optional[str] = None
    repository_name: str
    estimated_scope: TaskScope
    comments: list[str] = Field(default_factory=list)
    confluence_docs: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    linked_issue_summaries: list[str] = Field(default_factory=list)
    previous_review_feedback: Optional[str] = None
    issue_type: Optional[str] = None
    reporter: Optional[str] = None
    base_branch: str = "main"
    priority: Optional[str] = None


# --- Code Context ---


class CodeFile(BaseModel):
    path: str
    content: str
    line_count: int = 0
    language: str = ""
    is_test: bool = False

    @model_validator(mode="after")
    def compute_line_count(self):
        if self.line_count == 0 and self.content:
            self.line_count = len(self.content.splitlines())
        return self


class SkippedFile(BaseModel):
    path: str
    reason: str  # "too_large", "binary", "lock_file", "generated"


class CodeContext(BaseModel):
    files: list[CodeFile] = Field(default_factory=list)
    test_files: list[CodeFile] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    repository_name: str
    file_tree: Optional[str] = None
    skipped_files: list[SkippedFile] = Field(default_factory=list)


# --- Code Change ---


class FileChange(BaseModel):
    path: str
    new_content: Optional[str] = None
    change_type: ChangeType
    explanation: str

    @model_validator(mode="after")
    def validate_content_for_change_type(self):
        if self.change_type == ChangeType.DELETE:
            if self.new_content is not None:
                raise ValueError(
                    "new_content must be None for DELETE changes"
                )
        else:
            # CREATE or MODIFY require content
            if self.new_content is None:
                raise ValueError(
                    f"new_content is required for {self.change_type.value} changes"
                )
        return self


class CodeChange(BaseModel):
    changes: list[FileChange]
    test_changes: list[FileChange] = Field(default_factory=list)
    commit_message: str
    pr_title: str
    pr_description: str
    unfulfilled_criteria: list[str] = Field(default_factory=list)


# --- Review Result ---


class ReviewFinding(BaseModel):
    file_path: str
    line_range: Optional[str] = None
    severity: FindingSeverity
    category: str  # "security", "logic", "style", "performance", "test"
    message: str


class ReviewResult(BaseModel):
    verdict: ReviewVerdict
    score: int = Field(ge=1, le=10)
    findings: list[ReviewFinding] = Field(default_factory=list)
    feedback_for_rewrite: Optional[str] = None
    acceptance_criteria_met: bool = False

    @model_validator(mode="after")
    def validate_approve_conditions(self):
        if self.verdict == ReviewVerdict.APPROVE:
            critical_count = sum(
                1 for f in self.findings if f.severity == FindingSeverity.CRITICAL
            )
            if self.score < 7:
                raise ValueError("APPROVE requires score >= 7")
            if critical_count > 0:
                raise ValueError("APPROVE requires zero CRITICAL findings")
            if not self.acceptance_criteria_met:
                raise ValueError("APPROVE requires all acceptance criteria met")
        return self


# --- Pipeline Result ---


class PipelineResult(BaseModel):
    issue_key: str
    success: bool
    pr_url: Optional[str] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    dry_run: bool = False


# --- LLM ---


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
