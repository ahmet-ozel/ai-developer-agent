"""CodeReviewer Agent - reviews generated code changes and produces a verdict.

Uses mcp-agent Agent + AugmentedLLM pattern with server_names=[] (no MCP
servers needed). Reviews code for security issues, quality, and correctness.
Produces a ReviewResult with verdict (APPROVE/REQUEST_CHANGES/REJECT),
quality score (1-10), findings with severity categorization, and
hardcoded secret detection.

Requirements: 5.1, 5.2, 5.8, 5.9, 12.6
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    CodeChange,
    CodeContext,
    FindingSeverity,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    TaskContext,
)

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
# Hardcoded secret detection patterns
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("Hardcoded password", re.compile(r"""password\s*=\s*["'][^"']+["']""")),
    ("Hardcoded secret", re.compile(r"""secret\s*=\s*["'][^"']+["']""")),
    ("Hardcoded API key", re.compile(r"""api_key\s*=\s*["'][^"']+["']""")),
]

# ---------------------------------------------------------------------------
# Instruction prompt for the CodeReviewer agent
# ---------------------------------------------------------------------------

CODE_REVIEWER_INSTRUCTION = """You are a code reviewer agent. Your job is to review code changes
produced by a code writer agent and provide a structured review result.

Review the code for:
1. Security issues (hardcoded secrets, SQL injection, XSS, etc.)
2. Logic errors and bugs
3. Code style and best practices
4. Performance issues
5. Test coverage and quality

Categorize each finding with severity:
- CRITICAL: Must fix - security vulnerability, logic error, crash
- WARNING: Should fix - bad practice, edge case not handled
- SUGGESTION: Nice to have - readability, optimization
- GOOD: Positive feedback - well-implemented patterns

Assign a quality score from 1 to 10.

Determine a verdict:
- APPROVE: Score >= 7, zero CRITICAL findings, all acceptance criteria met
- REQUEST_CHANGES: Fixable issues found, provide feedback for rewrite
- REJECT: Fundamental issues that cannot be fixed with simple changes

Return your response as a JSON object with this exact schema:
{
  "verdict": "approve|request_changes|reject",
  "score": 8,
  "findings": [
    {
      "file_path": "src/example.py",
      "line_range": "10-15",
      "severity": "critical|warning|suggestion|good",
      "category": "security|logic|style|performance|test",
      "message": "Description of the finding"
    }
  ],
  "feedback_for_rewrite": "Optional feedback for the code writer if verdict is request_changes",
  "acceptance_criteria_met": true
}"""


# ---------------------------------------------------------------------------
# CodeReviewerAgent
# ---------------------------------------------------------------------------


class CodeReviewerAgent:
    """Reviews generated code using strong tier LLM without MCP servers.

    Produces a ReviewResult with verdict, quality score, findings with
    severity categorization, and hardcoded secret detection.
    """

    def __init__(self, settings: Settings, llm_router: LLMRouter) -> None:
        self._settings = settings
        self._llm_router = llm_router

    async def review_code(
        self,
        task_context: TaskContext,
        code_context: CodeContext,
        code_change: CodeChange,
    ) -> ReviewResult:
        """Review code changes and produce a ReviewResult.

        Args:
            task_context: The Jira task details and requirements.
            code_context: The relevant source files and tech stack.
            code_change: The proposed code changes to review.

        Returns:
            A ReviewResult with verdict, score, findings, and feedback.
        """
        # Run hardcoded secret detection first
        secret_findings = self._check_hardcoded_secrets(code_change)

        llm_class = self._llm_router.get_llm_class("strong")

        agent = Agent(
            name="code_reviewer",
            instruction=CODE_REVIEWER_INSTRUCTION,
            server_names=[],
        )

        async with agent:
            llm = await agent.attach_llm(llm_class)
            prompt = self._build_prompt(task_context, code_context, code_change)
            result = await llm.generate_str(prompt)
            review = self._parse_response(result)

        # Merge secret findings into the review
        if secret_findings:
            review.findings.extend(secret_findings)
            # If there are CRITICAL secret findings, cannot APPROVE
            has_critical = any(
                f.severity == FindingSeverity.CRITICAL for f in secret_findings
            )
            if has_critical and review.verdict == ReviewVerdict.APPROVE:
                review.verdict = ReviewVerdict.REQUEST_CHANGES
                review.feedback_for_rewrite = (
                    (review.feedback_for_rewrite or "")
                    + "\nHardcoded secrets detected. Remove all secrets and use environment variables."
                ).strip()

        return review

    def _build_prompt(
        self,
        task_context: TaskContext,
        code_context: CodeContext,
        code_change: CodeChange,
    ) -> str:
        """Build a comprehensive review prompt for the LLM."""
        sections: list[str] = []

        # Task details
        sections.append("## Task Details")
        sections.append(f"Issue: {task_context.issue_key}")
        sections.append(f"Summary: {task_context.summary}")
        sections.append(f"Description: {task_context.description}")
        if task_context.acceptance_criteria:
            sections.append(f"Acceptance Criteria: {task_context.acceptance_criteria}")

        # Existing code context
        if code_context.files:
            sections.append("\n## Existing Source Files")
            for f in code_context.files:
                sections.append(f"\n### {f.path} ({f.language})")
                sections.append(f"```\n{f.content}\n```")

        # Proposed changes
        sections.append("\n## Proposed Changes")
        sections.append(f"Commit: {code_change.commit_message}")
        for change in code_change.changes:
            sections.append(f"\n### {change.path} ({change.change_type.value})")
            sections.append(f"Explanation: {change.explanation}")
            if change.new_content:
                sections.append(f"```\n{change.new_content}\n```")

        # Test changes
        if code_change.test_changes:
            sections.append("\n## Test Changes")
            for change in code_change.test_changes:
                sections.append(f"\n### {change.path} ({change.change_type.value})")
                sections.append(f"Explanation: {change.explanation}")
                if change.new_content:
                    sections.append(f"```\n{change.new_content}\n```")

        # Unfulfilled criteria
        if code_change.unfulfilled_criteria:
            sections.append("\n## Unfulfilled Criteria")
            for criterion in code_change.unfulfilled_criteria:
                sections.append(f"- {criterion}")

        # Instructions
        sections.append("\n## Review Instructions")
        sections.append(
            "Review the proposed changes against the task requirements. "
            "Check for security issues, logic errors, code style, and test coverage. "
            "Check for hardcoded secrets, API keys, passwords, and credentials. "
            "Return your review as a JSON object."
        )

        return "\n".join(sections)

    @staticmethod
    def _check_hardcoded_secrets(code_change: CodeChange) -> list[ReviewFinding]:
        """Scan new_content in code changes for known secret patterns.

        Returns a list of CRITICAL security findings for any detected secrets.
        """
        findings: list[ReviewFinding] = []

        all_changes = list(code_change.changes) + list(code_change.test_changes)
        for change in all_changes:
            if change.new_content is None:
                continue
            lines = change.new_content.splitlines()
            for line_num, line in enumerate(lines, start=1):
                for pattern_name, pattern in _SECRET_PATTERNS:
                    if pattern.search(line):
                        findings.append(
                            ReviewFinding(
                                file_path=change.path,
                                line_range=str(line_num),
                                severity=FindingSeverity.CRITICAL,
                                category="security",
                                message=f"{pattern_name} detected: {line.strip()[:80]}",
                            )
                        )

        return findings

    @staticmethod
    def _parse_response(raw: str) -> ReviewResult:
        """Parse LLM JSON response into a ReviewResult model.

        Handles JSON embedded in markdown code blocks and validates
        the structure via Pydantic.
        """
        cleaned = _extract_json(raw)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"CodeReviewer LLM response is not valid JSON: {exc}"
            ) from exc

        return ReviewResult.model_validate(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    stripped = text.strip()

    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                return candidate

    return stripped
