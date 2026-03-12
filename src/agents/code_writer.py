"""CodeWriter Agent — generates code changes from task and code context.

Uses mcp-agent Agent + AugmentedLLM pattern with server_names=[] (no MCP
servers needed). Produces full file content (not diffs), test files,
conventional commits format commit messages, PR title/description, and
tracks unfulfilled acceptance criteria.

Supports review feedback for regeneration in the review loop.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    ChangeType,
    CodeChange,
    CodeContext,
    FileChange,
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
# Instruction prompt for the CodeWriter agent
# ---------------------------------------------------------------------------

CODE_WRITER_INSTRUCTION = """You are a code writer agent. Your job is to generate code changes
based on a Jira task description and existing code context.

Rules:
1. Produce COMPLETE file content for each changed file — never output diffs.
2. Generate or update test files covering your changes (happy path + error cases).
3. Use Conventional Commits format for the commit message (e.g. "feat(auth): add token refresh").
4. Generate a clear PR title and description.
5. Respect existing code style and conventions.
6. Address every acceptance criterion. If a criterion cannot be fulfilled, note it in unfulfilled_criteria.

CRITICAL: Return ONLY a valid JSON object. No markdown code blocks. No explanation text.
Keep file contents concise to avoid response truncation.

JSON schema:
{
  "changes": [{"path": "...", "new_content": "...", "change_type": "create|modify|delete", "explanation": "..."}],
  "test_changes": [{"path": "...", "new_content": "...", "change_type": "create|modify", "explanation": "..."}],
  "commit_message": "type(scope): description",
  "pr_title": "...",
  "pr_description": "...",
  "unfulfilled_criteria": ["..."]
}"""


# ---------------------------------------------------------------------------
# CodeWriterAgent
# ---------------------------------------------------------------------------


class CodeWriterAgent:
    """Generates code changes using strong tier LLM without MCP servers.

    Produces full file content (not diffs), test files, conventional commits
    format commit messages, and tracks unfulfilled acceptance criteria.
    """

    def __init__(self, settings: Settings, llm_router: LLMRouter) -> None:
        self._settings = settings
        self._llm_router = llm_router

    async def write_code(
        self,
        task_context: TaskContext,
        code_context: CodeContext,
        review_feedback: str | None = None,
    ) -> CodeChange:
        """Generate code changes for the given task and code context.

        Args:
            task_context: The Jira task details and requirements.
            code_context: The relevant source files and tech stack.
            review_feedback: Optional feedback from a previous review iteration.

        Returns:
            A CodeChange with file changes, test changes, commit message,
            PR title/description, and any unfulfilled criteria.
        """
        llm_class = self._llm_router.get_llm_class("strong")

        agent = Agent(
            name="code_writer",
            instruction=CODE_WRITER_INSTRUCTION,
            server_names=[],
        )

        last_exc: Exception | None = None
        for attempt in range(2):
            async with agent:
                llm = await agent.attach_llm(llm_class)
                prompt = self._build_prompt(task_context, code_context, review_feedback)
                if attempt > 0:
                    prompt += (
                        "\n\nIMPORTANT: Your previous response was not valid JSON. "
                        "Return ONLY a valid JSON object. No markdown, no explanation. "
                        "Keep file contents short if needed to avoid truncation."
                    )
                result = await llm.generate_str(prompt)
                try:
                    return self._parse_response(result)
                except (ValueError, Exception) as exc:
                    last_exc = exc
                    logger.warning(
                        "CodeWriter JSON parse failed (attempt %d/2): %s",
                        attempt + 1, exc,
                    )
        raise last_exc  # type: ignore[misc]

    def _build_prompt(
        self,
        task_context: TaskContext,
        code_context: CodeContext,
        review_feedback: str | None = None,
    ) -> str:
        """Build a comprehensive prompt for the LLM.

        Includes task details, existing code, tech stack, and optionally
        review feedback for regeneration.
        """
        sections: list[str] = []

        # Task details
        sections.append("## Task Details")
        sections.append(f"Issue: {task_context.issue_key}")
        sections.append(f"Summary: {task_context.summary}")
        sections.append(f"Description: {task_context.description}")
        if task_context.acceptance_criteria:
            sections.append(f"Acceptance Criteria: {task_context.acceptance_criteria}")
        if task_context.labels:
            sections.append(f"Labels: {', '.join(task_context.labels)}")
        if task_context.priority:
            sections.append(f"Priority: {task_context.priority}")

        # Tech stack
        if code_context.tech_stack:
            sections.append(f"\n## Tech Stack: {', '.join(code_context.tech_stack)}")

        # Existing source files
        if code_context.files:
            sections.append("\n## Existing Source Files")
            for f in code_context.files:
                sections.append(f"\n### {f.path} ({f.language})")
                sections.append(f"```\n{f.content}\n```")

        # Existing test files
        if code_context.test_files:
            sections.append("\n## Existing Test Files")
            for f in code_context.test_files:
                sections.append(f"\n### {f.path}")
                sections.append(f"```\n{f.content}\n```")

        # Review feedback (for regeneration)
        if review_feedback:
            sections.append("\n## Review Feedback (Please Address)")
            sections.append(review_feedback)

        # Instructions
        sections.append("\n## Instructions")
        sections.append(
            "Generate the code changes as a JSON object. "
            "Produce COMPLETE file content (not diffs). "
            "Include test files in test_changes. "
            "Use Conventional Commits format for commit_message. "
            "Address every acceptance criterion or list unfulfilled ones."
        )

        return "\n".join(sections)

    @staticmethod
    def _parse_response(raw: str) -> CodeChange:
        """Parse LLM JSON response into a CodeChange model.

        Handles JSON embedded in markdown code blocks, attempts to fix
        truncated JSON, and validates the structure via Pydantic.
        """
        logger.debug("CodeWriter raw response length: %d chars", len(raw))
        cleaned = _extract_json(raw)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError) as exc:
            # Log first 500 chars for debugging
            logger.error(
                "CodeWriter JSON parse failed. First 500 chars: %s",
                cleaned[:500],
            )
            raise ValueError(
                f"CodeWriter LLM response is not valid JSON: {exc}"
            ) from exc

        return CodeChange.model_validate(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code blocks.

    Also attempts to fix common LLM truncation issues (unclosed strings,
    missing brackets).
    """
    stripped = text.strip()

    # Try to extract from ```json ... ``` or ``` ... ```
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            # Remove optional language tag (e.g. "json\n{...")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                return _try_fix_truncated_json(candidate)

    if stripped.startswith("{"):
        return _try_fix_truncated_json(stripped)

    return stripped


def _try_fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON from LLM responses.

    Common issues: unclosed strings, missing closing brackets/braces.
    """
    import json as _json

    # First try as-is
    try:
        _json.loads(text)
        return text
    except _json.JSONDecodeError:
        pass

    # Strategy 1: close unclosed string literal, then close brackets
    fixed = text.rstrip()
    # If ends mid-string (no closing quote), close it
    in_string = False
    escape = False
    for ch in fixed:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        fixed += '"'

    # Count open brackets/braces and close them
    opens = {'[': 0, '{': 0}
    closes = {']': '[', '}': '{'}
    in_str = False
    esc = False
    for ch in fixed:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in opens:
            opens[ch] += 1
        elif ch in closes:
            opens[closes[ch]] -= 1

    # Close any unclosed arrays then objects
    fixed += ']' * max(0, opens['['])
    fixed += '}' * max(0, opens['{'])

    try:
        _json.loads(fixed)
        return fixed
    except _json.JSONDecodeError:
        pass

    # Strategy 2: find the last complete top-level object
    # by scanning for balanced braces
    return text
