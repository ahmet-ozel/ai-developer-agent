"""Confluence Publisher — publishes pipeline results to Confluence.

Creates a Confluence page documenting each AI pipeline run, including
issue summary, changed files, review findings, and PR link.
Gracefully disabled when Confluence credentials are not configured.
Never fails the pipeline — all errors are caught and logged.

Requirements: FR-5, FR-6, FR-7, FR-8, CP-3, CP-4, CP-5
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    CodeChange,
    PipelineResult,
    ReviewResult,
    TaskContext,
)

logger = logging.getLogger(__name__)

_MAX_DIFF_CHARS = 5000

try:
    from mcp_agent.agents.agent import Agent
except ImportError:  # pragma: no cover

    class Agent:  # type: ignore[no-redef]
        """Placeholder when mcp-agent is not installed."""

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


class ConfluencePublisher:
    """Publishes pipeline results to Confluence via mcp-atlassian.

    Disabled silently when confluence_enabled=False or credentials missing.
    All publish errors are caught — the pipeline is never failed.
    """

    def __init__(self, config: Settings, llm_router: LLMRouter | None = None) -> None:
        self._config = config
        self._llm_router = llm_router
        self._enabled: bool = (
            config.confluence_enabled
            and bool(config.confluence_url)
            and config.confluence_api_token is not None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(
        self,
        result: PipelineResult,
        task_ctx: TaskContext,
        code_change: CodeChange | None,
        review: ReviewResult | None,
        pr_url: str | None,
        duration_s: float = 0.0,
    ) -> str | None:
        """Create a Confluence page documenting the pipeline run.

        Returns the page URL on success, None if disabled or on error.
        Never raises — all exceptions are caught and logged as warnings.
        """
        if not self._enabled:
            logger.debug(
                "Confluence publisher disabled — skipping page creation for %s",
                task_ctx.issue_key,
            )
            return None

        try:
            title = self._build_page_title(task_ctx)
            content = self._build_page_content(
                result, task_ctx, code_change, review, pr_url, duration_s
            )
            page_url = await self._create_page(title, content)
            if page_url:
                await self._add_labels(page_url, task_ctx)
            return page_url
        except Exception:
            logger.warning(
                "Failed to publish Confluence page for %s — pipeline continues",
                task_ctx.issue_key,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Page title
    # ------------------------------------------------------------------

    def _build_page_title(self, task_ctx: TaskContext) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"[AI-Agent] {task_ctx.issue_key}: {task_ctx.summary} — {timestamp}"

    # ------------------------------------------------------------------
    # Page content builder
    # ------------------------------------------------------------------

    def _build_page_content(
        self,
        result: PipelineResult,
        task_ctx: TaskContext,
        code_change: CodeChange | None,
        review: ReviewResult | None,
        pr_url: str | None,
        duration_s: float,
    ) -> str:
        """Build Confluence wiki markup for the pipeline run page."""
        jira_url = self._config.jira_url.rstrip("/")
        issue_link = f"[{task_ctx.issue_key}|{jira_url}/browse/{task_ctx.issue_key}]"

        sections: list[str] = []

        # Header info panel
        status_icon = "(/)" if result.success else "(x)"
        status_text = "SUCCESS" if result.success else f"FAILED at {result.failure_stage or 'unknown'}"
        sections.append(
            f"{{info}}\n"
            f"*Jira Issue:* {issue_link}\n"
            f"*Summary:* {task_ctx.summary}\n"
            f"*Status:* {status_icon} {status_text}\n"
            f"*Duration:* {duration_s:.1f}s\n"
            f"{{info}}"
        )

        # Failure reason
        if not result.success and result.failure_reason:
            sections.append(
                f"h2. Failure Details\n"
                f"{{warning}}\n{result.failure_reason}\n{{warning}}"
            )

        # Changed files
        if code_change:
            all_changes = code_change.changes + code_change.test_changes
            if all_changes:
                rows = "\n".join(
                    f"|| {fc.path} || {fc.change_type.value} ||"
                    for fc in all_changes
                )
                sections.append(
                    f"h2. Changed Files\n"
                    f"|| File || Action ||\n{rows}"
                )

            # Diff summary (truncated)
            diff_parts: list[str] = []
            for fc in all_changes:
                if fc.new_content:
                    diff_parts.append(f"--- {fc.path} ---\n{fc.new_content}")
            if diff_parts:
                full_diff = "\n\n".join(diff_parts)
                if len(full_diff) > _MAX_DIFF_CHARS:
                    full_diff = full_diff[:_MAX_DIFF_CHARS] + "\n... [truncated]"
                sections.append(
                    f"h2. Code Diff Summary\n"
                    f"{{code}}\n{full_diff}\n{{code}}"
                )

        # Review findings
        if review:
            verdict = review.verdict.value.upper()
            findings_text = ""
            if review.findings:
                findings_text = "\n".join(
                    f"* [{f.severity.value}] {f.file_path}: {f.message}"
                    for f in review.findings
                )
            sections.append(
                f"h2. Code Review\n"
                f"*Verdict:* {verdict} (score: {review.score}/10)\n"
                + (f"\n{findings_text}" if findings_text else "")
            )

        # PR link
        if pr_url:
            sections.append(f"h2. Pull Request\n[{pr_url}|{pr_url}]")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # MCP helpers
    # ------------------------------------------------------------------

    async def _create_page(self, title: str, content: str) -> str | None:
        """Create a Confluence page via mcp-atlassian and return its URL."""
        if self._llm_router is None:
            logger.warning("No LLM router configured for ConfluencePublisher")
            return None

        confluence_agent = Agent(
            name="confluence_publisher",
            instruction="Use Confluence tools to create pages and add labels.",
            server_names=["atlassian"],
        )
        async with confluence_agent:
            llm = await confluence_agent.attach_llm(
                self._llm_router.get_llm_class("fast")
            )
            space_key = self._config.confluence_space_key or "~personal"
            parent_id = self._config.confluence_parent_page_id

            parent_clause = (
                f" under parent page ID {parent_id}" if parent_id else ""
            )
            prompt = (
                f"Create a Confluence page in space '{space_key}'{parent_clause} "
                f"with title: {title!r}\n\n"
                f"Content (wiki markup):\n{content}\n\n"
                f"Return the URL of the created page."
            )
            result = await llm.generate_str(prompt)

        match = re.search(r"https?://\S+", result)
        return match.group(0).rstrip(".,)") if match else None

    async def _add_labels(self, page_url: str, task_ctx: TaskContext) -> None:
        """Add labels to the Confluence page."""
        if self._llm_router is None:
            return

        labels = [
            "ai-agent",
            task_ctx.issue_key.lower().replace("-", "_"),
            task_ctx.repository_name.lower().replace("-", "_").replace("/", "_"),
        ]
        confluence_agent = Agent(
            name="confluence_labeler",
            instruction="Use Confluence tools to add labels to pages.",
            server_names=["atlassian"],
        )
        async with confluence_agent:
            llm = await confluence_agent.attach_llm(
                self._llm_router.get_llm_class("fast")
            )
            prompt = (
                f"Add these labels to the Confluence page at {page_url}: "
                f"{', '.join(labels)}"
            )
            await llm.generate_str(prompt)
