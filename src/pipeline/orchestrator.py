"""Pipeline Orchestrator — manages the deterministic pipeline flow.

Runs the full pipeline: TaskReader → scope check → task type filter →
CodeFinder → token budget → review loop → PR creation. Handles Jira
communication (comments, transitions) with dry-run support and secret
masking. Uses try/finally for cleanup guarantee.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 11.2, 11.4, 11.5, 12.7
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from typing import Any

from src.agents.code_finder import CodeFinderAgent
from src.agents.code_writer import CodeWriterAgent
from src.agents.code_reviewer import CodeReviewerAgent
from src.agents.task_reader import TaskReaderAgent, should_skip_task
from src.clients.gitlab_client import GitLabClient
from src.clients.bitbucket_client import BitbucketClient
from src.config.mcp_servers import get_active_git_server_name
from src.config.settings import Settings
from src.pipeline.confluence_publisher import ConfluencePublisher
from src.pipeline.llm_router import LLMRouter
from src.pipeline.logging import PipelineLogger
from src.pipeline.models import (
    CodeChange,
    CodeContext,
    FileChange,
    FindingSeverity,
    PipelineResult,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    TaskContext,
    TaskScope,
)
from src.pipeline.token_budget import trim_code_context
from src.utils.git_helpers import generate_branch_name
from src.utils.jira_helpers import format_jira_comment, mask_secrets

# ---------------------------------------------------------------------------
# Try importing mcp-agent Agent class for Jira helper agent pattern.
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
# PipelineOrchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Deterministic pipeline orchestrator.

    Accepts Settings and LLMRouter, runs the full pipeline for a given
    issue_key, and returns a PipelineResult.
    """

    def __init__(self, config: Settings, llm_router: LLMRouter) -> None:
        self._config = config
        self._llm_router = llm_router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, issue_key: str) -> PipelineResult:
        """Execute the full pipeline for *issue_key*.

        Flow:
        a. Create PipelineLogger
        b. Comment on Jira: "AI Developer started processing"
        c. Transition Jira to "In Progress"
        d. TaskReader → read task
        e. Scope check (LARGE → halt, comment on Jira)
        f. Task type filtering (should_skip_task)
        g. CodeFinder → find code
        h. Token budget → trim_code_context
        i. Review loop
        j. PR creation
        k. Confluence documentation
        l. Comment on Jira: "AI Developer completed"
        m. Return PipelineResult
        """
        pl = PipelineLogger(issue_key)
        pl.log_event("INFO", "Pipeline started", agent_name="orchestrator", stage="init")

        # Track state for Confluence publisher (called on all exit paths)
        _task_ctx: TaskContext | None = None
        _code_change: CodeChange | None = None
        _review: ReviewResult | None = None
        _pr_url: str | None = None
        _t_start = time.monotonic()

        async def _publish_to_confluence(result: PipelineResult) -> None:
            """Publish pipeline result to Confluence (non-blocking)."""
            if _task_ctx is None:
                return
            duration_s = time.monotonic() - _t_start
            confluence = ConfluencePublisher(
                config=self._config, llm_router=self._llm_router
            )
            try:
                doc_url = await confluence.publish(
                    result, _task_ctx, _code_change, _review, _pr_url, duration_s
                )
                if doc_url:
                    await self._comment_on_jira(
                        issue_key,
                        format_jira_comment("orchestrator", "docs", f"📄 Documentation: {doc_url}"),
                    )
            except Exception:
                logger.warning(
                    "Confluence publish failed for %s — pipeline continues",
                    issue_key,
                    exc_info=True,
                )

        try:
            # b. Comment on Jira: started
            await self._comment_on_jira(
                issue_key,
                format_jira_comment(
                    "orchestrator",
                    "init",
                    "AI Developer started processing this task.",
                ),
            )

            # c. Transition Jira → In Progress (non-blocking)
            if self._config.jira_transition_in_progress:
                try:
                    await self._transition_jira(
                        issue_key, self._config.jira_transition_in_progress
                    )
                except Exception:
                    logger.warning(
                        "Jira transition (in_progress) failed for %s — continuing pipeline",
                        issue_key,
                        exc_info=True,
                    )

            # d. TaskReader → read task
            pl.log_stage_start("task_reader", "task_reader")
            t0 = time.monotonic()
            task_reader = TaskReaderAgent(
                settings=self._config, llm_router=self._llm_router
            )
            _task_ctx = await task_reader.read_task(issue_key)
            elapsed = (time.monotonic() - t0) * 1000
            pl.log_stage_end("task_reader", "task_reader", elapsed)

            # Task 11.4: populate previous_review_feedback for reassignment context
            if _task_ctx.previous_review_feedback is None:
                _task_ctx = _task_ctx.model_copy(
                    update={
                        "previous_review_feedback": await self._get_previous_review_feedback(issue_key)
                    }
                )

            # e. Scope check — LARGE → halt
            if _task_ctx.estimated_scope == TaskScope.LARGE:
                msg = (
                    "Task scope estimated as LARGE. "
                    "Recommend breaking into smaller subtasks."
                )
                pl.log_event(
                    "WARNING", msg, agent_name="orchestrator", stage="scope_check"
                )
                await self._comment_on_jira(
                    issue_key,
                    format_jira_comment("orchestrator", "scope_check", msg),
                )
                result = PipelineResult(
                    issue_key=issue_key,
                    success=False,
                    failure_stage="scope_check",
                    failure_reason=msg,
                    dry_run=self._config.dry_run,
                )
                await _publish_to_confluence(result)
                return result

            # f. Task type filtering
            skip, reason = should_skip_task(_task_ctx, self._config)
            if skip:
                pl.log_event(
                    "INFO",
                    f"Task skipped: {reason}",
                    agent_name="orchestrator",
                    stage="task_filter",
                )
                result = PipelineResult(
                    issue_key=issue_key,
                    success=False,
                    failure_stage="task_filter",
                    failure_reason=reason,
                    dry_run=self._config.dry_run,
                )
                await _publish_to_confluence(result)
                return result

            # g. CodeFinder → find code
            pl.log_stage_start("code_finder", "code_finder")
            t0 = time.monotonic()
            code_finder = CodeFinderAgent(
                settings=self._config, llm_router=self._llm_router
            )
            code_ctx = await code_finder.find_code(_task_ctx)
            elapsed = (time.monotonic() - t0) * 1000
            pl.log_stage_end("code_finder", "code_finder", elapsed)

            # Sync qualified repo name back to task context so PR creation uses owner/repo
            if code_ctx.repository_name and code_ctx.repository_name != _task_ctx.repository_name:
                _task_ctx = _task_ctx.model_copy(
                    update={"repository_name": code_ctx.repository_name}
                )

            # h. Token budget → trim
            code_ctx = trim_code_context(
                code_ctx, self._config.max_context_tokens, _task_ctx
            )

            # i. Review loop
            _code_change, _review = await self._run_review_loop(
                _task_ctx, code_ctx, self._config.max_review_retries
            )

            # Check review verdict
            if _review.verdict == ReviewVerdict.REJECT:
                msg = f"Code review REJECTED (score={_review.score}). Findings: {_review.feedback_for_rewrite or 'N/A'}"
                pl.log_event(
                    "WARNING", msg, agent_name="orchestrator", stage="review"
                )
                await self._comment_on_jira(
                    issue_key,
                    format_jira_comment("orchestrator", "review", msg),
                )
                result = PipelineResult(
                    issue_key=issue_key,
                    success=False,
                    failure_stage="review",
                    failure_reason=msg,
                    dry_run=self._config.dry_run,
                )
                await _publish_to_confluence(result)
                return result

            if _review.verdict != ReviewVerdict.APPROVE:
                msg = f"Review loop exhausted after {self._config.max_review_retries} retries."
                pl.log_event(
                    "WARNING", msg, agent_name="orchestrator", stage="review"
                )
                await self._comment_on_jira(
                    issue_key,
                    format_jira_comment("orchestrator", "review", msg),
                )
                result = PipelineResult(
                    issue_key=issue_key,
                    success=False,
                    failure_stage="review",
                    failure_reason=msg,
                    dry_run=self._config.dry_run,
                )
                await _publish_to_confluence(result)
                return result

            # j. PR creation
            _pr_url = await self._create_pull_request(_task_ctx, _code_change)

            # k. Comment on Jira: completed
            completion_msg = (
                f"AI Developer completed processing. PR: {_pr_url}"
                if _pr_url
                else "AI Developer completed processing."
            )
            await self._comment_on_jira(
                issue_key,
                format_jira_comment("orchestrator", "complete", completion_msg),
            )

            # Transition Jira → In Review (non-blocking)
            if self._config.jira_transition_in_review:
                try:
                    await self._transition_jira(
                        issue_key, self._config.jira_transition_in_review
                    )
                except Exception:
                    logger.warning(
                        "Jira transition (in_review) failed for %s — continuing pipeline",
                        issue_key,
                        exc_info=True,
                    )

            pl.log_event(
                "INFO",
                f"Pipeline completed successfully. PR: {_pr_url}",
                agent_name="orchestrator",
                stage="complete",
            )

            result = PipelineResult(
                issue_key=issue_key,
                success=True,
                pr_url=_pr_url,
                dry_run=self._config.dry_run,
            )
            # k. Confluence documentation
            await _publish_to_confluence(result)
            return result

        except Exception:
            # Unhandled exception catch-all
            tb = traceback.format_exc()
            pl.log_event(
                "ERROR",
                f"Unhandled exception:\n{tb}",
                agent_name="orchestrator",
                stage="error",
            )
            # Comment generic error on Jira (mask secrets from traceback)
            try:
                masked_tb = self._mask_secrets(
                    "An unexpected error occurred while processing this task. "
                    "The team has been notified."
                )
                await self._comment_on_jira(
                    issue_key,
                    format_jira_comment("orchestrator", "error", masked_tb),
                )
            except Exception:
                logger.exception(
                    "Failed to comment error on Jira for %s", issue_key
                )

            result = PipelineResult(
                issue_key=issue_key,
                success=False,
                failure_stage="error",
                failure_reason="Unhandled exception occurred",
                dry_run=self._config.dry_run,
            )
            await _publish_to_confluence(result)
            return result

    # ------------------------------------------------------------------
    # Jira communication helpers
    # ------------------------------------------------------------------

    async def _comment_on_jira(self, issue_key: str, message: str) -> None:
        """Add a comment on the Jira issue.

        In dry-run mode, logs the comment instead of writing to Jira.
        Uses format_jira_comment + mask_secrets, and the helper agent
        pattern with mcp-atlassian.
        """
        masked = self._mask_secrets(message)

        if self._config.dry_run:
            logger.info("[DRY-RUN] Would comment on %s: %s", issue_key, masked)
            return

        try:
            jira_agent = Agent(
                name="jira_commenter",
                instruction="Use jira_add_comment tool to add comments.",
                server_names=["atlassian"],
            )
            async with jira_agent:
                llm = await jira_agent.attach_llm(
                    self._llm_router.get_llm_class("fast")
                )
                await llm.generate_str(
                    f"Add comment to {issue_key}: {masked}"
                )
        except Exception:
            logger.exception(
                "Failed to comment on Jira issue %s", issue_key
            )

    async def _transition_jira(self, issue_key: str, transition_id: str) -> None:
        """Transition a Jira issue status. Non-blocking — logs errors but
        does not crash the pipeline.

        In dry-run mode, logs the transition instead of executing it.
        """
        if self._config.dry_run:
            logger.info(
                "[DRY-RUN] Would transition %s with transition_id=%s",
                issue_key,
                transition_id,
            )
            return

        try:
            jira_agent = Agent(
                name="jira_transitioner",
                instruction="Use jira_transition_issue tool to transition issues.",
                server_names=["atlassian"],
            )
            async with jira_agent:
                llm = await jira_agent.attach_llm(
                    self._llm_router.get_llm_class("fast")
                )
                await llm.generate_str(
                    f"Transition issue {issue_key} using transition ID {transition_id}"
                )
        except Exception:
            logger.warning(
                "Jira transition failed for %s (transition_id=%s) — continuing pipeline",
                issue_key,
                transition_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Secret masking helper
    # ------------------------------------------------------------------

    def _mask_secrets(self, text: str) -> str:
        """Mask configured secrets from text before logging or commenting."""
        secrets: list[str] = []
        for attr in (
            "jira_api_token",
            "jira_webhook_secret",
            "llm_fast_api_key",
            "llm_strong_api_key",
        ):
            val = getattr(self._config, attr, None)
            if val is not None:
                secret_val = val.get_secret_value() if hasattr(val, "get_secret_value") else str(val)
                if secret_val:
                    secrets.append(secret_val)

        # Also mask optional git provider secrets
        for attr in ("bitbucket_app_password", "github_token", "gitlab_token"):
            val = getattr(self._config, attr, None)
            if val is not None:
                secret_val = val.get_secret_value() if hasattr(val, "get_secret_value") else str(val)
                if secret_val:
                    secrets.append(secret_val)

        return mask_secrets(text, secrets)

    # ------------------------------------------------------------------
    # Placeholder methods for Tasks 11.2 and 11.3
    # ------------------------------------------------------------------

    async def _run_review_loop(
        self,
        task_ctx: TaskContext,
        code_ctx: CodeContext,
        max_retries: int,
    ) -> tuple[CodeChange, ReviewResult]:
        """Run the CodeWriter ↔ CodeReviewer review loop.

        Runs up to max_retries + 1 iterations. On REQUEST_CHANGES the
        reviewer's feedback is passed back to the writer. On APPROVE or
        REJECT returns immediately. After exhausting retries returns the
        last (code_change, review) pair — the caller checks the verdict.
        """
        code_writer = CodeWriterAgent(
            settings=self._config, llm_router=self._llm_router
        )
        code_reviewer = CodeReviewerAgent(
            settings=self._config, llm_router=self._llm_router
        )

        feedback: str | None = None
        code_change: CodeChange | None = None
        review: ReviewResult | None = None

        for attempt in range(max_retries + 1):
            # a. Write code (pass feedback on retries)
            code_change = await code_writer.write_code(
                task_ctx, code_ctx, review_feedback=feedback
            )

            # b. Max file change limit check (BEFORE reviewer)
            total = len(code_change.changes) + len(code_change.test_changes)
            if total > self._config.max_file_changes:
                msg = (
                    f"Too many file changes: {total} > {self._config.max_file_changes}. "
                    "Reduce scope."
                )
                logger.warning(
                    "Max file change limit exceeded for %s: %d > %d",
                    task_ctx.issue_key,
                    total,
                    self._config.max_file_changes,
                )
                await self._comment_on_jira(
                    task_ctx.issue_key,
                    format_jira_comment("orchestrator", "review", msg),
                )
                reject_review = ReviewResult(
                    verdict=ReviewVerdict.REJECT,
                    score=1,
                    findings=[
                        ReviewFinding(
                            file_path="*",
                            severity=FindingSeverity.CRITICAL,
                            category="scope",
                            message=msg,
                        )
                    ],
                    feedback_for_rewrite="Too many file changes. Reduce scope.",
                    acceptance_criteria_met=False,
                )
                return code_change, reject_review

            # c. Review code
            review = await code_reviewer.review_code(task_ctx, code_ctx, code_change)

            # d. APPROVE → done
            if review.verdict == ReviewVerdict.APPROVE:
                return code_change, review

            # e. REJECT → return immediately (caller halts)
            if review.verdict == ReviewVerdict.REJECT:
                return code_change, review

            # f. REQUEST_CHANGES → set feedback and continue
            feedback = review.feedback_for_rewrite

        # Exhausted retries — return last pair
        return code_change, review  # type: ignore[return-value]

    async def _create_pull_request(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
    ) -> str | None:
        """Create a branch, commit files, and open a PR.

        Uses direct REST API clients for GitLab/Bitbucket (MCP servers are
        broken). Falls back to MCP agent pattern for GitHub.

        Returns the PR URL or None.
        """
        branch_name = generate_branch_name(
            self._config.branch_pattern, task_ctx.issue_key
        )
        all_changes = code_change.changes + code_change.test_changes
        n_files = len(all_changes)

        # 1. Dry-run mode
        if self._config.dry_run:
            logger.info(
                "[DRY-RUN] Would create branch '%s', commit %d files, open PR: %s",
                branch_name,
                n_files,
                code_change.pr_title,
            )
            return None

        # 2. auto_create_pr disabled → comment changes summary on Jira
        if not self._config.auto_create_pr:
            changes_summary = "Changes summary (no PR created):\n" + "\n".join(
                f"- {fc.path} ({fc.change_type.value})" for fc in all_changes
            )
            await self._comment_on_jira(
                task_ctx.issue_key,
                format_jira_comment("orchestrator", "pr_creation", changes_summary),
            )
            return None

        # 3. Route to direct API or MCP based on provider
        provider = self._config.git_provider
        if provider in ("gitlab", "bitbucket"):
            return await self._create_pr_direct(task_ctx, code_change, branch_name, all_changes)
        else:
            return await self._create_pr_mcp(task_ctx, code_change, branch_name, all_changes)

    async def _create_pr_direct(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
        branch_name: str,
        all_changes: list[FileChange],
    ) -> str | None:
        """Create PR via direct REST API (GitLab/Bitbucket)."""
        provider = self._config.git_provider
        pr_title = f"[AI-BOT] {task_ctx.issue_key}: {task_ctx.summary}"
        pr_body = self._build_pr_description(task_ctx, code_change)

        try:
            if provider == "gitlab":
                return await self._create_pr_gitlab(
                    task_ctx, code_change, branch_name, all_changes, pr_title, pr_body
                )
            else:
                return await self._create_pr_bitbucket(
                    task_ctx, code_change, branch_name, all_changes, pr_title, pr_body
                )
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "already exists" in exc_msg or "branch exists" in exc_msg:
                suffix = str(int(time.time()))
                branch_name = generate_branch_name(
                    self._config.branch_pattern, task_ctx.issue_key, suffix=suffix
                )
                logger.warning(
                    "Branch collision for %s — retrying with suffix %s",
                    task_ctx.issue_key, suffix,
                )
                if provider == "gitlab":
                    return await self._create_pr_gitlab(
                        task_ctx, code_change, branch_name, all_changes, pr_title, pr_body
                    )
                else:
                    return await self._create_pr_bitbucket(
                        task_ctx, code_change, branch_name, all_changes, pr_title, pr_body
                    )
            raise

    async def _create_pr_gitlab(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
        branch_name: str,
        all_changes: list[FileChange],
        pr_title: str,
        pr_body: str,
    ) -> str | None:
        """GitLab: create branch → commit files → create MR."""
        client = GitLabClient(self._config)
        repo = task_ctx.repository_name

        # 1. Create branch
        await client.create_branch(repo, branch_name)
        logger.info("Created GitLab branch: %s", branch_name)

        # 2. Commit all files in a single commit
        actions = []
        for fc in all_changes:
            action = "create" if fc.change_type.value == "create" else "update"
            if fc.change_type.value == "delete":
                action = "delete"
            entry: dict[str, str] = {
                "action": action,
                "file_path": fc.path,
            }
            if action != "delete":
                entry["content"] = fc.new_content or ""
            actions.append(entry)

        await client.commit_files(repo, branch_name, code_change.commit_message, actions)
        logger.info("Committed %d files to %s", len(actions), branch_name)

        # 3. Create merge request
        mr = await client.create_merge_request(
            repo,
            source_branch=branch_name,
            title=pr_title,
            description=pr_body,
            draft=self._config.pr_draft_mode,
        )
        mr_url = mr.get("web_url")
        logger.info("Created GitLab MR: %s", mr_url)
        return mr_url

    async def _create_pr_bitbucket(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
        branch_name: str,
        all_changes: list[FileChange],
        pr_title: str,
        pr_body: str,
    ) -> str | None:
        """Bitbucket: commit files (auto-creates branch) → create PR."""
        client = BitbucketClient(self._config)
        repo = task_ctx.repository_name

        # 1. Create branch
        await client.create_branch(repo, branch_name)
        logger.info("Created Bitbucket branch: %s", branch_name)

        # 2. Commit files
        files_dict = {}
        for fc in all_changes:
            if fc.change_type.value != "delete":
                files_dict[fc.path] = fc.new_content or ""

        await client.commit_files(repo, branch_name, code_change.commit_message, files_dict)
        logger.info("Committed %d files to %s", len(files_dict), branch_name)

        # 3. Create pull request
        pr = await client.create_pull_request(
            repo,
            source_branch=branch_name,
            title=pr_title,
            description=pr_body,
        )
        pr_url = pr.get("links", {}).get("html", {}).get("href")
        logger.info("Created Bitbucket PR: %s", pr_url)
        return pr_url

    async def _create_pr_mcp(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
        branch_name: str,
        all_changes: list[FileChange],
    ) -> str | None:
        """Create PR via MCP agent pattern (GitHub)."""
        git_server = get_active_git_server_name(self._config)

        async def _attempt_create_pr(b_name: str) -> str | None:
            git_agent = Agent(
                name="pr_creator",
                instruction="Use git tools to create branch, commit files, and open PR.",
                server_names=[git_server],
            )
            async with git_agent:
                llm = await git_agent.attach_llm(
                    self._llm_router.get_llm_class("fast")
                )
                prompt = self._build_pr_prompt(task_ctx, code_change, b_name, all_changes)
                result = await llm.generate_str(prompt)
            return self._extract_pr_url(result)

        try:
            pr_url = await _attempt_create_pr(branch_name)
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "already exists" in exc_msg or "branch exists" in exc_msg:
                suffix = str(int(time.time()))
                branch_name = generate_branch_name(
                    self._config.branch_pattern, task_ctx.issue_key, suffix=suffix
                )
                logger.warning(
                    "Branch collision for %s — retrying with suffix %s",
                    task_ctx.issue_key, suffix,
                )
                pr_url = await _attempt_create_pr(branch_name)
            else:
                raise

        return pr_url

    def _build_pr_description(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
        review: ReviewResult | None = None,
    ) -> str:
        """Build structured markdown PR description."""
        sections: list[str] = []

        # Summary
        sections.append(f"## Summary\n{task_ctx.summary}")

        # Changes table
        all_changes = code_change.changes + code_change.test_changes
        if all_changes:
            rows = "\n".join(
                f"| {fc.path} | {fc.change_type.value} |"
                for fc in all_changes
            )
            sections.append(
                f"## Changes\n| File | Action |\n|------|--------|\n{rows}"
            )

        # Review section (only if review is provided)
        if review is not None:
            verdict_line = (
                f"Verdict: {review.verdict.value.upper()} (score: {review.score}/10)"
            )
            findings_lines = "\n".join(
                f"- [{f.severity.value}] {f.message}"
                for f in review.findings
            )
            review_body = verdict_line
            if findings_lines:
                review_body += f"\n{findings_lines}"
            sections.append(f"## Review\n{review_body}")

        # Jira link
        jira_url = self._config.jira_url.rstrip("/")
        sections.append(
            f"## Jira\n[{task_ctx.issue_key}]({jira_url}/browse/{task_ctx.issue_key})"
        )

        return "\n\n".join(sections)

    def _build_pr_prompt(
        self,
        task_ctx: TaskContext,
        code_change: CodeChange,
        branch_name: str,
        all_changes: list[FileChange],
        review: ReviewResult | None = None,
    ) -> str:
        """Build the prompt for the PR creator agent.

        Includes the FULL file contents so the Git MCP agent can actually
        commit the generated code — not just file names/sizes.
        """
        # Build per-file sections with full content
        file_sections: list[str] = []
        for fc in all_changes:
            content = fc.new_content or ""
            action = fc.change_type.value
            file_sections.append(
                f"### File: {fc.path} (action={action})\n"
                f"```\n{content}\n```"
            )
        files_detail = "\n\n".join(file_sections)

        pr_title = f"[AI-BOT] {task_ctx.issue_key}: {task_ctx.summary}"
        pr_body = self._build_pr_description(task_ctx, code_change, review)

        extra_instructions: list[str] = []
        if self._config.pr_draft_mode:
            extra_instructions.append("- Create PR as draft")
        if self._config.pr_reviewer:
            extra_instructions.append(
                f"- Assign {self._config.pr_reviewer} as reviewers"
            )

        extras = "\n".join(extra_instructions)
        if extras:
            extras = f"\n{extras}"

        return (
            f"Create a pull request for repository '{task_ctx.repository_name}':\n"
            f"1. Create branch '{branch_name}' from '{self._config.git_base_branch}'\n"
            f"2. Commit the following files with message '{code_change.commit_message}'.\n"
            f"   Write EXACTLY the content shown for each file:\n\n"
            f"{files_detail}\n\n"
            f"3. Open PR titled '{pr_title}' targeting '{self._config.git_base_branch}'\n"
            f"   Description:\n{pr_body}\n"
            f"{extras}"
            f"\nReturn the PR URL."
        )

    def _extract_pr_url(self, result: str) -> str | None:
        """Extract the first HTTPS URL from a result string."""
        match = re.search(r"https?://\S+", result)
        return match.group(0).rstrip(".,)") if match else None

    # ------------------------------------------------------------------
    # Task 11.4 — Reassignment: read previous AI comments as context
    # ------------------------------------------------------------------

    async def _get_previous_review_feedback(self, issue_key: str) -> str | None:
        """Read previous AI comments from Jira to use as context for reassignment."""
        if self._config.dry_run:
            return None
        try:
            jira_agent = Agent(
                name="jira_reader",
                instruction="Use jira_get_issue tool to read issue comments.",
                server_names=["atlassian"],
            )
            async with jira_agent:
                llm = await jira_agent.attach_llm(
                    self._llm_router.get_llm_class("fast")
                )
                result = await llm.generate_str(
                    f"Get the last 3 comments from Jira issue {issue_key} that were added by the AI developer bot. "
                    f"Return them as plain text summary."
                )
                return result if result.strip() else None
        except Exception:
            logger.warning(
                "Failed to get previous review feedback for %s",
                issue_key,
                exc_info=True,
            )
            return None
