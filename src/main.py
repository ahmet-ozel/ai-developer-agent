"""FastAPI application entry point.

Creates the FastAPI app, loads Settings, wires up the webhook router,
and optionally starts a Jira polling background task.

Trigger modes (set via TRIGGER_MODE in .env):
- "webhook": Jira sends HTTP POST to /webhook/jira (needs ngrok or public URL)
- "polling": Agent polls Jira REST API every N seconds (no ngrok needed)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI

from src.app import run_pipeline
from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.webhook.server import WebhookServer
from src.webhook.task_lock import TaskLock

logger = logging.getLogger(__name__)

# Track already-processed issues in polling mode
_processed_issues: set[str] = set()


# ---------------------------------------------------------------------------
# Jira Polling
# ---------------------------------------------------------------------------


async def _poll_jira(settings: Settings, task_lock: TaskLock) -> None:
    """Background task: poll Jira for bot-assigned issues."""
    bot = settings.jira_bot_username
    interval = settings.poll_interval_seconds
    jira_url = settings.jira_url.rstrip("/")
    auth = (settings.jira_username, settings.jira_api_token.get_secret_value())

    logger.info("Polling mode active - checking Jira every %ds for tasks assigned to '%s'", interval, bot)

    while True:
        try:
            # Build JQL with project scope if configured
            project_filter = f'project = {settings.jira_project_key} AND ' if settings.jira_project_key else ''
            jql = f'{project_filter}assignee = "{bot}" AND status != Done ORDER BY updated DESC'
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{jira_url}/rest/api/3/search/jql",
                    auth=auth,
                    json={"jql": jql, "maxResults": 10, "fields": ["summary", "status", "assignee", "comment"]},
                )

            if resp.status_code == 200:
                issues = resp.json().get("issues", [])
                for issue in issues:
                    key = issue["key"]
                    if key in _processed_issues:
                        continue

                    # Check if AI bot already completed this task (has completion comment)
                    if _has_completion_comment(issue):
                        logger.info("Skipping %s - already has AI bot completion comment", key)
                        _processed_issues.add(key)
                        continue

                    acquired = await task_lock.acquire(key)
                    if not acquired:
                        continue

                    summary = issue["fields"]["summary"]
                    logger.info("Poll found new task: %s - %s", key, summary)
                    _processed_issues.add(key)

                    # Run pipeline in background so polling continues
                    asyncio.create_task(_run_and_release(key, settings, task_lock))
            else:
                logger.warning("Jira poll failed: HTTP %s - %s", resp.status_code, resp.text[:200])

        except Exception:
            logger.exception("Error during Jira poll cycle")

        await asyncio.sleep(interval)


def _has_completion_comment(issue: dict) -> bool:
    """Check if an issue already has an AI bot completion or PR comment.

    Looks for comments containing 'AI Developer completed' or 'PR:' patterns
    that indicate the pipeline already ran successfully for this issue.
    """
    try:
        comments = issue.get("fields", {}).get("comment", {}).get("comments", [])
        for comment in comments:
            body = ""
            # ADF format
            body_obj = comment.get("body", {})
            if isinstance(body_obj, dict):
                # Extract text from ADF content
                for block in body_obj.get("content", []):
                    for inline in block.get("content", []):
                        body += inline.get("text", "")
            elif isinstance(body_obj, str):
                body = body_obj

            if "AI Developer completed" in body or "[AI-BOT]" in body:
                return True
    except Exception:
        pass
    return False


async def _run_and_release(issue_key: str, settings: Settings, task_lock: TaskLock) -> None:
    """Run pipeline and release task lock afterwards."""
    try:
        await run_pipeline(issue_key, settings)
        logger.info("Pipeline completed for %s", issue_key)
    except Exception:
        logger.exception("Pipeline failed for %s", issue_key)
    finally:
        await task_lock.release(issue_key)

# ---------------------------------------------------------------------------
# Factory - builds the fully-wired FastAPI application
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        settings: Optional pre-built Settings instance. When *None* the
            settings are loaded from environment variables / ``.env`` file.
    """
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    # -- lifespan: credential validation + optional polling on startup -----

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Run credential validation at startup, start poller if configured."""
        await validate_credentials(settings)

        poll_task = None
        if settings.trigger_mode == "polling":
            poll_task = asyncio.create_task(_poll_jira(settings, task_lock))
            logger.info("Trigger mode: POLLING (interval=%ds)", settings.poll_interval_seconds)
        else:
            logger.info("Trigger mode: WEBHOOK (waiting for POST /webhook/jira)")

        yield

        if poll_task:
            poll_task.cancel()

    app = FastAPI(title="AI Developer Agent", version="0.1.0", lifespan=_lifespan)

    # -- shared instances ---------------------------------------------------
    task_lock = TaskLock()
    llm_router = LLMRouter(config=settings)

    # Store on app.state so tests / startup hooks can access them
    app.state.settings = settings
    app.state.task_lock = task_lock
    app.state.llm_router = llm_router

    # -- pipeline callback --------------------------------------------------

    async def _pipeline_callback(issue_key: str) -> None:
        """Thin wrapper that forwards to ``run_pipeline``."""
        await run_pipeline(issue_key, settings)  # type: ignore[arg-type]

    # -- webhook router -----------------------------------------------------
    webhook_server = WebhookServer(
        settings=settings,
        task_lock=task_lock,
        pipeline_callback=_pipeline_callback,
    )
    app.include_router(webhook_server.router)

    return app


# ---------------------------------------------------------------------------
# Credential validation (non-fatal - logs warnings, never crashes)
# ---------------------------------------------------------------------------


async def validate_credentials(settings: Settings) -> list[str]:
    """Check that configured credentials look plausible.

    Returns a list of warning messages (empty means all checks passed).
    This does NOT make network calls - it only validates that the expected
    fields are non-empty for the selected providers.
    """
    warnings: list[str] = []

    # Jira credentials
    if not settings.jira_url:
        warnings.append("JIRA_URL is empty")
    if not settings.jira_username:
        warnings.append("JIRA_USERNAME is empty")
    if not settings.jira_api_token.get_secret_value():
        warnings.append("JIRA_API_TOKEN is empty")

    # Git provider credentials (already validated by Settings model_validator,
    # but we double-check here for startup logging)
    provider = settings.git_provider
    if provider == "bitbucket":
        if not settings.bitbucket_workspace:
            warnings.append("BITBUCKET_WORKSPACE is empty")
    elif provider == "github":
        if not settings.github_token:
            warnings.append("GITHUB_TOKEN is empty")
    elif provider == "gitlab":
        if not settings.gitlab_token:
            warnings.append("GITLAB_TOKEN is empty")

    # LLM provider credentials
    if not settings.llm_fast_api_key.get_secret_value():
        warnings.append("LLM_FAST_API_KEY is empty")
    if not settings.llm_strong_api_key.get_secret_value():
        warnings.append("LLM_STRONG_API_KEY is empty")

    for w in warnings:
        logger.warning("Credential validation: %s", w)

    if not warnings:
        logger.info("All credential checks passed")

    return warnings
