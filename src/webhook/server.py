"""FastAPI webhook server for Jira event handling.

Provides a WebhookServer class that creates an APIRouter with:
- POST /webhook/jira - signature validation, event parsing, bot assignment check,
  task lock, and async pipeline triggering via BackgroundTask.
- GET /health - simple health check returning HTTP 200.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from src.config.settings import Settings
from src.webhook.models import WebhookEvent
from src.webhook.task_lock import TaskLock
from src.webhook.validators import WebhookValidator

logger = logging.getLogger(__name__)


class WebhookServer:
    """FastAPI router for Jira webhook handling.

    Args:
        settings: Application settings (provides jira_webhook_secret, jira_bot_username).
        task_lock: In-memory task lock for deduplication.
        pipeline_callback: Async callable that receives an issue_key and runs the pipeline.
    """

    def __init__(
        self,
        settings: Settings,
        task_lock: TaskLock,
        pipeline_callback: Callable[[str], Awaitable[Any]],
    ) -> None:
        self._settings = settings
        self._task_lock = task_lock
        self._pipeline_callback = pipeline_callback
        self._validator = WebhookValidator()
        self._router = APIRouter()
        self._register_routes()

    @property
    def router(self) -> APIRouter:
        """The FastAPI APIRouter with webhook and health endpoints."""
        return self._router

    def _register_routes(self) -> None:
        """Wire up the endpoint handlers to the router."""
        self._router.add_api_route(
            "/webhook/jira",
            self._handle_webhook,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/health",
            self._health_check,
            methods=["GET"],
        )

    async def _handle_webhook(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        """POST /webhook/jira - main webhook endpoint.

        Flow:
        1. Read raw body bytes
        2. Validate HMAC signature → 401 if invalid
        3. Parse JSON payload → 400 if malformed
        4. Parse into WebhookEvent → 400 if invalid
        5. Check bot assignment → 200 "ignored" if not
        6. Check event type → 200 "ignored" if not jira:issue_updated
        7. Acquire task lock → 200 "ignored" if already locked
        8. Enqueue pipeline via BackgroundTask
        9. Return 200 "pipeline enqueued"
        """
        # 1. Read raw body
        body = await request.body()

        # 2. Validate signature
        signature = request.headers.get("x-hub-signature", "")
        secret = self._settings.jira_webhook_secret.get_secret_value()
        if not self._validator.validate_signature(body, signature, secret):
            logger.warning("Rejected webhook: invalid signature")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid webhook signature"},
            )

        # 3. Parse JSON
        try:
            payload: dict[str, Any] = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Rejected webhook: malformed JSON payload - %s", exc)
            return JSONResponse(
                status_code=400,
                content={"detail": "Malformed JSON payload"},
            )

        # 4. Parse into WebhookEvent
        try:
            event: WebhookEvent = self._validator.parse_event(payload)
        except (ValueError, KeyError) as exc:
            logger.warning("Rejected webhook: invalid event payload - %s", exc)
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid event payload: {exc}"},
            )

        # 5. Check bot assignment
        bot_username = self._settings.jira_bot_username
        if not self._validator.is_bot_assignment(event, bot_username):
            logger.info(
                "Ignored webhook for %s: assignee is not bot", event.issue_key
            )
            return JSONResponse(
                status_code=200,
                content={"status": "ignored", "reason": "not bot assignment"},
            )

        # 6. Check event type
        if event.webhook_event != "jira:issue_updated":
            logger.info(
                "Ignored webhook for %s: event type %s",
                event.issue_key,
                event.webhook_event,
            )
            return JSONResponse(
                status_code=200,
                content={"status": "ignored", "reason": "unsupported event type"},
            )

        # 7. Acquire task lock
        acquired = await self._task_lock.acquire(event.issue_key)
        if not acquired:
            logger.info(
                "Ignored webhook for %s: task already locked", event.issue_key
            )
            return JSONResponse(
                status_code=200,
                content={"status": "ignored", "reason": "task already processing"},
            )

        # 8. Enqueue pipeline
        background_tasks.add_task(self._run_pipeline, event.issue_key)

        # 9. Return success
        logger.info("Pipeline enqueued for %s", event.issue_key)
        return JSONResponse(
            status_code=200,
            content={"status": "pipeline enqueued", "issue_key": event.issue_key},
        )

    async def _run_pipeline(self, issue_key: str) -> None:
        """Run the pipeline callback and release the task lock afterwards."""
        try:
            await self._pipeline_callback(issue_key)
        finally:
            await self._task_lock.release(issue_key)

    async def _health_check(self) -> JSONResponse:
        """GET /health - returns HTTP 200 with status ok."""
        return JSONResponse(status_code=200, content={"status": "ok"})
