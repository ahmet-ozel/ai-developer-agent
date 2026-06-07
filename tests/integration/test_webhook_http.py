"""HTTP integration tests for the Webhook Server.

Tests the full FastAPI request/response cycle including signature validation,
event routing, bot assignment checks, task locking, and health endpoint.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.config.settings import Settings
from src.webhook.server import WebhookServer
from src.webhook.task_lock import TaskLock

# ---------------------------------------------------------------------------
# Helpers - reuse the Settings construction pattern from test_settings.py
# ---------------------------------------------------------------------------

_COMMON: dict = {
    "jira_url": "https://jira.example.com",
    "jira_username": "ai-developer",
    "jira_api_token": "jira-secret-token",
    "jira_webhook_secret": "webhook-secret",
    "jira_bot_username": "ai-developer",
    "llm_fast_provider": "openai",
    "llm_fast_model": "gpt-4o-mini",
    "llm_fast_api_key": "sk-fast-key",
    "llm_strong_provider": "anthropic",
    "llm_strong_model": "claude-sonnet-4-20250514",
    "llm_strong_api_key": "sk-strong-key",
}

_BITBUCKET_CREDS: dict = {
    "git_provider": "bitbucket",
    "bitbucket_workspace": "my-workspace",
    "bitbucket_username": "bb-user",
    "bitbucket_app_password": "bb-app-password",
}

WEBHOOK_SECRET = "webhook-secret"
BOT_USERNAME = "ai-developer"


def _compute_signature(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 hex digest matching WebhookValidator logic."""
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _make_jira_payload(
    *,
    webhook_event: str = "jira:issue_updated",
    issue_key: str = "PROJ-1",
    assignee_name: str | None = BOT_USERNAME,
) -> dict:
    """Build a minimal valid Jira webhook payload."""
    return {
        "webhookEvent": webhook_event,
        "issue": {
            "key": issue_key,
            "fields": {
                "assignee": {"name": assignee_name} if assignee_name else None,
                "issuetype": {"name": "Story"},
                "project": {"key": "PROJ"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_callback() -> AsyncMock:
    """Mock async pipeline callback."""
    return AsyncMock()


@pytest.fixture
def app_and_settings(pipeline_callback: AsyncMock) -> tuple[FastAPI, Settings, TaskLock]:
    """Create a FastAPI app with the WebhookServer router mounted."""
    settings = Settings(**{**_COMMON, **_BITBUCKET_CREDS})
    task_lock = TaskLock()
    server = WebhookServer(
        settings=settings,
        task_lock=task_lock,
        pipeline_callback=pipeline_callback,
    )
    app = FastAPI()
    app.include_router(server.router)
    return app, settings, task_lock


@pytest.fixture
async def client(app_and_settings: tuple[FastAPI, Settings, TaskLock]) -> AsyncClient:
    """httpx AsyncClient wired to the ASGI app."""
    app, _, _ = app_and_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# POST /webhook/jira - valid signature + bot assignee → 200 pipeline enqueued
# ---------------------------------------------------------------------------


class TestWebhookPipelineEnqueued:
    """Req 1.1, 1.3, 1.4, 1.10: Valid signature + bot assignee + issue_updated → pipeline enqueued."""

    async def test_valid_request_returns_200_pipeline_enqueued(
        self, client: AsyncClient, pipeline_callback: AsyncMock
    ):
        payload = _make_jira_payload()
        body = json.dumps(payload).encode()
        sig = _compute_signature(body, WEBHOOK_SECRET)

        resp = await client.post(
            "/webhook/jira",
            content=body,
            headers={"x-hub-signature": sig, "content-type": "application/json"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pipeline enqueued"
        assert data["issue_key"] == "PROJ-1"


# ---------------------------------------------------------------------------
# POST /webhook/jira - invalid signature → 401
# ---------------------------------------------------------------------------


class TestWebhookInvalidSignature:
    """Req 1.1, 1.2: Invalid HMAC signature → 401."""

    async def test_invalid_signature_returns_401(self, client: AsyncClient):
        payload = _make_jira_payload()
        body = json.dumps(payload).encode()

        resp = await client.post(
            "/webhook/jira",
            content=body,
            headers={"x-hub-signature": "bad-signature", "content-type": "application/json"},
        )

        assert resp.status_code == 401
        assert "Invalid webhook signature" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /webhook/jira - malformed JSON → 400
# ---------------------------------------------------------------------------


class TestWebhookMalformedPayload:
    """Req 1.6: Malformed JSON body → 400."""

    async def test_malformed_json_returns_400(self, client: AsyncClient):
        body = b"this is not json{{"
        sig = _compute_signature(body, WEBHOOK_SECRET)

        resp = await client.post(
            "/webhook/jira",
            content=body,
            headers={"x-hub-signature": sig, "content-type": "application/json"},
        )

        assert resp.status_code == 400
        assert "Malformed JSON payload" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /webhook/jira - non-bot assignee → 200 ignored
# ---------------------------------------------------------------------------


class TestWebhookNonBotAssignee:
    """Req 1.5: Valid signature but assignee is not the bot → 200 ignored."""

    async def test_non_bot_assignee_returns_200_ignored(
        self, client: AsyncClient, pipeline_callback: AsyncMock
    ):
        payload = _make_jira_payload(assignee_name="john.doe")
        body = json.dumps(payload).encode()
        sig = _compute_signature(body, WEBHOOK_SECRET)

        resp = await client.post(
            "/webhook/jira",
            content=body,
            headers={"x-hub-signature": sig, "content-type": "application/json"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"
        pipeline_callback.assert_not_called()


# ---------------------------------------------------------------------------
# POST /webhook/jira - duplicate task (lock held) → 200 ignored
# ---------------------------------------------------------------------------


class TestWebhookDuplicateTask:
    """Req 1.9: Task lock already held → 200 ignored."""

    async def test_duplicate_task_returns_200_ignored(
        self,
        client: AsyncClient,
        app_and_settings: tuple[FastAPI, Settings, TaskLock],
        pipeline_callback: AsyncMock,
    ):
        _, _, task_lock = app_and_settings
        # Pre-acquire the lock for PROJ-1
        await task_lock.acquire("PROJ-1")

        payload = _make_jira_payload(issue_key="PROJ-1")
        body = json.dumps(payload).encode()
        sig = _compute_signature(body, WEBHOOK_SECRET)

        resp = await client.post(
            "/webhook/jira",
            content=body,
            headers={"x-hub-signature": sig, "content-type": "application/json"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"
        assert "already processing" in resp.json()["reason"]
        pipeline_callback.assert_not_called()


# ---------------------------------------------------------------------------
# POST /webhook/jira - wrong event type → 200 ignored
# ---------------------------------------------------------------------------


class TestWebhookWrongEventType:
    """Req 1.7: Unsupported event type (e.g. jira:issue_created) → 200 ignored."""

    async def test_wrong_event_type_returns_200_ignored(
        self, client: AsyncClient, pipeline_callback: AsyncMock
    ):
        payload = _make_jira_payload(webhook_event="jira:issue_created")
        body = json.dumps(payload).encode()
        sig = _compute_signature(body, WEBHOOK_SECRET)

        resp = await client.post(
            "/webhook/jira",
            content=body,
            headers={"x-hub-signature": sig, "content-type": "application/json"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"
        assert "unsupported event type" in resp.json()["reason"]
        pipeline_callback.assert_not_called()


# ---------------------------------------------------------------------------
# GET /health → 200
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Req 1.8: Health check returns 200 with status ok."""

    async def test_health_returns_200(self, client: AsyncClient):
        resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
