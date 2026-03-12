"""E2E test for the webhook server — starts real FastAPI, sends real webhook payload.

Tests the full webhook flow: signature validation → event parsing → bot check → pipeline enqueue.
Does NOT require external services — uses the real app with DRY_RUN=true.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os

import httpx
import pytest

from tests.e2e.conftest import requires_jira


def _make_jira_payload(issue_key: str, assignee: str) -> dict:
    """Build a realistic Jira webhook payload."""
    return {
        "webhookEvent": "jira:issue_updated",
        "issue": {
            "key": issue_key,
            "fields": {
                "summary": "E2E test issue",
                "assignee": {
                    "name": assignee,
                    "displayName": assignee,
                    "accountId": "e2e-test-account",
                },
                "issuetype": {"name": "Task"},
                "status": {"name": "To Do"},
                "description": "E2E test description",
            },
        },
        "changelog": {
            "items": [
                {
                    "field": "assignee",
                    "fromString": None,
                    "toString": assignee,
                }
            ]
        },
    }


def _sign_payload(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature matching Jira webhook format."""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class TestWebhookServerLive:
    """Start the real FastAPI app and send webhook payloads."""

    @pytest.fixture
    def app(self, monkeypatch):
        """Create a real FastAPI app with DRY_RUN=true.

        Uses monkeypatch so env vars are automatically cleaned up after each test.
        """
        env_defaults = {
            "DRY_RUN": "true",
            "JIRA_URL": "https://test.atlassian.net",
            "JIRA_USERNAME": "test@test.com",
            "JIRA_API_TOKEN": "test-token",
            "JIRA_WEBHOOK_SECRET": "e2e-test-secret",
            "JIRA_BOT_USERNAME": "ai-developer-bot",
            "GIT_PROVIDER": "github",
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_OWNER": "test-owner",
            "LLM_FAST_PROVIDER": "openai",
            "LLM_FAST_MODEL": "gpt-4o-mini",
            "LLM_FAST_API_KEY": "sk-test",
            "LLM_STRONG_PROVIDER": "openai",
            "LLM_STRONG_MODEL": "gpt-4o",
            "LLM_STRONG_API_KEY": "sk-test",
        }
        for key, value in env_defaults.items():
            monkeypatch.setenv(key, os.environ.get(key, value))

        from src.main import create_app
        from src.config.settings import Settings

        settings = Settings()  # type: ignore[call-arg]
        return create_app(settings)

    @pytest.mark.asyncio
    async def test_health_endpoint(self, app) -> None:
        """GET /health returns 200."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_valid_webhook_enqueues_pipeline(self, app) -> None:
        """Valid webhook with correct signature and bot assignee → pipeline enqueued."""
        secret = os.environ.get("JIRA_WEBHOOK_SECRET", "e2e-test-secret")
        bot = os.environ.get("JIRA_BOT_USERNAME", "ai-developer-bot")

        payload = _make_jira_payload("TEST-1", bot)
        body = json.dumps(payload).encode()
        signature = _sign_payload(body, secret)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/webhook/jira",
                content=body,
                headers={
                    "content-type": "application/json",
                    "x-hub-signature": signature,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pipeline enqueued"
        assert data["issue_key"] == "TEST-1"

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self, app) -> None:
        """Invalid signature → 401."""
        payload = _make_jira_payload("TEST-2", "ai-developer-bot")
        body = json.dumps(payload).encode()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/webhook/jira",
                content=body,
                headers={
                    "content-type": "application/json",
                    "x-hub-signature": "sha256=invalid",
                },
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_bot_assignee_ignored(self, app) -> None:
        """Webhook for non-bot assignee → ignored."""
        secret = os.environ.get("JIRA_WEBHOOK_SECRET", "e2e-test-secret")

        payload = _make_jira_payload("TEST-3", "some-other-user")
        body = json.dumps(payload).encode()
        signature = _sign_payload(body, secret)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/webhook/jira",
                content=body,
                headers={
                    "content-type": "application/json",
                    "x-hub-signature": signature,
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"
