"""Tests for src/main.py — FastAPI app creation and credential validation."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.config.settings import Settings
from src.main import create_app, validate_credentials


# ---------------------------------------------------------------------------
# Helpers
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
    "git_provider": "bitbucket",
    "bitbucket_workspace": "my-workspace",
    "bitbucket_username": "bb-user",
    "bitbucket_app_password": "bb-app-password",
}


def _make_settings(**overrides: object) -> Settings:
    return Settings(**{**_COMMON, **overrides})


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    """create_app should return a valid FastAPI application."""

    def test_returns_fastapi_instance(self) -> None:
        from fastapi import FastAPI

        settings = _make_settings()
        app = create_app(settings)
        assert isinstance(app, FastAPI)

    def test_app_has_webhook_route(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        routes = [r.path for r in app.routes]
        assert "/webhook/jira" in routes

    def test_app_has_health_route(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_app_state_has_settings(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        assert app.state.settings is settings

    def test_app_state_has_task_lock(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        from src.webhook.task_lock import TaskLock

        assert isinstance(app.state.task_lock, TaskLock)

    def test_app_state_has_llm_router(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        from src.pipeline.llm_router import LLMRouter

        assert isinstance(app.state.llm_router, LLMRouter)


# ---------------------------------------------------------------------------
# Health endpoint via create_app
# ---------------------------------------------------------------------------


class TestHealthViaCreateApp:
    """Health endpoint should work through the fully wired app."""

    async def test_health_returns_200(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# validate_credentials
# ---------------------------------------------------------------------------


class TestValidateCredentials:
    """validate_credentials should return warnings for missing/empty creds."""

    async def test_all_valid_returns_empty(self) -> None:
        settings = _make_settings()
        warnings = await validate_credentials(settings)
        assert warnings == []

    async def test_github_provider_valid(self) -> None:
        settings = _make_settings(
            git_provider="github",
            github_token="tok",
            github_owner="owner",
        )
        warnings = await validate_credentials(settings)
        assert warnings == []

    async def test_gitlab_provider_valid(self) -> None:
        settings = _make_settings(
            git_provider="gitlab",
            gitlab_token="tok",
            gitlab_group="grp",
        )
        warnings = await validate_credentials(settings)
        assert warnings == []
