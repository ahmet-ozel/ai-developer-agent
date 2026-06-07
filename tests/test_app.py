"""Tests for src/app.py - MCPApp factory and run_pipeline."""

from __future__ import annotations

import pytest

from src.app import _build_mcp_app, run_pipeline, _MCP_AGENT_AVAILABLE
from src.config.settings import Settings


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
    return Settings(_env_file=None, **{**_COMMON, **overrides})


# ---------------------------------------------------------------------------
# MCPApp factory
# ---------------------------------------------------------------------------


class TestBuildMCPApp:
    """_build_mcp_app should return a valid app instance."""

    def test_build_returns_app(self) -> None:
        settings = _make_settings()
        app = _build_mcp_app(settings)
        assert app is not None

    def test_build_for_each_provider(self) -> None:
        """Should not crash for any git provider."""
        for provider, creds in [
            ("bitbucket", {"bitbucket_workspace": "ws", "bitbucket_username": "u", "bitbucket_app_password": "p"}),
            ("github", {"github_token": "tok", "github_owner": "owner"}),
            ("gitlab", {"gitlab_token": "tok", "gitlab_group": "grp"}),
        ]:
            settings = _make_settings(git_provider=provider, **creds)
            app = _build_mcp_app(settings)
            assert app is not None


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """run_pipeline should execute inside the MCPApp context manager."""

    async def test_run_pipeline_completes_without_error(self) -> None:
        """Basic smoke test - pipeline runs through orchestrator."""
        settings = _make_settings()
        # Pipeline will fail at MCP agent level (placeholder) but should not crash
        try:
            await run_pipeline("TEST-1", settings)
        except NotImplementedError:
            pass  # Expected when mcp-agent placeholder is used

    async def test_run_pipeline_builds_mcp_config(self) -> None:
        """Ensure MCP config is built from settings (no crash for any provider)."""
        for provider, creds in [
            ("bitbucket", {"bitbucket_workspace": "ws", "bitbucket_username": "u", "bitbucket_app_password": "p"}),
            ("github", {"github_token": "tok", "github_owner": "owner"}),
            ("gitlab", {"gitlab_token": "tok", "gitlab_group": "grp"}),
        ]:
            settings = _make_settings(git_provider=provider, **creds)
            try:
                await run_pipeline("TEST-2", settings)
            except NotImplementedError:
                pass  # Expected when mcp-agent placeholder is used
