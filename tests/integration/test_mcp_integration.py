"""MCP integration tests.

Tests MCP server configuration generation, provider switching,
invalid provider handling, and credential validation.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.7, 10.8
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config.mcp_servers import (
    ConfigurationError,
    MCPServerConfigBuilder,
    get_active_git_server_name,
)
from src.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    defaults = {
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-dev",
        "jira_api_token": "jira-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-dev",
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast",
        "llm_strong_provider": "openai",
        "llm_strong_model": "gpt-4o",
        "llm_strong_api_key": "sk-strong",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _bitbucket_settings(**overrides: Any) -> Settings:
    return _make_settings(
        git_provider="bitbucket",
        bitbucket_workspace="my-workspace",
        bitbucket_username="bb-user",
        bitbucket_app_password="bb-app-password",
        **overrides,
    )


def _github_settings(**overrides: Any) -> Settings:
    return _make_settings(
        git_provider="github",
        github_token="ghp_testtoken",
        github_owner="test-owner",
        **overrides,
    )


def _gitlab_settings(**overrides: Any) -> Settings:
    return _make_settings(
        git_provider="gitlab",
        gitlab_token="glpat-testtoken",
        gitlab_group="test-group",
        **overrides,
    )


# ---------------------------------------------------------------------------
# MCPServerConfigBuilder — config generation
# ---------------------------------------------------------------------------


class TestMCPConfigGeneration:
    """Req 7.1–7.6: Correct config generated for each git provider."""

    def test_bitbucket_config_structure(self) -> None:
        settings = _bitbucket_settings()
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)

        assert "mcpServers" in config
        servers = config["mcpServers"]
        assert "atlassian" in servers
        assert "bitbucket" in servers

        bb = servers["bitbucket"]
        assert bb["command"] == "mcp-bitbucket"
        assert bb["env"]["BITBUCKET_USERNAME"] == "bb-user"
        assert bb["env"]["BITBUCKET_WORKSPACE"] == "my-workspace"
        assert bb["env"]["BITBUCKET_APP_PASSWORD"] == "bb-app-password"

    def test_github_config_structure(self) -> None:
        settings = _github_settings()
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)

        servers = config["mcpServers"]
        assert "atlassian" in servers
        assert "github" in servers

        gh = servers["github"]
        assert gh["command"] == "npx"
        assert "@modelcontextprotocol/server-github" in gh["args"]
        assert gh["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_testtoken"

    def test_gitlab_config_structure(self) -> None:
        settings = _gitlab_settings()
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)

        servers = config["mcpServers"]
        assert "atlassian" in servers
        assert "gitlab" in servers

        gl = servers["gitlab"]
        assert gl["command"] == "npx"
        assert "@modelcontextprotocol/server-gitlab" in gl["args"]
        assert gl["env"]["GITLAB_PERSONAL_ACCESS_TOKEN"] == "glpat-testtoken"
        assert "api/v4" in gl["env"]["GITLAB_API_URL"]

    def test_atlassian_config_always_present(self) -> None:
        """Atlassian server is always included regardless of git provider."""
        for settings in [_bitbucket_settings(), _github_settings(), _gitlab_settings()]:
            builder = MCPServerConfigBuilder()
            config = builder.build(settings)
            atlassian = config["mcpServers"]["atlassian"]
            assert atlassian["command"] == "mcp-atlassian"
            assert "--jira-url" in atlassian["args"]
            assert atlassian["env"]["JIRA_USERNAME"] == "ai-dev"
            assert atlassian["env"]["JIRA_API_TOKEN"] == "jira-token"


# ---------------------------------------------------------------------------
# Provider Switching
# ---------------------------------------------------------------------------


class TestProviderSwitching:
    """Req 7.1–7.4: Switching provider changes the active git server."""

    def test_bitbucket_active_server_name(self) -> None:
        settings = _bitbucket_settings()
        assert get_active_git_server_name(settings) == "bitbucket"

    def test_github_active_server_name(self) -> None:
        settings = _github_settings()
        assert get_active_git_server_name(settings) == "github"

    def test_gitlab_active_server_name(self) -> None:
        settings = _gitlab_settings()
        assert get_active_git_server_name(settings) == "gitlab"

    def test_each_provider_produces_unique_server_key(self) -> None:
        """Each provider maps to a distinct server name."""
        names = {
            get_active_git_server_name(_bitbucket_settings()),
            get_active_git_server_name(_github_settings()),
            get_active_git_server_name(_gitlab_settings()),
        }
        assert len(names) == 3


# ---------------------------------------------------------------------------
# Invalid Provider
# ---------------------------------------------------------------------------


class TestInvalidProvider:
    """Req 7.5: Unsupported provider raises ConfigurationError."""

    def test_invalid_provider_raises_configuration_error(self) -> None:
        """Directly test get_active_git_server_name with a mock settings object."""
        from unittest.mock import MagicMock

        fake_settings = MagicMock()
        fake_settings.git_provider = "mercurial"

        with pytest.raises(ConfigurationError, match="Unsupported git_provider"):
            get_active_git_server_name(fake_settings)

    def test_builder_invalid_provider_raises_configuration_error(self) -> None:
        """MCPServerConfigBuilder._git_config raises ConfigurationError for unknown provider."""
        from unittest.mock import MagicMock

        fake_settings = MagicMock()
        fake_settings.git_provider = "svn"
        fake_settings.jira_url = "https://jira.example.com"
        fake_settings.jira_username = "ai-dev"
        fake_settings.jira_api_token = MagicMock()
        fake_settings.jira_api_token.get_secret_value.return_value = "token"

        builder = MCPServerConfigBuilder()
        with pytest.raises(ConfigurationError, match="Unsupported git_provider"):
            builder._git_config(fake_settings)


# ---------------------------------------------------------------------------
# Credential Validation
# ---------------------------------------------------------------------------


class TestCredentialValidation:
    """Req 10.7, 10.8: Missing credentials raise ValidationError at Settings creation."""

    def test_bitbucket_missing_workspace_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="bitbucket_workspace"):
            _make_settings(
                git_provider="bitbucket",
                bitbucket_username="bb-user",
                bitbucket_app_password="bb-pass",
                # bitbucket_workspace intentionally omitted
            )

    def test_bitbucket_missing_app_password_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="bitbucket_app_password"):
            _make_settings(
                git_provider="bitbucket",
                bitbucket_workspace="my-workspace",
                bitbucket_username="bb-user",
                # bitbucket_app_password intentionally omitted
            )

    def test_github_missing_token_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="github_token"):
            _make_settings(
                git_provider="github",
                github_owner="test-owner",
                # github_token intentionally omitted
            )

    def test_github_missing_owner_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="github_owner"):
            _make_settings(
                git_provider="github",
                github_token="ghp_testtoken",
                # github_owner intentionally omitted
            )

    def test_gitlab_missing_token_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="gitlab_token"):
            _make_settings(
                git_provider="gitlab",
                gitlab_group="test-group",
                # gitlab_token intentionally omitted
            )

    def test_gitlab_missing_group_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="gitlab_group"):
            _make_settings(
                git_provider="gitlab",
                gitlab_token="glpat-testtoken",
                # gitlab_group intentionally omitted
            )

    def test_valid_bitbucket_credentials_accepted(self) -> None:
        settings = _bitbucket_settings()
        assert settings.git_provider == "bitbucket"
        assert settings.bitbucket_workspace == "my-workspace"

    def test_valid_github_credentials_accepted(self) -> None:
        settings = _github_settings()
        assert settings.git_provider == "github"
        assert settings.github_owner == "test-owner"

    def test_valid_gitlab_credentials_accepted(self) -> None:
        settings = _gitlab_settings()
        assert settings.git_provider == "gitlab"
        assert settings.gitlab_group == "test-group"


# ---------------------------------------------------------------------------
# Atlassian MCP Mock — simulated Jira operations
# ---------------------------------------------------------------------------


class TestAtlassianMCPMock:
    """Req 10.8: Atlassian MCP mock simulates Jira operations correctly."""

    @pytest.mark.asyncio
    async def test_mock_atlassian_get_issue(self, mock_atlassian_mcp: Any) -> None:
        result = await mock_atlassian_mcp.call_tool(
            "jira_get_issue", issue_key="TEST-1"
        )
        assert result["key"] == "TEST-1"
        assert "fields" in result
        assert result["fields"]["summary"] == "Test task"

    @pytest.mark.asyncio
    async def test_mock_atlassian_add_comment(self, mock_atlassian_mcp: Any) -> None:
        result = await mock_atlassian_mcp.call_tool(
            "jira_add_comment", issue_key="TEST-1", body="Hello"
        )
        assert result["id"] == "12345"

    @pytest.mark.asyncio
    async def test_mock_atlassian_transition(self, mock_atlassian_mcp: Any) -> None:
        result = await mock_atlassian_mcp.call_tool(
            "jira_transition_issue", issue_key="TEST-1", transition_id="21"
        )
        assert result is None  # transition returns None (fire-and-forget)


# ---------------------------------------------------------------------------
# Git MCP Mock — simulated Git operations
# ---------------------------------------------------------------------------


class TestGitMCPMock:
    """Req 7.6: Git MCP mock simulates file tree, content, branch, commit, PR."""

    @pytest.mark.asyncio
    async def test_mock_git_get_file_tree(self, mock_git_mcp: Any) -> None:
        result = await mock_git_mcp.call_tool("get_file_tree")
        assert "src/" in result
        assert "tests/" in result

    @pytest.mark.asyncio
    async def test_mock_git_get_file_content(self, mock_git_mcp: Any) -> None:
        result = await mock_git_mcp.call_tool("get_file_content", path="src/main.py")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_mock_git_create_branch(self, mock_git_mcp: Any) -> None:
        result = await mock_git_mcp.call_tool(
            "create_branch", name="feature/TEST-1-ai"
        )
        assert result["name"] == "feature/TEST-1-ai"

    @pytest.mark.asyncio
    async def test_mock_git_create_pull_request(self, mock_git_mcp: Any) -> None:
        result = await mock_git_mcp.call_tool(
            "create_pull_request",
            title="Fix auth",
            description="Updated auth handler",
        )
        assert "url" in result
        assert result["url"].startswith("https://")
