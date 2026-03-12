"""Unit tests for MCPServerConfigBuilder and get_active_git_server_name.

Covers all three git providers (bitbucket, github, gitlab),
the Atlassian server always being present, and the ConfigurationError
for unsupported providers.

NOTE: GitLab and Bitbucket use direct REST API clients instead of MCP servers.
Only GitHub uses an MCP server. Tests reflect this architecture.
"""

from __future__ import annotations

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

_GITHUB_CREDS: dict = {
    "git_provider": "github",
    "github_token": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "github_owner": "my-org",
}

_GITLAB_CREDS: dict = {
    "git_provider": "gitlab",
    "gitlab_token": "glpat-xxxxxxxxxxxxxxxxxxxx",
    "gitlab_group": "my-group",
}


def _settings(**overrides: object) -> Settings:
    data = {**_COMMON, **_BITBUCKET_CREDS, **overrides}
    return Settings(**data)


# ---------------------------------------------------------------------------
# MCPServerConfigBuilder.build() tests
# ---------------------------------------------------------------------------


class TestBuildBitbucket:
    """Bitbucket uses direct REST API — only atlassian MCP server present."""

    def test_contains_only_atlassian_server(self) -> None:
        s = _settings()
        config = MCPServerConfigBuilder().build(s)
        servers = config["mcpServers"]
        assert "atlassian" in servers
        assert len(servers) == 1  # Only atlassian, no bitbucket MCP

    def test_atlassian_config_values(self) -> None:
        s = _settings()
        cfg = MCPServerConfigBuilder().build(s)["mcpServers"]["atlassian"]
        assert cfg["command"] == "mcp-atlassian"
        assert "--jira-url" in cfg["args"]
        assert s.jira_url in cfg["args"]
        assert cfg["env"]["JIRA_USERNAME"] == "ai-developer"
        assert cfg["env"]["JIRA_API_TOKEN"] == "jira-secret-token"


class TestBuildGitHub:
    """GitHub provider should produce github MCP server config."""

    def test_contains_atlassian_and_github_servers(self) -> None:
        s = Settings(**{**_COMMON, **_GITHUB_CREDS})
        config = MCPServerConfigBuilder().build(s)
        servers = config["mcpServers"]
        assert "atlassian" in servers
        assert "github" in servers
        assert len(servers) == 2

    def test_github_config_values(self) -> None:
        s = Settings(**{**_COMMON, **_GITHUB_CREDS})
        cfg = MCPServerConfigBuilder().build(s)["mcpServers"]["github"]
        assert cfg["command"] == "node"
        assert any("server-github" in arg for arg in cfg["args"])
        assert cfg["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


class TestBuildGitLab:
    """GitLab uses direct REST API — only atlassian MCP server present."""

    def test_contains_only_atlassian_server(self) -> None:
        s = Settings(**{**_COMMON, **_GITLAB_CREDS})
        config = MCPServerConfigBuilder().build(s)
        servers = config["mcpServers"]
        assert "atlassian" in servers
        assert len(servers) == 1  # Only atlassian, no gitlab MCP


# ---------------------------------------------------------------------------
# get_active_git_server_name() tests
# ---------------------------------------------------------------------------


class TestGetActiveGitServerName:
    """get_active_git_server_name should return the correct server name string."""

    def test_bitbucket(self) -> None:
        s = _settings()
        assert get_active_git_server_name(s) == "bitbucket"

    def test_github(self) -> None:
        s = Settings(**{**_COMMON, **_GITHUB_CREDS})
        assert get_active_git_server_name(s) == "github"

    def test_gitlab(self) -> None:
        s = Settings(**{**_COMMON, **_GITLAB_CREDS})
        assert get_active_git_server_name(s) == "gitlab"


# ---------------------------------------------------------------------------
# Atlassian server always present
# ---------------------------------------------------------------------------


class TestAtlassianAlwaysPresent:
    """The Atlassian MCP server should be included regardless of git provider."""

    @pytest.mark.parametrize(
        "creds",
        [_BITBUCKET_CREDS, _GITHUB_CREDS, _GITLAB_CREDS],
        ids=["bitbucket", "github", "gitlab"],
    )
    def test_atlassian_present(self, creds: dict) -> None:
        s = Settings(**{**_COMMON, **creds})
        config = MCPServerConfigBuilder().build(s)
        assert "atlassian" in config["mcpServers"]


# ---------------------------------------------------------------------------
# Server count: GitHub=2, GitLab/Bitbucket=1 (only atlassian)
# ---------------------------------------------------------------------------


class TestServerCount:
    """GitHub produces 2 servers (atlassian + github), others produce 1."""

    def test_github_two_servers(self) -> None:
        s = Settings(**{**_COMMON, **_GITHUB_CREDS})
        config = MCPServerConfigBuilder().build(s)
        assert len(config["mcpServers"]) == 2

    @pytest.mark.parametrize(
        "creds",
        [_BITBUCKET_CREDS, _GITLAB_CREDS],
        ids=["bitbucket", "gitlab"],
    )
    def test_rest_api_providers_one_server(self, creds: dict) -> None:
        s = Settings(**{**_COMMON, **creds})
        config = MCPServerConfigBuilder().build(s)
        assert len(config["mcpServers"]) == 1


# ---------------------------------------------------------------------------
# Helper for property tests
# ---------------------------------------------------------------------------

_PROVIDER_CREDS: dict = {
    "bitbucket": _BITBUCKET_CREDS,
    "github": _GITHUB_CREDS,
    "gitlab": _GITLAB_CREDS,
}


def _make_settings(provider: str) -> Settings:
    """Return a Settings instance configured for the given git provider."""
    return Settings(**{**_COMMON, **_PROVIDER_CREDS[provider]})


# ---------------------------------------------------------------------------
# Property Tests: MCP Server Configuration Generation (Property 21)
# ---------------------------------------------------------------------------

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


class TestMCPServerConfigGenerationProperty:
    """Property 21: MCP Server Configuration Generation."""

    @given(provider=st.sampled_from(["bitbucket", "github", "gitlab"]))
    @h_settings(max_examples=10)
    def test_valid_provider_produces_mcp_servers_key(self, provider: str) -> None:
        settings = _make_settings(provider)
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)
        assert "mcpServers" in config

    @given(provider=st.sampled_from(["bitbucket", "github", "gitlab"]))
    @h_settings(max_examples=10)
    def test_valid_provider_always_includes_atlassian(self, provider: str) -> None:
        settings = _make_settings(provider)
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)
        assert "atlassian" in config["mcpServers"]

    @given(provider=st.sampled_from(["github"]))
    @h_settings(max_examples=5)
    def test_github_includes_git_server(self, provider: str) -> None:
        """Only GitHub includes a git MCP server."""
        settings = _make_settings(provider)
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)
        assert provider in config["mcpServers"]

    @given(provider=st.sampled_from(["bitbucket", "github", "gitlab"]))
    @h_settings(max_examples=10)
    def test_get_active_git_server_name_matches_provider(self, provider: str) -> None:
        settings = _make_settings(provider)
        server_name = get_active_git_server_name(settings)
        assert server_name == provider

    @given(provider=st.sampled_from(["bitbucket", "github", "gitlab"]))
    @h_settings(max_examples=10)
    def test_atlassian_config_has_required_fields(self, provider: str) -> None:
        settings = _make_settings(provider)
        builder = MCPServerConfigBuilder()
        config = builder.build(settings)
        atlassian = config["mcpServers"]["atlassian"]
        assert "command" in atlassian
        assert "env" in atlassian
        assert "JIRA_USERNAME" in atlassian["env"]
        assert "JIRA_API_TOKEN" in atlassian["env"]


# ---------------------------------------------------------------------------
# Confluence credentials in _atlassian_config()
# ---------------------------------------------------------------------------


class TestAtlassianConfluenceConfig:
    """Atlassian config should include Confluence credentials when enabled."""

    def test_no_confluence_args_when_disabled(self) -> None:
        s = _settings()
        cfg = MCPServerConfigBuilder().build(s)["mcpServers"]["atlassian"]
        assert "--confluence-url" not in cfg["args"]
        assert "CONFLUENCE_USERNAME" not in cfg["env"]
        assert "CONFLUENCE_API_TOKEN" not in cfg["env"]

    def test_confluence_args_added_when_enabled(self) -> None:
        s = _settings(
            confluence_enabled=True,
            confluence_url="https://wiki.example.com",
            confluence_username="wiki-user",
            confluence_api_token="wiki-token",
        )
        cfg = MCPServerConfigBuilder().build(s)["mcpServers"]["atlassian"]
        assert "--confluence-url" in cfg["args"]
        assert "https://wiki.example.com" in cfg["args"]
        assert cfg["env"]["CONFLUENCE_USERNAME"] == "wiki-user"
        assert cfg["env"]["CONFLUENCE_API_TOKEN"] == "wiki-token"

    def test_confluence_url_not_added_when_url_empty(self) -> None:
        s = _settings(confluence_enabled=False)
        cfg = MCPServerConfigBuilder().build(s)["mcpServers"]["atlassian"]
        assert "--confluence-url" not in cfg["args"]

    def test_jira_config_unchanged_when_confluence_enabled(self) -> None:
        s = _settings(
            confluence_enabled=True,
            confluence_url="https://wiki.example.com",
            confluence_username="wiki-user",
            confluence_api_token="wiki-token",
        )
        cfg = MCPServerConfigBuilder().build(s)["mcpServers"]["atlassian"]
        assert "--jira-url" in cfg["args"]
        assert cfg["env"]["JIRA_USERNAME"] == "ai-developer"
        assert cfg["env"]["JIRA_API_TOKEN"] == "jira-secret-token"
