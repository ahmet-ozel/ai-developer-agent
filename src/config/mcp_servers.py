"""MCP Server configuration builder.

Generates mcp-agent compatible server configurations based on the
selected git provider and Jira/Atlassian credentials from Settings.
"""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings


class ConfigurationError(Exception):
    """Raised when MCP server configuration cannot be built."""


class MCPServerConfigBuilder:
    """Builds MCP server configuration dicts from application Settings.

    The output format matches mcp-agent's expected ``mcpServers`` structure.
    """

    def build(self, settings: Settings) -> dict[str, Any]:
        """Return the full MCP server configuration dictionary.

        Always includes the Atlassian (Jira) server.
        For GitHub, also includes the GitHub MCP server.
        GitLab and Bitbucket use direct REST API clients instead of MCP.
        """
        servers: dict[str, Any] = {
            "atlassian": self._atlassian_config(settings),
        }
        # Only GitHub uses MCP server; GitLab/Bitbucket use direct REST API
        if settings.git_provider == "github":
            git_server_name = get_active_git_server_name(settings)
            servers[git_server_name] = self._git_config(settings)
        return {"mcpServers": servers}

    # ------------------------------------------------------------------
    # Atlassian (Jira) config - always present
    # ------------------------------------------------------------------

    def _atlassian_config(self, settings: Settings) -> dict[str, Any]:
        config: dict[str, Any] = {
            "command": "mcp-atlassian",
            "args": ["--jira-url", settings.jira_url],
            "env": {
                "JIRA_USERNAME": settings.jira_username,
                "JIRA_API_TOKEN": settings.jira_api_token.get_secret_value(),
            },
        }
        # Add Confluence credentials when enabled
        if settings.confluence_enabled and settings.confluence_url:
            config["args"].extend(["--confluence-url", settings.confluence_url])
            config["env"]["CONFLUENCE_USERNAME"] = settings.confluence_username
            if settings.confluence_api_token:
                config["env"]["CONFLUENCE_API_TOKEN"] = (
                    settings.confluence_api_token.get_secret_value()
                )
        return config

    # ------------------------------------------------------------------
    # Git provider config - dispatched by provider type
    # ------------------------------------------------------------------

    def _git_config(self, settings: Settings) -> dict[str, Any]:
        if settings.git_provider == "bitbucket":
            return self._bitbucket_config(settings)
        elif settings.git_provider == "github":
            return self._github_config(settings)
        elif settings.git_provider == "gitlab":
            return self._gitlab_config(settings)
        raise ConfigurationError(
            f"Unsupported git_provider: {settings.git_provider}. "
            f"Supported: bitbucket, github, gitlab"
        )

    def _bitbucket_config(self, settings: Settings) -> dict[str, Any]:
        # Prefer new API token; fall back to legacy app password
        token = (
            settings.bitbucket_api_token.get_secret_value()
            if settings.bitbucket_api_token
            else (settings.bitbucket_app_password.get_secret_value() if settings.bitbucket_app_password else "")
        )
        return {
            "command": "mcp-bitbucket",
            "args": [],
            "env": {
                "BITBUCKET_USERNAME": settings.bitbucket_username,
                "BITBUCKET_APP_PASSWORD": token,  # mcp-bitbucket still uses this env var name
                "BITBUCKET_WORKSPACE": settings.bitbucket_workspace,
            },
        }

    def _github_config(self, settings: Settings) -> dict[str, Any]:
        return {
            "command": "node",
            "args": ["/usr/local/lib/node_modules/@modelcontextprotocol/server-github/dist/index.js"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token.get_secret_value(),
            },
        }

    def _gitlab_config(self, settings: Settings) -> dict[str, Any]:
        return {
            "command": "node",
            "args": ["/usr/local/lib/node_modules/@modelcontextprotocol/server-gitlab/dist/index.js"],
            "env": {
                "GITLAB_PERSONAL_ACCESS_TOKEN": settings.gitlab_token.get_secret_value(),
                "GITLAB_API_URL": f"{settings.gitlab_url}/api/v4",
            },
        }


def get_active_git_server_name(settings: Settings) -> str:
    """Return the MCP server name for the active git provider.

    Used by agents to populate their ``server_names`` list.
    """
    mapping = {
        "bitbucket": "bitbucket",
        "github": "github",
        "gitlab": "gitlab",
    }
    provider = settings.git_provider
    if provider not in mapping:
        raise ConfigurationError(
            f"Unsupported git_provider: {provider}. "
            f"Supported: bitbucket, github, gitlab"
        )
    return mapping[provider]
