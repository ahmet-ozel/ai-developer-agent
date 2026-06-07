"""MCPApp factory and pipeline entry point.

Builds an MCPApp per-pipeline-run with the correct MCP server config
derived from Settings. Ensures proper MCP server lifecycle management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.config.mcp_servers import MCPServerConfigBuilder
from src.pipeline.llm_router import LLMRouter
from src.pipeline.orchestrator import PipelineOrchestrator

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mcp-agent imports - graceful fallback when not installed
# ---------------------------------------------------------------------------

_MCP_AGENT_AVAILABLE = False

try:
    from mcp_agent.app import MCPApp
    from mcp_agent.config import (
        Settings as MCPSettings,
        MCPSettings as MCPServerBlock,
        MCPServerSettings,
        LoggerSettings,
    )
    _MCP_AGENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


class _PlaceholderApp:
    """Lightweight placeholder when mcp-agent is not installed."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def run(self) -> "_PlaceholderApp":
        return self

    async def __aenter__(self) -> "_PlaceholderApp":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Build MCPApp with programmatic config from Settings
# ---------------------------------------------------------------------------


def _build_mcp_app(settings: Settings) -> Any:
    """Create an MCPApp configured with the correct MCP servers.

    Translates our Settings → mcp-agent's MCPSettings programmatically
    so no YAML file is needed at runtime.
    """
    if not _MCP_AGENT_AVAILABLE:
        logger.warning(
            "mcp-agent not installed - using placeholder. "
            "Install with: pip install mcp-agent"
        )
        return _PlaceholderApp()

    # Build our internal MCP config dict
    config_builder = MCPServerConfigBuilder()
    mcp_config = config_builder.build(settings)
    servers_dict = mcp_config.get("mcpServers", {})

    # Convert to mcp-agent MCPServerSettings
    mcp_servers: dict[str, MCPServerSettings] = {}
    for name, srv in servers_dict.items():
        mcp_servers[name] = MCPServerSettings(
            command=srv.get("command", ""),
            args=srv.get("args", []),
            env=srv.get("env", {}),
        )

    # Build OpenAI config for mcp-agent
    openai_config: dict[str, Any] = {}
    if settings.llm_fast_provider == "openai" or settings.llm_strong_provider == "openai":
        # Use the fast tier key as default (mcp-agent reads OPENAI_API_KEY from env too)
        openai_config["default_model"] = settings.llm_fast_model
        openai_config["api_key"] = settings.llm_fast_api_key.get_secret_value()

    # Build mcp-agent Settings
    mcp_settings = MCPSettings(
        name="ai-developer-agent",
        logger=LoggerSettings(transports=["console"], level="info"),
        mcp=MCPServerBlock(servers=mcp_servers),
    )

    # Set OpenAI config if available
    if openai_config:
        try:
            from mcp_agent.config import OpenAISettings
            mcp_settings.openai = OpenAISettings(**openai_config)
        except (ImportError, Exception):
            pass

    return MCPApp(settings=mcp_settings)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


async def run_pipeline(issue_key: str, settings: Settings) -> None:
    """Run the full pipeline for *issue_key* inside the MCPApp context.

    Steps:
    1. Build MCPApp with programmatic config from Settings.
    2. Enter MCPApp async context (manages MCP server lifecycle).
    3. Execute the pipeline orchestrator.
    4. Guarantee cleanup via try/finally.
    """
    mcp_app = _build_mcp_app(settings)

    server_names = list(
        MCPServerConfigBuilder().build(settings).get("mcpServers", {}).keys()
    )
    logger.info(
        "Pipeline starting for issue %s (servers: %s, mcp-agent: %s)",
        issue_key,
        server_names,
        "available" if _MCP_AGENT_AVAILABLE else "placeholder",
    )

    try:
        async with mcp_app.run() as running_app:
            logger.info("MCPApp context entered for issue %s", issue_key)

            llm_router = LLMRouter(config=settings)
            orchestrator = PipelineOrchestrator(config=settings, llm_router=llm_router)
            result = await orchestrator.run(issue_key)

            if result.success:
                logger.info(
                    "Pipeline completed for issue %s (PR: %s)",
                    issue_key,
                    result.pr_url or "N/A",
                )
            else:
                logger.warning(
                    "Pipeline failed for issue %s: stage=%s reason=%s",
                    issue_key,
                    result.failure_stage,
                    result.failure_reason,
                )
    except Exception:
        logger.exception("Unhandled exception in pipeline for issue %s", issue_key)
        raise
    finally:
        logger.info("MCP server cleanup complete for issue %s", issue_key)
