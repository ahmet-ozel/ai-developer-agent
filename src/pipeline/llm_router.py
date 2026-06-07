"""LLM Router - provider selection, tier routing, and fallback chain.

Routes LLM requests to the correct mcp-agent AugmentedLLM subclass based on
the configured tier (fast/strong) and provider (openai, anthropic, google, vllm).
Provides fallback chain support when the primary provider fails.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from src.config.settings import Settings

# ---------------------------------------------------------------------------
# Try importing mcp-agent AugmentedLLM classes.  They may not be installed
# in every environment (e.g. tests), so we create lightweight placeholders
# when the real classes are unavailable.
# ---------------------------------------------------------------------------

try:
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
except ImportError:  # pragma: no cover

    class OpenAIAugmentedLLM:  # type: ignore[no-redef]
        """Placeholder for OpenAIAugmentedLLM when mcp-agent is not installed."""


try:
    from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
except ImportError:  # pragma: no cover

    class AnthropicAugmentedLLM:  # type: ignore[no-redef]
        """Placeholder for AnthropicAugmentedLLM when mcp-agent is not installed."""


try:
    from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
except ImportError:  # pragma: no cover

    class GoogleAugmentedLLM:  # type: ignore[no-redef]
        """Placeholder for GoogleAugmentedLLM when mcp-agent is not installed."""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AllProvidersFailedError(Exception):
    """Raised when every provider in the fallback chain has failed."""

    def __init__(self, tier: str, attempted_providers: list[str], errors: list[str] | None = None):
        self.tier = tier
        self.attempted_providers = attempted_providers
        self.errors = errors or []
        providers_str = ", ".join(attempted_providers)
        super().__init__(
            f"All providers failed for tier '{tier}'. "
            f"Attempted: [{providers_str}]"
        )


# ---------------------------------------------------------------------------
# Provider  AugmentedLLM class mapping
# ---------------------------------------------------------------------------

_PROVIDER_LLM_MAP: dict[str, type] = {
    "openai": OpenAIAugmentedLLM,
    "vllm": OpenAIAugmentedLLM,  # vLLM uses OpenAI-compatible API
    "anthropic": AnthropicAugmentedLLM,
    "google": GoogleAugmentedLLM,
}


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------


class LLMRouter:
    """LLM provider selection, tier routing, and fallback chain.

    Returns mcp-agent AugmentedLLM *classes* - it does not make LLM calls
    itself.  Agents use the returned class with ``agent.attach_llm(cls)``.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config

    # -- public API --------------------------------------------------------

    def get_llm_class(self, tier: Literal["fast", "strong"]) -> type:
        """Return the AugmentedLLM subclass for *tier*'s configured provider."""
        provider = self._get_provider_for_tier(tier)
        return self._provider_to_llm_class(provider)

    def get_model_override(self, tier: Literal["fast", "strong"]) -> dict[str, Any]:
        """Return model name and optional base_url / api_key overrides.

        Agents pass these overrides to ``attach_llm`` so the correct model
        and endpoint are used.
        """
        if tier == "fast":
            result: dict[str, Any] = {"model": self._config.llm_fast_model}
            if self._config.llm_fast_endpoint:
                result["base_url"] = self._config.llm_fast_endpoint
            if self._config.llm_fast_api_key:
                result["api_key"] = self._config.llm_fast_api_key.get_secret_value()
            return result

        # strong tier
        result = {"model": self._config.llm_strong_model}
        if self._config.llm_strong_endpoint:
            result["base_url"] = self._config.llm_strong_endpoint
        if self._config.llm_strong_api_key:
            result["api_key"] = self._config.llm_strong_api_key.get_secret_value()
        return result

    async def call_with_fallback(
        self,
        tier: Literal["fast", "strong"],
        agent: Any,
        prompt: str,
    ) -> str:
        """Try the primary provider, then each fallback. Raise on total failure."""
        chain = self._get_chain_for_tier(tier)
        errors: list[str] = []

        for provider in chain:
            try:
                llm_class = self._provider_to_llm_class(provider)
                llm = await agent.attach_llm(llm_class)
                return await llm.generate_str(prompt)
            except Exception as exc:
                error_msg = f"{provider}: {exc}"
                errors.append(error_msg)
                logger.warning("LLM fallback - %s failed: %s", provider, exc)
                continue

        raise AllProvidersFailedError(
            tier=tier,
            attempted_providers=chain,
            errors=errors,
        )

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _provider_to_llm_class(provider: str) -> type:
        """Map a provider string to its AugmentedLLM class."""
        try:
            return _PROVIDER_LLM_MAP[provider]
        except KeyError:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: {', '.join(sorted(_PROVIDER_LLM_MAP))}"
            ) from None

    def _get_provider_for_tier(self, tier: str) -> str:
        """Return the primary provider name for the given tier."""
        if tier == "fast":
            return self._config.llm_fast_provider
        if tier == "strong":
            return self._config.llm_strong_provider
        raise ValueError(f"Unknown tier: '{tier}'. Supported: fast, strong")

    def _get_chain_for_tier(self, tier: str) -> list[str]:
        """Return the ordered fallback chain for *tier*.

        The chain starts with the primary provider, followed by the
        configured ``llm_fallback_chain`` entries.
        """
        primary = self._get_provider_for_tier(tier)
        chain = [primary]
        for provider in self._config.llm_fallback_chain:
            if provider not in chain:
                chain.append(provider)
        return chain
