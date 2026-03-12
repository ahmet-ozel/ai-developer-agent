"""Unit tests for LLMRouter — provider routing, model overrides, and fallback chain.

Tests mock the mcp-agent AugmentedLLM imports so the routing logic can be
verified without a real mcp-agent installation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.pipeline.llm_router import (
    AllProvidersFailedError,
    AnthropicAugmentedLLM,
    GoogleAugmentedLLM,
    LLMRouter,
    OpenAIAugmentedLLM,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON: dict = {
    "jira_url": "https://jira.example.com",
    "jira_username": "ai-developer",
    "jira_api_token": "jira-secret-token",
    "jira_webhook_secret": "webhook-secret",
    "jira_bot_username": "ai-developer",
    "git_provider": "bitbucket",
    "bitbucket_workspace": "ws",
    "bitbucket_username": "user",
    "bitbucket_app_password": "pass",
}


def _settings(**overrides: object) -> Settings:
    defaults = {
        **_COMMON,
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast",
        "llm_strong_provider": "anthropic",
        "llm_strong_model": "claude-sonnet-4-20250514",
        "llm_strong_api_key": "sk-strong",
        "llm_fallback_chain": [],
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# get_llm_class
# ---------------------------------------------------------------------------


class TestGetLLMClass:
    """get_llm_class should return the correct AugmentedLLM subclass."""

    def test_fast_tier_openai(self) -> None:
        router = LLMRouter(_settings(llm_fast_provider="openai"))
        assert router.get_llm_class("fast") is OpenAIAugmentedLLM

    def test_fast_tier_anthropic(self) -> None:
        router = LLMRouter(_settings(llm_fast_provider="anthropic"))
        assert router.get_llm_class("fast") is AnthropicAugmentedLLM

    def test_strong_tier_anthropic(self) -> None:
        router = LLMRouter(_settings(llm_strong_provider="anthropic"))
        assert router.get_llm_class("strong") is AnthropicAugmentedLLM

    def test_strong_tier_openai(self) -> None:
        router = LLMRouter(_settings(llm_strong_provider="openai"))
        assert router.get_llm_class("strong") is OpenAIAugmentedLLM

    def test_google_provider(self) -> None:
        router = LLMRouter(_settings(llm_fast_provider="google"))
        assert router.get_llm_class("fast") is GoogleAugmentedLLM

    def test_vllm_maps_to_openai(self) -> None:
        router = LLMRouter(_settings(llm_fast_provider="vllm"))
        assert router.get_llm_class("fast") is OpenAIAugmentedLLM

    def test_unknown_provider_raises(self) -> None:
        router = LLMRouter(_settings(llm_fast_provider="unknown-llm"))
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            router.get_llm_class("fast")


# ---------------------------------------------------------------------------
# get_model_override
# ---------------------------------------------------------------------------


class TestGetModelOverride:
    """get_model_override should return model name and optional endpoint/key."""

    def test_fast_tier_basic(self) -> None:
        router = LLMRouter(_settings())
        result = router.get_model_override("fast")
        assert result["model"] == "gpt-4o-mini"
        assert "api_key" in result
        assert result["api_key"] == "sk-fast"

    def test_strong_tier_basic(self) -> None:
        router = LLMRouter(_settings())
        result = router.get_model_override("strong")
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["api_key"] == "sk-strong"

    def test_fast_tier_with_endpoint(self) -> None:
        router = LLMRouter(
            _settings(llm_fast_endpoint="http://localhost:8000/v1")
        )
        result = router.get_model_override("fast")
        assert result["base_url"] == "http://localhost:8000/v1"

    def test_strong_tier_with_endpoint(self) -> None:
        router = LLMRouter(
            _settings(llm_strong_endpoint="http://localhost:9000/v1")
        )
        result = router.get_model_override("strong")
        assert result["base_url"] == "http://localhost:9000/v1"

    def test_no_endpoint_means_no_base_url_key(self) -> None:
        router = LLMRouter(_settings())
        result = router.get_model_override("fast")
        assert "base_url" not in result


# ---------------------------------------------------------------------------
# _provider_to_llm_class
# ---------------------------------------------------------------------------


class TestProviderToLLMClass:
    """Static mapping helper should resolve known providers."""

    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("openai", OpenAIAugmentedLLM),
            ("anthropic", AnthropicAugmentedLLM),
            ("google", GoogleAugmentedLLM),
            ("vllm", OpenAIAugmentedLLM),
        ],
    )
    def test_known_providers(self, provider: str, expected: type) -> None:
        assert LLMRouter._provider_to_llm_class(provider) is expected

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMRouter._provider_to_llm_class("deepseek")


# ---------------------------------------------------------------------------
# call_with_fallback
# ---------------------------------------------------------------------------


class TestCallWithFallback:
    """call_with_fallback should try providers in order and raise on total failure."""

    @pytest.mark.asyncio
    async def test_primary_succeeds(self) -> None:
        router = LLMRouter(_settings())
        mock_agent = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.generate_str = AsyncMock(return_value="hello")
        mock_agent.attach_llm = AsyncMock(return_value=mock_llm)

        result = await router.call_with_fallback("fast", mock_agent, "say hi")
        assert result == "hello"
        # attach_llm called once (primary succeeded)
        assert mock_agent.attach_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self) -> None:
        router = LLMRouter(
            _settings(
                llm_fast_provider="openai",
                llm_fallback_chain=["anthropic"],
            )
        )
        mock_agent = AsyncMock()

        # First call (openai) fails, second (anthropic) succeeds
        mock_llm_fail = AsyncMock()
        mock_llm_fail.generate_str = AsyncMock(side_effect=ConnectionError("timeout"))
        mock_llm_ok = AsyncMock()
        mock_llm_ok.generate_str = AsyncMock(return_value="fallback result")

        mock_agent.attach_llm = AsyncMock(side_effect=[mock_llm_fail, mock_llm_ok])

        result = await router.call_with_fallback("fast", mock_agent, "prompt")
        assert result == "fallback result"
        assert mock_agent.attach_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self) -> None:
        router = LLMRouter(
            _settings(
                llm_fast_provider="openai",
                llm_fallback_chain=["anthropic"],
            )
        )
        mock_agent = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.generate_str = AsyncMock(side_effect=ConnectionError("down"))
        mock_agent.attach_llm = AsyncMock(return_value=mock_llm)

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await router.call_with_fallback("fast", mock_agent, "prompt")

        err = exc_info.value
        assert err.tier == "fast"
        assert "openai" in err.attempted_providers
        assert "anthropic" in err.attempted_providers
        assert len(err.errors) == 2

    @pytest.mark.asyncio
    async def test_no_fallback_chain_single_attempt(self) -> None:
        router = LLMRouter(_settings(llm_fallback_chain=[]))
        mock_agent = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.generate_str = AsyncMock(side_effect=RuntimeError("fail"))
        mock_agent.attach_llm = AsyncMock(return_value=mock_llm)

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await router.call_with_fallback("fast", mock_agent, "prompt")

        assert len(exc_info.value.attempted_providers) == 1

    @pytest.mark.asyncio
    async def test_duplicate_provider_in_chain_deduped(self) -> None:
        """If the primary is also in fallback_chain, it should only appear once."""
        router = LLMRouter(
            _settings(
                llm_fast_provider="openai",
                llm_fallback_chain=["openai", "anthropic"],
            )
        )
        chain = router._get_chain_for_tier("fast")
        assert chain == ["openai", "anthropic"]


# ---------------------------------------------------------------------------
# AllProvidersFailedError
# ---------------------------------------------------------------------------


class TestAllProvidersFailedError:
    """Custom exception should carry tier and provider info."""

    def test_attributes(self) -> None:
        err = AllProvidersFailedError(
            tier="strong",
            attempted_providers=["openai", "anthropic"],
            errors=["openai: timeout", "anthropic: 500"],
        )
        assert err.tier == "strong"
        assert err.attempted_providers == ["openai", "anthropic"]
        assert len(err.errors) == 2
        assert "strong" in str(err)

    def test_default_errors_empty(self) -> None:
        err = AllProvidersFailedError(tier="fast", attempted_providers=["openai"])
        assert err.errors == []


# =========================================================================
# Property Tests (Hypothesis)
# =========================================================================

import asyncio

import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


_providers = st.sampled_from(["openai", "anthropic", "google", "vllm"])
_tiers = st.sampled_from(["fast", "strong"])

_PROVIDER_TO_CLASS = {
    "openai": OpenAIAugmentedLLM,
    "anthropic": AnthropicAugmentedLLM,
    "google": GoogleAugmentedLLM,
    "vllm": OpenAIAugmentedLLM,  # vllm maps to OpenAI-compatible
}


class TestLLMTierRoutingProperty:
    """Property 22: LLM Tier Routing.

    Validates: Requirements 8.2, 8.3, 8.4, 8.8
    """

    @given(provider=_providers)
    @h_settings(max_examples=20)
    def test_fast_tier_returns_correct_class(self, provider: str) -> None:
        """Fast tier always returns the correct AugmentedLLM class for the provider."""
        router = LLMRouter(_settings(llm_fast_provider=provider))
        result = router.get_llm_class("fast")
        assert result is _PROVIDER_TO_CLASS[provider]

    @given(provider=_providers)
    @h_settings(max_examples=20)
    def test_strong_tier_returns_correct_class(self, provider: str) -> None:
        """Strong tier always returns the correct AugmentedLLM class for the provider."""
        router = LLMRouter(_settings(llm_strong_provider=provider))
        result = router.get_llm_class("strong")
        assert result is _PROVIDER_TO_CLASS[provider]

    @given(provider=_providers)
    @h_settings(max_examples=20)
    def test_provider_to_llm_class_consistent(self, provider: str) -> None:
        """_provider_to_llm_class always returns the same class for the same provider."""
        result1 = LLMRouter._provider_to_llm_class(provider)
        result2 = LLMRouter._provider_to_llm_class(provider)
        assert result1 is result2

    @given(
        fast_provider=_providers,
        strong_provider=_providers,
    )
    @h_settings(max_examples=20)
    def test_tier_routing_is_independent(
        self, fast_provider: str, strong_provider: str
    ) -> None:
        """Fast and strong tier routing are independent of each other."""
        router = LLMRouter(_settings(
            llm_fast_provider=fast_provider,
            llm_strong_provider=strong_provider,
        ))
        fast_class = router.get_llm_class("fast")
        strong_class = router.get_llm_class("strong")
        assert fast_class is _PROVIDER_TO_CLASS[fast_provider]
        assert strong_class is _PROVIDER_TO_CLASS[strong_provider]


class TestLLMFallbackChainProperty:
    """Property 23: LLM Fallback Chain.

    Validates: Requirements 8.5, 8.6, 11.6
    """

    @given(
        primary=_providers,
        fallback_chain=st.lists(_providers, min_size=0, max_size=3),
    )
    @h_settings(max_examples=50)
    def test_chain_starts_with_primary(
        self, primary: str, fallback_chain: list
    ) -> None:
        """The fallback chain always starts with the primary provider."""
        router = LLMRouter(_settings(
            llm_fast_provider=primary,
            llm_fallback_chain=fallback_chain,
        ))
        chain = router._get_chain_for_tier("fast")
        assert chain[0] == primary

    @given(
        primary=_providers,
        fallback_chain=st.lists(_providers, min_size=1, max_size=3),
    )
    @h_settings(max_examples=50)
    def test_chain_has_no_duplicates(
        self, primary: str, fallback_chain: list
    ) -> None:
        """The fallback chain never contains duplicate providers."""
        router = LLMRouter(_settings(
            llm_fast_provider=primary,
            llm_fallback_chain=fallback_chain,
        ))
        chain = router._get_chain_for_tier("fast")
        assert len(chain) == len(set(chain))

    @given(
        primary=_providers,
        fallback_chain=st.lists(_providers, min_size=0, max_size=3),
    )
    @h_settings(max_examples=50)
    def test_chain_length_at_least_one(
        self, primary: str, fallback_chain: list
    ) -> None:
        """The fallback chain always has at least one provider (the primary)."""
        router = LLMRouter(_settings(
            llm_fast_provider=primary,
            llm_fallback_chain=fallback_chain,
        ))
        chain = router._get_chain_for_tier("fast")
        assert len(chain) >= 1

    @given(
        primary=_providers,
        fallback_chain=st.lists(
            st.sampled_from(["openai", "anthropic", "google"]),
            min_size=1,
            max_size=2,
        ),
    )
    @h_settings(max_examples=20)
    def test_all_fail_raises_all_providers_failed(
        self, primary: str, fallback_chain: list
    ) -> None:
        """When all providers fail, AllProvidersFailedError is raised."""
        from unittest.mock import AsyncMock

        router = LLMRouter(_settings(
            llm_fast_provider=primary,
            llm_fallback_chain=fallback_chain,
        ))
        mock_agent = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.generate_str = AsyncMock(side_effect=ConnectionError("down"))
        mock_agent.attach_llm = AsyncMock(return_value=mock_llm)

        async def _run() -> None:
            with pytest.raises(AllProvidersFailedError) as exc_info:
                await router.call_with_fallback("fast", mock_agent, "prompt")
            err = exc_info.value
            assert err.tier == "fast"
            assert len(err.attempted_providers) >= 1

        asyncio.get_event_loop().run_until_complete(_run())
