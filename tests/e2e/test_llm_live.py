"""E2E tests for LLM provider connectivity — real API calls.

Requires LLM_FAST_PROVIDER, LLM_FAST_MODEL, LLM_FAST_API_KEY in .env.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.conftest import requires_llm


@requires_llm
class TestLLMConnectivity:
    """Verify LLM API is reachable and returns valid responses."""

    @pytest.mark.asyncio
    async def test_fast_tier_completion(self) -> None:
        """Fast tier LLM returns a valid completion."""
        provider = os.getenv("LLM_FAST_PROVIDER", "")
        model = os.getenv("LLM_FAST_MODEL", "")
        api_key = os.getenv("LLM_FAST_API_KEY", "")
        endpoint = os.getenv("LLM_FAST_ENDPOINT", "")

        response_text = await self._call_llm(provider, model, api_key, endpoint)
        assert len(response_text) > 0, "LLM returned empty response"

    @pytest.mark.asyncio
    async def test_strong_tier_completion(self) -> None:
        """Strong tier LLM returns a valid completion."""
        provider = os.getenv("LLM_STRONG_PROVIDER", "")
        model = os.getenv("LLM_STRONG_MODEL", "")
        api_key = os.getenv("LLM_STRONG_API_KEY", "")
        endpoint = os.getenv("LLM_STRONG_ENDPOINT", "")

        if not all([provider, model, api_key]):
            pytest.skip("Strong tier LLM not configured")

        response_text = await self._call_llm(provider, model, api_key, endpoint)
        assert len(response_text) > 0, "LLM returned empty response"

    @pytest.mark.asyncio
    async def test_fast_tier_tool_calling(self) -> None:
        """Fast tier LLM supports tool/function calling."""
        provider = os.getenv("LLM_FAST_PROVIDER", "")
        model = os.getenv("LLM_FAST_MODEL", "")
        api_key = os.getenv("LLM_FAST_API_KEY", "")
        endpoint = os.getenv("LLM_FAST_ENDPOINT", "")

        if provider not in ("openai", "vllm"):
            pytest.skip("Tool calling test only for OpenAI-compatible providers")

        base_url = endpoint or "https://api.openai.com/v1"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "description": "Perform arithmetic",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "expression": {"type": "string"},
                                    },
                                    "required": ["expression"],
                                },
                            },
                        }
                    ],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200, f"Tool calling failed: {resp.text[:200]}"

    async def _call_llm(
        self, provider: str, model: str, api_key: str, endpoint: str
    ) -> str:
        """Make a minimal LLM call and return the response text."""
        if provider in ("openai", "vllm"):
            base_url = endpoint or "https://api.openai.com/v1"
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with exactly: ALIVE"}],
                        "max_tokens": 10,
                    },
                )
            assert resp.status_code == 200, f"LLM call failed: {resp.text[:200]}"
            return resp.json()["choices"][0]["message"]["content"]

        elif provider == "anthropic":
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "Reply with exactly: ALIVE"}],
                    },
                )
            assert resp.status_code == 200, f"LLM call failed: {resp.text[:200]}"
            return resp.json()["content"][0]["text"]

        elif provider == "google":
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents": [{"parts": [{"text": "Reply with exactly: ALIVE"}]}]},
                )
            assert resp.status_code == 200, f"LLM call failed: {resp.text[:200]}"
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        else:
            pytest.skip(f"Unknown provider: {provider}")
            return ""
