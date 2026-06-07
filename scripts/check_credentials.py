#!/usr/bin/env python3
"""Credential health check - verifies all configured services are reachable.

Usage:
    python scripts/check_credentials.py

Reads .env and tests real API connectivity for:
- Jira Cloud (REST API v3)
- GitHub / GitLab / Bitbucket (depending on GIT_PROVIDER)
- LLM provider (fast + strong tiers)

Exit code 0 = all checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")


results: list[bool] = []


async def check_jira() -> None:
    header("Jira Cloud")
    url = os.getenv("JIRA_URL", "")
    username = os.getenv("JIRA_USERNAME", "")
    token = os.getenv("JIRA_API_TOKEN", "")

    if not all([url, username, token]):
        fail("JIRA_URL, JIRA_USERNAME, or JIRA_API_TOKEN is empty")
        results.append(False)
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{url.rstrip('/')}/rest/api/3/myself",
                auth=(username, token),
            )
        if resp.status_code == 200:
            data = resp.json()
            ok(f"Connected as: {data.get('displayName', 'unknown')} ({data.get('emailAddress', '')})")
            results.append(True)
        else:
            fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
            results.append(False)
    except Exception as e:
        fail(f"Connection error: {e}")
        results.append(False)

    # Check bot user exists
    bot = os.getenv("JIRA_BOT_USERNAME", "")
    if bot:
        ok(f"Bot username configured: {bot}")
    else:
        warn("JIRA_BOT_USERNAME is empty - webhook filtering won't work")


async def check_github() -> None:
    header("GitHub")
    token = os.getenv("GITHUB_TOKEN", "")
    owner = os.getenv("GITHUB_OWNER", "")

    if not token:
        fail("GITHUB_TOKEN is empty")
        results.append(False)
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            )
        if resp.status_code == 200:
            data = resp.json()
            ok(f"Authenticated as: {data.get('login', 'unknown')}")
            results.append(True)

            # Check scopes
            scopes = resp.headers.get("x-oauth-scopes", "")
            if "repo" in scopes:
                ok(f"Token scopes: {scopes}")
            else:
                warn(f"Token scopes: {scopes} - 'repo' scope recommended")
        else:
            fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
            results.append(False)
    except Exception as e:
        fail(f"Connection error: {e}")
        results.append(False)

    if owner:
        ok(f"Owner/org: {owner}")
    else:
        warn("GITHUB_OWNER is empty")


async def check_gitlab() -> None:
    header("GitLab")
    token = os.getenv("GITLAB_TOKEN", "")
    url = os.getenv("GITLAB_URL", "https://gitlab.com")

    if not token:
        fail("GITLAB_TOKEN is empty")
        results.append(False)
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{url.rstrip('/')}/api/v4/user",
                headers={"PRIVATE-TOKEN": token},
            )
        if resp.status_code == 200:
            data = resp.json()
            ok(f"Authenticated as: {data.get('username', 'unknown')}")
            results.append(True)
        else:
            fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
            results.append(False)
    except Exception as e:
        fail(f"Connection error: {e}")
        results.append(False)


async def check_bitbucket() -> None:
    header("Bitbucket")
    username = os.getenv("BITBUCKET_USERNAME", "")
    password = os.getenv("BITBUCKET_APP_PASSWORD", "")
    workspace = os.getenv("BITBUCKET_WORKSPACE", "")

    if not all([username, password]):
        fail("BITBUCKET_USERNAME or BITBUCKET_APP_PASSWORD is empty")
        results.append(False)
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.bitbucket.org/2.0/user",
                auth=(username, password),
            )
        if resp.status_code == 200:
            data = resp.json()
            ok(f"Authenticated as: {data.get('display_name', 'unknown')}")
            results.append(True)
        else:
            fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
            results.append(False)
    except Exception as e:
        fail(f"Connection error: {e}")
        results.append(False)

    if workspace:
        ok(f"Workspace: {workspace}")
    else:
        warn("BITBUCKET_WORKSPACE is empty")


async def check_llm() -> None:
    header("LLM Providers")

    for tier in ("fast", "strong"):
        provider = os.getenv(f"LLM_{tier.upper()}_PROVIDER", "")
        model = os.getenv(f"LLM_{tier.upper()}_MODEL", "")
        api_key = os.getenv(f"LLM_{tier.upper()}_API_KEY", "")
        endpoint = os.getenv(f"LLM_{tier.upper()}_ENDPOINT", "")

        if not provider:
            warn(f"LLM_{tier.upper()}_PROVIDER is empty - skipping")
            continue

        if not api_key:
            fail(f"LLM_{tier.upper()}_API_KEY is empty")
            results.append(False)
            continue

        # Test with a minimal completion request
        try:
            if provider in ("openai", "vllm"):
                base_url = endpoint or "https://api.openai.com/v1"
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Say 'ok'"}],
                            "max_tokens": 5,
                        },
                    )
                if resp.status_code == 200:
                    ok(f"{tier} tier: {provider}/{model} - working")
                    results.append(True)
                else:
                    fail(f"{tier} tier: {provider}/{model} - HTTP {resp.status_code}: {resp.text[:200]}")
                    results.append(False)

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
                            "max_tokens": 5,
                            "messages": [{"role": "user", "content": "Say 'ok'"}],
                        },
                    )
                if resp.status_code == 200:
                    ok(f"{tier} tier: {provider}/{model} - working")
                    results.append(True)
                else:
                    fail(f"{tier} tier: {provider}/{model} - HTTP {resp.status_code}: {resp.text[:200]}")
                    results.append(False)

            elif provider == "google":
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                        json={"contents": [{"parts": [{"text": "Say 'ok'"}]}]},
                    )
                if resp.status_code == 200:
                    ok(f"{tier} tier: {provider}/{model} - working")
                    results.append(True)
                else:
                    fail(f"{tier} tier: {provider}/{model} - HTTP {resp.status_code}: {resp.text[:200]}")
                    results.append(False)
            else:
                warn(f"{tier} tier: unknown provider '{provider}' - skipping connectivity check")

        except Exception as e:
            fail(f"{tier} tier: {provider}/{model} - {e}")
            results.append(False)


async def main() -> None:
    print(f"{BOLD}{'='*60}")
    print("  AI Developer Agent - Credential Health Check")
    print(f"{'='*60}{RESET}")

    # Always check Jira
    await check_jira()

    # Check active git provider
    provider = os.getenv("GIT_PROVIDER", "github")
    if provider == "github":
        await check_github()
    elif provider == "gitlab":
        await check_gitlab()
    elif provider == "bitbucket":
        await check_bitbucket()
    else:
        warn(f"Unknown GIT_PROVIDER: {provider}")

    # Check LLM
    await check_llm()

    # Summary
    header("Summary")
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    total = len(results)
    print(f"  {passed}/{total} checks passed, {failed} failed")

    if failed > 0:
        print(f"\n  {RED}Some checks failed. Fix the issues above before running e2e tests.{RESET}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}All checks passed! Ready for e2e testing.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
