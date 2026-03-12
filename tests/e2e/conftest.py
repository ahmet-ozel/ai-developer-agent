"""E2E test configuration — skips tests if credentials are missing.

Run e2e tests with: pytest tests/e2e/ -v
Credentials are read from .env file in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env for e2e tests — these need real credentials
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


def _has_jira_creds() -> bool:
    return all([
        os.getenv("JIRA_URL"),
        os.getenv("JIRA_USERNAME"),
        os.getenv("JIRA_API_TOKEN"),
    ])


def _has_github_creds() -> bool:
    return all([
        os.getenv("GITHUB_TOKEN"),
        os.getenv("GITHUB_OWNER"),
    ])


def _has_llm_creds() -> bool:
    return all([
        os.getenv("LLM_FAST_API_KEY"),
        os.getenv("LLM_FAST_PROVIDER"),
        os.getenv("LLM_FAST_MODEL"),
    ])


requires_jira = pytest.mark.skipif(
    not _has_jira_creds(), reason="Jira credentials not configured"
)
requires_github = pytest.mark.skipif(
    not _has_github_creds(), reason="GitHub credentials not configured"
)
requires_llm = pytest.mark.skipif(
    not _has_llm_creds(), reason="LLM credentials not configured"
)
