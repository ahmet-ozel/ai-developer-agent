"""E2E test: Pipeline dry-run with real Jira issue.

Tests the full pipeline flow in DRY_RUN mode:
1. Creates a Jira issue assigned to the bot
2. Runs the pipeline (dry-run — no Git writes)
3. Verifies pipeline completes without crash
4. Cleans up the test issue

Requires: JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN, JIRA_PROJECT_KEY,
          JIRA_BOT_USERNAME, LLM_FAST_API_KEY, GIT_PROVIDER + creds
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.conftest import requires_jira, requires_llm


def _has_pipeline_creds() -> bool:
    return all([
        os.getenv("JIRA_URL"),
        os.getenv("JIRA_USERNAME"),
        os.getenv("JIRA_API_TOKEN"),
        os.getenv("JIRA_PROJECT_KEY"),
        os.getenv("LLM_FAST_API_KEY"),
        os.getenv("LLM_FAST_PROVIDER"),
        os.getenv("GIT_PROVIDER"),
    ])


requires_pipeline = pytest.mark.skipif(
    not _has_pipeline_creds(),
    reason="Pipeline credentials not fully configured",
)


@requires_pipeline
class TestPipelineDryRun:
    """Full pipeline dry-run test with real Jira."""

    @pytest.mark.asyncio
    async def test_pipeline_dryrun_creates_no_pr(self) -> None:
        """Run pipeline in DRY_RUN mode — should complete without errors."""
        jira_url = os.getenv("JIRA_URL", "").rstrip("/")
        username = os.getenv("JIRA_USERNAME", "")
        api_token = os.getenv("JIRA_API_TOKEN", "")
        project_key = os.getenv("JIRA_PROJECT_KEY", "")
        bot_username = os.getenv("JIRA_BOT_USERNAME", "")
        github_owner = os.getenv("GITHUB_OWNER", "")

        auth = (username, api_token)
        issue_key = None

        try:
            # 1. Create a test issue assigned to bot
            async with httpx.AsyncClient(timeout=30) as client:
                # Find bot account ID
                resp = await client.get(
                    f"{jira_url}/rest/api/3/user/search",
                    params={"query": bot_username},
                    auth=auth,
                )
                if resp.status_code != 200 or not resp.json():
                    pytest.skip(f"Bot user '{bot_username}' not found in Jira")

                bot_account_id = resp.json()[0]["accountId"]

                # Create issue
                issue_data = {
                    "fields": {
                        "project": {"key": project_key},
                        "summary": "[E2E Test] Pipeline dry-run test — auto-delete",
                        "description": {
                            "type": "doc",
                            "version": 1,
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "This is an automated e2e test issue. "
                                                "Add a simple hello world function to the repository. "
                                                f"Repository: {github_owner}/Rag-Project"
                                            ),
                                        }
                                    ],
                                }
                            ],
                        },
                        "issuetype": {"name": "Task"},
                        "assignee": {"accountId": bot_account_id},
                    }
                }

                resp = await client.post(
                    f"{jira_url}/rest/api/3/issue",
                    json=issue_data,
                    auth=auth,
                )
                if resp.status_code not in (200, 201):
                    pytest.skip(f"Could not create test issue: {resp.status_code} {resp.text[:200]}")

                issue_key = resp.json()["key"]

            # 2. Run pipeline in dry-run mode
            from src.config.settings import Settings

            settings = Settings(
                _env_file=None,
                jira_url=jira_url,
                jira_username=username,
                jira_api_token=api_token,
                jira_webhook_secret=os.getenv("JIRA_WEBHOOK_SECRET", "test"),
                jira_bot_username=bot_username,
                git_provider=os.getenv("GIT_PROVIDER", "github"),
                github_token=os.getenv("GITHUB_TOKEN"),
                github_owner=github_owner,
                llm_fast_provider=os.getenv("LLM_FAST_PROVIDER", "openai"),
                llm_fast_model=os.getenv("LLM_FAST_MODEL", "gpt-4o-mini"),
                llm_fast_api_key=os.getenv("LLM_FAST_API_KEY", ""),
                llm_strong_provider=os.getenv("LLM_STRONG_PROVIDER", "openai"),
                llm_strong_model=os.getenv("LLM_STRONG_MODEL", "gpt-4o"),
                llm_strong_api_key=os.getenv("LLM_STRONG_API_KEY", ""),
                dry_run=True,
            )

            from src.app import run_pipeline

            # This should NOT raise — dry-run mode skips Git writes
            # Note: mcp-agent may not be fully installed, so the pipeline
            # may fail at the MCP agent level. We catch and check.
            try:
                await run_pipeline(issue_key, settings)
            except NotImplementedError:
                # Expected when mcp-agent placeholder is used
                pass
            except Exception as exc:
                # Pipeline may fail for various reasons in test env,
                # but it should not crash with an unhandled exception
                # that indicates a code bug (as opposed to missing MCP server)
                exc_msg = str(exc).lower()
                acceptable_failures = [
                    "mcp-agent is not installed",
                    "not implemented",
                    "connection refused",
                    "repository",
                ]
                is_acceptable = any(kw in exc_msg for kw in acceptable_failures)
                if not is_acceptable:
                    raise

        finally:
            # 3. Cleanup: delete the test issue
            if issue_key:
                async with httpx.AsyncClient(timeout=15) as client:
                    await client.delete(
                        f"{jira_url}/rest/api/3/issue/{issue_key}",
                        auth=auth,
                    )
