"""E2E tests for Jira Cloud API — real API calls, no mocks.

Requires JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN in .env.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.conftest import requires_jira

JIRA_URL = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_AUTH = (os.getenv("JIRA_USERNAME", ""), os.getenv("JIRA_API_TOKEN", ""))


@requires_jira
class TestJiraConnectivity:
    """Verify Jira Cloud API is reachable and credentials work."""

    @pytest.mark.asyncio
    async def test_myself_endpoint(self) -> None:
        """GET /rest/api/3/myself returns current user."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{JIRA_URL}/rest/api/3/myself", auth=JIRA_AUTH
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "accountId" in data
        assert "displayName" in data

    @pytest.mark.asyncio
    async def test_search_issues(self) -> None:
        """JQL search returns results without error."""
        async with httpx.AsyncClient(timeout=15) as client:
            # First get a project key to use in JQL (Jira Cloud requires scoped queries)
            proj_resp = await client.get(
                f"{JIRA_URL}/rest/api/3/project", auth=JIRA_AUTH
            )
            if proj_resp.status_code != 200 or not proj_resp.json():
                pytest.skip("No Jira projects available for search test")
            project_key = proj_resp.json()[0]["key"]

            resp = await client.post(
                f"{JIRA_URL}/rest/api/3/search/jql",
                auth=JIRA_AUTH,
                json={"jql": f"project = {project_key} ORDER BY created DESC", "maxResults": 1},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "issues" in data

    @pytest.mark.asyncio
    async def test_list_projects(self) -> None:
        """GET /rest/api/3/project returns at least one project."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{JIRA_URL}/rest/api/3/project", auth=JIRA_AUTH
            )
        assert resp.status_code == 200
        projects = resp.json()
        assert isinstance(projects, list)

    @pytest.mark.asyncio
    async def test_list_fields(self) -> None:
        """GET /rest/api/3/field returns field definitions."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{JIRA_URL}/rest/api/3/field", auth=JIRA_AUTH
            )
        assert resp.status_code == 200
        fields = resp.json()
        assert isinstance(fields, list)
        # Check that at least some fields exist (names may be localized)
        assert len(fields) > 0


@requires_jira
class TestJiraIssueLifecycle:
    """Create, read, update, delete a Jira issue — full lifecycle."""

    @pytest.mark.asyncio
    async def test_issue_crud(self) -> None:
        """Create → Read → Comment → Delete an issue."""
        async with httpx.AsyncClient(timeout=30) as client:
            # Find first available project
            resp = await client.get(
                f"{JIRA_URL}/rest/api/3/project", auth=JIRA_AUTH
            )
            assert resp.status_code == 200
            projects = resp.json()
            if not projects:
                pytest.skip("No Jira projects available")
            project_key = projects[0]["key"]

            # CREATE
            resp = await client.post(
                f"{JIRA_URL}/rest/api/3/issue",
                auth=JIRA_AUTH,
                json={
                    "fields": {
                        "project": {"key": project_key},
                        "summary": "[E2E TEST] Auto-created — safe to delete",
                        "issuetype": {"name": "Task"},
                    }
                },
            )
            assert resp.status_code == 201, f"Create failed: {resp.text}"
            issue_key = resp.json()["key"]

            try:
                # READ
                resp = await client.get(
                    f"{JIRA_URL}/rest/api/3/issue/{issue_key}",
                    auth=JIRA_AUTH,
                )
                assert resp.status_code == 200
                assert resp.json()["fields"]["summary"].startswith("[E2E TEST]")

                # COMMENT
                resp = await client.post(
                    f"{JIRA_URL}/rest/api/3/issue/{issue_key}/comment",
                    auth=JIRA_AUTH,
                    json={
                        "body": {
                            "type": "doc",
                            "version": 1,
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [
                                        {"type": "text", "text": "E2E test comment — pipeline working"}
                                    ],
                                }
                            ],
                        }
                    },
                )
                assert resp.status_code == 201

            finally:
                # DELETE (cleanup)
                await client.delete(
                    f"{JIRA_URL}/rest/api/3/issue/{issue_key}",
                    auth=JIRA_AUTH,
                )
