"""E2E tests for GitLab API — real API calls, no mocks.

Requires GITLAB_TOKEN, GITLAB_GROUP, GITLAB_URL in .env.
"""

from __future__ import annotations

import os

import httpx
import pytest

GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "")
GITLAB_URL = os.getenv("GITLAB_URL", "https://gitlab.com")
GITLAB_GROUP = os.getenv("GITLAB_GROUP", "")

requires_gitlab = pytest.mark.skipif(
    not all([GITLAB_TOKEN, GITLAB_GROUP]),
    reason="GitLab credentials not configured",
)

HEADERS = {"PRIVATE-TOKEN": GITLAB_TOKEN}


@requires_gitlab
class TestGitLabConnectivity:
    """Verify GitLab API is reachable and token works."""

    @pytest.mark.asyncio
    async def test_authenticated_user(self) -> None:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(f"{GITLAB_URL}/api/v4/user")
        assert resp.status_code == 200
        data = resp.json()
        assert "username" in data

    @pytest.mark.asyncio
    async def test_list_projects(self) -> None:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"{GITLAB_URL}/api/v4/projects",
                params={"membership": True, "per_page": 5},
            )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_list_group_projects(self) -> None:
        """List projects in the configured group."""
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"{GITLAB_URL}/api/v4/groups/{GITLAB_GROUP}/projects",
                params={"per_page": 5},
            )
        if resp.status_code == 404:
            pytest.skip(f"Group '{GITLAB_GROUP}' not found")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@requires_gitlab
class TestGitLabRepoOperations:
    """Test repo-level operations on GitLab."""

    @pytest.mark.asyncio
    async def test_find_rag_project(self) -> None:
        """Find the Rag Project repo in the group."""
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"{GITLAB_URL}/api/v4/groups/{GITLAB_GROUP}/projects",
                params={"search": "Rag", "per_page": 10},
            )
        if resp.status_code == 404:
            pytest.skip(f"Group '{GITLAB_GROUP}' not found")
        assert resp.status_code == 200
        projects = resp.json()
        # Should find at least one project
        if not projects:
            pytest.skip("No 'Rag' projects found in group")
        names = [p["name"] for p in projects]
        assert any("rag" in n.lower() for n in names), f"Projects found: {names}"

    @pytest.mark.asyncio
    async def test_read_repo_tree(self) -> None:
        """Read file tree from a project."""
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            # First find the project
            resp = await client.get(
                f"{GITLAB_URL}/api/v4/groups/{GITLAB_GROUP}/projects",
                params={"search": "Rag", "per_page": 5},
            )
        if resp.status_code == 404 or not resp.json():
            pytest.skip("Rag project not found")

        project_id = resp.json()[0]["id"]

        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"{GITLAB_URL}/api/v4/projects/{project_id}/repository/tree",
            )
        assert resp.status_code == 200
        tree = resp.json()
        assert isinstance(tree, list)
        assert len(tree) > 0
