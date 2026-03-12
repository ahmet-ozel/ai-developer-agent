"""E2E tests for GitHub API — real API calls, no mocks.

Requires GITHUB_TOKEN, GITHUB_OWNER in .env.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.conftest import requires_github

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


@requires_github
class TestGitHubConnectivity:
    """Verify GitHub API is reachable and token works."""

    @pytest.mark.asyncio
    async def test_authenticated_user(self) -> None:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get("https://api.github.com/user")
        assert resp.status_code == 200
        data = resp.json()
        assert "login" in data

    @pytest.mark.asyncio
    async def test_list_repos(self) -> None:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                "https://api.github.com/user/repos",
                params={"per_page": 5},
            )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_token_scopes(self) -> None:
        """Token should have 'repo' scope for full functionality."""
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get("https://api.github.com/user")
        assert resp.status_code == 200
        scopes = resp.headers.get("x-oauth-scopes", "")
        # At minimum, repo scope is needed
        assert "repo" in scopes, f"Token scopes: {scopes} — 'repo' scope required"


@requires_github
class TestGitHubRepoOperations:
    """Test repo-level operations (branch, file read, PR)."""

    TEST_REPO = "Rag-Project"

    @pytest.mark.asyncio
    async def test_read_repo_contents(self) -> None:
        """Read root contents of test repo."""
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}/contents/"
            )
        if resp.status_code == 404:
            pytest.skip(f"Test repo {GITHUB_OWNER}/{self.TEST_REPO} not found")
        assert resp.status_code == 200
        files = resp.json()
        assert isinstance(files, list)
        assert len(files) > 0

    @pytest.mark.asyncio
    async def test_read_file_content(self) -> None:
        """Read a specific file from test repo."""
        # First list contents to find a file
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}/contents/"
            )
        if resp.status_code == 404:
            pytest.skip("Test repo not found")
        assert resp.status_code == 200
        files = resp.json()
        # Find any file (not directory)
        target = next((f for f in files if f["type"] == "file"), None)
        if target is None:
            pytest.skip("No files found in repo root")

        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}/contents/{target['name']}"
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["encoding"] == "base64"
        assert len(data["content"]) > 0

    @pytest.mark.asyncio
    async def test_create_and_delete_branch(self) -> None:
        """Create a test branch and delete it."""
        import time

        branch_name = f"e2e-test-{int(time.time())}"

        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            # Get default branch SHA
            resp = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}"
            )
            if resp.status_code == 404:
                pytest.skip("Test repo not found")
            repo_data = resp.json()
            default_branch = repo_data.get("default_branch", "main")

            resp = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}/git/ref/heads/{default_branch}"
            )
            if resp.status_code == 404:
                pytest.skip(f"Default branch '{default_branch}' not found")
            assert resp.status_code == 200
            sha = resp.json()["object"]["sha"]

            # Create branch
            resp = await client.post(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}/git/refs",
                json={"ref": f"refs/heads/{branch_name}", "sha": sha},
            )
            assert resp.status_code == 201, f"Branch create failed: {resp.text}"

            # Delete branch (cleanup)
            resp = await client.delete(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{self.TEST_REPO}/git/refs/heads/{branch_name}"
            )
            assert resp.status_code == 204
