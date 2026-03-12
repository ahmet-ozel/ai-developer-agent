"""Direct Bitbucket Cloud REST API client — bypasses MCP server entirely.

Provides all Git operations needed by the pipeline:
- Repository file tree listing (via src endpoint)
- File content retrieval
- Branch creation
- File commits
- Pull request creation

Uses httpx async client with Bitbucket Cloud API v2.0.
Authenticates via Atlassian API token (same token as Jira).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import Settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bitbucket.org/2.0"


class BitbucketClientError(Exception):
    """Raised on Bitbucket API errors."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"Bitbucket API {status}: {message}")


class BitbucketClient:
    """Async Bitbucket Cloud REST API v2 client."""

    def __init__(self, settings: Settings) -> None:
        self._workspace = settings.bitbucket_workspace or ""
        self._username = settings.bitbucket_username or ""
        # Prefer API token (new), fall back to app password (legacy)
        self._token = ""
        if settings.bitbucket_api_token:
            self._token = settings.bitbucket_api_token.get_secret_value()
        elif settings.bitbucket_app_password:
            self._token = settings.bitbucket_app_password.get_secret_value()
        self._default_branch = settings.git_base_branch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth(self) -> tuple[str, str]:
        return (self._username, self._token)

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url, auth=self._auth(), **kwargs)
        if resp.status_code >= 400:
            raise BitbucketClientError(resp.status_code, resp.text[:500])
        return resp

    async def _get(self, path: str, **params: Any) -> Any:
        resp = await self._request("GET", f"{BASE_URL}{path}", params=params)
        return resp.json()

    async def _post(self, path: str, json_body: dict | None = None, **kwargs: Any) -> Any:
        resp = await self._request("POST", f"{BASE_URL}{path}", json=json_body, **kwargs)
        return resp.json()

    def _repo_path(self, repo: str) -> str:
        """Return workspace/repo slug."""
        if "/" in repo:
            return repo
        return f"{self._workspace}/{repo}"

    # ------------------------------------------------------------------
    # Repository tree
    # ------------------------------------------------------------------

    async def get_file_tree(self, repo: str, ref: str | None = None) -> str:
        """Return newline-separated file paths."""
        rp = self._repo_path(repo)
        ref = ref or self._default_branch
        all_paths: list[str] = []

        # First request with params
        first_url = f"{BASE_URL}/repositories/{rp}/src/{ref}/"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(first_url, auth=self._auth(), params={"pagelen": 100})
        if resp.status_code >= 400:
            raise BitbucketClientError(resp.status_code, resp.text[:500])
        data = resp.json()
        for item in data.get("values", []):
            item_path = item.get("path", "")
            if item_path:
                all_paths.append(item_path)

        # Follow pagination
        next_url = data.get("next")
        while next_url:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(next_url, auth=self._auth())
            if resp.status_code >= 400:
                break
            data = resp.json()
            for item in data.get("values", []):
                item_path = item.get("path", "")
                if item_path:
                    all_paths.append(item_path)
            next_url = data.get("next")

        return "\n".join(sorted(all_paths))

    # ------------------------------------------------------------------
    # File content
    # ------------------------------------------------------------------

    async def get_file_content(self, repo: str, file_path: str, ref: str | None = None) -> str:
        """Return raw file content."""
        rp = self._repo_path(repo)
        ref = ref or self._default_branch
        url = f"{BASE_URL}/repositories/{rp}/src/{ref}/{file_path}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, auth=self._auth())
        if resp.status_code >= 400:
            raise BitbucketClientError(resp.status_code, resp.text[:500])
        return resp.text

    # ------------------------------------------------------------------
    # Branch creation
    # ------------------------------------------------------------------

    async def create_branch(self, repo: str, branch: str, ref: str | None = None) -> dict:
        """Create a new branch from ref (default: main branch's latest commit)."""
        rp = self._repo_path(repo)
        ref = ref or self._default_branch

        # Bitbucket needs the commit hash, not branch name, for target
        # First get the latest commit hash of the ref branch
        try:
            branch_info = await self._get(f"/repositories/{rp}/refs/branches/{ref}")
            commit_hash = branch_info["target"]["hash"]
        except Exception:
            # If ref is already a hash, use it directly
            commit_hash = ref

        body = {
            "name": branch,
            "target": {"hash": commit_hash},
        }
        return await self._post(f"/repositories/{rp}/refs/branches", body)

    # ------------------------------------------------------------------
    # Commit files (via src endpoint)
    # ------------------------------------------------------------------

    async def commit_files(
        self,
        repo: str,
        branch: str,
        message: str,
        files: dict[str, str],
    ) -> dict:
        """Commit multiple files. files = {path: content}."""
        rp = self._repo_path(repo)
        # Bitbucket uses multipart form for src endpoint
        data: dict[str, str] = {
            "message": message,
            "branch": branch,
        }
        file_data: list[tuple[str, tuple[str, str]]] = []
        for path, content in files.items():
            file_data.append((path, (path, content)))

        url = f"{BASE_URL}/repositories/{rp}/src"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, auth=self._auth(), data=data, files=file_data)
        if resp.status_code >= 400:
            raise BitbucketClientError(resp.status_code, resp.text[:500])
        return resp.json() if resp.text.strip() else {"status": "ok"}

    # ------------------------------------------------------------------
    # Pull request creation
    # ------------------------------------------------------------------

    async def create_pull_request(
        self,
        repo: str,
        source_branch: str,
        target_branch: str | None = None,
        title: str = "",
        description: str = "",
    ) -> dict:
        """Create a pull request."""
        rp = self._repo_path(repo)
        target = target_branch or self._default_branch
        body = {
            "title": title,
            "description": description,
            "source": {"branch": {"name": source_branch}},
            "destination": {"branch": {"name": target}},
            "close_source_branch": True,
        }
        return await self._post(f"/repositories/{rp}/pullrequests", body)
