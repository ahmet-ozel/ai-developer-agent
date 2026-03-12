"""Direct GitLab REST API client — bypasses MCP server entirely.

Provides all Git operations needed by the pipeline:
- Repository file tree listing
- File content retrieval
- Branch creation
- File commits (create/update/delete via commits API)
- Merge request creation

Uses httpx async client with the GitLab v4 API.
"""

from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import quote as url_quote

import httpx

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class GitLabClientError(Exception):
    """Raised on GitLab API errors."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"GitLab API {status}: {message}")


class GitLabClient:
    """Async GitLab REST API v4 client."""

    def __init__(self, settings: Settings) -> None:
        self._base = settings.gitlab_url.rstrip("/") + "/api/v4"
        self._token = settings.gitlab_token.get_secret_value() if settings.gitlab_token else ""
        self._headers = {"PRIVATE-TOKEN": self._token}
        self._group = settings.gitlab_group or ""
        self._default_branch = settings.git_base_branch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_path(self, repo: str) -> str:
        """Return URL-encoded project path (group/repo)."""
        if "/" not in repo:
            repo = f"{self._group}/{repo}"
        return url_quote(repo, safe="")

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(
                method, f"{self._base}{path}", headers=self._headers, **kwargs
            )
        if resp.status_code >= 400:
            raise GitLabClientError(resp.status_code, resp.text[:500])
        return resp

    async def _get(self, path: str, **params: Any) -> Any:
        resp = await self._request("GET", path, params=params)
        return resp.json()

    async def _post(self, path: str, json_body: dict) -> Any:
        resp = await self._request("POST", path, json=json_body)
        return resp.json()

    # ------------------------------------------------------------------
    # Repository tree
    # ------------------------------------------------------------------

    async def get_file_tree(self, repo: str, ref: str | None = None) -> str:
        """Return newline-separated file paths in the repo."""
        pid = self._project_path(repo)
        ref = ref or self._default_branch
        all_paths: list[str] = []
        page = 1
        while True:
            items = await self._get(
                f"/projects/{pid}/repository/tree",
                ref=ref, recursive="true", per_page=100, page=page,
            )
            if not items:
                break
            for item in items:
                all_paths.append(item["path"])
            if len(items) < 100:
                break
            page += 1
        return "\n".join(sorted(all_paths))

    # ------------------------------------------------------------------
    # File content
    # ------------------------------------------------------------------

    async def get_file_content(self, repo: str, file_path: str, ref: str | None = None) -> str:
        """Return decoded file content."""
        pid = self._project_path(repo)
        ref = ref or self._default_branch
        encoded_path = url_quote(file_path, safe="")
        data = await self._get(
            f"/projects/{pid}/repository/files/{encoded_path}",
            ref=ref,
        )
        content_b64 = data.get("content", "")
        return base64.b64decode(content_b64).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Branch creation
    # ------------------------------------------------------------------

    async def create_branch(self, repo: str, branch: str, ref: str | None = None) -> dict:
        """Create a new branch from ref (default: main)."""
        pid = self._project_path(repo)
        ref = ref or self._default_branch
        return await self._post(
            f"/projects/{pid}/repository/branches",
            {"branch": branch, "ref": ref},
        )

    # ------------------------------------------------------------------
    # Commit files (create/update/delete in a single commit)
    # ------------------------------------------------------------------

    async def commit_files(
        self,
        repo: str,
        branch: str,
        message: str,
        actions: list[dict[str, str]],
    ) -> dict:
        """Create a commit with multiple file actions.

        Each action: {"action": "create"|"update"|"delete", "file_path": "...", "content": "..."}
        """
        pid = self._project_path(repo)
        return await self._post(
            f"/projects/{pid}/repository/commits",
            {
                "branch": branch,
                "commit_message": message,
                "actions": actions,
            },
        )

    # ------------------------------------------------------------------
    # Merge request creation
    # ------------------------------------------------------------------

    async def create_merge_request(
        self,
        repo: str,
        source_branch: str,
        target_branch: str | None = None,
        title: str = "",
        description: str = "",
        draft: bool = True,
    ) -> dict:
        """Create a merge request and return the full MR object."""
        pid = self._project_path(repo)
        target = target_branch or self._default_branch
        body: dict[str, Any] = {
            "source_branch": source_branch,
            "target_branch": target,
            "title": title,
            "description": description,
        }
        # GitLab uses "Draft: " prefix for draft MRs
        if draft and not title.startswith("Draft:"):
            body["title"] = f"Draft: {title}"
        return await self._post(f"/projects/{pid}/merge_requests", body)
