"""CodeFinder Agent — finds relevant code files and produces CodeContext.

Uses mcp-agent Agent + AugmentedLLM pattern with
server_names=[get_active_git_server_name(settings)] to interact with the
Git MCP server. Identifies relevant files via fast tier LLM, fetches their
contents, detects tech stack, and filters out skippable/oversized files.

Includes exponential backoff retry (max 3 attempts).

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Optional

from src.config.mcp_servers import get_active_git_server_name
from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    CodeContext,
    CodeFile,
    SkippedFile,
    TaskContext,
)
from src.utils.git_helpers import detect_tech_stack, detect_test_file_path, is_skippable_file

# Direct API clients (bypass broken MCP servers)
from src.clients.gitlab_client import GitLabClient
from src.clients.bitbucket_client import BitbucketClient

# ---------------------------------------------------------------------------
# Try importing mcp-agent Agent class. Falls back to placeholder when
# mcp-agent is not installed (e.g. in test environments).
# ---------------------------------------------------------------------------

try:
    from mcp_agent.agents.agent import Agent
except ImportError:  # pragma: no cover

    class Agent:  # type: ignore[no-redef]
        """Placeholder for mcp-agent Agent when not installed."""

        def __init__(self, **kwargs: Any) -> None:
            self.name = kwargs.get("name", "")
            self.instruction = kwargs.get("instruction", "")
            self.server_names = kwargs.get("server_names", [])

        async def __aenter__(self) -> "Agent":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def attach_llm(self, llm_class: type) -> Any:
            raise NotImplementedError("mcp-agent is not installed")


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LINE_COUNT = 1000

# ---------------------------------------------------------------------------
# Retry helper (same pattern as task_reader.py)
# ---------------------------------------------------------------------------

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def _is_retryable(exc: Exception) -> bool:
    """Check whether an exception is retryable."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status and status in {429, 502, 503, 504}:
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "too many requests", "temporarily unavailable")):
        return True
    return False


async def _retry_with_backoff(
    func: Any,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Exponential backoff retry with ±25% jitter."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.25 * (random.random() * 2 - 1)
            logger.warning(
                "CodeFinder retry attempt %d/%d after error: %s (delay=%.2fs)",
                attempt + 1,
                max_retries,
                exc,
                delay + jitter,
            )
            await asyncio.sleep(delay + jitter)
    raise RuntimeError("Retry loop exhausted unexpectedly")  # pragma: no cover


# ---------------------------------------------------------------------------
# Instruction prompt for the CodeFinder agent
# ---------------------------------------------------------------------------

CODE_FINDER_INSTRUCTION = """You are a code finder agent. Your job is to find relevant source
files in a repository for a given task.

When asked to find files:
1. Use the Git MCP tools to get the repository file tree
2. Analyze the task description to identify relevant files
3. Return a JSON list of file paths that are most relevant to the task

Return ONLY a JSON array of file path strings, e.g.: ["src/auth/login.py", "src/auth/session.py"]"""

FILE_IDENTIFICATION_PROMPT = """Given the following repository file tree and task description,
identify the most relevant source files that would need to be read or modified.

Task Summary: {summary}
Task Description: {description}
Acceptance Criteria: {acceptance_criteria}

Repository File Tree:
{file_tree}

Return ONLY a JSON array of file paths (max {max_files} files), ordered by relevance.
Example: ["src/auth/login.py", "src/utils/helpers.py"]"""


# ---------------------------------------------------------------------------
# CodeFinderAgent
# ---------------------------------------------------------------------------


class CodeFinderAgent:
    """Finds relevant code files via Git MCP and produces CodeContext.

    Uses mcp-agent Agent pattern with server_names=[get_active_git_server_name(settings)]
    and fast tier LLM for file identification.
    """

    def __init__(self, settings: Settings, llm_router: LLMRouter) -> None:
        self._settings = settings
        self._llm_router = llm_router

    async def find_code(self, task_context: TaskContext) -> CodeContext:
        """Find relevant code files and build a CodeContext.

        Retries with exponential backoff (max 3 attempts) on transient errors.
        """
        return await _retry_with_backoff(
            lambda: self._find_code_impl(task_context),
            max_retries=3,
        )

    async def _find_code_impl(self, task_context: TaskContext) -> CodeContext:
        """Core implementation: find files and build CodeContext.

        Uses direct REST API clients for GitLab/Bitbucket (MCP servers are
        broken). Falls back to MCP agent pattern for GitHub.
        """
        llm_class = self._llm_router.get_llm_class("fast")
        provider = self._settings.git_provider

        # Build fully-qualified repo reference (owner/repo or group/repo)
        repo_name = task_context.repository_name
        qualified_repo = self._qualify_repo_name(repo_name)

        # --- Direct API path for GitLab / Bitbucket ---
        if provider in ("gitlab", "bitbucket"):
            return await self._find_code_direct(task_context, qualified_repo, llm_class)

        # --- MCP agent path for GitHub ---
        return await self._find_code_mcp(task_context, qualified_repo, llm_class)

    async def _find_code_direct(
        self, task_context: TaskContext, qualified_repo: str, llm_class: type
    ) -> CodeContext:
        """Find code using direct REST API client (GitLab/Bitbucket)."""
        provider = self._settings.git_provider

        # Pick the right client
        if provider == "gitlab":
            client: Any = GitLabClient(self._settings)
        else:
            client = BitbucketClient(self._settings)

        # 1. Get repository file tree
        file_tree = await client.get_file_tree(qualified_repo)
        logger.info("Got file tree for %s (%d files)", qualified_repo, len(file_tree.splitlines()))

        # 2. Detect tech stack from file tree
        tech_stack = detect_tech_stack(file_tree)

        # 3. Use LLM to identify relevant files (no MCP needed — pure LLM call)
        max_files = self._settings.max_files_per_task
        prompt = FILE_IDENTIFICATION_PROMPT.format(
            summary=task_context.summary,
            description=task_context.description or "No description",
            acceptance_criteria=task_context.acceptance_criteria or "None specified",
            file_tree=file_tree,
            max_files=max_files,
        )

        # Use a simple Agent with no server_names for pure LLM call
        agent = Agent(
            name="code_finder_llm",
            instruction="You identify relevant files from a file tree.",
            server_names=[],
        )
        async with agent:
            llm = await agent.attach_llm(llm_class)
            relevant_files_raw = await llm.generate_str(prompt)

        relevant_paths = self._parse_file_list(relevant_files_raw)

        # 4. Filter and fetch file contents via direct API
        files: list[CodeFile] = []
        test_files: list[CodeFile] = []
        skipped_files: list[SkippedFile] = []

        for path in relevant_paths:
            if len(files) + len(test_files) >= max_files:
                break

            skippable, reason = is_skippable_file(path)
            if skippable:
                skipped_files.append(SkippedFile(path=path, reason=reason))
                continue

            try:
                content = await client.get_file_content(qualified_repo, path)
            except Exception:
                logger.warning("Could not fetch file: %s", path)
                skipped_files.append(SkippedFile(path=path, reason="fetch_error"))
                continue

            file_size_kb = len(content.encode("utf-8")) / 1024
            if file_size_kb > self._settings.max_file_size_kb:
                skipped_files.append(SkippedFile(path=path, reason="too_large"))
                continue

            line_count = len(content.splitlines())
            if line_count > MAX_LINE_COUNT:
                skipped_files.append(SkippedFile(path=path, reason="too_large"))
                continue

            language = self._detect_language(path)
            is_test = self._is_test_file(path)

            code_file = CodeFile(
                path=path, content=content, line_count=line_count,
                language=language, is_test=is_test,
            )
            if is_test:
                test_files.append(code_file)
            else:
                files.append(code_file)

        # 5. Find corresponding test files
        for source_file in list(files):
            test_path = detect_test_file_path(source_file.path, tech_stack)
            if test_path and not any(tf.path == test_path for tf in test_files):
                if len(files) + len(test_files) >= max_files:
                    break
                if test_path in file_tree:
                    try:
                        test_content = await client.get_file_content(qualified_repo, test_path)
                        test_line_count = len(test_content.splitlines())
                        if test_line_count <= MAX_LINE_COUNT:
                            test_files.append(CodeFile(
                                path=test_path, content=test_content,
                                line_count=test_line_count,
                                language=self._detect_language(test_path), is_test=True,
                            ))
                    except Exception:
                        logger.debug("Could not fetch test file: %s", test_path)

        return CodeContext(
            files=files, test_files=test_files, tech_stack=tech_stack,
            repository_name=qualified_repo, file_tree=file_tree,
            skipped_files=skipped_files,
        )

    async def _find_code_mcp(
        self, task_context: TaskContext, qualified_repo: str, llm_class: type
    ) -> CodeContext:
        """Find code using MCP agent pattern (GitHub)."""
        git_server = get_active_git_server_name(self._settings)

        agent = Agent(
            name="code_finder",
            instruction=CODE_FINDER_INSTRUCTION,
            server_names=[git_server],
        )

        async with agent:
            llm = await agent.attach_llm(llm_class)

            # 1. Get repository file tree
            file_tree_raw = await llm.generate_str(
                f"Get the file tree for repository '{qualified_repo}'. "
                "Return the raw file tree output."
            )
            file_tree = file_tree_raw.strip()

            # 2. Detect tech stack from file tree
            tech_stack = detect_tech_stack(file_tree)

            # 3. Use LLM to identify relevant files
            max_files = self._settings.max_files_per_task
            prompt = FILE_IDENTIFICATION_PROMPT.format(
                summary=task_context.summary,
                description=task_context.description or "No description",
                acceptance_criteria=task_context.acceptance_criteria or "None specified",
                file_tree=file_tree,
                max_files=max_files,
            )
            relevant_files_raw = await llm.generate_str(prompt)
            relevant_paths = self._parse_file_list(relevant_files_raw)

            # 4. Filter and fetch file contents
            files: list[CodeFile] = []
            test_files: list[CodeFile] = []
            skipped_files: list[SkippedFile] = []

            for path in relevant_paths:
                if len(files) + len(test_files) >= max_files:
                    break

                skippable, reason = is_skippable_file(path)
                if skippable:
                    skipped_files.append(SkippedFile(path=path, reason=reason))
                    continue

                content = await llm.generate_str(
                    f"Get the content of file '{path}' from repository "
                    f"'{qualified_repo}'. Return ONLY the raw file content."
                )

                file_size_kb = len(content.encode("utf-8")) / 1024
                if file_size_kb > self._settings.max_file_size_kb:
                    skipped_files.append(SkippedFile(path=path, reason="too_large"))
                    continue

                line_count = len(content.splitlines())
                if line_count > MAX_LINE_COUNT:
                    skipped_files.append(SkippedFile(path=path, reason="too_large"))
                    continue

                language = self._detect_language(path)
                is_test = self._is_test_file(path)

                code_file = CodeFile(
                    path=path, content=content, line_count=line_count,
                    language=language, is_test=is_test,
                )
                if is_test:
                    test_files.append(code_file)
                else:
                    files.append(code_file)

            # 5. Find corresponding test files
            for source_file in list(files):
                test_path = detect_test_file_path(source_file.path, tech_stack)
                if test_path and not any(tf.path == test_path for tf in test_files):
                    if len(files) + len(test_files) >= max_files:
                        break
                    if test_path in file_tree:
                        try:
                            test_content = await llm.generate_str(
                                f"Get the content of file '{test_path}' from repository "
                                f"'{qualified_repo}'. Return ONLY the raw file content."
                            )
                            test_line_count = len(test_content.splitlines())
                            if test_line_count <= MAX_LINE_COUNT:
                                test_files.append(CodeFile(
                                    path=test_path, content=test_content,
                                    line_count=test_line_count,
                                    language=self._detect_language(test_path),
                                    is_test=True,
                                ))
                        except Exception:
                            logger.debug("Could not fetch test file: %s", test_path)

            return CodeContext(
                files=files, test_files=test_files, tech_stack=tech_stack,
                repository_name=qualified_repo, file_tree=file_tree,
                skipped_files=skipped_files,
            )

    def _qualify_repo_name(self, repo_name: str) -> str:
        """Return owner/repo (or group/repo) qualified name for the active git provider.

        If the repo_name already contains a '/' it is returned as-is.
        Otherwise the owner/group from settings is prepended.
        The repo slug is normalized (lowercased, spaces → hyphens).
        """
        # Normalize: "Rag Project" → "rag-project"
        import re
        repo_name = repo_name.strip()
        repo_name = re.sub(r'[^a-zA-Z0-9/_.-]', '-', repo_name)
        repo_name = re.sub(r'-+', '-', repo_name).strip('-').lower()

        if "/" in repo_name:
            return repo_name
        provider = self._settings.git_provider
        if provider == "github" and self._settings.github_owner:
            return f"{self._settings.github_owner}/{repo_name}"
        if provider == "gitlab" and self._settings.gitlab_group:
            return f"{self._settings.gitlab_group}/{repo_name}"
        if provider == "bitbucket" and self._settings.bitbucket_workspace:
            return f"{self._settings.bitbucket_workspace}/{repo_name}"
        return repo_name

    @staticmethod
    def _parse_file_list(raw: str) -> list[str]:
        """Parse LLM response into a list of file paths.

        Handles JSON arrays and plain text (one path per line).
        """
        raw = raw.strip()
        # Try JSON array first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if isinstance(p, str) and p.strip()]
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: one path per line
        paths = []
        for line in raw.splitlines():
            line = line.strip().strip('"').strip("'").strip(",").strip()
            if line and not line.startswith("#") and not line.startswith("//"):
                paths.append(line)
        return paths

    @staticmethod
    def _detect_language(path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".sh": "shell",
            ".md": "markdown",
        }
        lower = path.lower()
        for ext, lang in ext_map.items():
            if lower.endswith(ext):
                return lang
        return ""

    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Check if a file path looks like a test file."""
        lower = path.lower().replace("\\", "/")
        basename = lower.rsplit("/", 1)[-1]

        # Python: test_*.py or *_test.py
        if basename.startswith("test_") and basename.endswith(".py"):
            return True
        if basename.endswith("_test.py"):
            return True

        # TypeScript/JavaScript: *.test.ts, *.spec.ts, *.test.js, *.spec.js
        for suffix in (".test.ts", ".spec.ts", ".test.js", ".spec.js", ".test.tsx", ".spec.tsx"):
            if basename.endswith(suffix):
                return True

        # Go: *_test.go
        if basename.endswith("_test.go"):
            return True

        # Rust: in tests/ directory
        if "/tests/" in lower or lower.startswith("tests/"):
            return True

        # Java: *Test.java
        if basename.endswith("test.java"):
            return True

        # __tests__ directory
        if "/__tests__/" in lower:
            return True

        return False
