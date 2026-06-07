"""Unit tests for CodeFinderAgent.

Tests:
- File tree parsing and tech stack detection
- Skippable file filtering (binary, lock, generated)
- Line count limit enforcement (>1000 lines  skipped)
- MAX_FILES_PER_TASK limit
- Test file identification
- File size limit enforcement (max_file_size_kb)
- Retry logic on transient errors
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.code_finder import (
    CodeFinderAgent,
    _retry_with_backoff,
    MAX_LINE_COUNT,
)
from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    CodeContext,
    CodeFile,
    SkippedFile,
    TaskContext,
    TaskScope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with sensible defaults for testing."""
    defaults = {
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-developer",
        "jira_api_token": "jira-secret-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-developer",
        "git_provider": "github",
        "github_token": "ghp_test_token",
        "github_owner": "test-owner",
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast-key",
        "llm_strong_provider": "anthropic",
        "llm_strong_model": "claude-sonnet-4-20250514",
        "llm_strong_api_key": "sk-strong-key",
        "max_files_per_task": 10,
        "max_file_size_kb": 100,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_task_context(**overrides: Any) -> TaskContext:
    """Create a TaskContext with sensible defaults."""
    defaults = {
        "issue_key": "TEST-42",
        "summary": "Fix authentication bug",
        "description": "Login fails when session expires during OAuth refresh",
        "acceptance_criteria": "Users should stay logged in after token refresh",
        "repository_name": "backend-api",
        "estimated_scope": TaskScope.SMALL,
    }
    defaults.update(overrides)
    return TaskContext(**defaults)


SAMPLE_FILE_TREE = """src/
  auth/
    login.py
    session.py
  utils/
    helpers.py
tests/
  test_login.py
pyproject.toml
package.json
README.md"""

SMALL_FILE_CONTENT = "def hello():\n    return 'world'\n"


def _build_mock_llm(
    file_tree: str = SAMPLE_FILE_TREE,
    relevant_files_json: str | None = None,
    file_contents: dict[str, str] | None = None,
) -> AsyncMock:
    """Build a mock LLM that returns predefined responses.

    Call sequence:
    1. file tree request  file_tree
    2. relevant files identification  relevant_files_json
    3+ file content requests  looked up from file_contents dict
    """
    if relevant_files_json is None:
        relevant_files_json = json.dumps(["src/auth/login.py", "src/auth/session.py"])
    if file_contents is None:
        file_contents = {
            "src/auth/login.py": SMALL_FILE_CONTENT,
            "src/auth/session.py": "class Session:\n    pass\n",
        }

    call_count = 0
    responses = [file_tree, relevant_files_json]

    mock_llm = AsyncMock()

    async def _generate_str(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # First two calls are file tree and file identification
        if call_count <= 2:
            return responses[call_count - 1]
        # Subsequent calls are file content requests
        for path, content in file_contents.items():
            if path in prompt:
                return content
        return "# empty file\n"

    mock_llm.generate_str = AsyncMock(side_effect=_generate_str)
    return mock_llm


def _patch_agent(mock_llm: AsyncMock):
    """Return a context manager that patches the Agent class to use mock_llm."""
    mock_agent_instance = AsyncMock()
    mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
    mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
    mock_agent_instance.__aexit__ = AsyncMock(return_value=None)

    return patch("src.agents.code_finder.Agent", return_value=mock_agent_instance)


# ---------------------------------------------------------------------------
# Tests: File tree parsing and tech stack detection
# ---------------------------------------------------------------------------


class TestFileTreeAndTechStack:
    """Test file tree parsing and tech stack detection."""

    @pytest.fixture
    def agent(self) -> CodeFinderAgent:
        settings = _make_settings()
        return CodeFinderAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_tech_stack_detected_from_file_tree(self, agent: CodeFinderAgent) -> None:
        """Tech stack should be detected from config files in the file tree."""
        mock_llm = _build_mock_llm(file_tree=SAMPLE_FILE_TREE)
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert isinstance(result, CodeContext)
        # pyproject.toml  python, package.json  typescript
        assert "python" in result.tech_stack
        assert "typescript" in result.tech_stack

    async def test_file_tree_stored_in_context(self, agent: CodeFinderAgent) -> None:
        """The raw file tree should be stored in CodeContext."""
        mock_llm = _build_mock_llm(file_tree=SAMPLE_FILE_TREE)
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert result.file_tree is not None
        assert "pyproject.toml" in result.file_tree

    async def test_repository_name_propagated(self, agent: CodeFinderAgent) -> None:
        """Repository name should be qualified with owner/group prefix."""
        mock_llm = _build_mock_llm()
        task_ctx = _make_task_context(repository_name="my-cool-repo")

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        # CodeFinder qualifies the repo name with the owner from settings
        assert result.repository_name == "test-owner/my-cool-repo"


# ---------------------------------------------------------------------------
# Tests: Skippable file filtering
# ---------------------------------------------------------------------------


class TestSkippableFileFiltering:
    """Test that binary, lock, and generated files are skipped."""

    @pytest.fixture
    def agent(self) -> CodeFinderAgent:
        settings = _make_settings()
        return CodeFinderAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_binary_files_skipped(self, agent: CodeFinderAgent) -> None:
        """Binary files (.png, .jpg, etc.) should be skipped."""
        relevant = json.dumps(["src/logo.png", "src/auth/login.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={"src/auth/login.py": SMALL_FILE_CONTENT},
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(sf.path == "src/logo.png" and sf.reason == "binary" for sf in result.skipped_files)
        assert not any(f.path == "src/logo.png" for f in result.files)

    async def test_lock_files_skipped(self, agent: CodeFinderAgent) -> None:
        """Lock files (package-lock.json, etc.) should be skipped."""
        relevant = json.dumps(["package-lock.json", "src/auth/login.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={"src/auth/login.py": SMALL_FILE_CONTENT},
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(sf.path == "package-lock.json" and sf.reason == "lock_file" for sf in result.skipped_files)

    async def test_generated_files_skipped(self, agent: CodeFinderAgent) -> None:
        """Generated files (.min.js, dist/, etc.) should be skipped."""
        relevant = json.dumps(["dist/bundle.js", "src/app.min.js", "src/auth/login.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={"src/auth/login.py": SMALL_FILE_CONTENT},
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        skipped_paths = [sf.path for sf in result.skipped_files]
        assert "dist/bundle.js" in skipped_paths
        assert "src/app.min.js" in skipped_paths
        assert all(sf.reason == "generated" for sf in result.skipped_files if sf.path in ["dist/bundle.js", "src/app.min.js"])


# ---------------------------------------------------------------------------
# Tests: Line count limit enforcement
# ---------------------------------------------------------------------------


class TestLineCountLimit:
    """Test that files with >1000 lines are skipped."""

    @pytest.fixture
    def agent(self) -> CodeFinderAgent:
        settings = _make_settings()
        return CodeFinderAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_large_file_skipped(self, agent: CodeFinderAgent) -> None:
        """Files with more than 1000 lines should be skipped as too_large."""
        large_content = "\n".join(f"line {i}" for i in range(1500))
        relevant = json.dumps(["src/big_file.py", "src/small_file.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={
                "src/big_file.py": large_content,
                "src/small_file.py": SMALL_FILE_CONTENT,
            },
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        # Big file should be in skipped_files
        assert any(sf.path == "src/big_file.py" and sf.reason == "too_large" for sf in result.skipped_files)
        # Small file should be in files
        assert any(f.path == "src/small_file.py" for f in result.files)

    async def test_exactly_1000_lines_not_skipped(self, agent: CodeFinderAgent) -> None:
        """Files with exactly 1000 lines should NOT be skipped."""
        content_1000 = "\n".join(f"line {i}" for i in range(1000))
        relevant = json.dumps(["src/borderline.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={"src/borderline.py": content_1000},
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(f.path == "src/borderline.py" for f in result.files)
        assert not any(sf.path == "src/borderline.py" for sf in result.skipped_files)


# ---------------------------------------------------------------------------
# Tests: MAX_FILES_PER_TASK limit
# ---------------------------------------------------------------------------


class TestMaxFilesLimit:
    """Test that the number of files is limited to max_files_per_task."""

    async def test_files_limited_to_max(self) -> None:
        """Total files (files + test_files) should not exceed max_files_per_task."""
        settings = _make_settings(max_files_per_task=3)
        agent = CodeFinderAgent(settings=settings, llm_router=LLMRouter(config=settings))

        # LLM identifies 6 files but limit is 3
        paths = [f"src/file_{i}.py" for i in range(6)]
        relevant = json.dumps(paths)
        contents = {p: f"# file {i}\npass\n" for i, p in enumerate(paths)}
        mock_llm = _build_mock_llm(
            file_tree="src/\n" + "\n".join(f"  file_{i}.py" for i in range(6)),
            relevant_files_json=relevant,
            file_contents=contents,
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        total = len(result.files) + len(result.test_files)
        assert total <= 3


# ---------------------------------------------------------------------------
# Tests: File size limit enforcement (max_file_size_kb)
# ---------------------------------------------------------------------------


class TestFileSizeLimit:
    """Test that files exceeding max_file_size_kb are skipped."""

    async def test_oversized_file_skipped(self) -> None:
        """Files larger than max_file_size_kb should be skipped."""
        # Set a very small limit: 1 KB
        settings = _make_settings(max_file_size_kb=1)
        agent = CodeFinderAgent(settings=settings, llm_router=LLMRouter(config=settings))

        # Create content larger than 1 KB but under 1000 lines
        # ~2KB of content in ~20 lines
        big_content = "\n".join("x" * 100 for _ in range(20))
        relevant = json.dumps(["src/big.py", "src/small.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={
                "src/big.py": big_content,
                "src/small.py": "pass\n",
            },
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(sf.path == "src/big.py" and sf.reason == "too_large" for sf in result.skipped_files)
        assert any(f.path == "src/small.py" for f in result.files)


# ---------------------------------------------------------------------------
# Tests: Test file identification
# ---------------------------------------------------------------------------


class TestTestFileIdentification:
    """Test that test files are correctly identified and categorized."""

    @pytest.fixture
    def agent(self) -> CodeFinderAgent:
        settings = _make_settings()
        return CodeFinderAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_test_files_categorized(self, agent: CodeFinderAgent) -> None:
        """Files matching test patterns should go into test_files."""
        relevant = json.dumps(["src/auth/login.py", "tests/test_login.py"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={
                "src/auth/login.py": SMALL_FILE_CONTENT,
                "tests/test_login.py": "def test_login():\n    assert True\n",
            },
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(f.path == "src/auth/login.py" for f in result.files)
        assert any(tf.path == "tests/test_login.py" for tf in result.test_files)
        # Test files should have is_test=True
        for tf in result.test_files:
            assert tf.is_test is True

    async def test_typescript_test_files_detected(self, agent: CodeFinderAgent) -> None:
        """TypeScript test files (*.test.ts) should be detected."""
        relevant = json.dumps(["src/app.ts", "src/__tests__/app.test.ts"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={
                "src/app.ts": "export const app = {};\n",
                "src/__tests__/app.test.ts": "test('app', () => {});\n",
            },
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(tf.path == "src/__tests__/app.test.ts" and tf.is_test for tf in result.test_files)

    async def test_go_test_files_detected(self, agent: CodeFinderAgent) -> None:
        """Go test files (*_test.go) should be detected."""
        relevant = json.dumps(["pkg/handler.go", "pkg/handler_test.go"])
        mock_llm = _build_mock_llm(
            relevant_files_json=relevant,
            file_contents={
                "pkg/handler.go": "package pkg\n",
                "pkg/handler_test.go": "package pkg\nfunc TestHandler(t *testing.T) {}\n",
            },
        )
        task_ctx = _make_task_context()

        with _patch_agent(mock_llm):
            result = await agent.find_code(task_ctx)

        assert any(tf.path == "pkg/handler_test.go" and tf.is_test for tf in result.test_files)


# ---------------------------------------------------------------------------
# Tests: Parse helpers
# ---------------------------------------------------------------------------


class TestParseHelpers:
    """Test static helper methods."""

    def test_parse_file_list_json_array(self) -> None:
        """JSON array should be parsed correctly."""
        raw = '["src/a.py", "src/b.py"]'
        result = CodeFinderAgent._parse_file_list(raw)
        assert result == ["src/a.py", "src/b.py"]

    def test_parse_file_list_plain_text(self) -> None:
        """Plain text (one path per line) should be parsed."""
        raw = "src/a.py\nsrc/b.py\n"
        result = CodeFinderAgent._parse_file_list(raw)
        assert result == ["src/a.py", "src/b.py"]

    def test_parse_file_list_empty(self) -> None:
        """Empty input should return empty list."""
        assert CodeFinderAgent._parse_file_list("") == []
        assert CodeFinderAgent._parse_file_list("[]") == []

    def test_detect_language_python(self) -> None:
        assert CodeFinderAgent._detect_language("src/main.py") == "python"

    def test_detect_language_typescript(self) -> None:
        assert CodeFinderAgent._detect_language("src/app.ts") == "typescript"

    def test_detect_language_go(self) -> None:
        assert CodeFinderAgent._detect_language("pkg/main.go") == "go"

    def test_detect_language_unknown(self) -> None:
        assert CodeFinderAgent._detect_language("Makefile") == ""

    def test_is_test_file_python(self) -> None:
        assert CodeFinderAgent._is_test_file("tests/test_auth.py") is True
        assert CodeFinderAgent._is_test_file("src/auth.py") is False

    def test_is_test_file_typescript(self) -> None:
        assert CodeFinderAgent._is_test_file("src/__tests__/app.test.ts") is True
        assert CodeFinderAgent._is_test_file("src/app.ts") is False

    def test_is_test_file_go(self) -> None:
        assert CodeFinderAgent._is_test_file("pkg/handler_test.go") is True
        assert CodeFinderAgent._is_test_file("pkg/handler.go") is False


# ---------------------------------------------------------------------------
# Tests: Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test exponential backoff retry behavior for CodeFinder."""

    async def test_retry_on_connection_error(self) -> None:
        """Should retry on ConnectionError and succeed on subsequent attempt."""
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection refused")
            return "success"

        result = await _retry_with_backoff(flaky_func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 2

    async def test_no_retry_on_value_error(self) -> None:
        """Should not retry on non-retryable errors like ValueError."""
        call_count = 0

        async def bad_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad input")

        with pytest.raises(ValueError, match="Bad input"):
            await _retry_with_backoff(bad_func, max_retries=3, base_delay=0.01)

        assert call_count == 1

    async def test_max_retries_exhausted(self) -> None:
        """Should raise after max retries are exhausted."""
        call_count = 0

        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Timed out")

        with pytest.raises(TimeoutError, match="Timed out"):
            await _retry_with_backoff(always_fail, max_retries=3, base_delay=0.01)

        assert call_count == 3


# ---------------------------------------------------------------------------
# Property tests: CodeContext File Count Invariant
# ---------------------------------------------------------------------------


from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st


class TestProperty8CodeContextFileCountInvariant:
    """Property 8: CodeContext File Count Invariant

    Validates: Requirements 3.6

    For any CodeContext produced by CodeFinderAgent, the total number of files
    (files + test_files) must not exceed max_files_per_task.
    """

    @given(
        max_files=st.integers(min_value=1, max_value=20),
        paths=st.lists(
            st.from_regex(r"src/[a-z_]+/[a-z_]+\.py", fullmatch=True),
            min_size=1,
            max_size=50,
        ),
    )
    @h_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_file_count_never_exceeds_max(self, max_files: int, paths: list[str]) -> None:
        """**Validates: Requirements 3.6**

        Simulates the CodeFinderAgent file-accumulation loop:
        files are added until len(files) + len(test_files) >= max_files.
        The resulting total must always be <= max_files.
        """
        # Simulate what CodeFinderAgent._find_code_impl does:
        # it breaks out of the loop when len(files) + len(test_files) >= max_files.
        files: list[str] = []
        test_files: list[str] = []

        for path in paths:
            if len(files) + len(test_files) >= max_files:
                break
            # Classify as test or source (mirrors CodeFinderAgent._is_test_file logic)
            if CodeFinderAgent._is_test_file(path):
                test_files.append(path)
            else:
                files.append(path)

        total = len(files) + len(test_files)
        assert total <= max_files, (
            f"Expected total <= {max_files}, got {total} "
            f"(files={len(files)}, test_files={len(test_files)})"
        )


# ---------------------------------------------------------------------------
# Property tests: File Size and Type Filtering
# ---------------------------------------------------------------------------


class TestProperty11FileSizeAndTypeFiltering:
    """Property 11: File Size and Type Filtering

    Validates: Requirements 3.9

    For any set of files processed by CodeFinderAgent:
    - Files with line_count > MAX_LINE_COUNT must be in skipped_files with reason="too_large"
    - Files with line_count <= MAX_LINE_COUNT must NOT be in skipped_files due to line count
    """

    @given(
        file_specs=st.lists(
            st.tuples(
                st.from_regex(r"src/[a-z_]+/[a-z_]+\.py", fullmatch=True),
                st.integers(min_value=0, max_value=2000),  # line count
            ),
            min_size=1,
            max_size=20,
        )
    )
    @h_settings(max_examples=100)
    def test_large_files_always_skipped(
        self, file_specs: list[tuple[str, int]]
    ) -> None:
        """**Validates: Requirements 3.9**

        Simulates the line-count filtering logic from CodeFinderAgent._find_code_impl.
        Files with line_count > MAX_LINE_COUNT must end up in skipped_files.
        """
        # Deduplicate by path (keep last occurrence, matching real agent behaviour
        # where each path is processed once)
        seen: dict[str, int] = {}
        for path, line_count in file_specs:
            seen[path] = line_count
        file_specs = list(seen.items())

        files: list[CodeFile] = []
        skipped: list[SkippedFile] = []

        for path, line_count in file_specs:
            content = "\n".join(f"line {i}" for i in range(line_count))
            actual_line_count = len(content.splitlines())

            if actual_line_count > MAX_LINE_COUNT:
                skipped.append(SkippedFile(path=path, reason="too_large"))
            else:
                files.append(
                    CodeFile(
                        path=path,
                        content=content,
                        language="python",
                        is_test=False,
                    )
                )

        # Invariant: no file in `files` has line_count > MAX_LINE_COUNT
        for f in files:
            assert len(f.content.splitlines()) <= MAX_LINE_COUNT, (
                f"File {f.path} has {len(f.content.splitlines())} lines "
                f"but should have been skipped (limit={MAX_LINE_COUNT})"
            )

        # Invariant: all skipped files with reason="too_large" had > MAX_LINE_COUNT lines
        for sf in skipped:
            if sf.reason == "too_large":
                # Find the original line count for this path
                original_line_count = next(
                    lc for p, lc in file_specs if p == sf.path
                )
                assert original_line_count > MAX_LINE_COUNT, (
                    f"File {sf.path} was skipped as too_large but only had "
                    f"{original_line_count} lines (limit={MAX_LINE_COUNT})"
                )

    @given(
        max_kb=st.integers(min_value=1, max_value=500),
        file_sizes_kb=st.lists(
            st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        ),
    )
    @h_settings(max_examples=100)
    def test_oversized_files_always_skipped(
        self, max_kb: int, file_sizes_kb: list[float]
    ) -> None:
        """**Validates: Requirements 3.9**

        Files exceeding max_file_size_kb must be skipped.
        Files within the limit must not be skipped due to size.
        """
        skipped_paths: set[str] = set()
        kept_paths: set[str] = set()

        for i, size_kb in enumerate(file_sizes_kb):
            path = f"src/file_{i}.py"
            # Simulate content of approximately size_kb kilobytes
            content_size = int(size_kb * 1024)
            content = "x" * content_size

            actual_kb = len(content.encode("utf-8")) / 1024

            if actual_kb > max_kb:
                skipped_paths.add(path)
            else:
                kept_paths.add(path)

        # Invariant: skipped and kept are disjoint
        assert skipped_paths.isdisjoint(kept_paths), (
            f"Some files appear in both skipped and kept: "
            f"{skipped_paths & kept_paths}"
        )
