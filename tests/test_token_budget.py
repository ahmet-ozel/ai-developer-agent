"""Unit tests for src/pipeline/token_budget.py.

Covers:
- estimate_tokens basic calculation
- trim_code_context within budget returns unchanged
- trim_code_context over budget removes files
- Result files are subset of original
"""

from __future__ import annotations

from src.pipeline.models import CodeContext, CodeFile, TaskContext, TaskScope
from src.pipeline.token_budget import (
    estimate_tokens,
    trim_code_context,
    _file_tokens,
    _total_tokens,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_code_file(path: str, content: str, is_test: bool = False) -> CodeFile:
    return CodeFile(path=path, content=content, is_test=is_test)


def _make_context(
    files: list[CodeFile] | None = None,
    test_files: list[CodeFile] | None = None,
) -> CodeContext:
    return CodeContext(
        files=files or [],
        test_files=test_files or [],
        repository_name="test-repo",
    )


def _make_task_ctx() -> TaskContext:
    return TaskContext(
        issue_key="TEST-1",
        summary="Test task",
        description="A test task",
        repository_name="test-repo",
        estimated_scope=TaskScope.SMALL,
    )


# =========================================================================
# estimate_tokens
# =========================================================================


class TestEstimateTokens:
    def test_basic_calculation(self):
        assert estimate_tokens("abcd") == 1  # 4 chars -> 1 token

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("ab") == 0  # 2 // 4 == 0

    def test_longer_string(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25  # 100 // 4

    def test_exact_multiple(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


# =========================================================================
# trim_code_context - within budget
# =========================================================================


class TestTrimCodeContextWithinBudget:
    def test_returns_unchanged_when_within_budget(self):
        f1 = _make_code_file("a.py", "x" * 40)  # ~10 tokens
        ctx = _make_context(files=[f1])
        result = trim_code_context(ctx, max_tokens=10000)

        assert len(result.files) == len(ctx.files)
        assert result.files[0].path == "a.py"
        assert result.test_files == []

    def test_returns_unchanged_with_test_files(self):
        f1 = _make_code_file("a.py", "x" * 40)
        t1 = _make_code_file("test_a.py", "y" * 40, is_test=True)
        ctx = _make_context(files=[f1], test_files=[t1])
        result = trim_code_context(ctx, max_tokens=10000)

        assert len(result.files) == 1
        assert len(result.test_files) == 1

    def test_empty_context_within_budget(self):
        ctx = _make_context()
        result = trim_code_context(ctx, max_tokens=100)
        assert result.files == []
        assert result.test_files == []


# =========================================================================
# trim_code_context - over budget
# =========================================================================


class TestTrimCodeContextOverBudget:
    def test_removes_test_files_first(self):
        f1 = _make_code_file("a.py", "x" * 40)  # ~10 tokens
        t1 = _make_code_file("test_a.py", "y" * 400)  # ~100 tokens
        ctx = _make_context(files=[f1], test_files=[t1])

        # Budget allows only ~20 tokens - test file must go
        result = trim_code_context(ctx, max_tokens=20)

        assert len(result.files) == 1
        assert result.files[0].path == "a.py"
        assert len(result.test_files) == 0

    def test_removes_largest_test_file_first(self):
        f1 = _make_code_file("a.py", "x" * 40)  # ~10 tokens
        t_small = _make_code_file("test_small.py", "s" * 40)  # ~10 tokens
        t_large = _make_code_file("test_large.py", "L" * 400)  # ~100 tokens
        ctx = _make_context(files=[f1], test_files=[t_small, t_large])

        # Budget allows ~30 tokens - large test file removed, small kept
        result = trim_code_context(ctx, max_tokens=30)

        assert len(result.files) == 1
        assert len(result.test_files) == 1
        assert result.test_files[0].path == "test_small.py"

    def test_removes_source_files_after_test_files(self):
        f_large = _make_code_file("big.py", "B" * 4000)  # ~1000 tokens
        f_small = _make_code_file("small.py", "s" * 40)  # ~10 tokens
        ctx = _make_context(files=[f_large, f_small])

        # Budget allows ~20 tokens - large source file removed
        result = trim_code_context(ctx, max_tokens=20)

        assert len(result.files) == 1
        assert result.files[0].path == "small.py"

    def test_removes_all_files_if_needed(self):
        f1 = _make_code_file("a.py", "x" * 400)
        t1 = _make_code_file("test_a.py", "y" * 400)
        ctx = _make_context(files=[f1], test_files=[t1])

        result = trim_code_context(ctx, max_tokens=0)

        assert result.files == []
        assert result.test_files == []

    def test_preserves_metadata(self):
        f1 = _make_code_file("a.py", "x" * 4000)
        ctx = CodeContext(
            files=[f1],
            test_files=[],
            tech_stack=["python"],
            repository_name="my-repo",
            file_tree="a.py\nb.py",
        )
        result = trim_code_context(ctx, max_tokens=5)

        assert result.repository_name == "my-repo"
        assert result.tech_stack == ["python"]
        assert result.file_tree == "a.py\nb.py"


# =========================================================================
# Result files are subset of original
# =========================================================================


class TestResultSubset:
    def test_result_files_subset_of_original(self):
        files = [
            _make_code_file(f"file_{i}.py", "c" * (100 * (i + 1)))
            for i in range(5)
        ]
        test_files = [
            _make_code_file(f"test_{i}.py", "t" * (100 * (i + 1)), is_test=True)
            for i in range(3)
        ]
        ctx = _make_context(files=files, test_files=test_files)

        # Tight budget - some files will be removed
        result = trim_code_context(ctx, max_tokens=200)

        original_paths = {f.path for f in ctx.files}
        result_paths = {f.path for f in result.files}
        assert result_paths.issubset(original_paths)

        original_test_paths = {f.path for f in ctx.test_files}
        result_test_paths = {f.path for f in result.test_files}
        assert result_test_paths.issubset(original_test_paths)

    def test_with_task_context_param(self):
        """task_ctx parameter is accepted without error."""
        f1 = _make_code_file("a.py", "x" * 40)
        ctx = _make_context(files=[f1])
        task_ctx = _make_task_ctx()

        result = trim_code_context(ctx, max_tokens=10000, task_ctx=task_ctx)
        assert len(result.files) == 1


# ---------------------------------------------------------------------------
# Property tests: Token Budget Enforcement
# ---------------------------------------------------------------------------


from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


class TestProperty30TokenBudgetEnforcement:
    """Property 30: Token Budget Enforcement

    Validates: Requirements 15.1, 15.2

    After trim_code_context:
    1. Total tokens <= max_tokens (when possible)
    2. Result files are a subset of original files
    3. Result test_files are a subset of original test_files
    """

    @given(
        file_contents=st.lists(
            st.text(min_size=0, max_size=400),
            min_size=0,
            max_size=10,
        ),
        test_contents=st.lists(
            st.text(min_size=0, max_size=400),
            min_size=0,
            max_size=5,
        ),
        max_tokens=st.integers(min_value=0, max_value=10000),
    )
    @h_settings(max_examples=100)
    def test_result_within_budget(
        self,
        file_contents: list[str],
        test_contents: list[str],
        max_tokens: int,
    ) -> None:
        """**Validates: Requirements 15.1**

        After trimming, total tokens must be <= max_tokens.
        Exception: if even a single file exceeds max_tokens, we can't trim further.
        """
        files = [
            CodeFile(path=f"src/file_{i}.py", content=c, is_test=False)
            for i, c in enumerate(file_contents)
        ]
        test_files = [
            CodeFile(path=f"tests/test_{i}.py", content=c, is_test=True)
            for i, c in enumerate(test_contents)
        ]
        ctx = CodeContext(files=files, test_files=test_files, repository_name="test")

        result = trim_code_context(ctx, max_tokens=max_tokens)

        # Calculate actual token usage
        total = sum(_file_tokens(f) for f in result.files) + sum(
            _file_tokens(f) for f in result.test_files
        )

        # The result must be within budget
        assert total <= max_tokens, (
            f"Expected total tokens ({total}) <= max_tokens ({max_tokens}) "
            f"after trimming. Got {len(result.files)} files and "
            f"{len(result.test_files)} test files."
        )

    @given(
        file_contents=st.lists(
            st.text(min_size=1, max_size=400),
            min_size=1,
            max_size=10,
        ),
        test_contents=st.lists(
            st.text(min_size=1, max_size=400),
            min_size=0,
            max_size=5,
        ),
        max_tokens=st.integers(min_value=0, max_value=10000),
    )
    @h_settings(max_examples=100)
    def test_result_files_subset_of_original(
        self,
        file_contents: list[str],
        test_contents: list[str],
        max_tokens: int,
    ) -> None:
        """**Validates: Requirements 15.2**

        Result files must always be a subset of original files.
        No new files should be introduced by trimming.
        """
        files = [
            CodeFile(path=f"src/file_{i}.py", content=c, is_test=False)
            for i, c in enumerate(file_contents)
        ]
        test_files = [
            CodeFile(path=f"tests/test_{i}.py", content=c, is_test=True)
            for i, c in enumerate(test_contents)
        ]
        ctx = CodeContext(files=files, test_files=test_files, repository_name="test")

        result = trim_code_context(ctx, max_tokens=max_tokens)

        # Result files must be a subset of original files
        original_paths = {f.path for f in ctx.files}
        result_paths = {f.path for f in result.files}
        assert result_paths.issubset(original_paths), (
            f"Result files {result_paths} are not a subset of original {original_paths}"
        )

        # Result test_files must be a subset of original test_files
        original_test_paths = {f.path for f in ctx.test_files}
        result_test_paths = {f.path for f in result.test_files}
        assert result_test_paths.issubset(original_test_paths), (
            f"Result test_files {result_test_paths} are not a subset of original {original_test_paths}"
        )

    @given(
        file_contents=st.lists(
            st.text(min_size=1, max_size=400),
            min_size=1,
            max_size=10,
        ),
        test_contents=st.lists(
            st.text(min_size=1, max_size=400),
            min_size=0,
            max_size=5,
        ),
    )
    @h_settings(max_examples=100)
    def test_large_budget_returns_all_files(
        self,
        file_contents: list[str],
        test_contents: list[str],
    ) -> None:
        """**Validates: Requirements 15.1, 15.2**

        With a very large budget, all files should be preserved.
        """
        files = [
            CodeFile(path=f"src/file_{i}.py", content=c, is_test=False)
            for i, c in enumerate(file_contents)
        ]
        test_files = [
            CodeFile(path=f"tests/test_{i}.py", content=c, is_test=True)
            for i, c in enumerate(test_contents)
        ]
        ctx = CodeContext(files=files, test_files=test_files, repository_name="test")

        # Use a very large budget that should never be exceeded
        result = trim_code_context(ctx, max_tokens=10_000_000)

        assert len(result.files) == len(ctx.files), (
            f"Expected all {len(ctx.files)} files preserved with large budget, "
            f"got {len(result.files)}"
        )
        assert len(result.test_files) == len(ctx.test_files), (
            f"Expected all {len(ctx.test_files)} test files preserved with large budget, "
            f"got {len(result.test_files)}"
        )
