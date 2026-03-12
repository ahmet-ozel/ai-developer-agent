"""Token budget controller for the AI Developer Agent pipeline.

Provides token estimation and CodeContext trimming to stay within
the configured MAX_CONTEXT_TOKENS limit before sending data to LLM agents.
"""

from __future__ import annotations

from src.pipeline.models import CodeContext, CodeFile, TaskContext


def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple character/4 heuristic.

    This avoids a tiktoken dependency. The ~10-15% margin of error
    is acceptable because the token budget already includes a safety margin.
    """
    return len(text) // 4


def _file_tokens(f: CodeFile) -> int:
    """Estimate tokens for a single CodeFile (path + content)."""
    return estimate_tokens(f.path) + estimate_tokens(f.content)


def _total_tokens(ctx: CodeContext) -> int:
    """Estimate total tokens across all files in a CodeContext."""
    return sum(_file_tokens(f) for f in ctx.files) + sum(
        _file_tokens(f) for f in ctx.test_files
    )


def trim_code_context(
    code_ctx: CodeContext,
    max_tokens: int,
    task_ctx: TaskContext | None = None,
) -> CodeContext:
    """Trim a CodeContext to fit within *max_tokens*.

    Strategy (least-relevant first):
    1. Remove test_files first (largest files removed first).
    2. Then remove source files (largest first).
    3. Return a new CodeContext with the trimmed file lists.

    The result files are always a subset of the original.
    """
    if _total_tokens(code_ctx) <= max_tokens:
        return code_ctx

    # Start with copies sorted largest-first so we can pop from the end
    # (i.e. remove the largest files first).
    remaining_test_files: list[CodeFile] = sorted(
        code_ctx.test_files, key=_file_tokens, reverse=True
    )
    remaining_files: list[CodeFile] = sorted(
        code_ctx.files, key=_file_tokens, reverse=True
    )

    def _current_total() -> int:
        return sum(_file_tokens(f) for f in remaining_files) + sum(
            _file_tokens(f) for f in remaining_test_files
        )

    # Phase 1: drop test files (largest first)
    while remaining_test_files and _current_total() > max_tokens:
        remaining_test_files.pop(0)  # remove largest

    # Phase 2: drop source files (largest first)
    while remaining_files and _current_total() > max_tokens:
        remaining_files.pop(0)  # remove largest

    return CodeContext(
        files=remaining_files,
        test_files=remaining_test_files,
        tech_stack=code_ctx.tech_stack,
        repository_name=code_ctx.repository_name,
        file_tree=code_ctx.file_tree,
        skipped_files=code_ctx.skipped_files,
    )
