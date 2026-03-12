"""Git helper utilities for branch naming, test file detection, tech stack detection, and file filtering."""

from __future__ import annotations

from typing import Optional


def generate_branch_name(
    pattern: str, issue_key: str, suffix: Optional[str] = None
) -> str:
    """Generate a branch name by substituting issue_key into the pattern.

    Args:
        pattern: Branch name pattern with ``{issue_key}`` placeholder,
                 e.g. ``"feature/{issue_key}-ai"``.
        issue_key: Jira issue key, e.g. ``"PROJ-123"``.
        suffix: Optional suffix appended with ``-`` for uniqueness on retry.

    Returns:
        The generated branch name.
    """
    name = pattern.format(issue_key=issue_key)
    if suffix:
        name = f"{name}-{suffix}"
    return name


# ---------------------------------------------------------------------------
# Test file path detection
# ---------------------------------------------------------------------------

_TECH_STACK_TEST_CONVENTIONS: dict[str, str] = {
    "python": "python",
    "typescript": "typescript",
    "javascript": "typescript",  # same convention
    "go": "go",
    "rust": "rust",
    "java": "java",
}


def detect_test_file_path(
    source_path: str, tech_stack: list[str]
) -> Optional[str]:
    """Return the conventional test file path for *source_path*.

    Conventions:
        - Python:     ``src/foo/bar.py``  → ``tests/test_bar.py``
        - TypeScript: ``src/foo/bar.ts``  → ``src/foo/__tests__/bar.test.ts``
        - Go:         ``pkg/foo/bar.go``  → ``pkg/foo/bar_test.go``
        - Rust:       ``src/foo/bar.rs``  → ``tests/test_bar.rs``
        - Java:       ``src/main/java/com/Foo.java`` → ``src/test/java/com/FooTest.java``

    Returns ``None`` when the tech stack is unknown or empty.
    """
    # Determine the primary language from the tech stack
    lang = _resolve_language(tech_stack)
    if lang is None:
        return None

    if lang == "python":
        return _python_test_path(source_path)
    if lang == "typescript":
        return _typescript_test_path(source_path)
    if lang == "go":
        return _go_test_path(source_path)
    if lang == "rust":
        return _rust_test_path(source_path)
    if lang == "java":
        return _java_test_path(source_path)

    return None


def _resolve_language(tech_stack: list[str]) -> Optional[str]:
    """Map the first recognised tech stack entry to a canonical language key."""
    for entry in tech_stack:
        normalised = entry.lower().strip()
        if normalised in _TECH_STACK_TEST_CONVENTIONS:
            return _TECH_STACK_TEST_CONVENTIONS[normalised]
    return None


def _python_test_path(source_path: str) -> str:
    parts = _posix_split(source_path)
    basename = parts[-1]
    name, _ = _split_ext(basename)
    return f"tests/test_{name}.py"


def _typescript_test_path(source_path: str) -> str:
    parts = _posix_split(source_path)
    dirname = "/".join(parts[:-1]) if len(parts) > 1 else ""
    basename = parts[-1]
    name, ext = _split_ext(basename)
    return f"{dirname}/__tests__/{name}.test{ext}" if dirname else f"__tests__/{name}.test{ext}"


def _go_test_path(source_path: str) -> str:
    parts = _posix_split(source_path)
    dirname = "/".join(parts[:-1]) if len(parts) > 1 else ""
    basename = parts[-1]
    name, ext = _split_ext(basename)
    return f"{dirname}/{name}_test{ext}" if dirname else f"{name}_test{ext}"


def _rust_test_path(source_path: str) -> str:
    parts = _posix_split(source_path)
    basename = parts[-1]
    name, _ = _split_ext(basename)
    return f"tests/test_{name}.rs"


def _java_test_path(source_path: str) -> str:
    # src/main/java/com/Foo.java → src/test/java/com/FooTest.java
    path = source_path.replace("src/main/", "src/test/", 1)
    parts = _posix_split(path)
    dirname = "/".join(parts[:-1]) if len(parts) > 1 else ""
    basename = parts[-1]
    name, ext = _split_ext(basename)
    return f"{dirname}/{name}Test{ext}" if dirname else f"{name}Test{ext}"


def _posix_split(path: str) -> list[str]:
    """Split a path using forward slashes (platform-independent)."""
    return path.replace("\\", "/").split("/")


def _split_ext(basename: str) -> tuple[str, str]:
    """Split filename into (name, ext) without os.path dependency."""
    dot = basename.rfind(".")
    if dot <= 0:
        return basename, ""
    return basename[:dot], basename[dot:]


# ---------------------------------------------------------------------------
# Tech stack detection
# ---------------------------------------------------------------------------

_CONFIG_FILE_TO_STACK: dict[str, str] = {
    "package.json": "typescript",
    "pyproject.toml": "python",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
}


def detect_tech_stack(file_tree: str) -> list[str]:
    """Detect technology stacks from a repository file tree string.

    Scans *file_tree* for known configuration file names and returns a
    deduplicated list of detected stacks.

    Args:
        file_tree: Newline-separated file tree output (e.g. from ``git ls-tree``).

    Returns:
        List of detected tech stack identifiers (e.g. ``["python", "typescript"]``).
    """
    detected: list[str] = []
    for config_file, stack in _CONFIG_FILE_TO_STACK.items():
        if config_file in file_tree and stack not in detected:
            detected.append(stack)
    return detected


# ---------------------------------------------------------------------------
# Skippable file detection
# ---------------------------------------------------------------------------

_BINARY_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".pyc", ".pyo", ".so", ".dll"})
_LOCK_FILES = frozenset({"package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock", "pnpm-lock.yaml"})
_GENERATED_EXTENSIONS = frozenset({".min.js", ".min.css", ".map"})
_GENERATED_DIRS = ("dist/", "build/")


def is_skippable_file(path: str) -> tuple[bool, str]:
    """Determine whether a file should be skipped during code analysis.

    Returns:
        A tuple ``(should_skip, reason)`` where *reason* is one of
        ``"binary"``, ``"lock_file"``, ``"generated"``, or ``""`` (not skippable).
    """
    basename = path.replace("\\", "/").rsplit("/", 1)[-1]
    dot = basename.rfind(".")
    ext = basename[dot:] if dot > 0 else ""

    # Binary check
    if ext.lower() in _BINARY_EXTENSIONS:
        return True, "binary"

    # Lock file check
    if basename in _LOCK_FILES:
        return True, "lock_file"

    # Generated file check — by extension
    # Need to check compound extensions like .min.js
    if _has_generated_extension(path):
        return True, "generated"

    # Generated file check — by directory
    normalised = path.replace("\\", "/")
    for gen_dir in _GENERATED_DIRS:
        if normalised.startswith(gen_dir) or f"/{gen_dir}" in normalised:
            return True, "generated"

    return False, ""


def _has_generated_extension(path: str) -> bool:
    """Check for compound generated extensions like ``.min.js``."""
    lower = path.lower()
    return any(lower.endswith(ext) for ext in _GENERATED_EXTENSIONS)
