"""Unit tests for src/utils/git_helpers.py.

Covers:
- generate_branch_name: pattern substitution, suffix handling
- detect_test_file_path: Python, TypeScript, Go, Rust, Java conventions
- detect_tech_stack: config file detection
- is_skippable_file: binary, lock, generated, and normal files
"""

from __future__ import annotations

import pytest

from src.utils.git_helpers import (
    detect_tech_stack,
    detect_test_file_path,
    generate_branch_name,
    is_skippable_file,
)


# =========================================================================
# generate_branch_name
# =========================================================================


class TestGenerateBranchName:
    def test_basic_substitution(self):
        result = generate_branch_name("feature/{issue_key}-ai", "PROJ-123")
        assert result == "feature/PROJ-123-ai"

    def test_with_suffix(self):
        result = generate_branch_name(
            "feature/{issue_key}-ai", "PROJ-123", suffix="1700000000"
        )
        assert result == "feature/PROJ-123-ai-1700000000"

    def test_no_suffix(self):
        result = generate_branch_name("fix/{issue_key}", "BUG-42")
        assert result == "fix/BUG-42"

    def test_suffix_none_explicitly(self):
        result = generate_branch_name("feature/{issue_key}", "X-1", suffix=None)
        assert result == "feature/X-1"

    def test_empty_suffix_not_appended(self):
        result = generate_branch_name("feature/{issue_key}", "X-1", suffix="")
        # Empty string is falsy, so no suffix appended
        assert result == "feature/X-1"

    def test_pattern_preserves_prefix(self):
        result = generate_branch_name("hotfix/{issue_key}-auto", "SEC-99")
        assert result.startswith("hotfix/")


# =========================================================================
# detect_test_file_path
# =========================================================================


class TestDetectTestFilePath:
    # --- Python ---
    def test_python_convention(self):
        result = detect_test_file_path("src/foo/bar.py", ["python"])
        assert result == "tests/test_bar.py"

    def test_python_nested_path(self):
        result = detect_test_file_path("src/auth/oauth.py", ["python"])
        assert result == "tests/test_oauth.py"

    # --- TypeScript ---
    def test_typescript_convention(self):
        result = detect_test_file_path("src/foo/bar.ts", ["typescript"])
        assert result == "src/foo/__tests__/bar.test.ts"

    def test_javascript_uses_ts_convention(self):
        result = detect_test_file_path("src/utils/helper.js", ["javascript"])
        assert result == "src/utils/__tests__/helper.test.js"

    # --- Go ---
    def test_go_convention(self):
        result = detect_test_file_path("pkg/foo/bar.go", ["go"])
        assert result == "pkg/foo/bar_test.go"

    # --- Rust ---
    def test_rust_convention(self):
        result = detect_test_file_path("src/foo/bar.rs", ["rust"])
        assert result == "tests/test_bar.rs"

    # --- Java ---
    def test_java_convention(self):
        result = detect_test_file_path(
            "src/main/java/com/Foo.java", ["java"]
        )
        assert result == "src/test/java/com/FooTest.java"

    # --- Unknown / empty ---
    def test_unknown_stack_returns_none(self):
        result = detect_test_file_path("src/foo.rb", ["ruby"])
        assert result is None

    def test_empty_stack_returns_none(self):
        result = detect_test_file_path("src/foo.py", [])
        assert result is None

    # --- Multiple stacks (first match wins) ---
    def test_multiple_stacks_first_wins(self):
        result = detect_test_file_path("src/foo/bar.py", ["python", "typescript"])
        assert result == "tests/test_bar.py"


# =========================================================================
# detect_tech_stack
# =========================================================================


class TestDetectTechStack:
    def test_detects_python(self):
        tree = "pyproject.toml\nsrc/\n  main.py"
        assert "python" in detect_tech_stack(tree)

    def test_detects_typescript(self):
        tree = "package.json\nsrc/\n  index.ts"
        assert "typescript" in detect_tech_stack(tree)

    def test_detects_go(self):
        tree = "go.mod\ngo.sum\nmain.go"
        assert "go" in detect_tech_stack(tree)

    def test_detects_rust(self):
        tree = "Cargo.toml\nsrc/\n  main.rs"
        assert "rust" in detect_tech_stack(tree)

    def test_detects_java(self):
        tree = "pom.xml\nsrc/main/java/App.java"
        assert "java" in detect_tech_stack(tree)

    def test_detects_multiple(self):
        tree = "package.json\npyproject.toml\ngo.mod"
        stacks = detect_tech_stack(tree)
        assert "typescript" in stacks
        assert "python" in stacks
        assert "go" in stacks

    def test_empty_tree(self):
        assert detect_tech_stack("") == []

    def test_no_config_files(self):
        tree = "src/\n  main.py\n  utils.py"
        assert detect_tech_stack(tree) == []

    def test_no_duplicates(self):
        tree = "package.json\npackage.json"
        stacks = detect_tech_stack(tree)
        assert stacks.count("typescript") == 1


# =========================================================================
# is_skippable_file
# =========================================================================


class TestIsSkippableFile:
    # --- Binary ---
    @pytest.mark.parametrize(
        "path",
        [
            "assets/logo.png",
            "images/photo.jpg",
            "fonts/roboto.woff",
            "favicon.ico",
            "module.pyc",
        ],
    )
    def test_binary_files(self, path: str):
        skip, reason = is_skippable_file(path)
        assert skip is True
        assert reason == "binary"

    # --- Lock files ---
    @pytest.mark.parametrize(
        "path",
        [
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            "Cargo.lock",
        ],
    )
    def test_lock_files(self, path: str):
        skip, reason = is_skippable_file(path)
        assert skip is True
        assert reason == "lock_file"

    # --- Generated ---
    @pytest.mark.parametrize(
        "path",
        [
            "static/app.min.js",
            "css/style.min.css",
            "js/bundle.js.map",
            "dist/index.js",
            "build/output.css",
        ],
    )
    def test_generated_files(self, path: str):
        skip, reason = is_skippable_file(path)
        assert skip is True
        assert reason == "generated"

    # --- Normal files ---
    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "src/utils/helpers.ts",
            "README.md",
            "Makefile",
            "src/auth/oauth.go",
        ],
    )
    def test_normal_files(self, path: str):
        skip, reason = is_skippable_file(path)
        assert skip is False
        assert reason == ""

    def test_nested_dist_directory(self):
        skip, reason = is_skippable_file("project/dist/bundle.js")
        assert skip is True
        assert reason == "generated"

    def test_nested_build_directory(self):
        skip, reason = is_skippable_file("frontend/build/static/main.js")
        assert skip is True
        assert reason == "generated"


# =========================================================================
# Property Tests (Hypothesis)
# =========================================================================

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st
from tests.conftest import jira_issue_keys


# =========================================================================
# Property 18: Branch Name Generation
# =========================================================================

_branch_patterns = st.sampled_from([
    "feature/{issue_key}-ai",
    "fix/{issue_key}",
    "hotfix/{issue_key}-auto",
    "chore/{issue_key}",
])

_suffixes = st.one_of(
    st.none(),
    st.from_regex(r"[0-9]{10}", fullmatch=True),
    st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
)


class TestBranchNameGenerationProperty:
    """Property 18: Branch Name Generation.

    Validates: Requirements 6.1, 12.3, 17.2
    """

    @given(pattern=_branch_patterns, issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_issue_key_always_in_branch_name(self, pattern: str, issue_key: str) -> None:
        """Generated branch name always contains the issue_key."""
        result = generate_branch_name(pattern, issue_key)
        assert issue_key in result

    @given(pattern=_branch_patterns, issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_pattern_prefix_preserved(self, pattern: str, issue_key: str) -> None:
        """Branch name preserves the prefix before {issue_key}."""
        prefix = pattern.split("{issue_key}")[0]
        result = generate_branch_name(pattern, issue_key)
        assert result.startswith(prefix)

    @given(
        pattern=_branch_patterns,
        issue_key=jira_issue_keys,
        suffix=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
    )
    @h_settings(max_examples=100)
    def test_suffix_appended_with_dash(self, pattern: str, issue_key: str, suffix: str) -> None:
        """When suffix is provided, it is appended with a dash separator."""
        result = generate_branch_name(pattern, issue_key, suffix=suffix)
        assert result.endswith(f"-{suffix}")

    @given(pattern=_branch_patterns, issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_no_suffix_no_trailing_dash(self, pattern: str, issue_key: str) -> None:
        """Without suffix, branch name does not end with a dash."""
        result = generate_branch_name(pattern, issue_key)
        assert not result.endswith("-")


# =========================================================================
# Property 33: Branch Name Uniqueness on Retry
# =========================================================================


class TestBranchNameUniquenessOnRetryProperty:
    """Property 33: Branch Name Uniqueness on Retry.

    Validates: Requirements 6.1, 17.2
    """

    @given(
        issue_key=jira_issue_keys,
        suffix1=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
        suffix2=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
    )
    @h_settings(max_examples=100)
    def test_different_suffixes_produce_different_names(
        self, issue_key: str, suffix1: str, suffix2: str
    ) -> None:
        """Different suffixes for the same issue_key produce different branch names."""
        from hypothesis import assume
        assume(suffix1 != suffix2)
        pattern = "feature/{issue_key}-ai"
        name1 = generate_branch_name(pattern, issue_key, suffix=suffix1)
        name2 = generate_branch_name(pattern, issue_key, suffix=suffix2)
        assert name1 != name2

    @given(issue_key=jira_issue_keys)
    @h_settings(max_examples=100)
    def test_with_suffix_differs_from_without(self, issue_key: str) -> None:
        """Branch name with suffix always differs from branch name without suffix."""
        pattern = "feature/{issue_key}-ai"
        base = generate_branch_name(pattern, issue_key)
        with_suffix = generate_branch_name(pattern, issue_key, suffix="retry1")
        assert base != with_suffix


# =========================================================================
# Property 9: Test File Detection
# =========================================================================

_python_paths = st.from_regex(r"src/[a-z_]+/[a-z_]+\.py", fullmatch=True)
_ts_paths = st.from_regex(r"src/[a-z_]+/[a-z_]+\.ts", fullmatch=True)
_go_paths = st.from_regex(r"pkg/[a-z_]+/[a-z_]+\.go", fullmatch=True)
_rust_paths = st.from_regex(r"src/[a-z_]+/[a-z_]+\.rs", fullmatch=True)
_java_paths = st.from_regex(r"src/main/java/com/[A-Z][a-z]+\.java", fullmatch=True)


class TestTestFileDetectionProperty:
    """Property 9: Test File Detection.

    Validates: Requirements 3.7
    """

    @given(path=_python_paths)
    @h_settings(max_examples=100)
    def test_python_test_path_starts_with_tests(self, path: str) -> None:
        """Python test file path always starts with 'tests/'."""
        result = detect_test_file_path(path, ["python"])
        assert result is not None
        assert result.startswith("tests/test_")
        assert result.endswith(".py")

    @given(path=_ts_paths)
    @h_settings(max_examples=100)
    def test_typescript_test_path_in_tests_dir(self, path: str) -> None:
        """TypeScript test file path always contains '__tests__' directory."""
        result = detect_test_file_path(path, ["typescript"])
        assert result is not None
        assert "__tests__" in result
        assert result.endswith(".test.ts")

    @given(path=_go_paths)
    @h_settings(max_examples=100)
    def test_go_test_path_ends_with_test_go(self, path: str) -> None:
        """Go test file path always ends with '_test.go'."""
        result = detect_test_file_path(path, ["go"])
        assert result is not None
        assert result.endswith("_test.go")

    @given(path=_rust_paths)
    @h_settings(max_examples=100)
    def test_rust_test_path_in_tests_dir(self, path: str) -> None:
        """Rust test file path always starts with 'tests/'."""
        result = detect_test_file_path(path, ["rust"])
        assert result is not None
        assert result.startswith("tests/test_")
        assert result.endswith(".rs")

    @given(
        path=st.from_regex(r"src/[a-z_]+/[a-z_]+\.rb", fullmatch=True),
    )
    @h_settings(max_examples=50)
    def test_unknown_stack_returns_none(self, path: str) -> None:
        """Unknown tech stack always returns None."""
        result = detect_test_file_path(path, ["ruby"])
        assert result is None

    @given(
        path=st.from_regex(r"src/[a-z_]+/[a-z_]+\.py", fullmatch=True),
    )
    @h_settings(max_examples=50)
    def test_empty_stack_returns_none(self, path: str) -> None:
        """Empty tech stack always returns None."""
        result = detect_test_file_path(path, [])
        assert result is None


# =========================================================================
# Property 10: Tech Stack Detection
# =========================================================================

_config_files = {
    "python": "pyproject.toml",
    "typescript": "package.json",
    "go": "go.mod",
    "rust": "Cargo.toml",
    "java": "pom.xml",
}


class TestTechStackDetectionProperty:
    """Property 10: Tech Stack Detection.

    Validates: Requirements 3.8
    """

    @given(stack=st.sampled_from(["python", "typescript", "go", "rust", "java"]))
    @h_settings(max_examples=50)
    def test_config_file_always_detected(self, stack: str) -> None:
        """File tree containing a known config file always detects the stack."""
        config_file = _config_files[stack]
        file_tree = f"{config_file}\nsrc/\n  main.py"
        result = detect_tech_stack(file_tree)
        assert stack in result

    @given(
        stacks=st.lists(
            st.sampled_from(["python", "typescript", "go", "rust", "java"]),
            min_size=2,
            max_size=5,
            unique=True,
        )
    )
    @h_settings(max_examples=50)
    def test_multiple_config_files_all_detected(self, stacks: list) -> None:
        """File tree with multiple config files detects all corresponding stacks."""
        file_tree = "\n".join(_config_files[s] for s in stacks)
        result = detect_tech_stack(file_tree)
        for stack in stacks:
            assert stack in result

    @given(
        stack=st.sampled_from(["python", "typescript", "go", "rust", "java"]),
    )
    @h_settings(max_examples=50)
    def test_no_duplicates_in_result(self, stack: str) -> None:
        """Detected stacks list never contains duplicates."""
        config_file = _config_files[stack]
        # Repeat the config file multiple times
        file_tree = f"{config_file}\n{config_file}\n{config_file}"
        result = detect_tech_stack(file_tree)
        assert len(result) == len(set(result))

    @given(
        noise=st.text(
            min_size=0,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
        )
    )
    @h_settings(max_examples=50)
    def test_empty_or_noise_tree_returns_empty_or_subset(self, noise: str) -> None:
        """File tree without known config files returns empty list."""
        # Ensure no config file names appear in the noise
        for config_file in _config_files.values():
            if config_file in noise:
                return  # skip this example
        result = detect_tech_stack(noise)
        assert result == []
