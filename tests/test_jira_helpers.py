"""Unit tests for src/utils/jira_helpers.py.

Covers:
- format_jira_comment: structured comment formatting with agent_name and stage
- mask_secrets: secret value replacement, edge cases, substring handling
"""

from __future__ import annotations

import pytest

from src.utils.jira_helpers import format_jira_comment, mask_secrets


# =========================================================================
# format_jira_comment
# =========================================================================


class TestFormatJiraComment:
    def test_basic_format(self):
        result = format_jira_comment("task_reader", "reading", "Started processing")
        assert result == "🤖 *task_reader* | Stage: reading\n\nStarted processing"

    def test_contains_agent_name(self):
        result = format_jira_comment("code_writer", "writing", "Generating code")
        assert "code_writer" in result

    def test_contains_stage(self):
        result = format_jira_comment("code_reviewer", "review", "Reviewing changes")
        assert "Stage: review" in result

    def test_contains_message(self):
        msg = "PR created: https://example.com/pr/1"
        result = format_jira_comment("orchestrator", "pr_creation", msg)
        assert msg in result

    def test_multiline_message(self):
        msg = "Line 1\nLine 2\nLine 3"
        result = format_jira_comment("agent", "stage", msg)
        assert msg in result

    def test_empty_message(self):
        result = format_jira_comment("agent", "stage", "")
        assert "🤖 *agent* | Stage: stage" in result

    def test_empty_agent_name(self):
        result = format_jira_comment("", "stage", "msg")
        assert "🤖 ** | Stage: stage" in result

    def test_empty_stage(self):
        result = format_jira_comment("agent", "", "msg")
        assert "Stage: \n" in result


# =========================================================================
# mask_secrets
# =========================================================================


class TestMaskSecrets:
    def test_single_secret(self):
        result = mask_secrets("token is sk-abc123", ["sk-abc123"])
        assert result == "token is ***"
        assert "sk-abc123" not in result

    def test_multiple_secrets(self):
        text = "user=admin pass=secret123 key=AKIA1234"
        result = mask_secrets(text, ["admin", "secret123", "AKIA1234"])
        assert "admin" not in result
        assert "secret123" not in result
        assert "AKIA1234" not in result

    def test_repeated_secret(self):
        result = mask_secrets("abc abc abc", ["abc"])
        assert result == "*** *** ***"

    def test_empty_secrets_list(self):
        text = "nothing to mask"
        assert mask_secrets(text, []) == text

    def test_empty_text(self):
        assert mask_secrets("", ["secret"]) == ""

    def test_empty_text_and_secrets(self):
        assert mask_secrets("", []) == ""

    def test_secret_not_present(self):
        text = "no secrets here"
        assert mask_secrets(text, ["missing"]) == text

    def test_substring_secrets_longer_first(self):
        """When one secret is a substring of another, the longer one should be masked first."""
        text = "value is supersecretkey"
        result = mask_secrets(text, ["secret", "supersecretkey"])
        assert "secret" not in result
        assert result == "value is ***"

    def test_overlapping_secrets(self):
        text = "abcdef"
        result = mask_secrets(text, ["abc", "cde"])
        # "abc" → "***" first pass gives "***def", then "cde" not found
        # But with longest-first: both are len 3, order may vary
        assert "abc" not in result

    def test_empty_string_in_secrets_list(self):
        """Empty strings in the secrets list should be ignored."""
        text = "keep this text"
        result = mask_secrets(text, ["", "keep"])
        assert result == "*** this text"

    def test_secret_at_boundaries(self):
        result = mask_secrets("SECRET at start", ["SECRET"])
        assert result == "*** at start"

        result = mask_secrets("at end SECRET", ["SECRET"])
        assert result == "at end ***"

    def test_special_characters_in_secret(self):
        """Secrets with regex-special characters should be handled correctly."""
        text = "password is p@$$w0rd!"
        result = mask_secrets(text, ["p@$$w0rd!"])
        assert "p@$$w0rd!" not in result
        assert result == "password is ***"


# =========================================================================
# Property Tests (Hypothesis)
# =========================================================================

from hypothesis import given, settings as h_settings, assume
from hypothesis import strategies as st


# =========================================================================
# Property 26: Secret Masking
# =========================================================================

_printable_text = st.text(
    min_size=0,
    max_size=500,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)
_secret_value = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
)


class TestSecretMaskingProperty:
    """Property 26: Secret Masking.

    Validates: Requirements 12.2, 12.8
    """

    @given(
        text=_printable_text,
        secret=_secret_value,
    )
    @h_settings(max_examples=100)
    def test_single_secret_never_in_output(self, text: str, secret: str) -> None:
        """After masking, the secret value never appears in the output.

        Edge case: if the secret itself is a substring of '***' (e.g. secret='*'),
        we skip — the replacement marker cannot be secret-free in that case.
        """
        from hypothesis import assume
        assume("***" not in secret and secret not in "***")
        result = mask_secrets(text, [secret])
        assert secret not in result

    @given(
        text=_printable_text,
        secrets=st.lists(_secret_value, min_size=1, max_size=5),
    )
    @h_settings(max_examples=100)
    def test_all_secrets_removed_from_output(self, text: str, secrets: list) -> None:
        """After masking, none of the secret values appear in the output."""
        result = mask_secrets(text, secrets)
        for secret in secrets:
            assert secret not in result

    @given(
        prefix=_printable_text,
        suffix=_printable_text,
        secret=_secret_value,
    )
    @h_settings(max_examples=100)
    def test_text_containing_secret_is_masked(
        self, prefix: str, suffix: str, secret: str
    ) -> None:
        """Text that contains the secret always has it replaced with ***."""
        text = prefix + secret + suffix
        result = mask_secrets(text, [secret])
        assert secret not in result
        assert "***" in result

    @given(text=_printable_text)
    @h_settings(max_examples=50)
    def test_empty_secrets_list_returns_unchanged(self, text: str) -> None:
        """Empty secrets list returns the original text unchanged."""
        result = mask_secrets(text, [])
        assert result == text

    @given(secrets=st.lists(_secret_value, min_size=1, max_size=5))
    @h_settings(max_examples=50)
    def test_empty_text_returns_empty(self, secrets: list) -> None:
        """Empty text always returns empty string."""
        result = mask_secrets("", secrets)
        assert result == ""

    @given(
        text=_printable_text,
        secret=_secret_value,
    )
    @h_settings(max_examples=100)
    def test_masked_output_contains_stars_when_secret_present(
        self, text: str, secret: str
    ) -> None:
        """When text contains the secret, output contains '***'."""
        text_with_secret = text + secret
        result = mask_secrets(text_with_secret, [secret])
        assert "***" in result


# =========================================================================
# Property 25: Jira Comment Format
# =========================================================================

_agent_names = st.text(
    min_size=0,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)
_stage_names = st.text(
    min_size=0,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)
_messages = st.text(
    min_size=0,
    max_size=500,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)


class TestJiraCommentFormatProperty:
    """Property 25: Jira Comment Format.

    Validates: Requirements 9.5
    """

    @given(agent_name=_agent_names, stage=_stage_names, message=_messages)
    @h_settings(max_examples=100)
    def test_comment_always_contains_agent_name(
        self, agent_name: str, stage: str, message: str
    ) -> None:
        """Every Jira comment always contains the agent_name."""
        result = format_jira_comment(agent_name, stage, message)
        assert agent_name in result

    @given(agent_name=_agent_names, stage=_stage_names, message=_messages)
    @h_settings(max_examples=100)
    def test_comment_always_contains_stage(
        self, agent_name: str, stage: str, message: str
    ) -> None:
        """Every Jira comment always contains the pipeline stage."""
        result = format_jira_comment(agent_name, stage, message)
        assert stage in result

    @given(agent_name=_agent_names, stage=_stage_names, message=_messages)
    @h_settings(max_examples=100)
    def test_comment_always_contains_message(
        self, agent_name: str, stage: str, message: str
    ) -> None:
        """Every Jira comment always contains the message body."""
        result = format_jira_comment(agent_name, stage, message)
        assert message in result

    @given(agent_name=_agent_names, stage=_stage_names, message=_messages)
    @h_settings(max_examples=100)
    def test_comment_has_robot_emoji(
        self, agent_name: str, stage: str, message: str
    ) -> None:
        """Every Jira comment starts with the robot emoji."""
        result = format_jira_comment(agent_name, stage, message)
        assert result.startswith("🤖")

    @given(agent_name=_agent_names, stage=_stage_names, message=_messages)
    @h_settings(max_examples=100)
    def test_comment_has_stage_label(
        self, agent_name: str, stage: str, message: str
    ) -> None:
        """Every Jira comment contains 'Stage:' label."""
        result = format_jira_comment(agent_name, stage, message)
        assert "Stage:" in result

    @given(agent_name=_agent_names, stage=_stage_names, message=_messages)
    @h_settings(max_examples=100)
    def test_comment_is_string(
        self, agent_name: str, stage: str, message: str
    ) -> None:
        """format_jira_comment always returns a string."""
        result = format_jira_comment(agent_name, stage, message)
        assert isinstance(result, str)
