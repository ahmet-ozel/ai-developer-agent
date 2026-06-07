"""Unit tests for Settings validation.

Tests valid configurations for each git provider and verifies that
missing or mismatched credentials raise ValidationError.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers - minimal required kwargs per provider
# ---------------------------------------------------------------------------

_COMMON: dict = {
    "jira_url": "https://jira.example.com",
    "jira_username": "ai-developer",
    "jira_api_token": "jira-secret-token",
    "jira_webhook_secret": "webhook-secret",
    "jira_bot_username": "ai-developer",
    "llm_fast_provider": "openai",
    "llm_fast_model": "gpt-4o-mini",
    "llm_fast_api_key": "sk-fast-key",
    "llm_strong_provider": "anthropic",
    "llm_strong_model": "claude-sonnet-4-20250514",
    "llm_strong_api_key": "sk-strong-key",
}

_BITBUCKET_CREDS: dict = {
    "git_provider": "bitbucket",
    "bitbucket_workspace": "my-workspace",
    "bitbucket_username": "bb-user",
    "bitbucket_app_password": "bb-app-password",
}

_GITHUB_CREDS: dict = {
    "git_provider": "github",
    "github_token": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "github_owner": "my-org",
}

_GITLAB_CREDS: dict = {
    "git_provider": "gitlab",
    "gitlab_token": "glpat-xxxxxxxxxxxxxxxxxxxx",
    "gitlab_group": "my-group",
}


def _make_settings(**overrides: object) -> Settings:
    """Build a Settings instance with Bitbucket defaults, applying overrides."""
    data = {**_COMMON, **_BITBUCKET_CREDS, **overrides}
    return Settings(_env_file=None, **data)


# ---------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------


class TestValidSettings:
    """Settings should be constructable with correct credentials."""

    def test_bitbucket_valid(self) -> None:
        s = Settings(_env_file=None, **{**_COMMON, **_BITBUCKET_CREDS})
        assert s.git_provider == "bitbucket"
        assert s.bitbucket_workspace == "my-workspace"

    def test_github_valid(self) -> None:
        s = Settings(_env_file=None, **{**_COMMON, **_GITHUB_CREDS})
        assert s.git_provider == "github"
        assert s.github_owner == "my-org"

    def test_gitlab_valid(self) -> None:
        s = Settings(_env_file=None, **{**_COMMON, **_GITLAB_CREDS})
        assert s.git_provider == "gitlab"
        assert s.gitlab_group == "my-group"

    def test_secret_fields_are_secret_str(self) -> None:
        s = _make_settings()
        assert s.jira_api_token.get_secret_value() == "jira-secret-token"
        assert s.jira_webhook_secret.get_secret_value() == "webhook-secret"
        assert s.llm_fast_api_key.get_secret_value() == "sk-fast-key"
        assert s.llm_strong_api_key.get_secret_value() == "sk-strong-key"

    def test_defaults(self) -> None:
        s = _make_settings()
        assert s.git_base_branch == "main"
        assert s.max_file_size_kb == 100
        assert s.max_review_retries == 2
        assert s.max_files_per_task == 10
        assert s.max_file_changes == 15
        assert s.max_context_tokens == 100000
        assert s.branch_pattern == "feature/{issue_key}-ai"
        assert s.auto_create_pr is True
        assert s.pr_auto_assign_reviewer is False
        assert s.dry_run is False
        assert s.skip_task_types == []
        assert s.allowed_task_types == []
        assert s.task_reader_llm_tier == "fast"
        assert s.code_writer_llm_tier == "strong"
        assert s.trigger_mode == "polling"
        assert s.poll_interval_seconds == 30

    def test_overridden_defaults(self) -> None:
        s = _make_settings(
            git_base_branch="develop",
            max_file_size_kb=200,
            dry_run=True,
            max_review_retries=5,
        )
        assert s.git_base_branch == "develop"
        assert s.max_file_size_kb == 200
        assert s.dry_run is True
        assert s.max_review_retries == 5

    def test_pr_settings_defaults(self) -> None:
        s = _make_settings()
        assert s.pr_reviewer == ""
        assert s.pr_draft_mode is True
        assert s.pr_auto_merge is False

    def test_pr_settings_overridden(self) -> None:
        s = _make_settings(
            pr_reviewer="alice,bob",
            pr_draft_mode=False,
            pr_auto_merge=True,
        )
        assert s.pr_reviewer == "alice,bob"
        assert s.pr_draft_mode is False
        assert s.pr_auto_merge is True

    def test_confluence_settings_defaults(self) -> None:
        s = _make_settings()
        assert s.confluence_enabled is False
        assert s.confluence_url == ""
        assert s.confluence_username == ""
        assert s.confluence_api_token is None
        assert s.confluence_space_key == ""
        assert s.confluence_parent_page_id == ""

    def test_confluence_settings_overridden(self) -> None:
        s = _make_settings(
            confluence_enabled=True,
            confluence_url="https://wiki.example.com",
            confluence_username="wiki-user",
            confluence_api_token="wiki-token",
            confluence_space_key="DEV",
            confluence_parent_page_id="12345",
        )
        assert s.confluence_enabled is True
        assert s.confluence_url == "https://wiki.example.com"
        assert s.confluence_username == "wiki-user"
        assert s.confluence_api_token.get_secret_value() == "wiki-token"
        assert s.confluence_space_key == "DEV"
        assert s.confluence_parent_page_id == "12345"


# ---------------------------------------------------------------------------
# Missing credential validation
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    """Settings should raise ValidationError when provider credentials are missing."""

    def test_bitbucket_missing_workspace(self) -> None:
        data = {**_COMMON, **_BITBUCKET_CREDS, "bitbucket_workspace": None}
        with pytest.raises(ValidationError, match="bitbucket_workspace"):
            Settings(_env_file=None, **data)

    def test_bitbucket_missing_username(self) -> None:
        data = {**_COMMON, **_BITBUCKET_CREDS, "bitbucket_username": None}
        with pytest.raises(ValidationError, match="bitbucket_username"):
            Settings(_env_file=None, **data)

    def test_bitbucket_missing_app_password(self) -> None:
        data = {**_COMMON, **_BITBUCKET_CREDS, "bitbucket_app_password": None}
        with pytest.raises(ValidationError, match="bitbucket_app_password"):
            Settings(_env_file=None, **data)

    def test_github_missing_token(self) -> None:
        data = {**_COMMON, **_GITHUB_CREDS, "github_token": None}
        with pytest.raises(ValidationError, match="github_token"):
            Settings(_env_file=None, **data)

    def test_github_missing_owner(self) -> None:
        data = {**_COMMON, **_GITHUB_CREDS, "github_owner": None}
        with pytest.raises(ValidationError, match="github_owner"):
            Settings(_env_file=None, **data)

    def test_gitlab_missing_token(self) -> None:
        data = {**_COMMON, **_GITLAB_CREDS, "gitlab_token": None}
        with pytest.raises(ValidationError, match="gitlab_token"):
            Settings(_env_file=None, **data)

    def test_gitlab_missing_group(self) -> None:
        data = {**_COMMON, **_GITLAB_CREDS, "gitlab_group": None}
        with pytest.raises(ValidationError, match="gitlab_group"):
            Settings(_env_file=None, **data)

    def test_invalid_git_provider(self) -> None:
        data = {**_COMMON, "git_provider": "svn"}
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **data)


# ---------------------------------------------------------------------------
# Confluence validation
# ---------------------------------------------------------------------------


class TestConfluenceValidation:
    """Confluence validation should require url and api_token when enabled."""

    def test_enabled_with_all_creds_succeeds(self) -> None:
        s = _make_settings(
            confluence_enabled=True,
            confluence_url="https://wiki.example.com",
            confluence_api_token="wiki-token",
        )
        assert s.confluence_enabled is True

    def test_enabled_missing_url_raises(self) -> None:
        with pytest.raises(ValidationError, match="confluence_url"):
            _make_settings(
                confluence_enabled=True,
                confluence_url="",
                confluence_api_token="wiki-token",
            )

    def test_enabled_missing_api_token_raises(self) -> None:
        with pytest.raises(ValidationError, match="confluence_api_token"):
            _make_settings(
                confluence_enabled=True,
                confluence_url="https://wiki.example.com",
                confluence_api_token=None,
            )

    def test_enabled_missing_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="confluence_url"):
            _make_settings(
                confluence_enabled=True,
                confluence_url="",
                confluence_api_token=None,
            )

    def test_disabled_no_creds_required(self) -> None:
        s = _make_settings(confluence_enabled=False)
        assert s.confluence_enabled is False


# ---------------------------------------------------------------------------
# conftest sample_settings fixture compatibility
# ---------------------------------------------------------------------------


class TestSampleSettingsFixtureCompat:
    """Ensure the sample_settings fixture dict can construct a valid Settings."""

    def test_sample_settings_creates_valid_instance(
        self, sample_settings: dict
    ) -> None:
        s = Settings(_env_file=None, **sample_settings)
        assert s.jira_url == "https://jira.example.com"
        assert s.git_provider == "bitbucket"
        assert s.llm_fast_provider == "openai"


# ---------------------------------------------------------------------------
# Property Tests: Settings Validation (Property 29)
# ---------------------------------------------------------------------------

import re
from pathlib import Path

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


class TestSettingsValidationProperty:
    """Property 29: Settings Validation.

    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.6
    """

    @given(
        provider=st.sampled_from(["bitbucket", "github", "gitlab"]),
    )
    @h_settings(max_examples=10)
    def test_valid_provider_with_correct_creds_succeeds(self, provider: str) -> None:
        """Valid provider + correct credentials always creates Settings successfully."""
        if provider == "bitbucket":
            creds = _BITBUCKET_CREDS
        elif provider == "github":
            creds = _GITHUB_CREDS
        else:
            creds = _GITLAB_CREDS
        s = Settings(_env_file=None, **{**_COMMON, **creds})
        assert s.git_provider == provider

    @given(
        provider=st.sampled_from(["bitbucket", "github", "gitlab"]),
        missing_field=st.sampled_from(["workspace_or_token", "username_or_owner", "password_or_group"]),
    )
    @h_settings(max_examples=30)
    def test_missing_credential_raises_validation_error(
        self, provider: str, missing_field: str
    ) -> None:
        """Missing required credential for any provider always raises ValidationError."""
        if provider == "bitbucket":
            # Test missing each of the 3 required bitbucket fields
            if missing_field == "workspace_or_token":
                data = {**_COMMON, **_BITBUCKET_CREDS, "bitbucket_workspace": None}
            elif missing_field == "username_or_owner":
                data = {**_COMMON, **_BITBUCKET_CREDS, "bitbucket_username": None}
            else:
                data = {**_COMMON, **_BITBUCKET_CREDS, "bitbucket_app_password": None}
        elif provider == "github":
            if missing_field == "workspace_or_token":
                data = {**_COMMON, **_GITHUB_CREDS, "github_token": None}
            else:
                data = {**_COMMON, **_GITHUB_CREDS, "github_owner": None}
        else:  # gitlab
            if missing_field == "workspace_or_token":
                data = {**_COMMON, **_GITLAB_CREDS, "gitlab_token": None}
            else:
                data = {**_COMMON, **_GITLAB_CREDS, "gitlab_group": None}

        with pytest.raises(ValidationError):
            Settings(_env_file=None, **data)

    @given(
        dry_run=st.booleans(),
        auto_create_pr=st.booleans(),
        max_review_retries=st.integers(min_value=0, max_value=10),
        max_files_per_task=st.integers(min_value=1, max_value=50),
    )
    @h_settings(max_examples=50)
    def test_boolean_and_int_settings_accepted(
        self,
        dry_run: bool,
        auto_create_pr: bool,
        max_review_retries: int,
        max_files_per_task: int,
    ) -> None:
        """Boolean and integer settings accept any valid value."""
        s = _make_settings(
            dry_run=dry_run,
            auto_create_pr=auto_create_pr,
            max_review_retries=max_review_retries,
            max_files_per_task=max_files_per_task,
        )
        assert s.dry_run == dry_run
        assert s.auto_create_pr == auto_create_pr
        assert s.max_review_retries == max_review_retries
        assert s.max_files_per_task == max_files_per_task


class TestEnvExampleKeyMatch:
    """Verify .env.example keys match Settings class fields.

    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.6
    """

    def test_env_example_keys_exist_in_settings(self) -> None:
        """Every KEY in .env.example must correspond to a Settings field."""
        env_example_path = Path(".env.example")
        if not env_example_path.exists():
            pytest.skip(".env.example not found")

        content = env_example_path.read_text()
        # Extract KEY names from lines like: KEY=value or KEY=
        keys = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                match = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
                if match:
                    keys.append(match.group(1).lower())

        settings_fields = set(Settings.model_fields.keys())

        missing_from_settings = []
        for key in keys:
            if key not in settings_fields:
                missing_from_settings.append(key)

        assert not missing_from_settings, (
            f"Keys in .env.example not found in Settings: {missing_from_settings}"
        )

    def test_settings_has_env_example_file(self) -> None:
        """The .env.example file must exist."""
        assert Path(".env.example").exists(), ".env.example file is missing"
