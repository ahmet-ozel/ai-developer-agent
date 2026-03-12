"""Application settings loaded from environment variables.

Uses Pydantic BaseSettings to validate and parse all configuration.
Git provider credentials are validated via a model_validator to ensure
the correct set of credentials is provided for the selected provider.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment variables for the AI Developer Agent pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Jira ---
    jira_url: str
    jira_username: str
    jira_api_token: SecretStr
    jira_webhook_secret: SecretStr
    jira_bot_username: str
    jira_transition_in_progress: Optional[str] = None
    jira_transition_in_review: Optional[str] = None

    # --- Git Provider (common) ---
    git_provider: Literal["bitbucket", "github", "gitlab"]
    git_base_branch: str = "main"

    # --- Bitbucket ---
    bitbucket_workspace: Optional[str] = None
    bitbucket_username: Optional[str] = None
    bitbucket_app_password: Optional[SecretStr] = None
    bitbucket_api_token: Optional[SecretStr] = None  # replaces app_password (Bitbucket Cloud API tokens)

    # --- GitHub ---
    github_token: Optional[SecretStr] = None
    github_owner: Optional[str] = None

    # --- GitLab ---
    gitlab_url: str = "https://gitlab.com"
    gitlab_token: Optional[SecretStr] = None
    gitlab_group: Optional[str] = None

    # --- LLM Fast Tier ---
    llm_fast_provider: str
    llm_fast_model: str
    llm_fast_api_key: SecretStr
    llm_fast_endpoint: Optional[str] = None

    # --- LLM Strong Tier ---
    llm_strong_provider: str
    llm_strong_model: str
    llm_strong_api_key: SecretStr
    llm_strong_endpoint: Optional[str] = None
    llm_fallback_chain: list[str] = []

    # --- Pipeline ---
    max_review_retries: int = 2
    max_files_per_task: int = 10
    max_file_changes: int = 15
    max_context_tokens: int = 100000
    max_file_size_kb: int = 100
    branch_pattern: str = "feature/{issue_key}-ai"
    auto_create_pr: bool = True
    pr_auto_assign_reviewer: bool = False
    dry_run: bool = False

    # --- PR Review ---
    pr_reviewer: str = ""
    pr_draft_mode: bool = True
    pr_auto_merge: bool = False

    # --- Confluence ---
    confluence_enabled: bool = False
    confluence_url: str = ""
    confluence_username: str = ""
    confluence_api_token: SecretStr | None = None
    confluence_space_key: str = ""
    confluence_parent_page_id: str = ""

    # --- Task Filtering ---
    skip_task_types: list[str] = []
    allowed_task_types: list[str] = []

    # --- Trigger Mode ---
    trigger_mode: Literal["webhook", "polling"] = "polling"
    poll_interval_seconds: int = 30
    jira_project_key: Optional[str] = None

    # --- LLM Tier Overrides ---
    task_reader_llm_tier: Literal["fast", "strong"] = "fast"
    code_finder_llm_tier: Literal["fast", "strong"] = "fast"
    code_writer_llm_tier: Literal["fast", "strong"] = "strong"
    code_reviewer_llm_tier: Literal["fast", "strong"] = "strong"

    @model_validator(mode="after")
    def validate_git_credentials(self) -> "Settings":
        """Ensure the correct credentials are present for the selected git provider."""
        if self.git_provider == "bitbucket":
            missing = []
            if not self.bitbucket_workspace:
                missing.append("bitbucket_workspace")
            if not self.bitbucket_username:
                missing.append("bitbucket_username")
            # Accept either api_token (new) or app_password (legacy)
            if not self.bitbucket_api_token and not self.bitbucket_app_password:
                missing.append("bitbucket_api_token (or bitbucket_app_password)")
            if missing:
                raise ValueError(
                    "Bitbucket provider requires: "
                    "bitbucket_workspace, bitbucket_username, bitbucket_api_token. "
                    f"Missing: {', '.join(missing)}"
                )
        elif self.git_provider == "github":
            missing = []
            if not self.github_token:
                missing.append("github_token")
            if not self.github_owner:
                missing.append("github_owner")
            if missing:
                raise ValueError(
                    "GitHub provider requires: github_token, github_owner. "
                    f"Missing: {', '.join(missing)}"
                )
        elif self.git_provider == "gitlab":
            missing = []
            if not self.gitlab_token:
                missing.append("gitlab_token")
            if not self.gitlab_group:
                missing.append("gitlab_group")
            if missing:
                raise ValueError(
                    "GitLab provider requires: gitlab_token, gitlab_group. "
                    f"Missing: {', '.join(missing)}"
                )
        return self

    @model_validator(mode="after")
    def validate_confluence_credentials(self) -> "Settings":
        """Ensure required Confluence credentials are present when enabled."""
        if self.confluence_enabled:
            missing = []
            if not self.confluence_url:
                missing.append("confluence_url")
            if not self.confluence_api_token:
                missing.append("confluence_api_token")
            if missing:
                raise ValueError(
                    "Confluence is enabled but missing required settings: "
                    f"{', '.join(missing)}"
                )
        return self
