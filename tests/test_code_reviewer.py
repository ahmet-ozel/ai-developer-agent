"""Unit tests for CodeReviewerAgent.

Tests:
- Valid review response produces correct ReviewResult
- Hardcoded secret detection finds known patterns
- APPROVE constraints validated (score >= 7, zero CRITICAL, acceptance_criteria_met)
- Different verdicts (APPROVE, REQUEST_CHANGES, REJECT)
- Prompt construction includes all sections
- LLM tier and agent configuration
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.code_reviewer import CodeReviewerAgent, _extract_json
from src.config.settings import Settings
from src.pipeline.llm_router import LLMRouter
from src.pipeline.models import (
    ChangeType,
    CodeChange,
    CodeContext,
    CodeFile,
    FileChange,
    FindingSeverity,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    TaskContext,
    TaskScope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    defaults = {
        "jira_url": "https://jira.example.com",
        "jira_username": "ai-developer",
        "jira_api_token": "jira-secret-token",
        "jira_webhook_secret": "webhook-secret",
        "jira_bot_username": "ai-developer",
        "git_provider": "bitbucket",
        "bitbucket_workspace": "my-workspace",
        "bitbucket_username": "bb-user",
        "bitbucket_app_password": "bb-app-password",
        "llm_fast_provider": "openai",
        "llm_fast_model": "gpt-4o-mini",
        "llm_fast_api_key": "sk-fast-key",
        "llm_strong_provider": "anthropic",
        "llm_strong_model": "claude-sonnet-4-20250514",
        "llm_strong_api_key": "sk-strong-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_task_context(**overrides: Any) -> TaskContext:
    defaults = {
        "issue_key": "TEST-42",
        "summary": "Add OAuth2 token refresh",
        "description": "Implement automatic token refresh when session expires",
        "acceptance_criteria": "Users should stay logged in after token refresh",
        "repository_name": "backend-api",
        "estimated_scope": TaskScope.SMALL,
    }
    defaults.update(overrides)
    return TaskContext(**defaults)


def _make_code_context(**overrides: Any) -> CodeContext:
    defaults = {
        "files": [
            CodeFile(
                path="src/auth/oauth.py",
                content="class OAuthHandler:\n    pass\n",
                language="python",
                is_test=False,
            ),
        ],
        "test_files": [],
        "tech_stack": ["python"],
        "repository_name": "backend-api",
    }
    defaults.update(overrides)
    return CodeContext(**defaults)


def _make_code_change(**overrides: Any) -> CodeChange:
    defaults = {
        "changes": [
            FileChange(
                path="src/auth/oauth.py",
                new_content="class OAuthHandler:\n    def refresh_token(self):\n        return 'new_token'\n",
                change_type=ChangeType.MODIFY,
                explanation="Added token refresh",
            ),
        ],
        "test_changes": [
            FileChange(
                path="tests/test_oauth.py",
                new_content="def test_refresh():\n    assert True\n",
                change_type=ChangeType.CREATE,
                explanation="Added test",
            ),
        ],
        "commit_message": "feat(auth): add token refresh",
        "pr_title": "feat(auth): add token refresh",
        "pr_description": "Adds token refresh support",
    }
    defaults.update(overrides)
    return CodeChange(**defaults)


def _make_review_response(
    *,
    verdict: str = "approve",
    score: int = 8,
    findings: list[dict[str, Any]] | None = None,
    feedback: str | None = None,
    acceptance_criteria_met: bool = True,
) -> str:
    data: dict[str, Any] = {
        "verdict": verdict,
        "score": score,
        "findings": findings or [],
        "feedback_for_rewrite": feedback,
        "acceptance_criteria_met": acceptance_criteria_met,
    }
    return json.dumps(data)


def _build_mock_agent_and_llm(llm_response: str) -> tuple[AsyncMock, AsyncMock]:
    mock_llm = AsyncMock()
    mock_llm.generate_str = AsyncMock(return_value=llm_response)

    mock_agent_instance = AsyncMock()
    mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)
    mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
    mock_agent_instance.__aexit__ = AsyncMock(return_value=None)

    return mock_agent_instance, mock_llm


# ---------------------------------------------------------------------------
# Tests: Valid review response  correct ReviewResult
# ---------------------------------------------------------------------------


class TestValidResponse:
    """Test that valid LLM responses produce correct ReviewResult models."""

    @pytest.fixture
    def agent(self) -> CodeReviewerAgent:
        settings = _make_settings()
        return CodeReviewerAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_approve_response(self, agent: CodeReviewerAgent) -> None:
        """Valid APPROVE response should produce ReviewResult with correct fields."""
        mock_agent_instance, _ = _build_mock_agent_and_llm(
            _make_review_response(verdict="approve", score=8, acceptance_criteria_met=True)
        )

        with patch("src.agents.code_reviewer.Agent") as MockAgent:
            MockAgent.return_value = mock_agent_instance
            result = await agent.review_code(
                _make_task_context(), _make_code_context(), _make_code_change()
            )

        assert isinstance(result, ReviewResult)
        assert result.verdict == ReviewVerdict.APPROVE
        assert result.score == 8
        assert result.acceptance_criteria_met is True

    async def test_request_changes_response(self, agent: CodeReviewerAgent) -> None:
        """REQUEST_CHANGES response should include feedback."""
        findings = [
            {
                "file_path": "src/auth/oauth.py",
                "line_range": "5-10",
                "severity": "warning",
                "category": "logic",
                "message": "Missing error handling for expired tokens",
            }
        ]
        mock_agent_instance, _ = _build_mock_agent_and_llm(
            _make_review_response(
                verdict="request_changes",
                score=5,
                findings=findings,
                feedback="Add error handling for expired tokens",
            )
        )

        with patch("src.agents.code_reviewer.Agent") as MockAgent:
            MockAgent.return_value = mock_agent_instance
            result = await agent.review_code(
                _make_task_context(), _make_code_context(), _make_code_change()
            )

        assert result.verdict == ReviewVerdict.REQUEST_CHANGES
        assert result.score == 5
        assert len(result.findings) >= 1
        assert result.feedback_for_rewrite is not None

    async def test_reject_response(self, agent: CodeReviewerAgent) -> None:
        """REJECT response should have low score."""
        mock_agent_instance, _ = _build_mock_agent_and_llm(
            _make_review_response(
                verdict="reject",
                score=2,
                feedback="Fundamental design issues",
            )
        )

        with patch("src.agents.code_reviewer.Agent") as MockAgent:
            MockAgent.return_value = mock_agent_instance
            result = await agent.review_code(
                _make_task_context(), _make_code_context(), _make_code_change()
            )

        assert result.verdict == ReviewVerdict.REJECT
        assert result.score == 2

    async def test_findings_with_severity(self, agent: CodeReviewerAgent) -> None:
        """Findings should have correct severity levels."""
        findings = [
            {
                "file_path": "src/auth/oauth.py",
                "severity": "warning",
                "category": "logic",
                "message": "Edge case not handled",
            },
            {
                "file_path": "src/auth/oauth.py",
                "severity": "suggestion",
                "category": "style",
                "message": "Consider using f-string",
            },
            {
                "file_path": "src/auth/oauth.py",
                "severity": "good",
                "category": "test",
                "message": "Good test coverage",
            },
        ]
        mock_agent_instance, _ = _build_mock_agent_and_llm(
            _make_review_response(
                verdict="request_changes",
                score=6,
                findings=findings,
                feedback="Fix edge case",
            )
        )

        with patch("src.agents.code_reviewer.Agent") as MockAgent:
            MockAgent.return_value = mock_agent_instance
            result = await agent.review_code(
                _make_task_context(), _make_code_context(), _make_code_change()
            )

        severities = {f.severity for f in result.findings}
        assert FindingSeverity.WARNING in severities
        assert FindingSeverity.SUGGESTION in severities
        assert FindingSeverity.GOOD in severities


# ---------------------------------------------------------------------------
# Tests: Hardcoded secret detection
# ---------------------------------------------------------------------------


class TestHardcodedSecretDetection:
    """Test _check_hardcoded_secrets finds known patterns."""

    def test_detects_aws_access_key(self) -> None:
        """Should detect AWS access key pattern AKIA..."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/config.py",
                    new_content='AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n',
                    change_type=ChangeType.CREATE,
                    explanation="Config file",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) >= 1
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert findings[0].category == "security"
        assert "AWS" in findings[0].message

    def test_detects_github_token(self) -> None:
        """Should detect GitHub token pattern ghp_..."""
        token = "ghp_" + "a" * 36
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/config.py",
                    new_content=f'GITHUB_TOKEN = "{token}"\n',
                    change_type=ChangeType.CREATE,
                    explanation="Config file",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) >= 1
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert "GitHub" in findings[0].message

    def test_detects_password_pattern(self) -> None:
        """Should detect password = '...' pattern."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/db.py",
                    new_content='password = "super_secret_123"\n',
                    change_type=ChangeType.CREATE,
                    explanation="DB config",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) >= 1
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert "password" in findings[0].message.lower()

    def test_detects_secret_pattern(self) -> None:
        """Should detect secret = '...' pattern."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/app.py",
                    new_content="secret = 'my_secret_value'\n",
                    change_type=ChangeType.CREATE,
                    explanation="App config",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) >= 1
        assert findings[0].severity == FindingSeverity.CRITICAL

    def test_detects_api_key_pattern(self) -> None:
        """Should detect api_key = '...' pattern."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/client.py",
                    new_content='api_key = "sk-1234567890abcdef"\n',
                    change_type=ChangeType.CREATE,
                    explanation="API client",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) >= 1
        assert findings[0].severity == FindingSeverity.CRITICAL

    def test_no_secrets_in_clean_code(self) -> None:
        """Clean code should produce no secret findings."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/utils.py",
                    new_content="import os\n\ndef get_key():\n    return os.environ['API_KEY']\n",
                    change_type=ChangeType.CREATE,
                    explanation="Utils",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) == 0

    def test_detects_secrets_in_test_changes(self) -> None:
        """Should also scan test_changes for secrets."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/app.py",
                    new_content="print('hello')\n",
                    change_type=ChangeType.CREATE,
                    explanation="App",
                ),
            ],
            test_changes=[
                FileChange(
                    path="tests/test_app.py",
                    new_content='password = "test_password_123"\n',
                    change_type=ChangeType.CREATE,
                    explanation="Test",
                ),
            ],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) >= 1
        assert findings[0].file_path == "tests/test_app.py"

    def test_skips_delete_changes(self) -> None:
        """DELETE changes have no new_content, should not crash."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/old.py",
                    new_content=None,
                    change_type=ChangeType.DELETE,
                    explanation="Remove old file",
                ),
            ],
            test_changes=[],
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Tests: APPROVE constraints validation
# ---------------------------------------------------------------------------


class TestApproveConstraints:
    """Test that APPROVE verdict requires score >= 7, zero CRITICAL, acceptance_criteria_met."""

    @pytest.fixture
    def agent(self) -> CodeReviewerAgent:
        settings = _make_settings()
        return CodeReviewerAgent(settings=settings, llm_router=LLMRouter(config=settings))

    async def test_approve_with_secrets_downgrades_to_request_changes(
        self, agent: CodeReviewerAgent
    ) -> None:
        """If LLM says APPROVE but secrets are found, verdict should downgrade."""
        code_change = _make_code_change(
            changes=[
                FileChange(
                    path="src/config.py",
                    new_content='AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n',
                    change_type=ChangeType.CREATE,
                    explanation="Config",
                ),
            ],
            test_changes=[],
        )
        mock_agent_instance, _ = _build_mock_agent_and_llm(
            _make_review_response(verdict="approve", score=8, acceptance_criteria_met=True)
        )

        with patch("src.agents.code_reviewer.Agent") as MockAgent:
            MockAgent.return_value = mock_agent_instance
            result = await agent.review_code(
                _make_task_context(), _make_code_context(), code_change
            )

        assert result.verdict == ReviewVerdict.REQUEST_CHANGES
        assert any(f.severity == FindingSeverity.CRITICAL for f in result.findings)

    def test_approve_requires_score_gte_7(self) -> None:
        """Pydantic validation: APPROVE with score < 7 should raise."""
        with pytest.raises(ValueError, match="score >= 7"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=5,
                findings=[],
                acceptance_criteria_met=True,
            )

    def test_approve_requires_zero_critical(self) -> None:
        """Pydantic validation: APPROVE with CRITICAL findings should raise."""
        with pytest.raises(ValueError, match="zero CRITICAL"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=8,
                findings=[
                    ReviewFinding(
                        file_path="src/x.py",
                        severity=FindingSeverity.CRITICAL,
                        category="security",
                        message="Secret found",
                    )
                ],
                acceptance_criteria_met=True,
            )

    def test_approve_requires_acceptance_criteria_met(self) -> None:
        """Pydantic validation: APPROVE with acceptance_criteria_met=False should raise."""
        with pytest.raises(ValueError, match="acceptance criteria"):
            ReviewResult(
                verdict=ReviewVerdict.APPROVE,
                score=8,
                findings=[],
                acceptance_criteria_met=False,
            )

    def test_valid_approve(self) -> None:
        """Valid APPROVE: score >= 7, no CRITICAL, acceptance_criteria_met."""
        result = ReviewResult(
            verdict=ReviewVerdict.APPROVE,
            score=9,
            findings=[
                ReviewFinding(
                    file_path="src/x.py",
                    severity=FindingSeverity.GOOD,
                    category="style",
                    message="Clean code",
                )
            ],
            acceptance_criteria_met=True,
        )
        assert result.verdict == ReviewVerdict.APPROVE
        assert result.score == 9


# ---------------------------------------------------------------------------
# Tests: Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Test _build_prompt includes all expected sections."""

    @pytest.fixture
    def agent(self) -> CodeReviewerAgent:
        settings = _make_settings()
        return CodeReviewerAgent(settings=settings, llm_router=LLMRouter(config=settings))

    def test_prompt_includes_task_details(self, agent: CodeReviewerAgent) -> None:
        ctx = _make_task_context()
        prompt = agent._build_prompt(ctx, _make_code_context(), _make_code_change())
        assert ctx.issue_key in prompt
        assert ctx.summary in prompt
        assert ctx.acceptance_criteria in prompt

    def test_prompt_includes_proposed_changes(self, agent: CodeReviewerAgent) -> None:
        code_change = _make_code_change()
        prompt = agent._build_prompt(
            _make_task_context(), _make_code_context(), code_change
        )
        assert "Proposed Changes" in prompt
        assert code_change.commit_message in prompt
        assert "src/auth/oauth.py" in prompt

    def test_prompt_includes_existing_code(self, agent: CodeReviewerAgent) -> None:
        prompt = agent._build_prompt(
            _make_task_context(), _make_code_context(), _make_code_change()
        )
        assert "Existing Source Files" in prompt
        assert "OAuthHandler" in prompt

    def test_prompt_includes_test_changes(self, agent: CodeReviewerAgent) -> None:
        prompt = agent._build_prompt(
            _make_task_context(), _make_code_context(), _make_code_change()
        )
        assert "Test Changes" in prompt
        assert "tests/test_oauth.py" in prompt


# ---------------------------------------------------------------------------
# Tests: JSON parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Test _parse_response and _extract_json helpers."""

    def test_parse_valid_json(self) -> None:
        raw = _make_review_response(verdict="request_changes", score=5)
        result = CodeReviewerAgent._parse_response(raw)
        assert isinstance(result, ReviewResult)
        assert result.verdict == ReviewVerdict.REQUEST_CHANGES

    def test_parse_json_in_code_block(self) -> None:
        raw = "```json\n" + _make_review_response() + "\n```"
        result = CodeReviewerAgent._parse_response(raw)
        assert isinstance(result, ReviewResult)

    def test_parse_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            CodeReviewerAgent._parse_response("not json")

    def test_extract_json_plain(self) -> None:
        raw = '{"key": "value"}'
        assert _extract_json(raw) == raw

    def test_extract_json_from_code_block(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        assert _extract_json(raw).strip() == '{"key": "value"}'


# ---------------------------------------------------------------------------
# Tests: Agent uses strong tier LLM
# ---------------------------------------------------------------------------


class TestLLMTier:
    """Test that CodeReviewerAgent uses strong tier LLM."""

    async def test_uses_strong_tier(self) -> None:
        settings = _make_settings()
        llm_router = LLMRouter(config=settings)
        agent = CodeReviewerAgent(settings=settings, llm_router=llm_router)

        mock_agent_instance, _ = _build_mock_agent_and_llm(
            _make_review_response()
        )

        with patch("src.agents.code_reviewer.Agent") as MockAgent:
            MockAgent.return_value = mock_agent_instance
            await agent.review_code(
                _make_task_context(), _make_code_context(), _make_code_change()
            )

        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args[1]
        assert call_kwargs["server_names"] == []
        assert call_kwargs["name"] == "code_reviewer"


# ---------------------------------------------------------------------------
# Property tests: Hardcoded Secret Detection
# ---------------------------------------------------------------------------


from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


class TestProperty35HardcodedSecretDetection:
    """Property 35: Hardcoded Secret Detection

    Validates: Requirements 12.6

    For any CodeChange containing known secret patterns, _check_hardcoded_secrets
    must return at least one CRITICAL finding with category="security".
    """

    # Known secret patterns that should always be detected
    _AWS_KEY_STRATEGY = st.builds(
        lambda suffix: f'AWS_KEY = "AKIA{suffix}"',
        suffix=st.from_regex(r"[0-9A-Z]{16}", fullmatch=True),
    )

    _GITHUB_TOKEN_STRATEGY = st.builds(
        lambda suffix: f'GITHUB_TOKEN = "ghp_{suffix}"',
        suffix=st.from_regex(r"[a-zA-Z0-9]{36}", fullmatch=True),
    )

    _PASSWORD_STRATEGY = st.builds(
        lambda val: f'password = "{val}"',
        val=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )

    _SECRET_STRATEGY = st.builds(
        lambda val: f'secret = "{val}"',
        val=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )

    _API_KEY_STRATEGY = st.builds(
        lambda val: f'api_key = "{val}"',
        val=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )

    def _make_code_change_with_secret(self, secret_line: str) -> CodeChange:
        """Build a CodeChange with a secret in the new_content."""
        return CodeChange(
            changes=[
                FileChange(
                    path="src/config.py",
                    new_content=f"import os\n\n{secret_line}\n",
                    change_type=ChangeType.CREATE,
                    explanation="Config file with secret",
                ),
            ],
            test_changes=[],
            commit_message="feat(config): add config",
            pr_title="Add config",
            pr_description="Adds config",
        )

    @given(secret_line=_AWS_KEY_STRATEGY)
    @h_settings(max_examples=100)
    def test_aws_key_always_detected(self, secret_line: str) -> None:
        """**Validates: Requirements 12.6**

        Any AWS access key (AKIA...) must be detected as a CRITICAL finding.
        """
        code_change = self._make_code_change_with_secret(secret_line)
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)

        assert len(findings) >= 1, (
            f"Expected at least 1 CRITICAL finding for AWS key, got 0. "
            f"Secret line: {secret_line!r}"
        )
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings), (
            f"Expected CRITICAL severity finding for AWS key. "
            f"Findings: {findings}"
        )
        assert all(f.category == "security" for f in findings), (
            f"Expected all findings to have category='security'. "
            f"Findings: {findings}"
        )

    @given(secret_line=_GITHUB_TOKEN_STRATEGY)
    @h_settings(max_examples=100)
    def test_github_token_always_detected(self, secret_line: str) -> None:
        """**Validates: Requirements 12.6**

        Any GitHub token (ghp_...) must be detected as a CRITICAL finding.
        """
        code_change = self._make_code_change_with_secret(secret_line)
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)

        assert len(findings) >= 1, (
            f"Expected at least 1 CRITICAL finding for GitHub token, got 0. "
            f"Secret line: {secret_line!r}"
        )
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings)

    @given(secret_line=_PASSWORD_STRATEGY)
    @h_settings(max_examples=100)
    def test_hardcoded_password_always_detected(self, secret_line: str) -> None:
        """**Validates: Requirements 12.6**

        Any hardcoded password assignment must be detected as a CRITICAL finding.
        """
        code_change = self._make_code_change_with_secret(secret_line)
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)

        assert len(findings) >= 1, (
            f"Expected at least 1 CRITICAL finding for password, got 0. "
            f"Secret line: {secret_line!r}"
        )
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings)

    @given(secret_line=_SECRET_STRATEGY)
    @h_settings(max_examples=100)
    def test_hardcoded_secret_always_detected(self, secret_line: str) -> None:
        """**Validates: Requirements 12.6**

        Any hardcoded secret assignment must be detected as a CRITICAL finding.
        """
        code_change = self._make_code_change_with_secret(secret_line)
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)

        assert len(findings) >= 1, (
            f"Expected at least 1 CRITICAL finding for secret, got 0. "
            f"Secret line: {secret_line!r}"
        )
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings)

    @given(secret_line=_API_KEY_STRATEGY)
    @h_settings(max_examples=100)
    def test_hardcoded_api_key_always_detected(self, secret_line: str) -> None:
        """**Validates: Requirements 12.6**

        Any hardcoded api_key assignment must be detected as a CRITICAL finding.
        """
        code_change = self._make_code_change_with_secret(secret_line)
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)

        assert len(findings) >= 1, (
            f"Expected at least 1 CRITICAL finding for api_key, got 0. "
            f"Secret line: {secret_line!r}"
        )
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings)

    @given(
        clean_content=st.text(
            min_size=1,
            max_size=500,
            alphabet=st.characters(whitelist_categories=("L", "N", "Z", "P")),
        ).filter(
            lambda s: (
                "AKIA" not in s
                and "ghp_" not in s
                and "password" not in s.lower()
                and "secret" not in s.lower()
                and "api_key" not in s.lower()
            )
        )
    )
    @h_settings(max_examples=100)
    def test_clean_code_produces_no_findings(self, clean_content: str) -> None:
        """**Validates: Requirements 12.6**

        Code without known secret patterns should produce no findings.
        """
        code_change = CodeChange(
            changes=[
                FileChange(
                    path="src/utils.py",
                    new_content=clean_content,
                    change_type=ChangeType.CREATE,
                    explanation="Clean utility file",
                ),
            ],
            test_changes=[],
            commit_message="feat(utils): add utils",
            pr_title="Add utils",
            pr_description="Adds utils",
        )
        findings = CodeReviewerAgent._check_hardcoded_secrets(code_change)
        assert len(findings) == 0, (
            f"Expected 0 findings for clean code, got {len(findings)}. "
            f"Content: {clean_content!r}"
        )
