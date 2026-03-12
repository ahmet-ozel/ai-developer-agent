"""Unit tests for WebhookValidator.

Covers signature validation, event parsing, and bot assignment detection.
Requirements: 1.1, 1.2, 1.3, 1.6, 1.7
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from src.webhook.models import WebhookEvent
from src.webhook.validators import WebhookValidator


@pytest.fixture
def validator() -> WebhookValidator:
    return WebhookValidator()


def _compute_signature(payload: bytes, secret: str) -> str:
    """Helper: compute HMAC-SHA256 hex digest for a payload."""
    return hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()


# ---------------------------------------------------------------------------
# validate_signature
# ---------------------------------------------------------------------------


class TestValidateSignature:
    """Requirement 1.1: Validate webhook signature against configured secret."""

    def test_valid_signature_returns_true(self, validator: WebhookValidator):
        payload = b'{"webhookEvent":"jira:issue_updated"}'
        secret = "my-secret"
        sig = _compute_signature(payload, secret)
        assert validator.validate_signature(payload, sig, secret) is True

    def test_invalid_signature_returns_false(self, validator: WebhookValidator):
        payload = b'{"webhookEvent":"jira:issue_updated"}'
        secret = "my-secret"
        assert validator.validate_signature(payload, "bad-signature", secret) is False

    def test_wrong_secret_returns_false(self, validator: WebhookValidator):
        payload = b'{"data": "test"}'
        sig = _compute_signature(payload, "correct-secret")
        assert validator.validate_signature(payload, sig, "wrong-secret") is False

    def test_empty_payload(self, validator: WebhookValidator):
        payload = b""
        secret = "s"
        sig = _compute_signature(payload, secret)
        assert validator.validate_signature(payload, sig, secret) is True

    def test_mutated_payload_returns_false(self, validator: WebhookValidator):
        payload = b"original"
        secret = "sec"
        sig = _compute_signature(payload, secret)
        assert validator.validate_signature(b"tampered", sig, secret) is False


# ---------------------------------------------------------------------------
# parse_event
# ---------------------------------------------------------------------------


def _make_payload(
    *,
    webhook_event: str = "jira:issue_updated",
    issue_key: str = "PROJ-1",
    assignee_name: str | None = "ai-developer",
    issue_type: str | None = "Story",
    project_key: str | None = "PROJ",
    previous_assignee: str | None = None,
) -> dict:
    """Build a minimal valid Jira webhook payload dict."""
    payload: dict = {
        "webhookEvent": webhook_event,
        "issue": {
            "key": issue_key,
            "fields": {
                "assignee": {"name": assignee_name} if assignee_name else None,
                "issuetype": {"name": issue_type} if issue_type else None,
                "project": {"key": project_key} if project_key else None,
            },
        },
    }
    if previous_assignee is not None:
        payload["changelog"] = {
            "items": [
                {
                    "field": "assignee",
                    "fromString": previous_assignee,
                }
            ]
        }
    return payload


class TestParseEvent:
    """Requirement 1.3: Parse event payload and extract assignee change info."""

    def test_valid_payload_extracts_all_fields(self, validator: WebhookValidator):
        payload = _make_payload(
            webhook_event="jira:issue_updated",
            issue_key="TEST-42",
            assignee_name="bot-user",
            issue_type="Bug",
            project_key="TEST",
            previous_assignee="john.doe",
        )
        event = validator.parse_event(payload)

        assert event.webhook_event == "jira:issue_updated"
        assert event.issue_key == "TEST-42"
        assert event.assignee == "bot-user"
        assert event.issue_type == "Bug"
        assert event.project_key == "TEST"
        assert event.previous_assignee == "john.doe"
        assert event.raw_payload == payload

    def test_missing_webhook_event_raises(self, validator: WebhookValidator):
        with pytest.raises(ValueError, match="webhookEvent"):
            validator.parse_event({"issue": {"key": "X-1", "fields": {}}})

    def test_missing_issue_raises(self, validator: WebhookValidator):
        with pytest.raises(ValueError, match="issue"):
            validator.parse_event({"webhookEvent": "jira:issue_updated"})

    def test_missing_issue_key_raises(self, validator: WebhookValidator):
        with pytest.raises(ValueError, match="issue.key"):
            validator.parse_event(
                {"webhookEvent": "jira:issue_updated", "issue": {"fields": {}}}
            )

    def test_none_assignee(self, validator: WebhookValidator):
        payload = _make_payload(assignee_name=None)
        event = validator.parse_event(payload)
        assert event.assignee is None

    def test_no_changelog_gives_none_previous_assignee(
        self, validator: WebhookValidator
    ):
        payload = _make_payload()
        event = validator.parse_event(payload)
        assert event.previous_assignee is None

    def test_empty_fields_still_parses(self, validator: WebhookValidator):
        payload = {
            "webhookEvent": "jira:issue_created",
            "issue": {"key": "A-1", "fields": {}},
        }
        event = validator.parse_event(payload)
        assert event.issue_key == "A-1"
        assert event.assignee is None
        assert event.issue_type is None
        assert event.project_key is None

    def test_non_dict_payload_raises(self, validator: WebhookValidator):
        with pytest.raises(ValueError):
            validator.parse_event(None)  # type: ignore[arg-type]

    def test_issue_as_non_dict_raises(self, validator: WebhookValidator):
        with pytest.raises(ValueError, match="issue"):
            validator.parse_event(
                {"webhookEvent": "jira:issue_updated", "issue": "not-a-dict"}
            )

    def test_assignee_with_account_id(self, validator: WebhookValidator):
        """Jira Cloud may send accountId instead of name."""
        payload = {
            "webhookEvent": "jira:issue_updated",
            "issue": {
                "key": "X-1",
                "fields": {
                    "assignee": {"accountId": "abc123"},
                },
            },
        }
        event = validator.parse_event(payload)
        assert event.assignee == "abc123"


# ---------------------------------------------------------------------------
# is_bot_assignment
# ---------------------------------------------------------------------------


class TestIsBotAssignment:
    """Requirement 1.7: Verify assignee is the bot user."""

    def test_matching_assignee_returns_true(self, validator: WebhookValidator):
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key="P-1",
            assignee="ai-developer",
        )
        assert validator.is_bot_assignment(event, "ai-developer") is True

    def test_different_assignee_returns_false(self, validator: WebhookValidator):
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key="P-1",
            assignee="john.doe",
        )
        assert validator.is_bot_assignment(event, "ai-developer") is False

    def test_none_assignee_returns_false(self, validator: WebhookValidator):
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key="P-1",
            assignee=None,
        )
        assert validator.is_bot_assignment(event, "ai-developer") is False

    def test_case_sensitive(self, validator: WebhookValidator):
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key="P-1",
            assignee="AI-Developer",
        )
        assert validator.is_bot_assignment(event, "ai-developer") is False


# =========================================================================
# Property Tests (Hypothesis)
# =========================================================================

import hashlib
import hmac as hmac_module

from hypothesis import given, settings as h_settings
from hypothesis import strategies as st


_payloads = st.binary(min_size=0, max_size=1000)
_secrets = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N")))
_issue_keys = st.from_regex(r"[A-Z]{2,10}-\d{1,5}", fullmatch=True)
_usernames = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N")))
_event_types = st.sampled_from([
    "jira:issue_created",
    "jira:issue_updated",
    "jira:issue_deleted",
    "comment_created",
])


class TestWebhookSignatureValidationProperty:
    """Property 1: Webhook Signature Validation Round-Trip.

    Validates: Requirements 1.1, 1.2, 12.1
    """

    @given(payload=_payloads, secret=_secrets)
    @h_settings(max_examples=100)
    def test_correct_signature_always_validates(self, payload: bytes, secret: str) -> None:
        """Correct HMAC-SHA256 signature always validates as True."""
        sig = hmac_module.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        validator = WebhookValidator()
        assert validator.validate_signature(payload, sig, secret) is True

    @given(payload=_payloads, secret=_secrets)
    @h_settings(max_examples=100)
    def test_wrong_secret_always_fails(self, payload: bytes, secret: str) -> None:
        """Signature computed with wrong secret always fails validation."""
        sig = hmac_module.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        wrong_secret = secret + "x"
        validator = WebhookValidator()
        assert validator.validate_signature(payload, sig, wrong_secret) is False

    @given(payload=_payloads, secret=_secrets)
    @h_settings(max_examples=100)
    def test_mutated_payload_always_fails(self, payload: bytes, secret: str) -> None:
        """Signature computed for original payload fails for mutated payload."""
        sig = hmac_module.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        mutated = payload + b"x"
        validator = WebhookValidator()
        assert validator.validate_signature(mutated, sig, secret) is False

    @given(payload=_payloads, secret=_secrets)
    @h_settings(max_examples=100)
    def test_bad_signature_string_always_fails(self, payload: bytes, secret: str) -> None:
        """Arbitrary non-matching signature string always fails."""
        validator = WebhookValidator()
        assert validator.validate_signature(payload, "bad-signature-xyz", secret) is False


class TestWebhookEventRoutingProperty:
    """Property 2: Webhook Event Routing.

    Validates: Requirements 1.4, 1.5, 1.7
    """

    @given(
        issue_key=_issue_keys,
        bot_username=_usernames,
    )
    @h_settings(max_examples=100)
    def test_bot_assignee_always_returns_true(
        self, issue_key: str, bot_username: str
    ) -> None:
        """is_bot_assignment always returns True when assignee matches bot_username."""
        from src.webhook.models import WebhookEvent
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key=issue_key,
            assignee=bot_username,
        )
        validator = WebhookValidator()
        assert validator.is_bot_assignment(event, bot_username) is True

    @given(
        issue_key=_issue_keys,
        bot_username=_usernames,
        other_username=_usernames,
    )
    @h_settings(max_examples=100)
    def test_different_assignee_always_returns_false(
        self, issue_key: str, bot_username: str, other_username: str
    ) -> None:
        """is_bot_assignment always returns False when assignee differs from bot_username."""
        from hypothesis import assume
        assume(other_username != bot_username)
        from src.webhook.models import WebhookEvent
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key=issue_key,
            assignee=other_username,
        )
        validator = WebhookValidator()
        assert validator.is_bot_assignment(event, bot_username) is False

    @given(issue_key=_issue_keys, bot_username=_usernames)
    @h_settings(max_examples=50)
    def test_none_assignee_always_returns_false(
        self, issue_key: str, bot_username: str
    ) -> None:
        """is_bot_assignment always returns False when assignee is None."""
        from src.webhook.models import WebhookEvent
        event = WebhookEvent(
            webhook_event="jira:issue_updated",
            issue_key=issue_key,
            assignee=None,
        )
        validator = WebhookValidator()
        assert validator.is_bot_assignment(event, bot_username) is False


class TestWebhookPayloadParsingProperty:
    """Property 3: Webhook Payload Parsing.

    Validates: Requirements 1.3, 1.6
    """

    @given(
        webhook_event=_event_types,
        issue_key=_issue_keys,
    )
    @h_settings(max_examples=100)
    def test_valid_payload_always_parses(
        self, webhook_event: str, issue_key: str
    ) -> None:
        """Valid minimal payload always parses successfully."""
        payload = {
            "webhookEvent": webhook_event,
            "issue": {
                "key": issue_key,
                "fields": {},
            },
        }
        validator = WebhookValidator()
        event = validator.parse_event(payload)
        assert event.webhook_event == webhook_event
        assert event.issue_key == issue_key

    @given(
        webhook_event=_event_types,
        issue_key=_issue_keys,
        assignee=st.one_of(st.none(), _usernames),
    )
    @h_settings(max_examples=100)
    def test_assignee_field_extracted_correctly(
        self, webhook_event: str, issue_key: str, assignee: str | None
    ) -> None:
        """Assignee field is always extracted correctly from payload."""
        payload = {
            "webhookEvent": webhook_event,
            "issue": {
                "key": issue_key,
                "fields": {
                    "assignee": {"name": assignee} if assignee else None,
                },
            },
        }
        validator = WebhookValidator()
        event = validator.parse_event(payload)
        assert event.assignee == assignee
