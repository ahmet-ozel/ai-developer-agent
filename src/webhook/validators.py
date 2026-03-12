"""Webhook payload validation for the AI Developer Agent.

Provides HMAC-SHA256 signature verification, event parsing,
and bot assignment detection for Jira webhook events.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from src.webhook.models import WebhookEvent


class WebhookValidator:
    """Validates and parses incoming Jira webhook payloads."""

    def validate_signature(
        self, payload: bytes, signature: str, secret: str
    ) -> bool:
        """Verify HMAC-SHA256 signature of the webhook payload.

        Args:
            payload: Raw request body bytes.
            signature: The signature string sent by Jira.
            secret: The shared webhook secret.

        Returns:
            True if the computed HMAC matches the provided signature.
        """
        computed = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed, signature)

    def parse_event(self, payload: dict[str, Any]) -> WebhookEvent:
        """Parse a Jira webhook payload dict into a WebhookEvent model.

        Extracts webhook_event, issue_key, assignee, issue_type,
        previous_assignee, project_key and stores the full payload
        as raw_payload.

        Args:
            payload: The parsed JSON body of the webhook request.

        Returns:
            A validated WebhookEvent instance.

        Raises:
            ValueError: If required fields are missing or malformed.
        """
        try:
            webhook_event = payload["webhookEvent"]
        except (KeyError, TypeError):
            raise ValueError("Missing required field: webhookEvent")

        issue = payload.get("issue")
        if not issue or not isinstance(issue, dict):
            raise ValueError("Missing required field: issue")

        issue_key = issue.get("key")
        if not issue_key:
            raise ValueError("Missing required field: issue.key")

        fields = issue.get("fields") or {}

        # Extract assignee — Jira sends nested object or None
        assignee_obj = fields.get("assignee")
        assignee: str | None = None
        if isinstance(assignee_obj, dict):
            assignee = (
                assignee_obj.get("name")
                or assignee_obj.get("accountId")
                or assignee_obj.get("displayName")
            )

        # Extract issue type
        issue_type_obj = fields.get("issuetype")
        issue_type: str | None = None
        if isinstance(issue_type_obj, dict):
            issue_type = issue_type_obj.get("name")

        # Extract project key
        project_obj = fields.get("project")
        project_key: str | None = None
        if isinstance(project_obj, dict):
            project_key = project_obj.get("key")

        # Extract previous assignee from changelog
        previous_assignee: str | None = None
        changelog = payload.get("changelog")
        if isinstance(changelog, dict):
            items = changelog.get("items") or []
            for item in items:
                if isinstance(item, dict) and item.get("field") == "assignee":
                    previous_assignee = item.get("fromString")
                    break

        return WebhookEvent(
            webhook_event=webhook_event,
            issue_key=issue_key,
            assignee=assignee,
            issue_type=issue_type,
            previous_assignee=previous_assignee,
            project_key=project_key,
            raw_payload=payload,
        )

    def is_bot_assignment(self, event: WebhookEvent, bot_username: str) -> bool:
        """Check whether the event's assignee matches the bot username.

        Args:
            event: A parsed WebhookEvent.
            bot_username: The configured bot user identifier.

        Returns:
            True if the event assignee equals bot_username.
        """
        return event.assignee == bot_username
