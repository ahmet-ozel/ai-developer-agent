"""Webhook data models for the AI Developer Agent.

Handles Jira webhook event parsing and validation.
"""

from typing import Optional

from pydantic import BaseModel


class WebhookEvent(BaseModel):
    webhook_event: str  # "jira:issue_created" | "jira:issue_updated"
    issue_key: str
    assignee: Optional[str] = None
    issue_type: Optional[str] = None
    previous_assignee: Optional[str] = None
    project_key: Optional[str] = None
    raw_payload: Optional[dict] = None
