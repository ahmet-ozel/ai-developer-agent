"""Jira helper utilities for comment formatting and secret masking."""

from __future__ import annotations


def format_jira_comment(agent_name: str, stage: str, message: str) -> str:
    """Format a structured Jira comment with agent name and pipeline stage.

    Produces a comment in the format::

         *{agent_name}* | Stage: {stage}

        {message}

    Args:
        agent_name: Name of the agent producing the comment.
        stage: Current pipeline stage identifier.
        message: The comment body text.

    Returns:
        Formatted Jira comment string.
    """
    return f" *{agent_name}* | Stage: {stage}\n\n{message}"


def mask_secrets(text: str, secrets: list[str]) -> str:
    """Replace every occurrence of each secret value in *text* with ``***``.

    Secrets are replaced longest-first so that a shorter secret that is a
    substring of a longer one does not partially mask the longer value.

    Args:
        text: The input text that may contain secret values.
        secrets: A list of secret strings to mask.

    Returns:
        The text with all secret occurrences replaced by ``***``.
    """
    if not text or not secrets:
        return text

    # Sort by length descending so longer secrets are masked first,
    # preventing partial replacement when one secret is a substring of another.
    sorted_secrets = sorted(
        (s for s in secrets if s),
        key=len,
        reverse=True,
    )

    for secret in sorted_secrets:
        text = text.replace(secret, "***")

    return text
