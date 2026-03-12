"""Structured logging infrastructure for the AI Developer Agent pipeline.

Provides JSON-formatted structured logging with required fields for
end-to-end traceability: timestamp, issue_key, agent_name, pipeline_stage,
log_level, and message. Also supports LLM usage metrics and pipeline stage
timing.

Requirements: 11.3, 11.7, 18.1, 18.2, 18.3, 18.4, 18.5
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Custom log formatter that outputs structured JSON log entries.

    Every entry includes: timestamp, issue_key, agent_name, pipeline_stage,
    log_level, and message. Extra fields are merged into the JSON object.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "issue_key": getattr(record, "issue_key", ""),
            "agent_name": getattr(record, "agent_name", ""),
            "pipeline_stage": getattr(record, "pipeline_stage", ""),
            "log_level": record.levelname,
            "message": record.getMessage(),
        }

        # Merge any extra fields attached to the record
        extra: dict[str, Any] = getattr(record, "extra_fields", {})
        if extra:
            entry.update(extra)

        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up JSON-formatted structured logging on the root logger.

    Replaces existing handlers with a single StreamHandler using
    :class:`JsonFormatter`.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


class PipelineLogger:
    """Wraps Python's logging module with structured JSON output.

    Stores the ``issue_key`` so every log entry is automatically tagged
    for end-to-end traceability.
    """

    def __init__(self, issue_key: str) -> None:
        self.issue_key = issue_key
        self._logger = logging.getLogger("pipeline")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _log(
        self,
        level: int,
        message: str,
        agent_name: str = "",
        stage: str = "",
        **extra: Any,
    ) -> None:
        """Emit a structured log record with the required fields."""
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.issue_key = self.issue_key  # type: ignore[attr-defined]
        record.agent_name = agent_name  # type: ignore[attr-defined]
        record.pipeline_stage = stage  # type: ignore[attr-defined]
        record.extra_fields = extra  # type: ignore[attr-defined]
        self._logger.handle(record)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_stage_start(self, stage: str, agent_name: str) -> None:
        """Log the start of a pipeline stage with a timestamp."""
        self._log(
            logging.INFO,
            f"Stage '{stage}' started",
            agent_name=agent_name,
            stage=stage,
            event="stage_start",
        )

    def log_stage_end(
        self, stage: str, agent_name: str, elapsed_ms: float
    ) -> None:
        """Log the completion of a pipeline stage with elapsed time."""
        self._log(
            logging.INFO,
            f"Stage '{stage}' completed in {elapsed_ms:.1f}ms",
            agent_name=agent_name,
            stage=stage,
            event="stage_end",
            elapsed_ms=elapsed_ms,
        )

    def log_llm_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Log LLM usage metrics for a single call."""
        self._log(
            logging.INFO,
            f"LLM call: {provider}/{model}",
            event="llm_usage",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def log_event(
        self,
        level: str,
        message: str,
        agent_name: str = "",
        stage: str = "",
        **extra: Any,
    ) -> None:
        """General-purpose structured log entry.

        ``level`` is a string like ``"INFO"``, ``"WARNING"``, ``"ERROR"``, etc.
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self._log(
            numeric_level,
            message,
            agent_name=agent_name,
            stage=stage,
            **extra,
        )
