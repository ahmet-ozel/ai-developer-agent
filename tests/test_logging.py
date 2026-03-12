"""Unit tests for structured logging infrastructure.

Covers:
- JsonFormatter produces valid JSON
- Log entries contain required fields (timestamp, issue_key, agent_name,
  pipeline_stage, log_level, message)
- LLM usage metrics are logged correctly
- Stage start/end logging works
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from src.pipeline.logging import JsonFormatter, PipelineLogger, configure_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CaptureHandler(logging.Handler):
    """Minimal handler that stores formatted log strings."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def _make_logger_with_capture() -> tuple[PipelineLogger, _CaptureHandler]:
    """Create a PipelineLogger whose output is captured for assertions."""
    handler = _CaptureHandler()
    handler.setFormatter(JsonFormatter())

    inner = logging.getLogger("pipeline.test")
    inner.handlers.clear()
    inner.addHandler(handler)
    inner.setLevel(logging.DEBUG)

    pl = PipelineLogger(issue_key="TEST-99")
    pl._logger = inner
    return pl, handler


# ---------------------------------------------------------------------------
# JsonFormatter tests
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    """JsonFormatter produces valid JSON with required fields."""

    def test_output_is_valid_json(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("INFO", "hello")
        raw = handler.records[0]
        parsed = json.loads(raw)  # must not raise
        assert isinstance(parsed, dict)

    def test_required_fields_present(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("WARNING", "something happened", agent_name="reader", stage="task_read")
        entry = json.loads(handler.records[0])

        required = {"timestamp", "issue_key", "agent_name", "pipeline_stage", "log_level", "message"}
        assert required.issubset(entry.keys())

    def test_issue_key_propagated(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("INFO", "test")
        entry = json.loads(handler.records[0])
        assert entry["issue_key"] == "TEST-99"

    def test_log_level_matches(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("ERROR", "boom")
        entry = json.loads(handler.records[0])
        assert entry["log_level"] == "ERROR"

    def test_extra_fields_merged(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("INFO", "with extras", custom_key="custom_val")
        entry = json.loads(handler.records[0])
        assert entry["custom_key"] == "custom_val"


# ---------------------------------------------------------------------------
# Stage start / end
# ---------------------------------------------------------------------------


class TestStageLogging:
    """log_stage_start and log_stage_end emit correct structured entries."""

    def test_stage_start_fields(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_stage_start("code_find", "code_finder")
        entry = json.loads(handler.records[0])

        assert entry["pipeline_stage"] == "code_find"
        assert entry["agent_name"] == "code_finder"
        assert entry["event"] == "stage_start"
        assert "started" in entry["message"].lower()

    def test_stage_end_fields(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_stage_end("code_find", "code_finder", elapsed_ms=123.4)
        entry = json.loads(handler.records[0])

        assert entry["pipeline_stage"] == "code_find"
        assert entry["agent_name"] == "code_finder"
        assert entry["event"] == "stage_end"
        assert entry["elapsed_ms"] == 123.4
        assert "123.4" in entry["message"]


# ---------------------------------------------------------------------------
# LLM usage metrics
# ---------------------------------------------------------------------------


class TestLLMUsageLogging:
    """log_llm_usage emits entries with provider, model, tokens, latency."""

    def test_llm_usage_fields(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_llm_usage(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
            latency_ms=1234.5,
        )
        entry = json.loads(handler.records[0])

        assert entry["provider"] == "openai"
        assert entry["model"] == "gpt-4o-mini"
        assert entry["input_tokens"] == 500
        assert entry["output_tokens"] == 200
        assert entry["latency_ms"] == 1234.5
        assert entry["event"] == "llm_usage"

    def test_llm_usage_message_contains_provider_model(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_llm_usage("anthropic", "claude-sonnet-4-20250514", 100, 50, 800.0)
        entry = json.loads(handler.records[0])
        assert "anthropic" in entry["message"]
        assert "claude-sonnet-4-20250514" in entry["message"]


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """configure_logging sets up JSON formatter on root logger."""

    def test_root_logger_has_json_handler(self) -> None:
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_configure_logging_clears_previous_handlers(self) -> None:
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        configure_logging()
        assert len(root.handlers) == 1


# ---------------------------------------------------------------------------
# General log_event
# ---------------------------------------------------------------------------


class TestLogEvent:
    """log_event supports arbitrary levels and extra kwargs."""

    def test_debug_level(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("DEBUG", "debug msg")
        entry = json.loads(handler.records[0])
        assert entry["log_level"] == "DEBUG"

    def test_default_empty_agent_and_stage(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("INFO", "bare event")
        entry = json.loads(handler.records[0])
        assert entry["agent_name"] == ""
        assert entry["pipeline_stage"] == ""

    def test_timestamp_is_iso_format(self) -> None:
        pl, handler = _make_logger_with_capture()
        pl.log_event("INFO", "ts check")
        entry = json.loads(handler.records[0])
        # ISO 8601 timestamps contain 'T' and end with timezone info
        assert "T" in entry["timestamp"]


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st


class TestLogEntryFormatProperties:
    """Property 31: Structured Log Entry Format

    Validates: Requirements 11.3, 11.7, 18.1, 18.2
    """

    REQUIRED_FIELDS = {"timestamp", "issue_key", "agent_name", "pipeline_stage", "log_level", "message"}

    # Property 31a: Any log_event call produces valid JSON with required fields
    @given(
        issue_key=st.text(min_size=1, max_size=20),
        message=st.text(min_size=1, max_size=100),
        level=st.sampled_from(["INFO", "WARNING", "ERROR", "DEBUG"]),
        agent_name=st.text(min_size=0, max_size=30),
        stage=st.text(min_size=0, max_size=30),
    )
    @settings(max_examples=100)
    def test_property_31a_log_event_valid_json_with_required_fields(
        self, issue_key: str, message: str, level: str, agent_name: str, stage: str
    ) -> None:
        """**Validates: Requirements 11.3, 11.7, 18.1, 18.2**

        Property 31a: Any log_event call produces valid JSON with required fields.
        """
        handler = _CaptureHandler()
        handler.setFormatter(JsonFormatter())

        inner = logging.getLogger(f"pipeline.prop31a.{issue_key[:8]}")
        inner.handlers.clear()
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        pl = PipelineLogger(issue_key=issue_key)
        pl._logger = inner

        pl.log_event(level, message, agent_name=agent_name, stage=stage)

        assert len(handler.records) == 1, "Expected exactly one log record"
        entry = json.loads(handler.records[0])  # must not raise
        assert isinstance(entry, dict)
        assert self.REQUIRED_FIELDS.issubset(entry.keys()), (
            f"Missing fields: {self.REQUIRED_FIELDS - entry.keys()}"
        )

    # Property 31b: log_stage_start always produces valid JSON with stage_start event
    @given(
        issue_key=st.text(min_size=1, max_size=20),
        stage=st.text(min_size=1, max_size=30),
        agent_name=st.text(min_size=1, max_size=30),
    )
    @settings(max_examples=100)
    def test_property_31b_log_stage_start_valid_json(
        self, issue_key: str, stage: str, agent_name: str
    ) -> None:
        """**Validates: Requirements 11.3, 11.7, 18.1, 18.2**

        Property 31b: log_stage_start always produces valid JSON with stage_start event.
        """
        handler = _CaptureHandler()
        handler.setFormatter(JsonFormatter())

        inner = logging.getLogger(f"pipeline.prop31b.{issue_key[:8]}")
        inner.handlers.clear()
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        pl = PipelineLogger(issue_key=issue_key)
        pl._logger = inner

        pl.log_stage_start(stage, agent_name)

        assert len(handler.records) == 1
        entry = json.loads(handler.records[0])  # must not raise
        assert isinstance(entry, dict)
        assert self.REQUIRED_FIELDS.issubset(entry.keys())
        assert entry["event"] == "stage_start"
        assert entry["pipeline_stage"] == stage
        assert entry["agent_name"] == agent_name

    # Property 31c: log_stage_end always includes elapsed_ms in output
    @given(
        issue_key=st.text(min_size=1, max_size=20),
        stage=st.text(min_size=1, max_size=30),
        agent_name=st.text(min_size=1, max_size=30),
        elapsed_ms=st.floats(min_value=0, max_value=100000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_property_31c_log_stage_end_includes_elapsed_ms(
        self, issue_key: str, stage: str, agent_name: str, elapsed_ms: float
    ) -> None:
        """**Validates: Requirements 11.3, 11.7, 18.1, 18.2**

        Property 31c: log_stage_end always includes elapsed_ms in output.
        """
        handler = _CaptureHandler()
        handler.setFormatter(JsonFormatter())

        inner = logging.getLogger(f"pipeline.prop31c.{issue_key[:8]}")
        inner.handlers.clear()
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        pl = PipelineLogger(issue_key=issue_key)
        pl._logger = inner

        pl.log_stage_end(stage, agent_name, elapsed_ms=elapsed_ms)

        assert len(handler.records) == 1
        entry = json.loads(handler.records[0])  # must not raise
        assert isinstance(entry, dict)
        assert self.REQUIRED_FIELDS.issubset(entry.keys())
        assert entry["event"] == "stage_end"
        assert "elapsed_ms" in entry
        assert entry["elapsed_ms"] == elapsed_ms

    # Property 31d: timestamp is always ISO 8601 format (contains 'T')
    @given(
        issue_key=st.text(min_size=1, max_size=20),
        message=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_property_31d_timestamp_is_iso8601(
        self, issue_key: str, message: str
    ) -> None:
        """**Validates: Requirements 11.3, 11.7, 18.1, 18.2**

        Property 31d: timestamp is always ISO 8601 format (contains 'T').
        """
        handler = _CaptureHandler()
        handler.setFormatter(JsonFormatter())

        inner = logging.getLogger(f"pipeline.prop31d.{issue_key[:8]}")
        inner.handlers.clear()
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        pl = PipelineLogger(issue_key=issue_key)
        pl._logger = inner

        pl.log_event("INFO", message)

        assert len(handler.records) == 1
        entry = json.loads(handler.records[0])
        assert "timestamp" in entry
        assert "T" in entry["timestamp"], (
            f"Timestamp '{entry['timestamp']}' is not ISO 8601 format (missing 'T')"
        )


class TestLLMUsageMetricsProperties:
    """Property 32: LLM Usage Metrics Logging

    Validates: Requirements 18.4
    """

    REQUIRED_BASE_FIELDS = {
        "timestamp",
        "issue_key",
        "agent_name",
        "pipeline_stage",
        "log_level",
        "message",
    }

    # Property 32a: log_llm_usage always produces valid JSON with all required LLM fields
    @given(
        provider=st.text(min_size=1, max_size=30),
        model=st.text(min_size=1, max_size=50),
        input_tokens=st.integers(min_value=0, max_value=1_000_000),
        output_tokens=st.integers(min_value=0, max_value=100_000),
        latency_ms=st.floats(min_value=0, max_value=300_000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_property_32a_llm_usage_valid_json_with_required_fields(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """**Validates: Requirements 18.4**

        Property 32a: log_llm_usage always produces valid JSON with all required LLM fields.
        """
        pl, handler = _make_logger_with_capture()
        pl.log_llm_usage(provider, model, input_tokens, output_tokens, latency_ms)

        assert len(handler.records) == 1, "Expected exactly one log record"
        entry = json.loads(handler.records[0])  # must not raise
        assert isinstance(entry, dict)

        # All 6 base required fields
        assert self.REQUIRED_BASE_FIELDS.issubset(entry.keys()), (
            f"Missing base fields: {self.REQUIRED_BASE_FIELDS - entry.keys()}"
        )

        # LLM-specific required fields
        assert "provider" in entry
        assert "model" in entry
        assert "input_tokens" in entry
        assert "output_tokens" in entry
        assert "latency_ms" in entry
        assert entry["event"] == "llm_usage"

    # Property 32b: input_tokens and output_tokens are preserved exactly
    @given(
        input_tokens=st.integers(min_value=0, max_value=1_000_000),
        output_tokens=st.integers(min_value=0, max_value=100_000),
    )
    @settings(max_examples=100)
    def test_property_32b_token_counts_preserved_exactly(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """**Validates: Requirements 18.4**

        Property 32b: input_tokens and output_tokens are preserved exactly in the log entry.
        """
        pl, handler = _make_logger_with_capture()
        pl.log_llm_usage("openai", "gpt-4o", input_tokens, output_tokens, 100.0)

        entry = json.loads(handler.records[0])
        assert entry["input_tokens"] == input_tokens, (
            f"input_tokens mismatch: expected {input_tokens}, got {entry['input_tokens']}"
        )
        assert entry["output_tokens"] == output_tokens, (
            f"output_tokens mismatch: expected {output_tokens}, got {entry['output_tokens']}"
        )

    # Property 32c: latency_ms is preserved exactly
    @given(
        latency_ms=st.floats(min_value=0, max_value=300_000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_property_32c_latency_ms_preserved_exactly(
        self,
        latency_ms: float,
    ) -> None:
        """**Validates: Requirements 18.4**

        Property 32c: latency_ms is preserved exactly in the log entry.
        """
        pl, handler = _make_logger_with_capture()
        pl.log_llm_usage("anthropic", "claude-3", 100, 50, latency_ms)

        entry = json.loads(handler.records[0])
        assert entry["latency_ms"] == latency_ms, (
            f"latency_ms mismatch: expected {latency_ms}, got {entry['latency_ms']}"
        )

    # Property 32d: provider and model appear in the message field
    @given(
        provider=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        ),
        model=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=100)
    def test_property_32d_provider_and_model_in_message(
        self,
        provider: str,
        model: str,
    ) -> None:
        """**Validates: Requirements 18.4**

        Property 32d: provider and model always appear in the log entry message field.
        """
        pl, handler = _make_logger_with_capture()
        pl.log_llm_usage(provider, model, 0, 0, 0.0)

        entry = json.loads(handler.records[0])
        assert provider in entry["message"], (
            f"provider '{provider}' not found in message: '{entry['message']}'"
        )
        assert model in entry["message"], (
            f"model '{model}' not found in message: '{entry['message']}'"
        )
