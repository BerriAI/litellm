"""
Tests for the `clickhouse` spend-log callback.
"""

import json
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


import pytest

import litellm
from litellm.integrations.clickhouse.clickhouse_spend_logger import (
    ClickHouseSpendLogger,
    parse_traceparent,
    spend_log_row_from_payload,
    strip_cache_hit_suffix,
)
from litellm.integrations.clickhouse.schema import SPEND_LOGS_TABLE
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.tracing.types import SpendLogRecord

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{SPAN_ID}-01"


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-abc123",
        "trace_id": "trace-1",
        "session_id": "",
        "call_type": "acompletion",
        "response_cost": 0.00042,
        "status": "success",
        "custom_llm_provider": "openai",
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1_700_000_000.123,
        "endTime": 1_700_000_001.456,
        "completionStartTime": 1_700_000_000.5,
        "model": "gpt-4o",
        "model_id": "model-uuid",
        "model_group": "gpt-4o-group",
        "api_base": "https://api.openai.com/v1",
        "metadata": {
            "user_api_key_hash": "hashed-key",
            "user_api_key_alias": "my-key",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "Team One",
            "user_api_key_org_id": "org-1",
            "user_api_key_user_id": "user-1",
            "user_api_key_end_user_id": None,
            "requester_custom_headers": {"traceparent": TRACEPARENT},
            "usage_object": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 7},
            },
        },
        "cache_hit": None,
        "request_tags": ["prod", "agent"],
        "end_user": "end-user-1",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {"choices": [{"message": {"content": "hello"}}]},
        "error_str": None,
        "hidden_params": {"usage_object": None},
    }
    return {**payload, **overrides}


def test_is_a_custom_batch_logger():
    assert issubclass(ClickHouseSpendLogger, CustomBatchLogger)
    assert ClickHouseSpendLogger.table == SPEND_LOGS_TABLE


def test_success_row_mapping():
    row = spend_log_row_from_payload(_payload(), {})  # type: ignore[arg-type]

    assert set(row) == set(SpendLogRecord.__annotations__)
    assert row["request_id"] == "chatcmpl-abc123"
    assert row["response_id"] == "chatcmpl-abc123"
    assert row["spend"] == 0.00042
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (20, 10, 30)
    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (5, 7)
    assert row["start_time"] == 1_700_000_000_123
    assert row["end_time"] == 1_700_000_001_456
    assert row["completion_start_time"] == 1_700_000_000_500
    assert row["status"] == "success"
    assert row["cache_hit"] is False
    assert row["api_key"] == "hashed-key"
    assert row["key_alias"] == "my-key"
    assert row["team_id"] == "team-1"
    assert row["team_alias"] == "Team One"
    assert row["organization_id"] == "org-1"
    assert row["user"] == "user-1"
    assert row["end_user"] == "end-user-1"
    assert row["model_group"] == "gpt-4o-group"
    assert row["session_id"] == "trace-1"
    assert (row["trace_id"], row["span_id"]) == (TRACE_ID, SPAN_ID)
    assert row["request_tags"] == ["prod", "agent"]
    assert json.loads(row["messages"]) == [{"role": "user", "content": "hi"}]
    assert json.loads(row["metadata"])["user_api_key_alias"] == "my-key"


def test_anthropic_cache_fields_are_used_as_fallback():
    usage = {"cache_read_input_tokens": 11, "cache_creation_input_tokens": 3}
    payload = _payload()
    payload["metadata"] = {**payload["metadata"], "usage_object": usage}

    row = spend_log_row_from_payload(payload, {})  # type: ignore[arg-type]

    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (11, 3)


def test_explicit_session_id_wins_over_trace_id():
    row = spend_log_row_from_payload(
        _payload(),  # type: ignore[arg-type]
        {"litellm_params": {"metadata": {"session_id": "sess-9"}}},
    )
    assert row["session_id"] == "sess-9"


def test_cache_hit_id_is_stripped_for_response_id():
    row = spend_log_row_from_payload(
        _payload(id="chatcmpl-abc123_cache_hit1727600000.123456", cache_hit=True),  # type: ignore[arg-type]
        {},
    )
    assert row["request_id"] == "chatcmpl-abc123_cache_hit1727600000.123456"
    assert row["response_id"] == "chatcmpl-abc123"
    assert row["cache_hit"] is True
    assert strip_cache_hit_suffix("chatcmpl-xyz") == "chatcmpl-xyz"


def test_parse_traceparent_valid_missing_malformed():
    assert parse_traceparent(TRACEPARENT) == (TRACE_ID, SPAN_ID)
    assert parse_traceparent(None) == ("", "")
    assert parse_traceparent("") == ("", "")
    assert parse_traceparent("not-a-traceparent") == ("", "")
    assert parse_traceparent(f"00-{TRACE_ID}-{SPAN_ID}") == ("", "")
    assert parse_traceparent(f"00-{'0' * 32}-{SPAN_ID}-01") == ("", "")


def test_traceparent_from_proxy_server_request_headers():
    payload = _payload()
    payload["metadata"] = {**payload["metadata"], "requester_custom_headers": None}
    kwargs = {"litellm_params": {"proxy_server_request": {"headers": {"Traceparent": TRACEPARENT}}}}

    row = spend_log_row_from_payload(payload, kwargs)  # type: ignore[arg-type]

    assert (row["trace_id"], row["span_id"]) == (TRACE_ID, SPAN_ID)


def test_turn_off_message_logging_blanks_messages_and_response():
    with patch.object(litellm, "turn_off_message_logging", True):
        row = spend_log_row_from_payload(_payload(), {})  # type: ignore[arg-type]
    assert row["messages"] == ""
    assert row["response"] == ""


@pytest.mark.asyncio
async def test_failure_event_maps_status_and_error():
    client = MagicMock()
    client.insert_json_each_row = AsyncMock()
    logger = ClickHouseSpendLogger(storage=client)
    payload = _payload(status="failure", error_str="RateLimitError: slow down", response_cost=0.0)

    await logger.async_log_failure_event({"standard_logging_object": payload}, None, None, None)

    assert len(logger.log_queue) == 1
    row = logger.log_queue[0]
    assert row["status"] == "failure"
    assert row["error_str"] == "RateLimitError: slow down"


@pytest.mark.asyncio
async def test_missing_payload_and_bad_payload_never_raise():
    logger = ClickHouseSpendLogger(storage=MagicMock())
    await logger.async_log_success_event({}, None, None, None)
    await logger.async_log_success_event({"standard_logging_object": "garbage"}, None, None, None)
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_trace_ingest_requests_are_not_logged_as_spend():
    # OTLP exports hit POST /v1/traces; they are not LLM calls and must not create spend rows
    logger = ClickHouseSpendLogger(storage=MagicMock())
    payload = _payload(call_type="/v1/traces", status="failure")

    await logger.async_log_failure_event({"standard_logging_object": payload}, None, None, None)

    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_clickhouse_callback_resolves_via_factory(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_URL", "http://localhost:8123")
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])

    created = litellm_logging._init_custom_logger_compatible_class("clickhouse", None, None)
    assert isinstance(created, ClickHouseSpendLogger)
    assert litellm_logging._init_custom_logger_compatible_class("clickhouse", None, None) is created
    assert litellm_logging.get_custom_logger_compatible_class("clickhouse") is created
