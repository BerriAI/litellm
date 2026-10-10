"""
Tests for the `clickhouse` spend-log callback.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.clickhouse.clickhouse_spend_logger import (
    ClickHouseSpendLogger,
    parse_traceparent,
    spend_log_row_from_payload,
    strip_cache_hit_suffix,
)
from litellm.integrations.clickhouse.context import lens_analysis
from litellm.integrations.clickhouse.schema import SPEND_LOGS_TABLE
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.secret_redaction import REDACTED
from litellm.tracing.types import SpendLogRecord
from litellm.types.utils import StandardLoggingPayload

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(Mapping[str, JsonValue])

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{SPAN_ID}-01"


class _StandardPayloadBuilder(Protocol):
    def __call__(
        self,
        *,
        kwargs: dict[str, object],
        init_response_obj: object,
        start_time: datetime,
        end_time: datetime,
        logging_obj: litellm_logging.Logging,
        status: Literal["success", "failure"],
    ) -> StandardLoggingPayload | None: ...


class _ClickHouseLogger(Protocol):
    log_queue: Sequence[Mapping[str, object]]

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None: ...

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None: ...


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-abc123",
        "litellm_call_id": "gateway-call",
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


def _standard_payload(
    *,
    response_cost: float | None,
    status: Literal["success", "failure"] = "success",
    metadata: Mapping[str, object] = MappingProxyType({}),
) -> StandardLoggingPayload:
    now: Final = datetime.now(timezone.utc)
    logging_obj: Final = litellm_logging.Logging(
        model="gpt-4o",
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=now,
        litellm_call_id="standard-payload-call",
        function_id="standard-payload-function",
    )
    kwargs: Final[dict[str, object]] = {
        "litellm_call_id": "standard-payload-call",
        "model": "gpt-4o",
        "messages": [],
        "call_type": "acompletion",
        "response_cost": response_cost,
        "litellm_params": {"metadata": dict(metadata)},
    }
    payload_builder: Final = cast(_StandardPayloadBuilder, litellm_logging.get_standard_logging_object_payload)
    payload: Final = payload_builder(
        kwargs=kwargs,
        init_response_obj={},
        start_time=now,
        end_time=now,
        logging_obj=logging_obj,
        status=status,
    )
    assert payload is not None
    return payload


def test_is_a_custom_batch_logger():
    assert issubclass(ClickHouseSpendLogger, CustomBatchLogger)
    assert ClickHouseSpendLogger.table == SPEND_LOGS_TABLE


def test_success_row_mapping():
    row: Final = spend_log_row_from_payload(cast(StandardLoggingPayload, _payload()), {"response_cost": 0.00042})

    assert set(row) == set(SpendLogRecord.__annotations__)
    assert row["request_id"] == "chatcmpl-abc123"
    assert row["response_id"] == "chatcmpl-abc123"
    assert row["litellm_call_id"] == "gateway-call"
    assert row["provider_request_id"] == ""
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


@pytest.mark.parametrize("header_source", ("response_headers", "additional_headers"))
def test_provider_request_id_stays_separate_from_message_and_gateway_ids(header_source: str) -> None:
    headers: Final = {"request-id": "req_native"}
    hidden: Final = {"additional_headers": headers} if header_source == "additional_headers" else {}
    kwargs: Final = {"response_cost": 0.00042, "response_headers": headers if header_source == "response_headers" else None}
    payload: Final = cast(StandardLoggingPayload, _payload(id="msg_native", hidden_params=hidden))
    row: Final = spend_log_row_from_payload(payload, kwargs)
    assert (row["provider_request_id"], row["response_id"], row["litellm_call_id"]) == (
        "req_native", "msg_native", "gateway-call"
    )


@pytest.mark.parametrize("status", ("success", "failure"))
@pytest.mark.asyncio
async def test_custom_request_metadata_is_redacted_before_clickhouse_logging(
    status: Literal["success", "failure"],
) -> None:
    custom: Final = {
        "project": "example",
        "labels": {"priority": 3, "enabled": False},
        "steps": ["plan", {"duration": 0}],
        "empty": None,
        "api_key": "caller-api-key",
        "auth": {"token": "nested-auth-token"},
        "prompt": "private prompt",
    }
    payload: Final = _standard_payload(
        response_cost=0.00042,
        status=status,
        metadata={**custom, "user_api_key_team_id": "payload-team"},
    )
    kwargs: Final = {
        "standard_logging_object": payload,
        "response_cost": 0.00042,
        "litellm_params": {
            "metadata": {**custom, "shared": "request", "user_api_key_team_id": "untrusted-team"},
            "litellm_metadata": {
                "integration": "agent",
                "shared": "model",
                "litellm_lens_internal": True,
                "user_api_key_auth": {"api_key": "internal-api-key"},
                "user_api_key_budget_reservation": {"token": "internal-token"},
                "proxy_server_request": {"headers": {"authorization": "internal-auth"}},
                "parent_otel_span": object(),
            },
        },
    }
    logger: Final = cast(_ClickHouseLogger, ClickHouseSpendLogger(storage=MagicMock()))

    if status == "success":
        await logger.async_log_success_event(kwargs, None, None, None)
    else:
        await logger.async_log_failure_event(kwargs, None, None, None)

    log_rows: Final = logger.log_queue
    assert len(log_rows) == 1
    metadata_json: Final = cast(str, log_rows[0]["metadata"])
    metadata: Final = _JSON_OBJECT_ADAPTER.validate_json(metadata_json)
    serialized_metadata: Final = json.dumps(metadata)
    assert metadata["api_key"] == REDACTED
    assert metadata["auth"] == REDACTED
    assert "caller-api-key" not in serialized_metadata
    assert "nested-auth-token" not in serialized_metadata
    assert metadata["project"] == "example"
    assert metadata["labels"] == {"priority": 3, "enabled": False}
    assert metadata["steps"] == ["plan", {"duration": 0}]
    assert metadata["prompt"] == "private prompt"
    assert metadata["integration"] == "agent"
    assert metadata["shared"] == "request"
    assert metadata["user_api_key_team_id"] == "payload-team"
    assert "user_api_key_auth" not in metadata
    assert "user_api_key_budget_reservation" not in metadata
    assert "proxy_server_request" not in metadata
    litellm_params: Final = cast(Mapping[str, object], kwargs["litellm_params"])
    request_metadata: Final = cast(Mapping[str, object], litellm_params["metadata"])
    assert request_metadata == {
        **custom,
        "shared": "request",
        "user_api_key_team_id": "untrusted-team",
    }
    assert log_rows[0]["team_id"] == "payload-team"


@pytest.mark.asyncio
async def test_turn_off_message_logging_omits_all_custom_request_metadata() -> None:
    custom: Final = {
        "project": "example",
        "api_key": "caller-api-key",
        "auth": {"token": "nested-auth-token"},
        "prompt": "private prompt",
    }
    payload: Final = _standard_payload(
        response_cost=0.00042,
        metadata={**custom, "user_api_key_team_id": "payload-team"},
    )
    kwargs: Final = {
        "standard_logging_object": payload,
        "response_cost": 0.00042,
        "litellm_params": {"metadata": {**custom, "user_api_key_team_id": "untrusted-team"}},
    }
    logger: Final = cast(_ClickHouseLogger, ClickHouseSpendLogger(storage=MagicMock()))

    with patch.object(litellm, "turn_off_message_logging", True):
        await logger.async_log_success_event(kwargs, None, None, None)

    log_rows: Final = logger.log_queue
    assert len(log_rows) == 1
    metadata_json: Final = cast(str, log_rows[0]["metadata"])
    metadata: Final = _JSON_OBJECT_ADAPTER.validate_json(metadata_json)
    standard_metadata: Final = cast(Mapping[str, object], payload["metadata"])
    assert metadata == {
        **standard_metadata,
        "litellm_lens_internal": False,
    }
    assert {"project", "api_key", "auth", "prompt"}.isdisjoint(metadata)


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
    assert row["litellm_call_id"] == "gateway-call"
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


@pytest.mark.asyncio
async def test_caller_tags_cannot_impersonate_internal_lens_analysis():
    import asyncio

    payload: Final = _payload(
        request_tags=["litellm-engine"],
        metadata={"litellm_lens_internal": True},
    )

    async def logged_internal():
        return spend_log_row_from_payload(payload, {})

    external: Final = spend_log_row_from_payload(payload, {})
    with lens_analysis():
        callback: Final = asyncio.create_task(logged_internal())
    internal: Final = await callback
    following: Final = spend_log_row_from_payload(payload, {})
    assert json.loads(external["metadata"])["litellm_lens_internal"] is False
    assert json.loads(internal["metadata"])["litellm_lens_internal"] is True
    assert json.loads(following["metadata"])["litellm_lens_internal"] is False
    assert external["request_tags"] == ["litellm-engine"]


def _minimal_payload(request_id: str, *, status: str, cost: float) -> dict[str, object]:
    return {
        "id": request_id,
        "call_type": "acompletion",
        "response_cost": cost,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "startTime": 1_700_000_000.123,
        "endTime": 1_700_000_001.456,
        "metadata": {"user_api_key_hash": "key-a", "user_api_key_team_id": "team-a"},
        "model": "test-model",
        "status": status,
    }


@pytest.mark.asyncio
async def test_success_and_failure_events_write_scoped_spend_rows():
    storage = MagicMock()
    storage.ensure_schema = AsyncMock()
    storage.insert_rows = AsyncMock()
    logger = ClickHouseSpendLogger(storage=storage)
    now = datetime.now(timezone.utc)

    await logger.async_log_success_event(
        {
            "standard_logging_object": _minimal_payload("response-1", status="success", cost=0.25),
            "response_cost": 0.25,
        },
        None,
        now,
        now,
    )
    await logger.async_log_failure_event(
        {
            "standard_logging_object": _minimal_payload("response-2_cache_hit123", status="failure", cost=0.0),
            "response_cost": 0.0,
        },
        None,
        now,
        now,
    )
    await logger.flush_queue()
    if logger._flush_task is not None:
        logger._flush_task.cancel()

    storage.ensure_schema.assert_not_awaited()
    assert storage.insert_rows.await_count == 1
    table, rows = storage.insert_rows.await_args.args
    assert table == "spend_logs"
    expected = [
        {
            "request_id": "response-1",
            "response_id": "response-1",
            "call_type": "acompletion",
            "api_key": "key-a",
            "team_id": "team-a",
            "model": "test-model",
            "spend": 0.25,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "start_time": 1_700_000_000_123,
            "end_time": 1_700_000_001_456,
            "status": "success",
            "cache_hit": False,
        },
        {
            "request_id": "response-2_cache_hit123",
            "response_id": "response-2",
            "call_type": "acompletion",
            "api_key": "key-a",
            "team_id": "team-a",
            "model": "test-model",
            "spend": 0.0,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "start_time": 1_700_000_000_123,
            "end_time": 1_700_000_001_456,
            "status": "failure",
            "cache_hit": False,
        },
    ]
    assert len(rows) == len(expected)
    for row, original_fields in zip(rows, expected):
        assert {key: row[key] for key in original_fields} == original_fields


@pytest.mark.asyncio
async def test_trace_ingest_and_invalid_payload_do_not_write_spend():
    storage = MagicMock()
    storage.ensure_schema = AsyncMock()
    logger = ClickHouseSpendLogger(storage=storage)
    now = datetime.now(timezone.utc)

    await logger.async_log_success_event(
        {"standard_logging_object": {**_minimal_payload("trace", status="success", cost=0), "call_type": "/v1/traces"}},
        None,
        now,
        now,
    )
    await logger.async_log_success_event({"standard_logging_object": "invalid"}, None, now, now)

    assert logger.log_queue == []
    storage.ensure_schema.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_line_item_success_event_does_not_write_spend(monkeypatch: pytest.MonkeyPatch):
    # A batch line item carries call_type=acompletion + litellm_params.batch_parent_id.
    # The aggregate aretrieve_batch row already bills the batch, so a per-line spend row
    # would bill it twice.
    monkeypatch.setattr(litellm, "store_batch_line_items_in_callbacks", True, raising=False)
    storage = MagicMock()
    logger = ClickHouseSpendLogger(storage=storage)
    now = datetime.now(timezone.utc)

    await logger.async_log_success_event(
        {
            "standard_logging_object": _minimal_payload("chatcmpl-line-1", status="success", cost=0.25),
            "call_type": "acompletion",
            "litellm_params": {"batch_parent_id": "batch-1"},
        },
        None,
        now,
        now,
    )

    assert logger.log_queue == []

    await logger.async_log_success_event(
        {
            "standard_logging_object": _minimal_payload("chatcmpl-live-1", status="success", cost=0.25),
            "call_type": "acompletion",
            "litellm_params": {},
        },
        None,
        now,
        now,
    )

    # The aggregate aretrieve_batch event bills the batch; it must still write one
    # row with the batch's full cost even while line items are skipped.
    await logger.async_log_success_event(
        {
            "standard_logging_object": {
                **_minimal_payload("batch-1", status="success", cost=0.0001032),
                "call_type": "aretrieve_batch",
            },
            "call_type": "aretrieve_batch",
            "response_cost": 0.0001032,
            "litellm_params": {},
        },
        None,
        now,
        now,
    )

    assert len(logger.log_queue) == 2
    assert logger.log_queue[0]["request_id"] == "chatcmpl-live-1"
    assert logger.log_queue[1]["call_type"] == "aretrieve_batch"
    assert logger.log_queue[1]["request_id"] == "batch-1"
    assert logger.log_queue[1]["spend"] == 0.0001032

    # A FAILED batch line is still part of the aggregate row's request counts and
    # must not add its own spend row; a genuine live failure still writes one.
    await logger.async_log_failure_event(
        {
            "standard_logging_object": _minimal_payload("chatcmpl-line-2", status="failure", cost=0.0),
            "call_type": "acompletion",
            "litellm_params": {"batch_parent_id": "batch-1"},
        },
        None,
        now,
        now,
    )
    await logger.async_log_failure_event(
        {
            "standard_logging_object": _minimal_payload("chatcmpl-live-2", status="failure", cost=0.0),
            "call_type": "acompletion",
            "litellm_params": {},
        },
        None,
        now,
        now,
    )

    assert len(logger.log_queue) == 3
    assert logger.log_queue[-1]["request_id"] == "chatcmpl-live-2"
    assert logger.log_queue[-1]["status"] == "failure"
    if logger._flush_task is not None:
        logger._flush_task.cancel()

@pytest.mark.parametrize(
    "status,llm_cost,guardrail_cost,expected",
    [
        ("success", None, 0.0, None),
        ("success", 0.0, 0.0, 0.0),
        ("success", 0.25, 0.0003, 0.2503),
        ("success", None, 0.0003, None),
        ("failure", 0.25, 0.0003, 0.2503),
    ],
)
def test_standard_payload_spend_preserves_unknown_and_known_costs(
    status: Literal["success", "failure"],
    llm_cost: float | None,
    guardrail_cost: float,
    expected: float | None,
) -> None:
    guardrail_information: Final = (
        [
            {
                "guardrail_name": "guardrail",
                "guardrail_status": "success",
                "guardrail_usage": {"topicPolicyUnits": 1, "contentPolicyUnits": 1},
                "guardrail_cost": guardrail_cost,
            }
        ]
        if guardrail_cost
        else []
    )
    payload: Final = _standard_payload(
        response_cost=llm_cost,
        status=status,
        metadata={"standard_logging_guardrail_information": guardrail_information},
    )
    row: Final = spend_log_row_from_payload(payload, {"response_cost": llm_cost})
    assert row["spend"] == expected
    assert json.loads(json.dumps(row, allow_nan=False))["spend"] == expected


@pytest.mark.parametrize("response_cost", (float("nan"), float("inf")))
def test_non_finite_payload_cost_is_logged_as_unknown(response_cost: float) -> None:
    payload: Final = cast(StandardLoggingPayload, _payload(response_cost=response_cost))
    row: Final = spend_log_row_from_payload(payload, {"response_cost": response_cost})
    assert row["spend"] is None


@pytest.mark.parametrize("status", ("success", "failure"))
def test_standard_payload_retains_gateway_call_id(status: Literal["success", "failure"]) -> None:
    payload: Final = _standard_payload(response_cost=0.0, status=status)
    row: Final = spend_log_row_from_payload(payload, {"response_cost": 0.0})
    assert row["litellm_call_id"] == payload["litellm_call_id"] == "standard-payload-call"
    assert row["request_id"] == payload["id"]
