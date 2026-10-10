from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.request_context import RequestAccumulator, bind_call_id, current_request
from litellm.telemetry.records import (
    AttemptRecord,
    BlockCounts,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TokenCounts,
    UIEvent,
)


@dataclass
class _RecordingSink:
    attempts: tuple[AttemptRecord, ...] = ()

    def set_instance(self, info: InstanceInfo) -> None: ...

    def record_request(self, record: RequestRecord) -> None: ...

    def record_attempt(self, record: AttemptRecord) -> None:
        self.attempts = (*self.attempts, record)

    def record_ui_event(self, event: UIEvent) -> None: ...

    async def flush(self) -> None: ...


_CALL_ID: Final = "call-1"


def _request() -> RequestAccumulator:
    request: Final = RequestAccumulator()
    request.bind(_CALL_ID)
    return request


def _payload(error_information: Mapping[str, object] | None = None) -> Mapping[str, object]:
    return {
        "custom_llm_provider": "anthropic",
        "model_id": "deployment-1",
        "stream": True,
        "startTime": 100.0,
        "endTime": 101.5,
        "completionStartTime": 100.25,
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "cache_hit": None,
        "metadata": {"usage_object": {"prompt_tokens_details": {"cached_tokens": 20}}},
        "error_information": error_information,
    }


@pytest.mark.asyncio
async def test_a_logged_stream_success_becomes_an_attempt_and_joins_the_in_flight_request() -> None:
    sink: Final = _RecordingSink()
    logger: Final = TelemetryAttemptLogger(
        lambda: sink, hash_deployment=lambda model_id: f"h({model_id})", blocks_enabled=lambda: True
    )
    request: Final = _request()
    token: Final = current_request.set(request)
    try:
        await logger.async_log_success_event(
            {
                "litellm_call_id": _CALL_ID,
                "standard_logging_object": _payload(),
                "messages": [{"role": "user", "content": "hi"}],
            },
            None,
            None,
            None,
        )
    finally:
        current_request.reset(token)

    expected: Final = AttemptRecord(
        provider="anthropic",
        provider_status=StatusClass.SUCCESS,
        stream=True,
        deployment_hash="h(deployment-1)",
        latency_to_first_token_ms=250.0,
    )
    assert sink.attempts == (expected,)
    (observation,) = request.observations
    assert observation.attempt == expected
    assert observation.succeeded
    assert observation.tokens == TokenCounts(input=30, output=7, cache_read=20)
    assert request.blocks == BlockCounts(total=1, by_type=((BlockType.TEXT, 1),))


@pytest.mark.parametrize(
    ("error_information", "expected_status"),
    [
        ({"error_code": "429"}, StatusClass.CLIENT_ERROR),
        ({"error_code": "503"}, StatusClass.SERVER_ERROR),
        ({"error_code": None}, StatusClass.NONE),
        (None, StatusClass.NONE),
    ],
)
@pytest.mark.asyncio
async def test_a_logged_failure_carries_the_provider_status_class(
    error_information: Mapping[str, object] | None, expected_status: StatusClass
) -> None:
    sink: Final = _RecordingSink()
    logger: Final = TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id)
    token: Final = current_request.set(_request())
    try:
        await logger.async_log_failure_event(
            {"litellm_call_id": _CALL_ID, "standard_logging_object": _payload(error_information)}, None, None, None
        )
    finally:
        current_request.reset(token)
    (attempt,) = sink.attempts
    assert attempt.provider_status is expected_status
    assert attempt.latency_to_first_token_ms is None


@pytest.mark.asyncio
async def test_a_call_without_a_standard_logging_payload_is_skipped() -> None:
    sink: Final = _RecordingSink()
    await TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id).async_log_success_event(
        {}, None, None, None
    )
    assert sink.attempts == ()


@pytest.mark.asyncio
async def test_a_proxy_side_reject_that_never_reached_a_provider_is_not_an_attempt() -> None:
    sink: Final = _RecordingSink()
    request: Final = _request()
    token: Final = current_request.set(request)
    try:
        await TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id).async_log_failure_event(
            {
                "litellm_call_id": _CALL_ID,
                "standard_logging_object": _payload({"error_code": "401"}),
                "litellm_params": {"proxy_rejected_before_routing": True},
            },
            None,
            None,
            None,
        )
    finally:
        current_request.reset(token)
    assert sink.attempts == ()
    assert request.observations == ()


@pytest.mark.asyncio
async def test_a_litellm_cache_hit_joins_the_request_but_is_not_a_provider_attempt() -> None:
    sink: Final = _RecordingSink()
    logger: Final = TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: f"h({model_id})")
    request: Final = _request()
    token: Final = current_request.set(request)
    try:
        await logger.async_log_success_event(
            {"litellm_call_id": _CALL_ID, "standard_logging_object": {**_payload(), "cache_hit": True}},
            None,
            None,
            None,
        )
    finally:
        current_request.reset(token)

    assert sink.attempts == ()
    (observation,) = request.observations
    assert observation.litellm_cache_hit


@pytest.mark.asyncio
async def test_a_call_outside_any_proxy_request_is_not_recorded() -> None:
    sink: Final = _RecordingSink()
    await TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id).async_log_success_event(
        {"litellm_call_id": _CALL_ID, "standard_logging_object": _payload()}, None, None, None
    )
    assert sink.attempts == ()


@pytest.mark.parametrize("call_id", ["guardrail-call", None])
@pytest.mark.asyncio
async def test_a_guardrail_or_moderation_call_inside_a_request_is_not_recorded(call_id: str | None) -> None:
    sink: Final = _RecordingSink()
    request: Final = _request()
    token: Final = current_request.set(request)
    try:
        await TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id).async_log_success_event(
            {"litellm_call_id": call_id, "standard_logging_object": _payload()}, None, None, None
        )
    finally:
        current_request.reset(token)
    assert sink.attempts == ()
    assert request.observations == ()


@pytest.mark.asyncio
async def test_calls_before_the_proxy_binds_its_call_id_are_not_recorded() -> None:
    sink: Final = _RecordingSink()
    request: Final = RequestAccumulator()
    token: Final = current_request.set(request)
    logger: Final = TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id)
    try:
        await logger.async_log_success_event(
            {"litellm_call_id": "pre-call-guardrail", "standard_logging_object": _payload()}, None, None, None
        )
        bind_call_id(_CALL_ID)
        bind_call_id("later-call")
        await logger.async_log_success_event(
            {"litellm_call_id": _CALL_ID, "standard_logging_object": _payload()}, None, None, None
        )
    finally:
        current_request.reset(token)
    assert len(sink.attempts) == 1
    assert len(request.observations) == 1


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.asyncio
async def test_blocks_are_counted_once_per_request_and_only_when_enabled(enabled: bool) -> None:
    counted: list[object] = []  # mutable-ok: records each messages payload handed to the block counter

    def _counter(messages: object) -> BlockCounts:
        counted.append(messages)
        return BlockCounts(total=1, by_type=((BlockType.TEXT, 1),))

    sink: Final = _RecordingSink()
    request: Final = _request()
    logger: Final = TelemetryAttemptLogger(
        lambda: sink, hash_deployment=lambda model_id: model_id, blocks_enabled=lambda: enabled, block_counter=_counter
    )
    kwargs: Final = {"litellm_call_id": _CALL_ID, "standard_logging_object": _payload(), "messages": ["m"]}
    token: Final = current_request.set(request)
    try:
        await logger.async_log_failure_event(kwargs, None, None, None)
        await logger.async_log_success_event(kwargs, None, None, None)
    finally:
        current_request.reset(token)
    assert counted == ([["m"]] if enabled else [])
    assert (request.blocks is not None) is enabled
