from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
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
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage


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


def _payload() -> Mapping[str, object]:
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


def _failed_call(provider: str = "anthropic", model_id: str = "deployment-1") -> dict[str, object]:
    return {
        "litellm_call_id": _CALL_ID,
        "model": f"{provider}/claude",
        "stream": True,
        "metadata": {"model_info": {"id": model_id}},
    }


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [
        (litellm.RateLimitError("429", "anthropic", "claude"), StatusClass.CLIENT_ERROR),
        (litellm.ServiceUnavailableError("503", "anthropic", "claude"), StatusClass.SERVER_ERROR),
        (ValueError("no status"), StatusClass.NONE),
        (litellm.Timeout("timed out", "claude", "anthropic"), StatusClass.TIMEOUT),
        (litellm.APIConnectionError("refused", "anthropic", "claude"), StatusClass.CONNECTION_ERROR),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_attempt_carries_the_provider_status_class(
    exception: Exception, expected_status: StatusClass
) -> None:
    sink: Final = _RecordingSink()
    logger: Final = TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: f"h({model_id})")
    token: Final = current_request.set(_request())
    try:
        await logger.async_post_call_failure_deployment_hook(_failed_call(), exception, None)
    finally:
        current_request.reset(token)
    assert sink.attempts == (
        AttemptRecord(
            provider="anthropic", provider_status=expected_status, stream=True, deployment_hash="h(deployment-1)"
        ),
    )


@pytest.mark.asyncio
async def test_every_failed_fallback_on_a_shared_logging_object_is_its_own_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink: Final = _RecordingSink()
    logger: Final = TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    router: Final = Router(
        model_list=[
            {
                "model_name": "a",
                "litellm_params": {"model": "anthropic/a", "api_key": "k", "mock_response": "litellm.RateLimitError"},
                "model_info": {"id": "a"},
            },
            {
                "model_name": "b",
                "litellm_params": {
                    "model": "bedrock/b",
                    "api_key": "k",
                    "mock_response": "litellm.InternalServerError",
                },
                "model_info": {"id": "b"},
            },
            {
                "model_name": "c",
                "litellm_params": {"model": "openai/c", "api_key": "k", "mock_timeout": True, "timeout": 0.01},
                "model_info": {"id": "c"},
            },
        ],
        fallbacks=[{"a": ["b", "c"]}],
        num_retries=0,
    )
    messages: Final[list[AllMessageValues]] = [ChatCompletionUserMessage(role="user", content="hi")]
    shared_logging_obj: Final = LitellmLogging(
        model="a",
        messages=messages,
        stream=False,
        call_type="acompletion",
        start_time=datetime(2026, 1, 1),
        litellm_call_id=_CALL_ID,
        function_id=_CALL_ID,
    )
    request: Final = _request()
    token: Final = current_request.set(request)
    try:
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router.acompletion's overloads return partially untyped responses
                model="a", messages=messages, litellm_call_id=_CALL_ID, litellm_logging_obj=shared_logging_obj
            )
    finally:
        current_request.reset(token)
    assert [(a.provider, a.deployment_hash, a.provider_status) for a in sink.attempts] == [
        ("anthropic", "a", StatusClass.CLIENT_ERROR),
        ("bedrock", "b", StatusClass.SERVER_ERROR),
        ("openai", "c", StatusClass.TIMEOUT),
    ]
    assert len(request.observations) == 3


@pytest.mark.asyncio
async def test_a_request_level_failure_event_is_not_counted_again() -> None:
    sink: Final = _RecordingSink()
    request: Final = _request()
    token: Final = current_request.set(request)
    try:
        logger = TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id)
        await logger.async_log_failure_event(  # pyright: ignore[reportUnknownMemberType]  # the base callback's signature is untyped
            _failed_call(), None, None, None
        )
    finally:
        current_request.reset(token)
    assert sink.attempts == ()
    assert request.observations == ()


@pytest.mark.asyncio
async def test_a_call_without_a_standard_logging_payload_is_skipped() -> None:
    sink: Final = _RecordingSink()
    await TelemetryAttemptLogger(lambda: sink, hash_deployment=lambda model_id: model_id).async_log_success_event(
        {}, None, None, None
    )
    assert sink.attempts == ()


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
        await logger.async_post_call_failure_deployment_hook(kwargs, ValueError("first attempt failed"), None)
        await logger.async_log_success_event(kwargs, None, None, None)
    finally:
        current_request.reset(token)
    assert counted == ([["m"]] if enabled else [])
    assert (request.blocks is not None) is enabled
