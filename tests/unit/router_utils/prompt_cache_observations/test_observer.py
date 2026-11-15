import asyncio
import datetime
import json
import logging
from collections.abc import AsyncIterator, Mapping
from itertools import product

import httpx
import pytest
import respx
from pydantic import JsonValue

import litellm
from litellm.router_rust.steps import callbacks as router_callbacks
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.responses.streaming_iterator import drain_logging_tasks
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.litellm_core_utils.upstream_cache_evidence import (
    DEPLOYMENT_CACHE_SETTINGS_KEY,
    SERVED_RUNTIME_KEY,
    UPSTREAM_STREAM_USAGE_KEY,
    AnthropicStreamUsage,
)
from litellm.router_utils.prompt_cache_observations.deployment import (
    DeploymentCacheSettings,
    SelectedDeployment,
    deployment_cache_identity,
    deployment_cache_settings,
    effective_destination,
)
from litellm.router_utils.prompt_cache_observations.observer import (
    VERIFIED_PATHS,
    DiagnosticLog,
    ObserverSettings,
    PromptCacheObserver,
    prefix_fingerprint_key,
)
from litellm.router_utils.prompt_cache_observations.prefix import outgoing_cacheable_prefix
from litellm.router_utils.prompt_cache_observations.types import (
    Assessed,
    CacheablePrefix,
    CacheCounters,
    ClientSurface,
    ObservationPath,
    PromptCacheObservation,
    Runtime,
    Unassessed,
    UnassessedReason,
    UsageUnknown,
    UsageUnknownReason,
)
from litellm.types.utils import ModelResponse

_KEY = b"observer-test-key"
_NOW = 1_700_000_000.0
_API_BASE = "https://api.anthropic.com/v1/messages"
_MODEL_INFO: Mapping[str, object] = {"id": "claude-east", "prompt_cache_observation": True}
_ATTEMPT_SETTINGS = deployment_cache_settings({"model": "anthropic/claude-test", "model_info": _MODEL_INFO})
_PATH = ObservationPath(
    surface=ClientSurface.CHAT,
    stream=False,
    runtime=Runtime.PYTHON,
)
_STREAM_PATH = ObservationPath(surface=_PATH.surface, stream=True, runtime=_PATH.runtime)
_MARKED_BLOCK: dict[str, JsonValue] = {"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}
_MARKED_BODY: dict[str, JsonValue] = {
    "model": "claude-test",
    "messages": [{"role": "user", "content": [_MARKED_BLOCK]}],
}
_UNMARKED_BODY: dict[str, JsonValue] = {"model": "claude-test", "messages": [{"role": "user", "content": "shared"}]}
_ZERO_USAGE_RESPONSE = json.dumps(
    {"stop_reason": "end_turn", "usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
)


class _RecordingSink:
    def __init__(self) -> None:
        self.calls: tuple[PromptCacheObservation, ...] = ()
        self.recorded = asyncio.Event()

    async def record(self, observation: PromptCacheObservation) -> None:
        self.calls = (*self.calls, observation)
        self.recorded.set()

    @property
    def observations(self) -> tuple[PromptCacheObservation, ...]:
        return self.calls


def _observer(
    sink: _RecordingSink, verified: frozenset[ObservationPath] = frozenset({_PATH}), key: bytes | None = _KEY
) -> PromptCacheObserver:
    return PromptCacheObserver(
        settings=ObserverSettings(verified_paths=verified, fingerprint_key=key), sink=sink, clock=lambda: _NOW
    )


def _kwargs(
    body: object = _MARKED_BODY,
    stream: bool = False,
    original_response: object = _ZERO_USAGE_RESPONSE,
    model_info: Mapping[str, object] = _MODEL_INFO,
    client_body: Mapping[str, object] | None = None,
    stream_usage: AnthropicStreamUsage | None = None,
    served_runtime: str | None = "python",
    settings: DeploymentCacheSettings | None = _ATTEMPT_SETTINGS,
    call_type: str = "acompletion",
    httpx_response: httpx.Response | None = None,
    provider: str = "anthropic",
    model: str = "claude-test",
    api_base: str = _API_BASE,
) -> dict[str, object]:
    return {
        "standard_logging_object": {
            "id": "row-1",
            "litellm_call_id": "call-1",
            "call_type": call_type,
            "stream": stream,
            "model": model,
            "custom_llm_provider": provider,
            "api_base": api_base,
            "model_id": model_info.get("id"),
        },
        "litellm_params": {
            "model_info": model_info,
            "proxy_server_request": {"body": client_body} if client_body is not None else None,
            "metadata": {},
        },
        "additional_args": {"complete_input_dict": body},
        "original_response": original_response,
        "httpx_response": httpx_response,
        UPSTREAM_STREAM_USAGE_KEY: stream_usage,
        SERVED_RUNTIME_KEY: served_runtime,
        DEPLOYMENT_CACHE_SETTINGS_KEY: settings,
    }


def _deployment(settings: DeploymentCacheSettings | None = _ATTEMPT_SETTINGS):
    return deployment_cache_identity(
        selected=SelectedDeployment(deployment_id="claude-east", base_model=None),
        provider="anthropic",
        provider_model="claude-test",
        destination=effective_destination(_API_BASE),
        settings=settings,
    )


def _prefix(body: Mapping[str, JsonValue] = _MARKED_BODY) -> CacheablePrefix:
    prefix = outgoing_cacheable_prefix(body, _KEY)
    assert isinstance(prefix, CacheablePrefix)
    return prefix


def _observation(
    outcome: Assessed | Unassessed, settings: DeploymentCacheSettings | None = _ATTEMPT_SETTINGS
) -> PromptCacheObservation:
    return PromptCacheObservation(call_id="call-1", deployment=_deployment(settings), observed_at=_NOW, outcome=outcome)


async def test_verified_path_records_the_upstream_counters_for_the_attempts_deployment() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(_kwargs(), ModelResponse(), None, None)
    assert sink.calls == (
        _observation(Assessed(path=_PATH, prefix=_prefix(), usage=CacheCounters(read_tokens=0, write_tokens=0))),
    )


async def test_unverified_path_is_recorded_as_unassessed_without_reading_usage() -> None:
    sink = _RecordingSink()
    await _observer(sink, verified=frozenset()).async_log_success_event(_kwargs(), ModelResponse(), None, None)
    assert sink.calls == (_observation(Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=_PATH, prefix=_prefix())),)


async def test_without_a_shared_salt_key_nothing_is_assessable() -> None:
    sink = _RecordingSink()
    await _observer(sink, key=prefix_fingerprint_key(None)).async_log_success_event(
        _kwargs(), ModelResponse(), None, None
    )
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.PREFIX_KEY_UNAVAILABLE, path=_PATH, prefix=None)),
    )


def test_prefix_key_is_derived_from_the_salt_so_workers_agree() -> None:
    assert prefix_fingerprint_key("salt") == prefix_fingerprint_key("salt") != prefix_fingerprint_key("other")


@pytest.mark.parametrize(
    "settings",
    [None, deployment_cache_settings({"model_info": {"id": "another-deployment", "prompt_cache_observation": True}})],
)
async def test_attempt_without_its_own_cache_settings_is_unassessed(settings: DeploymentCacheSettings | None) -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(_kwargs(settings=settings), ModelResponse(), None, None)
    assert sink.observations == (
        _observation(
            Unassessed(UnassessedReason.DEPLOYMENT_SETTINGS_UNAVAILABLE, path=_PATH, prefix=_prefix()),
            settings=None,
        ),
    )


async def test_attempt_without_a_runtime_stamp_is_unknown_not_python() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(_kwargs(served_runtime=None), ModelResponse(), None, None)
    assert sink.observations == (_observation(Unassessed(UnassessedReason.RUNTIME_UNKNOWN, path=None, prefix=None)),)


async def test_client_breakpoint_lost_before_the_upstream_call_is_reported_not_dropped() -> None:
    sink = _RecordingSink()
    client_body = {"messages": [{"role": "user", "content": [_MARKED_BLOCK]}]}
    await _observer(sink).async_log_success_event(
        _kwargs(body=_UNMARKED_BODY, client_body=client_body), ModelResponse(), None, None
    )
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.CACHE_INTENT_NOT_SENT, path=_PATH, prefix=None)),
    )


async def test_messages_attempt_reads_the_sent_request_and_raw_body_kept_on_the_provider_response() -> None:
    sink = _RecordingSink()
    messages_path = ObservationPath(surface=ClientSurface.MESSAGES, stream=False, runtime=Runtime.RUST)
    response = httpx.Response(
        200,
        content=_ZERO_USAGE_RESPONSE.encode(),
        request=httpx.Request("POST", _API_BASE, content=json.dumps(_MARKED_BODY).encode()),
    )
    await _observer(sink, verified=frozenset({messages_path})).async_log_success_event(
        _kwargs(
            body={},
            original_response=None,
            call_type="anthropic_messages",
            served_runtime="rust",
            httpx_response=response,
        ),
        {"id": "msg_1"},
        None,
        None,
    )
    assert sink.observations == (
        _observation(Assessed(path=messages_path, prefix=_prefix(), usage=CacheCounters(0, 0))),
    )


async def _unread_body() -> AsyncIterator[bytes]:
    yield json.dumps(_MARKED_BODY).encode()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=_ZERO_USAGE_RESPONSE.encode()),
        httpx.Response(
            200, content=_ZERO_USAGE_RESPONSE.encode(), request=httpx.Request("POST", _API_BASE, content=_unread_body())
        ),
    ],
    ids=["no-request-on-response", "request-body-not-read"],
)
async def test_a_caching_attempt_whose_sent_body_cannot_be_read_is_unassessed(response: httpx.Response) -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(
        _kwargs(body={}, client_body=_MARKED_BODY, httpx_response=response), ModelResponse(), None, None
    )
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.REQUEST_BODY_UNAVAILABLE, path=None, prefix=None)),
    )


async def test_a_breakpoint_the_prefix_cannot_compare_is_unassessed_without_a_prefix() -> None:
    sink = _RecordingSink()
    block = {**_MARKED_BLOCK, "cache_control": {"type": "ephemeral", "ttl": "2h"}}
    body = {**_MARKED_BODY, "messages": [{"role": "user", "content": [block]}]}
    await _observer(sink).async_log_success_event(_kwargs(body=body), ModelResponse(), None, None)
    assert sink.observations == (_observation(Unassessed(UnassessedReason.TTL_UNRECOGNIZED, path=_PATH, prefix=None)),)


@pytest.mark.parametrize(
    "httpx_response",
    [None, httpx.Response(200, stream=httpx.ByteStream(_ZERO_USAGE_RESPONSE.encode()))],
    ids=["no-provider-response", "provider-body-not-read"],
)
async def test_a_response_body_that_is_not_json_is_never_read_as_zero_usage(
    httpx_response: httpx.Response | None,
) -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(
        _kwargs(original_response="first stream response received", httpx_response=httpx_response),
        ModelResponse(),
        None,
        None,
    )
    assert sink.observations == (
        _observation(Assessed(path=_PATH, prefix=_prefix(), usage=UsageUnknown(UsageUnknownReason.BODY_MALFORMED))),
    )


async def test_rust_messages_attempt_without_model_info_is_selected_by_its_stamped_settings() -> None:
    sink = _RecordingSink()
    kwargs = _kwargs(model_info={}, served_runtime="rust")
    standard_logging_object = kwargs["standard_logging_object"]
    assert isinstance(standard_logging_object, dict)
    standard_logging_object["model_id"] = "claude-east"
    await _observer(sink).async_log_success_event(kwargs, ModelResponse(), None, None)
    match sink.observations:
        case (PromptCacheObservation(deployment=deployment, outcome=Unassessed(reason=reason, path=path)),):
            assert (deployment.deployment_id, reason, path is not None and path.runtime) == (
                "claude-east",
                UnassessedReason.PATH_NOT_VERIFIED,
                Runtime.RUST,
            )
        case observations:
            pytest.fail(f"expected the stamped deployment to be observed, got {observations}")


async def test_stream_usage_is_read_from_the_raw_events_and_absence_stays_unknown() -> None:
    sink = _RecordingSink()
    observer = _observer(sink, verified=frozenset({_STREAM_PATH}))
    reported = AnthropicStreamUsage(
        opening={"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        closing={"output_tokens": 3},
        stop_reason="end_turn",
        starts=1,
        stopped=True,
    )
    absent = AnthropicStreamUsage(
        opening={"input_tokens": 9}, closing={"output_tokens": 3}, stop_reason="end_turn", starts=1, stopped=True
    )
    for stream_usage in (reported, absent, None):
        await observer.async_log_success_event(
            _kwargs(stream=True, stream_usage=stream_usage), ModelResponse(), None, None
        )
    prefix = _prefix()
    assert sink.observations == (
        _observation(Assessed(path=_STREAM_PATH, prefix=prefix, usage=CacheCounters(read_tokens=0, write_tokens=0))),
        _observation(Assessed(path=_STREAM_PATH, prefix=prefix, usage=UsageUnknown(UsageUnknownReason.MISSING))),
        _observation(
            Assessed(path=_STREAM_PATH, prefix=prefix, usage=UsageUnknown(UsageUnknownReason.STREAM_USAGE_NOT_CAPTURED))
        ),
    )


async def test_failed_attempt_is_unassessed_for_its_prefix() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_failure_event(_kwargs(original_response=None), None, None, None)
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.UPSTREAM_FAILED, path=_PATH, prefix=_prefix())),
    )


async def test_failed_stream_attempt_keeps_its_stream_path() -> None:
    sink = _RecordingSink()
    kwargs = {**_kwargs(original_response=None), "stream": True}
    await _observer(sink, verified=frozenset({_STREAM_PATH})).async_log_failure_event(kwargs, None, None, None)
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.UPSTREAM_FAILED, path=_STREAM_PATH, prefix=_prefix())),
    )


async def test_calls_with_no_cache_signal_or_no_opt_in_record_nothing() -> None:
    sink = _RecordingSink()
    observer = _observer(sink)
    await observer.async_log_success_event(_kwargs(model_info={"id": "other"}), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(body=_UNMARKED_BODY), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(body=None), ModelResponse(), None, None)
    await observer.async_log_success_event({"unrelated": True}, ModelResponse(), None, None)
    assert sink.calls == ()


def _attempt_logging() -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="anthropic/claude-test",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="call-1",
    )


async def test_deployment_hook_keeps_the_attempts_cache_settings_before_the_gateway_consumes_them() -> None:
    logging_obj = _attempt_logging()
    points = [{"location": "message", "role": "system"}]
    attempt = {
        "model": "anthropic/claude-test",
        "model_info": _MODEL_INFO,
        "cache_control_injection_points": points,
        "litellm_logging_obj": logging_obj,
    }
    await _observer(_RecordingSink()).async_pre_call_deployment_hook(attempt, None)
    recorded = logging_obj.recorded_deployment_cache_settings()
    assert recorded == deployment_cache_settings(attempt)
    assert isinstance(recorded, DeploymentCacheSettings)
    assert dict(recorded.settings)["cache_control_injection_points"] == points


async def test_a_second_hook_pass_on_the_attempt_keeps_the_injection_settings_the_first_saw() -> None:
    logging_obj = _attempt_logging()
    points = [{"location": "message", "role": "system"}]
    outer = {"model_info": _MODEL_INFO, "cache_control_injection_points": points, "litellm_logging_obj": logging_obj}
    inner = {"model_info": _MODEL_INFO, "litellm_logging_obj": logging_obj}
    observer = _observer(_RecordingSink())
    await observer.async_pre_call_deployment_hook(outer, None)
    await observer.async_pre_call_deployment_hook(inner, None)
    recorded = logging_obj.recorded_deployment_cache_settings()
    assert isinstance(recorded, DeploymentCacheSettings)
    assert dict(recorded.settings)["cache_control_injection_points"] == points


@pytest.mark.parametrize(
    ("outcome", "label"),
    [
        (Assessed(path=_PATH, prefix=_prefix(), usage=CacheCounters(0, 0)), "assessed:counters"),
        (
            Assessed(path=_PATH, prefix=_prefix(), usage=UsageUnknown(UsageUnknownReason.MISSING)),
            "assessed:usage_unknown:missing",
        ),
        (Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=_PATH, prefix=None), "unassessed:path_not_verified"),
    ],
)
async def test_diagnostic_log_names_the_deployment_and_outcome_without_prompt_content(
    caplog: pytest.LogCaptureFixture, outcome: Assessed | Unassessed, label: str
) -> None:
    observation = _observation(outcome)
    with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
        await DiagnosticLog().record(observation)
    (message,) = (record.getMessage() for record in caplog.records if "prompt cache observation" in record.getMessage())
    assert message == (
        f"prompt cache observation deployment=claude-east config={observation.deployment.config_fingerprint} "
        f"outcome={label}"
    )


def _sse(events: tuple[Mapping[str, object], ...]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def _anthropic_stream(
    start_usage: Mapping[str, object], delta_usage: Mapping[str, object], terminal: str | None = None
) -> str:
    events: tuple[Mapping[str, object], ...] = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "usage": start_usage,
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": delta_usage},
    )
    return _sse(events) + (_sse(({"type": "message_stop"},)) if terminal is None else terminal)


async def _observe_live_stream(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, stream_body: str, may_fail: bool = False
) -> PromptCacheObservation:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    sink = _RecordingSink()
    monkeypatch.setattr(litellm, "callbacks", [_observer(sink, verified=frozenset({_STREAM_PATH}))])
    respx_mock.post(_API_BASE).mock(
        return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
    )
    try:
        response = await litellm.acompletion(
            model="anthropic/claude-test",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}],
                }
            ],
            stream=True,
            api_key="sk-unit-test",
            model_info=_MODEL_INFO,
        )
        assert isinstance(response, CustomStreamWrapper)
        _ = [chunk async for chunk in response]
    except litellm.APIError:
        if not may_fail:
            raise
    await asyncio.wait_for(sink.recorded.wait(), timeout=10)
    return sink.observations[-1]


async def test_live_stream_with_absent_cache_counters_is_unknown_though_normalized_usage_reads_zero(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    observation = await _observe_live_stream(
        monkeypatch, respx_mock, _anthropic_stream({"input_tokens": 9, "output_tokens": 1}, {"output_tokens": 2})
    )
    match observation.outcome:
        case Assessed(path=path, usage=usage):
            assert (path, usage) == (_STREAM_PATH, UsageUnknown(UsageUnknownReason.MISSING))
        case outcome:
            pytest.fail(f"expected an assessed stream observation, got {outcome}")


async def test_live_stream_with_reported_zero_counters_is_an_authoritative_miss(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    zeros = {"input_tokens": 9, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    observation = await _observe_live_stream(monkeypatch, respx_mock, _anthropic_stream(zeros, {"output_tokens": 2}))
    match observation.outcome:
        case Assessed(usage=usage):
            assert usage == CacheCounters(read_tokens=0, write_tokens=0)
        case outcome:
            pytest.fail(f"expected an assessed stream observation, got {outcome}")


@pytest.mark.parametrize(
    "terminal",
    [
        "",
        "event: message_stop\n",
        'event: message_stop\ndata: {"type": "message_st\n\n',
        'event: message_stop\ndata: {"type": "message_stop"}',
    ],
    ids=["cut-after-delta", "header-only", "malformed-data", "unterminated-frame"],
)
async def test_live_stream_without_a_complete_message_stop_frame_never_reads_as_a_miss(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, terminal: str
) -> None:
    zeros = {"input_tokens": 9, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    observation = await _observe_live_stream(
        monkeypatch, respx_mock, _anthropic_stream(zeros, {"output_tokens": 2}, terminal=terminal), may_fail=True
    )
    match observation.outcome:
        case Assessed(usage=UsageUnknown(reason=UsageUnknownReason.INCOMPLETE)):
            pass
        case Unassessed(reason=UnassessedReason.UPSTREAM_FAILED):
            pass
        case outcome:
            pytest.fail(f"a stream without message_stop must not be authoritative, got {outcome}")


async def test_router_check_observes_only_opted_in_deployments_and_reads_absent_counters_as_unknown(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [])
    sink = _RecordingSink()
    monkeypatch.setattr(router_callbacks, "prompt_cache_observer", lambda: _observer(sink, verified=VERIFIED_PATHS))
    respx_mock.post(_API_BASE).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 2},
            },
        )
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "observed",
                "litellm_params": {"model": "anthropic/claude-test", "api_key": "sk-unit-test"},
                "model_info": {"id": "observed-deployment", "prompt_cache_observation": True},
            },
            {
                "model_name": "unobserved",
                "litellm_params": {"model": "anthropic/claude-test", "api_key": "sk-unit-test"},
                "model_info": {"id": "unobserved-deployment"},
            },
        ],
        optional_pre_call_checks=["prompt_cache_observations"],
    )
    observers = tuple(cb for cb in router.optional_callbacks or () if isinstance(cb, PromptCacheObserver))
    assert len(observers) == 1, router.optional_callbacks
    for model in ("unobserved", "observed"):
        await router.acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}],
                }
            ],
        )
        await drain_logging_tasks()
        await GLOBAL_LOGGING_WORKER.flush()
    assert [observation.deployment.deployment_id for observation in sink.observations] == ["observed-deployment"]
    match sink.observations[0].outcome:
        case Assessed(path=path, prefix=CacheablePrefix(ttl_seconds=ttl, breakpoints=breakpoints), usage=usage):
            assert (path, ttl, breakpoints, usage) == (_PATH, 300, 1, UsageUnknown(UsageUnknownReason.MISSING))
        case outcome:
            pytest.fail(f"expected an assessed observation of the verified path, got {outcome}")


async def _client_stream(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, stream_body: str, model_info: Mapping[str, object]
) -> list[tuple[str | None, str | None]]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_observer(_RecordingSink(), verified=frozenset({_STREAM_PATH}))])
    respx_mock.post(_API_BASE).mock(
        return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
    )
    response = await litellm.acompletion(
        model="anthropic/claude-test",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}]}
        ],
        stream=True,
        api_key="sk-unit-test",
        model_info=model_info,
    )
    assert isinstance(response, CustomStreamWrapper)
    return [
        (chunk.choices[0].delta.content, chunk.choices[0].finish_reason) async for chunk in response if chunk.choices
    ]


async def test_observed_stream_with_a_provider_error_after_its_final_delta_streams_as_unobserved(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    zeros = {"input_tokens": 9, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    body = _anthropic_stream(
        zeros,
        {"output_tokens": 2},
        terminal='event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}\n\n',
    )
    unobserved = await _client_stream(monkeypatch, respx_mock, body, {"id": "claude-east"})
    observed = await _client_stream(monkeypatch, respx_mock, body, _MODEL_INFO)
    assert observed == unobserved


@pytest.mark.parametrize(
    ("provider", "model", "served_runtime", "reason"),
    [
        ("bedrock", "us.anthropic.claude-test-v1:0", "python", UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("vertex_ai", "claude-test", "python", UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("azure_ai", "claude-test", "python", UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("anthropic", "claude-test", "rust", UnassessedReason.PATH_NOT_VERIFIED),
    ],
)
async def test_paths_without_executed_evidence_stay_unassessed(
    provider: str, model: str, served_runtime: str, reason: UnassessedReason
) -> None:
    sink = _RecordingSink()
    await _observer(sink, verified=VERIFIED_PATHS).async_log_success_event(
        _kwargs(provider=provider, model=model, served_runtime=served_runtime), ModelResponse(), None, None
    )
    match sink.observations:
        case (PromptCacheObservation(outcome=Unassessed(reason=recorded)),):
            assert recorded == reason
        case recent:
            pytest.fail(f"expected the path to stay unassessed, got {recent}")


@pytest.mark.parametrize(
    ("call_type", "stream"), list(product(("acompletion", "anthropic_messages", "aresponses"), (False, True)))
)
async def test_one_deployment_keeps_one_identity_across_client_routes_and_stream_modes(
    call_type: str, stream: bool
) -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(
        _kwargs(call_type=call_type, stream=stream), ModelResponse(), None, None
    )
    (observation,) = sink.observations
    assert observation.deployment == _deployment()
