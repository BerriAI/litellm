import asyncio
import json
from collections.abc import Mapping

import httpx
import pytest
import respx
from pydantic import JsonValue

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.litellm_core_utils.upstream_stream_usage import UPSTREAM_STREAM_USAGE_KEY, AnthropicStreamUsage
from litellm.router_utils.prompt_cache_observations.deployment import (
    SelectedDeployment,
    deployment_cache_identity,
    effective_destination,
)
from litellm.router_utils.prompt_cache_observations.observer import (
    ObservationLog,
    ObserverSettings,
    PromptCacheObserver,
)
from litellm.router_utils.prompt_cache_observations.prefix import outgoing_cacheable_prefix
from litellm.router_utils.prompt_cache_observations.types import (
    Assessed,
    CacheablePrefix,
    CacheCounters,
    CacheIntent,
    ClientSurface,
    ObservationPath,
    ObservedCall,
    OutgoingPrefix,
    PromptCacheObservation,
    ProviderFamily,
    Runtime,
    Unassessed,
    UnassessedReason,
    UsageUnknown,
    UsageUnknownReason,
    WireShape,
)
from litellm.types.utils import ModelResponse

_KEY = b"observer-test-key"
_NOW = 1_700_000_000.0
_API_BASE = "https://api.anthropic.com/v1/messages"
_MODEL_INFO = {"id": "claude-east", "prompt_cache_observation": True}
_PATH = ObservationPath(
    provider=ProviderFamily.ANTHROPIC,
    wire=WireShape.ANTHROPIC_MESSAGES,
    surface=ClientSurface.CHAT,
    stream=False,
    runtime=Runtime.PYTHON,
)
_STREAM_PATH = ObservationPath(
    provider=_PATH.provider, wire=_PATH.wire, surface=_PATH.surface, stream=True, runtime=_PATH.runtime
)
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
        self.calls: tuple[ObservedCall, ...] = ()
        self.recorded = asyncio.Event()

    async def record(self, call: ObservedCall) -> None:
        self.calls = (*self.calls, call)
        self.recorded.set()

    @property
    def observations(self) -> tuple[PromptCacheObservation, ...]:
        return tuple(call.observation for call in self.calls)


def _observer(sink: _RecordingSink, verified: frozenset[ObservationPath] = frozenset({_PATH})) -> PromptCacheObserver:
    return PromptCacheObserver(
        settings=ObserverSettings(verified_paths=verified, fingerprint_key=_KEY), sink=sink, clock=lambda: _NOW
    )


def _kwargs(
    body: JsonValue = _MARKED_BODY,
    stream: bool = False,
    original_response: object = _ZERO_USAGE_RESPONSE,
    model_info: Mapping[str, object] = _MODEL_INFO,
    client_body: Mapping[str, object] | None = None,
    stream_usage: AnthropicStreamUsage | None = None,
) -> dict[str, object]:
    return {
        "standard_logging_object": {
            "id": "row-1",
            "litellm_call_id": "call-1",
            "call_type": "acompletion",
            "stream": stream,
            "model_id": model_info.get("id"),
            "model": "claude-test",
            "custom_llm_provider": "anthropic",
        },
        "litellm_params": {
            "model_info": model_info,
            "proxy_server_request": {"body": client_body} if client_body is not None else None,
            "metadata": {},
        },
        "additional_args": {"api_base": _API_BASE, "complete_input_dict": body},
        "original_response": original_response,
        UPSTREAM_STREAM_USAGE_KEY: stream_usage,
    }


def _deployment():
    return deployment_cache_identity(
        selected=SelectedDeployment(deployment_id="claude-east", base_model=None),
        provider="anthropic",
        provider_model="claude-test",
        destination=effective_destination(_API_BASE),
        fingerprint_key=_KEY,
    )


def _outgoing() -> OutgoingPrefix:
    outgoing = outgoing_cacheable_prefix(WireShape.ANTHROPIC_MESSAGES, _MARKED_BODY, _KEY)
    assert isinstance(outgoing, OutgoingPrefix)
    return outgoing


def _observation(
    outcome: Assessed | Unassessed, intent: CacheIntent = CacheIntent.UNATTRIBUTED
) -> PromptCacheObservation:
    return PromptCacheObservation(
        call_id="call-1", deployment=_deployment(), intent=intent, observed_at=_NOW, outcome=outcome
    )


async def test_verified_path_records_the_upstream_counters_for_the_attempts_deployment() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(_kwargs(), ModelResponse(), None, None)
    assert sink.calls == (
        ObservedCall(
            observation=_observation(
                Assessed(path=_PATH, prefix=_outgoing().prefix, usage=CacheCounters(read_tokens=0, write_tokens=0))
            ),
            count_request=_outgoing().count_request,
        ),
    )


async def test_unverified_path_is_recorded_as_unassessed_without_reading_usage() -> None:
    sink = _RecordingSink()
    await _observer(sink, verified=frozenset()).async_log_success_event(_kwargs(), ModelResponse(), None, None)
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=_PATH, prefix=_outgoing().prefix)),
    )


async def test_client_breakpoint_lost_before_the_upstream_call_is_reported_not_dropped() -> None:
    sink = _RecordingSink()
    client_body = {"messages": [{"role": "user", "content": [_MARKED_BLOCK]}]}
    await _observer(sink).async_log_success_event(
        _kwargs(body=_UNMARKED_BODY, client_body=client_body), ModelResponse(), None, None
    )
    assert sink.observations == (
        _observation(
            Unassessed(UnassessedReason.CACHE_INTENT_NOT_SENT, path=_PATH, prefix=None), intent=CacheIntent.CLIENT
        ),
    )


async def test_stream_usage_is_read_from_the_raw_events_and_absence_stays_unknown() -> None:
    sink = _RecordingSink()
    observer = _observer(sink, verified=frozenset({_STREAM_PATH}))
    reported = AnthropicStreamUsage(
        opening={"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        closing={"output_tokens": 3},
        stop_reason="end_turn",
    )
    absent = AnthropicStreamUsage(opening={"input_tokens": 9}, closing={"output_tokens": 3}, stop_reason="end_turn")
    await observer.async_log_success_event(_kwargs(stream=True, stream_usage=reported), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(stream=True, stream_usage=absent), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(stream=True), ModelResponse(), None, None)
    prefix = _outgoing().prefix
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
        _observation(Unassessed(UnassessedReason.UPSTREAM_FAILED, path=_PATH, prefix=_outgoing().prefix)),
    )


async def test_calls_with_no_cache_signal_or_no_opt_in_record_nothing() -> None:
    sink = _RecordingSink()
    observer = _observer(sink)
    await observer.async_log_success_event(_kwargs(model_info={"id": "other"}), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(body=_UNMARKED_BODY), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(body=None), ModelResponse(), None, None)
    await observer.async_log_success_event({"unrelated": True}, ModelResponse(), None, None)
    assert sink.calls == ()


async def test_observation_log_keeps_a_bounded_tail_and_counts_every_outcome() -> None:
    log = ObservationLog(capacity=2)
    observations = (
        _observation(Assessed(path=_PATH, prefix=_outgoing().prefix, usage=CacheCounters(0, 0))),
        _observation(Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=_PATH, prefix=None)),
        _observation(Assessed(path=_PATH, prefix=_outgoing().prefix, usage=UsageUnknown(UsageUnknownReason.MISSING))),
    )
    for observation in observations:
        await log.record(ObservedCall(observation=observation, count_request=_outgoing().count_request))
    assert log.recent == observations[1:]
    assert dict(log.coverage) == {
        "assessed:counters": 1,
        "unassessed:path_not_verified": 1,
        "assessed:usage_unknown:missing": 1,
    }


def _sse(events: tuple[Mapping[str, object], ...]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def _anthropic_stream(start_usage: Mapping[str, object], delta_usage: Mapping[str, object]) -> str:
    return _sse(
        (
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
            {"type": "message_stop"},
        )
    )


async def _observe_live_stream(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, stream_body: str
) -> PromptCacheObservation:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    sink = _RecordingSink()
    monkeypatch.setattr(litellm, "callbacks", [_observer(sink, verified=frozenset({_STREAM_PATH}))])
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
        model_info=_MODEL_INFO,
    )
    assert isinstance(response, CustomStreamWrapper)
    _ = [chunk async for chunk in response]
    await asyncio.wait_for(sink.recorded.wait(), timeout=10)
    return sink.observations[-1]


async def test_live_stream_with_absent_cache_counters_is_unknown_though_usage_reads_zero(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    observation = await _observe_live_stream(
        monkeypatch, respx_mock, _anthropic_stream({"input_tokens": 9, "output_tokens": 1}, {"output_tokens": 2})
    )
    match observation.outcome:
        case Assessed(path=path, usage=usage):
            assert (path, usage) == (_STREAM_PATH, UsageUnknown(UsageUnknownReason.MISSING))
        case _:
            pytest.fail(f"expected an assessed stream observation, got {observation}")


async def test_live_stream_with_reported_zero_counters_is_an_authoritative_miss(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    zeros = {"input_tokens": 9, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    observation = await _observe_live_stream(monkeypatch, respx_mock, _anthropic_stream(zeros, {"output_tokens": 2}))
    match observation.outcome:
        case Assessed(usage=usage):
            assert usage == CacheCounters(read_tokens=0, write_tokens=0)
        case _:
            pytest.fail(f"expected an assessed stream observation, got {observation}")


class _AfterObserver(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.logged.set()


async def test_router_check_observes_only_opted_in_deployments_and_never_trusts_config_as_verification(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [])
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
    log = observers[0].sink
    assert isinstance(log, ObservationLog)
    for model in ("unobserved", "observed"):
        after = _AfterObserver()
        monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, after])
        await router.acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}],
                }
            ],
        )
        await asyncio.wait_for(after.logged.wait(), timeout=10)
    assert [observation.deployment.deployment_id for observation in log.recent] == ["observed-deployment"]
    match log.recent[0].outcome:
        case Unassessed(reason=reason, path=path, prefix=CacheablePrefix(ttl_seconds=ttl, breakpoints=breakpoints)):
            assert (reason, path, ttl, breakpoints) == (UnassessedReason.PATH_NOT_VERIFIED, _PATH, 300, 1)
        case outcome:
            pytest.fail(f"expected an unverified-path observation with its prefix, got {outcome}")
