import json

from pydantic import JsonValue

from litellm.router_utils.prompt_cache_observations.observer import ObserverSettings, PromptCacheObserver
from litellm.router_utils.prompt_cache_observations.prefix import outgoing_cacheable_prefix
from litellm.router_utils.prompt_cache_observations.types import (
    Assessed,
    CacheablePrefix,
    CacheCounters,
    ClientSurface,
    DeploymentCacheIdentity,
    ObservationPath,
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
_DEPLOYMENT = DeploymentCacheIdentity(
    deployment_id="claude-east",
    config_fingerprint="config-1",
    provider="anthropic",
    provider_model="claude-test",
    min_prefix_tokens=1024,
    gateway_injects_breakpoints=False,
)
_PATH = ObservationPath(
    provider=ProviderFamily.ANTHROPIC,
    wire=WireShape.ANTHROPIC_MESSAGES,
    surface=ClientSurface.CHAT,
    stream=False,
    runtime=Runtime.PYTHON,
)
_MARKED_BODY: dict[str, JsonValue] = {
    "model": "claude-test",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}]}
    ],
}
_ZERO_USAGE_RESPONSE = json.dumps(
    {"stop_reason": "end_turn", "usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
)


class _RecordingSink:
    def __init__(self) -> None:
        self.observations: tuple[PromptCacheObservation, ...] = ()

    async def record(self, observation: PromptCacheObservation) -> None:
        self.observations = (*self.observations, observation)


def _observer(sink: _RecordingSink, verified: frozenset[ObservationPath] = frozenset({_PATH})) -> PromptCacheObserver:
    return PromptCacheObserver(
        settings=ObserverSettings(
            deployments={_DEPLOYMENT.deployment_id: _DEPLOYMENT}.get,
            verified_paths=verified,
            fingerprint_key=_KEY,
        ),
        sink=sink,
        clock=lambda: _NOW,
    )


def _kwargs(
    model_id: str = "claude-east",
    body: JsonValue = _MARKED_BODY,
    stream: bool = False,
    original_response: object = _ZERO_USAGE_RESPONSE,
) -> dict[str, object]:
    return {
        "standard_logging_object": {
            "id": "row-1",
            "litellm_call_id": "call-1",
            "call_type": "acompletion",
            "stream": stream,
            "model_id": model_id,
        },
        "additional_args": {"api_base": "https://api.anthropic.com/v1/messages", "complete_input_dict": body},
        "original_response": original_response,
    }


def _prefix() -> CacheablePrefix:
    prefix = outgoing_cacheable_prefix(WireShape.ANTHROPIC_MESSAGES, _MARKED_BODY, _KEY)
    assert isinstance(prefix, CacheablePrefix)
    return prefix


def _observation(outcome: Assessed | Unassessed) -> PromptCacheObservation:
    return PromptCacheObservation(call_id="call-1", deployment=_DEPLOYMENT, observed_at=_NOW, outcome=outcome)


async def test_verified_path_records_the_upstream_cache_counters_for_the_routed_deployment() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(_kwargs(), ModelResponse(), None, None)
    assert sink.observations == (
        _observation(Assessed(path=_PATH, prefix=_prefix(), usage=CacheCounters(read_tokens=0, write_tokens=0)))
    )


async def test_unverified_path_is_recorded_as_unassessed_without_reading_usage() -> None:
    sink = _RecordingSink()
    await _observer(sink, verified=frozenset()).async_log_success_event(_kwargs(), ModelResponse(), None, None)
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=_PATH, prefix=_prefix()))
    )


async def test_streamed_call_usage_is_unknown_rather_than_zero() -> None:
    stream_path = ObservationPath(
        provider=_PATH.provider, wire=_PATH.wire, surface=_PATH.surface, stream=True, runtime=_PATH.runtime
    )
    sink = _RecordingSink()
    await _observer(sink, verified=frozenset({stream_path})).async_log_success_event(
        _kwargs(stream=True, original_response="first stream response received"), ModelResponse(), None, None
    )
    assert sink.observations == (
        _observation(
            Assessed(
                path=stream_path, prefix=_prefix(), usage=UsageUnknown(UsageUnknownReason.STREAM_USAGE_NOT_CAPTURED)
            )
        )
    )


async def test_failed_call_is_unassessed_for_its_prefix() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_failure_event(_kwargs(original_response=None), None, None, None)
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.UPSTREAM_FAILED, path=_PATH, prefix=_prefix()))
    )


async def test_missing_outgoing_body_is_unassessed() -> None:
    sink = _RecordingSink()
    await _observer(sink).async_log_success_event(_kwargs(body=None), ModelResponse(), None, None)
    assert sink.observations == (
        _observation(Unassessed(UnassessedReason.REQUEST_BODY_UNAVAILABLE, path=None, prefix=None))
    )


async def test_unselected_deployment_and_uncached_request_record_nothing() -> None:
    sink = _RecordingSink()
    observer = _observer(sink)
    uncached: dict[str, JsonValue] = {"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}
    await observer.async_log_success_event(_kwargs(model_id="other-deployment"), ModelResponse(), None, None)
    await observer.async_log_success_event(_kwargs(body=uncached), ModelResponse(), None, None)
    await observer.async_log_success_event({"unrelated": True}, ModelResponse(), None, None)
    assert sink.observations == ()
