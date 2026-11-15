import hashlib
import os
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.litellm_core_utils.upstream_stream_usage import UPSTREAM_STREAM_USAGE_KEY, AnthropicStreamUsage

from .deployment import deployment_cache_identity, effective_destination, selected_deployment
from .intent import cache_intent
from .paths import classify_path
from .prefix import outgoing_cacheable_prefix
from .types import (
    Assessed,
    CacheIntent,
    DeploymentCacheIdentity,
    NoCacheBreakpoint,
    ObservationPath,
    ObservedCall,
    OutgoingPrefix,
    PrefixCountRequest,
    PrefixUnassessed,
    PromptCacheObservation,
    Unassessed,
    UnassessedReason,
    UpstreamCacheUsage,
    UsageUnknown,
    UsageUnknownReason,
    WireShape,
)
from .usage import anthropic_stream_cache_usage, upstream_cache_usage

VERIFIED_PATHS: Final[frozenset[ObservationPath]] = frozenset()
OBSERVATION_LOG_CAPACITY: Final = 1000

_REQUEST_BODY: Final = TypeAdapter(dict[str, JsonValue])


class ObservationSink(Protocol):
    async def record(self, call: ObservedCall) -> None: ...


class _LoggedRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    api_base: str | None = None
    complete_input_dict: object = None


class _LoggedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_call_id: str | None = None
    id: str
    call_type: str
    stream: bool | None = None
    model_id: str | None = None
    model: str = ""
    custom_llm_provider: str | None = None


class _LoggedCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", arbitrary_types_allowed=True)

    standard_logging_object: _LoggedPayload
    additional_args: _LoggedRequest = _LoggedRequest()
    litellm_params: Mapping[str, object] = MappingProxyType({})
    original_response: object = None
    upstream_stream_usage: AnthropicStreamUsage | None = None


@dataclass(frozen=True, slots=True)
class ObserverSettings:
    verified_paths: frozenset[ObservationPath]
    fingerprint_key: bytes


@dataclass(frozen=True, slots=True)
class _Call:
    logged: _LoggedCall
    deployment: DeploymentCacheIdentity
    intent: CacheIntent | None
    response: object
    succeeded: bool


def _request_body(raw: object) -> Mapping[str, JsonValue] | None:
    try:
        return _REQUEST_BODY.validate_python(raw) if isinstance(raw, dict) else None
    except ValidationError:
        return None


def _usage(call: _Call, path: ObservationPath) -> UpstreamCacheUsage:
    if not path.stream:
        return upstream_cache_usage(path.wire, call.logged.original_response)
    stream_usage: Final = call.logged.upstream_stream_usage
    if path.wire is not WireShape.ANTHROPIC_MESSAGES or stream_usage is None:
        return UsageUnknown(UsageUnknownReason.STREAM_USAGE_NOT_CAPTURED)
    return anthropic_stream_cache_usage(stream_usage)


def _assessed(
    settings: ObserverSettings, call: _Call, path: ObservationPath, outgoing: OutgoingPrefix
) -> Assessed | Unassessed:
    if not call.succeeded:
        return Unassessed(UnassessedReason.UPSTREAM_FAILED, path=path, prefix=outgoing.prefix)
    if path not in settings.verified_paths:
        return Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=path, prefix=outgoing.prefix)
    return Assessed(path=path, prefix=outgoing.prefix, usage=_usage(call, path))


def _outcome(settings: ObserverSettings, call: _Call) -> tuple[Assessed | Unassessed, PrefixCountRequest | None] | None:
    """None when the call neither asked for caching nor sent a breakpoint, so it says nothing about caching.
    Every other call is recorded, with the reason when it cannot be assessed."""
    body: Final = _request_body(call.logged.additional_args.complete_input_dict)
    if body is None:
        return (
            None if call.intent is None else (Unassessed(UnassessedReason.REQUEST_BODY_UNAVAILABLE, None, None), None)
        )
    path: Final = classify_path(
        provider=call.deployment.provider,
        provider_model=call.deployment.provider_model,
        call_type=call.logged.standard_logging_object.call_type,
        api_base=call.logged.additional_args.api_base,
        body=body,
        stream=call.logged.standard_logging_object.stream is True,
        response=call.response,
    )
    if isinstance(path, UnassessedReason):
        return None if call.intent is None else (Unassessed(path, path=None, prefix=None), None)
    match outgoing_cacheable_prefix(path.wire, body, settings.fingerprint_key):
        case NoCacheBreakpoint():
            lost: Final = Unassessed(UnassessedReason.CACHE_INTENT_NOT_SENT, path=path, prefix=None)
            return None if call.intent is None else (lost, None)
        case PrefixUnassessed(reason=reason):
            return Unassessed(reason, path=path, prefix=None), None
        case OutgoingPrefix() as outgoing:
            assessed: Final = _assessed(settings, call, path, outgoing)
            return assessed, outgoing.count_request if isinstance(assessed, Assessed) else None


def observe_call(
    settings: ObserverSettings, kwargs: Mapping[str, object], response: object, succeeded: bool, observed_at: float
) -> ObservedCall | None:
    """One observation for a selected deployment's attempt. The request is read from the body litellm sent
    upstream, so provider translation and gateway injection are applied, while intent is read from what the
    client sent and what the gateway recorded for this attempt, so a breakpoint lost on the way is visible."""
    try:
        logged: Final = _LoggedCall.model_validate(
            {**kwargs, "upstream_stream_usage": kwargs.get(UPSTREAM_STREAM_USAGE_KEY)}
        )
    except ValidationError:
        return None
    payload: Final = logged.standard_logging_object
    selected: Final = selected_deployment(logged.litellm_params.get("model_info"))
    if selected is None:
        return None
    deployment: Final = deployment_cache_identity(
        selected=selected,
        provider=payload.custom_llm_provider or "",
        provider_model=payload.model,
        destination=effective_destination(logged.additional_args.api_base),
        fingerprint_key=settings.fingerprint_key,
    )
    intent: Final = cache_intent(logged.litellm_params, selected.deployment_id)
    result: Final = _outcome(settings, _Call(logged, deployment, intent, response, succeeded))
    if result is None:
        return None
    outcome, count_request = result
    return ObservedCall(
        observation=PromptCacheObservation(
            call_id=payload.litellm_call_id or payload.id,
            deployment=deployment,
            intent=intent or CacheIntent.UNATTRIBUTED,
            observed_at=observed_at,
            outcome=outcome,
        ),
        count_request=count_request,
    )


def _coverage_key(observation: PromptCacheObservation) -> str:
    match observation.outcome:
        case Assessed(usage=UsageUnknown(reason=reason)):
            return f"assessed:usage_unknown:{reason.value}"
        case Assessed():
            return "assessed:counters"
        case Unassessed(reason=reason):
            return f"unassessed:{reason.value}"


class ObservationLog:
    """Bounded retained output: the most recent observations and a count per outcome and reason. The
    ephemeral count request is dropped here, so no prompt content is kept."""

    def __init__(self, capacity: int) -> None:
        self._capacity: Final = capacity
        self._recent: tuple[PromptCacheObservation, ...] = ()
        self._coverage: Mapping[str, int] = MappingProxyType({})

    @property
    def recent(self) -> tuple[PromptCacheObservation, ...]:
        return self._recent

    @property
    def coverage(self) -> Mapping[str, int]:
        return self._coverage

    async def record(self, call: ObservedCall) -> None:
        observation: Final = call.observation
        key: Final = _coverage_key(observation)
        self._recent = (*self._recent, observation)[-self._capacity :]
        self._coverage = MappingProxyType({**self._coverage, key: self._coverage.get(key, 0) + 1})
        verbose_router_logger.debug(
            "prompt cache observation deployment=%s config=%s outcome=%s intent=%s",
            observation.deployment.deployment_id,
            observation.deployment.config_fingerprint,
            key,
            observation.intent.value,
        )


class PromptCacheObserver(CustomLogger):
    def __init__(
        self, settings: ObserverSettings, sink: ObservationSink, clock: Callable[[], float] = time.time
    ) -> None:
        super().__init__()
        self._settings: Final = settings
        self._sink: Final = sink
        self._clock: Final = clock

    @property
    def sink(self) -> ObservationSink:
        return self._sink

    def exempt_from_message_redaction(self) -> bool:
        return True

    async def _observe(self, kwargs: Mapping[str, object], response: object, succeeded: bool) -> None:
        observed: Final = await offload_token_count(observe_call)(
            self._settings, kwargs, response, succeeded, self._clock()
        )
        if observed is not None:
            await self._sink.record(observed)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._observe(kwargs, response_obj, succeeded=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._observe(kwargs, response_obj, succeeded=False)


def _fingerprint_key(salt: str | None) -> bytes:
    """Keyed from the deployment's salt key so every worker derives the same fingerprints. Without one, the
    key is random per process: fingerprints still compare within a worker, never across workers."""
    if not salt:
        return secrets.token_bytes(32)
    return hashlib.blake2b(salt.encode(), digest_size=32, person=b"prompt-cache").digest()


def prompt_cache_observer() -> PromptCacheObserver:
    return PromptCacheObserver(
        settings=ObserverSettings(
            verified_paths=VERIFIED_PATHS, fingerprint_key=_fingerprint_key(os.getenv("LITELLM_SALT_KEY"))
        ),
        sink=ObservationLog(capacity=OBSERVATION_LOG_CAPACITY),
    )
