import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Final, Protocol, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.litellm_core_utils.upstream_cache_evidence import (
    DEPLOYMENT_CACHE_SETTINGS_KEY,
    SERVED_RUNTIME_KEY,
    UPSTREAM_STREAM_USAGE_KEY,
    RecordsDeploymentCacheSettings,
)
from litellm.types.utils import CallTypes

from .deployment import (
    DeploymentCacheSettings,
    SelectedDeployment,
    deployment_cache_identity,
    deployment_cache_settings,
    effective_destination,
    merged_cache_settings,
    selected_deployment,
)
from .intent import cache_requested
from .paths import classify_path
from .prefix import outgoing_cacheable_prefix
from .types import (
    Assessed,
    CacheablePrefix,
    CacheCounters,
    ClientSurface,
    DeploymentCacheIdentity,
    NoCacheBreakpoint,
    ObservationPath,
    PrefixUnassessed,
    PromptCacheObservation,
    Runtime,
    Unassessed,
    UnassessedReason,
    UpstreamCacheUsage,
    UsageUnknown,
)
from .usage import stream_cache_usage, upstream_cache_usage

_EXECUTED_CELLS: Final = (
    (ClientSurface.MESSAGES, Runtime.RUST),
    (ClientSurface.CHAT, Runtime.PYTHON),
    (ClientSurface.RESPONSES, Runtime.PYTHON),
)
VERIFIED_PATHS: Final[frozenset[ObservationPath]] = frozenset(
    ObservationPath(surface=surface, stream=stream, runtime=runtime)
    for (surface, runtime), stream in product(_EXECUTED_CELLS, (False, True))
)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BREAKPOINT_MARKER: Final = '"cache_control"'


class ObservationSink(Protocol):
    async def record(self, observation: PromptCacheObservation) -> None: ...


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
    model: str = ""
    custom_llm_provider: str | None = None
    api_base: str | None = None
    model_id: str | None = None


class _LoggedCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", arbitrary_types_allowed=True)

    standard_logging_object: _LoggedPayload
    additional_args: _LoggedRequest = _LoggedRequest()
    litellm_params: Mapping[str, object] = Field(default_factory=dict)
    original_response: object = None
    httpx_response: object = None
    stream: bool | None = None
    upstream_stream_usage: object = None
    served_runtime: object = None
    deployment_settings: object = None


@dataclass(frozen=True, slots=True)
class ObserverSettings:
    """`fingerprint_key` keys the prompt-prefix fingerprint. Without a key shared by every worker, prefixes from
    different workers could not be compared, so the observer records nothing assessable."""

    verified_paths: frozenset[ObservationPath]
    fingerprint_key: bytes | None


@dataclass(frozen=True, slots=True)
class _Call:
    logged: _LoggedCall
    body: Mapping[str, JsonValue] | None
    deployment: DeploymentCacheIdentity
    settings: DeploymentCacheSettings | None
    requested: bool
    response: object
    succeeded: bool


def _json_object(raw: object) -> Mapping[str, JsonValue] | None:
    try:
        match raw:
            case str() | bytes():
                return _JSON_OBJECT.validate_json(raw)
            case dict():
                return _JSON_OBJECT.validate_python(raw)
            case _:
                return None
    except ValidationError:
        return None


def _wire_request(response: object) -> httpx.Request | None:
    if not isinstance(response, httpx.Response):
        return None
    try:
        return response.request
    except RuntimeError:
        return None


def _wire_request_body(response: object) -> Mapping[str, JsonValue] | None:
    request: Final = _wire_request(response)
    try:
        return _json_object(request.content) if request is not None else None
    except httpx.RequestNotRead:
        return None


def _outgoing_body(logged: _LoggedCall) -> Mapping[str, JsonValue] | None:
    """The body the attempt sent. Messages paths replace the logged body after sending, so the request kept on
    the provider response is read when the logged one is empty."""
    logged_body: Final = _json_object(logged.additional_args.complete_input_dict)
    return logged_body if logged_body else _wire_request_body(logged.httpx_response)


def _destination(logged: _LoggedCall) -> str | None:
    """The URL the attempt went to. `post_call` replaces the pre-call arguments on several paths, so the
    endpoint the standard log and the attempt's params recorded come first."""
    params_base: Final = logged.litellm_params.get("api_base")
    request: Final = _wire_request(logged.httpx_response)
    candidates: Final = (
        logged.standard_logging_object.api_base,
        params_base if isinstance(params_base, str) else None,
        logged.additional_args.api_base,
        str(request.url) if request is not None else None,
    )
    return next((candidate for candidate in candidates if candidate), None)


def _raw_response(logged: _LoggedCall) -> object:
    if _json_object(logged.original_response) is not None:
        return logged.original_response
    response: Final = logged.httpx_response
    if not isinstance(response, httpx.Response):
        return logged.original_response
    try:
        return response.content
    except httpx.ResponseNotRead:
        return logged.original_response


def _usage(call: _Call, path: ObservationPath) -> UpstreamCacheUsage:
    if path.stream:
        return stream_cache_usage(call.logged.upstream_stream_usage)
    return upstream_cache_usage(_raw_response(call.logged))


def _assessed(
    settings: ObserverSettings, call: _Call, path: ObservationPath, prefix: CacheablePrefix
) -> Assessed | Unassessed:
    if not call.succeeded:
        return Unassessed(UnassessedReason.UPSTREAM_FAILED, path=path, prefix=prefix)
    if call.settings is None:
        return Unassessed(UnassessedReason.DEPLOYMENT_SETTINGS_UNAVAILABLE, path=path, prefix=prefix)
    if path not in settings.verified_paths:
        return Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=path, prefix=prefix)
    return Assessed(path=path, prefix=prefix, usage=_usage(call, path))


_Outcome: TypeAlias = Assessed | Unassessed


def _without_prefix(
    call: _Call, reason: UnassessedReason, path: ObservationPath | None, marked: bool = False
) -> _Outcome | None:
    return None if not call.requested and not marked else Unassessed(reason, path=path, prefix=None)


def _carries_breakpoint(body: Mapping[str, JsonValue]) -> bool:
    return _BREAKPOINT_MARKER in json.dumps(body)


def _outcome(settings: ObserverSettings, call: _Call) -> _Outcome | None:
    """None when the attempt neither asked for caching nor sent a breakpoint, so it says nothing about caching.
    Every other attempt is recorded, with the reason when it cannot be assessed."""
    body: Final = call.body
    if body is None:
        return _without_prefix(call, UnassessedReason.REQUEST_BODY_UNAVAILABLE, None)
    path: Final = classify_path(
        provider=call.deployment.provider,
        provider_model=call.deployment.provider_model,
        call_type=call.logged.standard_logging_object.call_type,
        body=body,
        stream=call.logged.stream is True or call.logged.standard_logging_object.stream is True,
        response=call.response,
        served_runtime=call.logged.served_runtime,
    )
    if isinstance(path, UnassessedReason):
        return _without_prefix(call, path, None, _carries_breakpoint(body))
    if settings.fingerprint_key is None:
        return _without_prefix(call, UnassessedReason.PREFIX_KEY_UNAVAILABLE, path, _carries_breakpoint(body))
    match outgoing_cacheable_prefix(body, settings.fingerprint_key):
        case NoCacheBreakpoint():
            return _without_prefix(call, UnassessedReason.CACHE_INTENT_NOT_SENT, path)
        case PrefixUnassessed(reason=reason):
            return Unassessed(reason, path=path, prefix=None)
        case CacheablePrefix() as prefix:
            return _assessed(settings, call, path, prefix)


def _attempt_settings(logged: _LoggedCall) -> DeploymentCacheSettings | None:
    """Settings the deployment hook stamped for this attempt's own deployment. The router names the deployment in
    the standard log on every route, while `litellm_params.model_info` is absent on some Rust-served ones and is
    the only name an SDK call that passes `model_info` carries."""
    settings: Final = logged.deployment_settings
    model_info: Final = as_str_mapping(logged.litellm_params.get("model_info")) or {}
    deployment_ids: Final = (logged.standard_logging_object.model_id, model_info.get("id"))
    if isinstance(settings, DeploymentCacheSettings) and settings.selected.deployment_id in deployment_ids:
        return settings
    return None


def _selected(logged: _LoggedCall, settings: DeploymentCacheSettings | None) -> SelectedDeployment | None:
    return settings.selected if settings is not None else selected_deployment(logged.litellm_params.get("model_info"))


def observe_call(
    settings: ObserverSettings, kwargs: Mapping[str, object], response: object, succeeded: bool, observed_at: float
) -> PromptCacheObservation | None:
    """One observation for an opted-in deployment's attempt. The request is the body that attempt sent, so
    gateway injection is applied, while whether caching was requested comes from what the client sent and what the
    gateway recorded for the attempt, so a breakpoint lost on the way is visible."""
    try:
        logged: Final = _LoggedCall.model_validate(
            {
                **kwargs,
                "upstream_stream_usage": kwargs.get(UPSTREAM_STREAM_USAGE_KEY),
                "served_runtime": kwargs.get(SERVED_RUNTIME_KEY),
                "deployment_settings": kwargs.get(DEPLOYMENT_CACHE_SETTINGS_KEY),
            }
        )
    except ValidationError:
        return None
    payload: Final = logged.standard_logging_object
    attempt_settings: Final = _attempt_settings(logged)
    selected: Final = _selected(logged, attempt_settings)
    if selected is None:
        return None
    provider: Final = payload.custom_llm_provider or ""
    body: Final = _outgoing_body(logged)
    deployment: Final = deployment_cache_identity(
        selected=selected,
        provider=provider,
        provider_model=payload.model,
        destination=effective_destination(_destination(logged)),
        settings=attempt_settings,
    )
    requested: Final = cache_requested(logged.litellm_params, selected.deployment_id)
    outcome: Final = _outcome(
        settings, _Call(logged, body, deployment, attempt_settings, requested, response, succeeded)
    )
    if outcome is None:
        return None
    return PromptCacheObservation(
        call_id=payload.litellm_call_id or payload.id,
        deployment=deployment,
        observed_at=observed_at,
        outcome=outcome,
    )


def _outcome_label(observation: PromptCacheObservation) -> str:
    match observation.outcome:
        case Assessed(usage=UsageUnknown(reason=reason)):
            return f"assessed:usage_unknown:{reason.value}"
        case Assessed():
            return "assessed:counters"
        case Unassessed(reason=reason):
            return f"unassessed:{reason.value}"


_PREFIX_ID_LENGTH: Final = 12
_UNKNOWN: Final = "-"


def _prefix_of(observation: PromptCacheObservation) -> CacheablePrefix | None:
    return observation.outcome.prefix


def _counters_of(observation: PromptCacheObservation) -> CacheCounters | None:
    outcome: Final = observation.outcome
    return outcome.usage if isinstance(outcome, Assessed) and isinstance(outcome.usage, CacheCounters) else None


class DiagnosticLog:
    """Writes each observation to the debug log and keeps nothing, so a worker never holds observation history.
    The prefix is named by the start of its keyed fingerprint, which cannot be turned back into prompt content."""

    async def record(self, observation: PromptCacheObservation) -> None:
        prefix: Final = _prefix_of(observation)
        counters: Final = _counters_of(observation)
        verbose_router_logger.debug(
            "prompt cache observation deployment=%s config=%s outcome=%s prefix=%s ttl=%s read=%s write=%s",
            observation.deployment.deployment_id,
            observation.deployment.config_fingerprint,
            _outcome_label(observation),
            prefix.fingerprint[:_PREFIX_ID_LENGTH] if prefix is not None else _UNKNOWN,
            prefix.ttl_seconds if prefix is not None else _UNKNOWN,
            counters.read_tokens if counters is not None else _UNKNOWN,
            counters.write_tokens if counters is not None else _UNKNOWN,
        )


class PromptCacheObserver(CustomLogger):
    def __init__(
        self, settings: ObserverSettings, sink: ObservationSink, clock: Callable[[], float] = time.time
    ) -> None:
        super().__init__()
        self._settings: Final = settings
        self._sink: Final = sink
        self._clock: Final = clock

    def exempt_from_message_redaction(self) -> bool:
        return True

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        """Runs once per attempt, before the gateway consumes the deployment's cache settings."""
        settings: Final = deployment_cache_settings(kwargs)
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if settings is not None and isinstance(logging_obj, RecordsDeploymentCacheSettings):
            logging_obj.record_deployment_cache_settings(
                merged_cache_settings(logging_obj.recorded_deployment_cache_settings(), settings)
            )

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


def prefix_fingerprint_key(salt: str | None) -> bytes | None:
    return hashlib.blake2b(salt.encode(), digest_size=32, person=b"prompt-cache").digest() if salt else None


def prompt_cache_observer() -> PromptCacheObserver:
    return PromptCacheObserver(
        settings=ObserverSettings(
            verified_paths=VERIFIED_PATHS, fingerprint_key=prefix_fingerprint_key(os.getenv("LITELLM_SALT_KEY"))
        ),
        sink=DiagnosticLog(),
    )
