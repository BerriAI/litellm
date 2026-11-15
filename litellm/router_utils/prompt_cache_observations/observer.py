import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.token_counter import offload_token_count

from .paths import classify_path
from .prefix import outgoing_cacheable_prefix
from .types import (
    Assessed,
    CacheablePrefix,
    DeploymentCacheIdentity,
    NoCacheBreakpoint,
    ObservationPath,
    PrefixUnassessed,
    PromptCacheObservation,
    Unassessed,
    UnassessedReason,
    UsageUnknown,
    UsageUnknownReason,
)
from .usage import upstream_cache_usage

DeploymentLookup: TypeAlias = Callable[[str], DeploymentCacheIdentity | None]

_REQUEST_BODY: Final = TypeAdapter(dict[str, JsonValue])


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
    model_id: str | None = None


class _LoggedCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    standard_logging_object: _LoggedPayload
    additional_args: _LoggedRequest = _LoggedRequest()
    original_response: object = None


@dataclass(frozen=True, slots=True)
class ObserverSettings:
    deployments: DeploymentLookup
    verified_paths: frozenset[ObservationPath]
    fingerprint_key: bytes


def _request_body(raw: object) -> Mapping[str, JsonValue] | None:
    try:
        return _REQUEST_BODY.validate_python(raw) if isinstance(raw, dict) else None
    except ValidationError:
        return None


def _outcome(
    settings: ObserverSettings,
    deployment: DeploymentCacheIdentity,
    call: _LoggedCall,
    response: object,
    succeeded: bool,
) -> Assessed | Unassessed | None:
    body: Final = _request_body(call.additional_args.complete_input_dict)
    if body is None:
        return Unassessed(UnassessedReason.REQUEST_BODY_UNAVAILABLE, path=None, prefix=None)
    path: Final = classify_path(
        provider=deployment.provider,
        provider_model=deployment.provider_model,
        call_type=call.standard_logging_object.call_type,
        api_base=call.additional_args.api_base,
        body=body,
        stream=call.standard_logging_object.stream is True,
        response=response,
    )
    if isinstance(path, UnassessedReason):
        return Unassessed(path, path=None, prefix=None)
    match outgoing_cacheable_prefix(path.wire, body, settings.fingerprint_key):
        case NoCacheBreakpoint():
            return None
        case PrefixUnassessed(reason=reason):
            return Unassessed(reason, path=path, prefix=None)
        case CacheablePrefix() as prefix:
            return _assessed(settings, call, path, prefix, succeeded)


def _assessed(
    settings: ObserverSettings, call: _LoggedCall, path: ObservationPath, prefix: CacheablePrefix, succeeded: bool
) -> Assessed | Unassessed:
    if not succeeded:
        return Unassessed(UnassessedReason.UPSTREAM_FAILED, path=path, prefix=prefix)
    if path not in settings.verified_paths:
        return Unassessed(UnassessedReason.PATH_NOT_VERIFIED, path=path, prefix=prefix)
    usage: Final = (
        UsageUnknown(UsageUnknownReason.STREAM_USAGE_NOT_CAPTURED)
        if path.stream
        else upstream_cache_usage(path.wire, call.original_response)
    )
    return Assessed(path=path, prefix=prefix, usage=usage)


def observe_call(
    settings: ObserverSettings, kwargs: Mapping[str, object], response: object, succeeded: bool, observed_at: float
) -> PromptCacheObservation | None:
    """One observation for a selected deployment's call that carried a cache breakpoint. The request is
    read from the body litellm sent upstream, so gateway-injected breakpoints and provider translation
    are already applied, and the deployment is the one the router finally sent it to."""
    try:
        call: Final = _LoggedCall.model_validate(kwargs)
    except ValidationError:
        return None
    model_id: Final = call.standard_logging_object.model_id
    deployment: Final = settings.deployments(model_id) if model_id is not None else None
    if deployment is None:
        return None
    outcome: Final = _outcome(settings, deployment, call, response, succeeded)
    if outcome is None:
        return None
    return PromptCacheObservation(
        call_id=call.standard_logging_object.litellm_call_id or call.standard_logging_object.id,
        deployment=deployment,
        observed_at=observed_at,
        outcome=outcome,
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

    async def _observe(self, kwargs: Mapping[str, object], response: object, succeeded: bool) -> None:
        observation: Final = await offload_token_count(observe_call)(
            self._settings, kwargs, response, succeeded, self._clock()
        )
        if observation is not None:
            await self._sink.record(observation)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._observe(kwargs, response_obj, succeeded=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._observe(kwargs, response_obj, succeeded=False)
