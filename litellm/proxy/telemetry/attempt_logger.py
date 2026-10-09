from collections.abc import Callable, Mapping
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.telemetry.blocks import count_blocks
from litellm.proxy.telemetry.request_context import AttemptObservation, RequestAccumulator, current_request
from litellm.telemetry.records import AttemptRecord, StatusClass, TokenCounts
from litellm.telemetry.sink import TelemetrySink

DeploymentHasher: TypeAlias = Callable[[str], str]


class _ErrorInformation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    error_code: str | None = None


class _PromptTokensDetails(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    cached_tokens: int | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_tokens_details: _PromptTokensDetails | None = None
    cache_read_input_tokens: int | None = None


class _Metadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    usage_object: _Usage | None = None


class _LoggedAttempt(BaseModel):
    """The slice of ``StandardLoggingPayload`` telemetry reads"""

    model_config = ConfigDict(frozen=True, extra="ignore")

    custom_llm_provider: str | None = None
    model_id: str | None = None
    stream: bool | None = None
    startTime: float
    endTime: float
    completionStartTime: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit: bool | None = None
    error_information: _ErrorInformation | None = None
    metadata: _Metadata | None = None


class _LoggedParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    proxy_rejected_before_routing: bool = False


def _rejected_before_routing(litellm_params: object) -> bool:
    try:
        return _LoggedParams.model_validate(litellm_params).proxy_rejected_before_routing
    except ValidationError:
        return False


def _status_code(error: _ErrorInformation | None) -> int | None:
    code: Final = error.error_code if error is not None else None
    return int(code) if code is not None and code.isdigit() else None


def _cache_read_tokens(metadata: _Metadata | None) -> int:
    usage: Final = metadata.usage_object if metadata is not None else None
    if usage is None:
        return 0
    details: Final = usage.prompt_tokens_details
    return (details.cached_tokens if details is not None else None) or usage.cache_read_input_tokens or 0


def observe(
    logged: _LoggedAttempt, *, succeeded: bool, messages: object, hash_deployment: DeploymentHasher
) -> AttemptObservation:
    stream: Final = logged.stream is True
    first_token_s: Final = logged.completionStartTime if stream and succeeded else None
    return AttemptObservation(
        attempt=AttemptRecord(
            provider=logged.custom_llm_provider or "unknown",
            provider_status=StatusClass.SUCCESS
            if succeeded
            else StatusClass.from_status_code(_status_code(logged.error_information)),
            stream=stream,
            deployment_hash=hash_deployment(logged.model_id) if logged.model_id else None,
            latency_to_first_token_ms=(first_token_s - logged.startTime) * 1000 if first_token_s is not None else None,
        ),
        succeeded=succeeded,
        tokens=TokenCounts(
            input=logged.prompt_tokens,
            output=logged.completion_tokens,
            cache_read=_cache_read_tokens(logged.metadata),
        ),
        litellm_cache_hit=logged.cache_hit is True,
        blocks=count_blocks(messages),
    )


class TelemetryAttemptLogger(CustomLogger):
    """Turns each logged provider call into an ``AttemptRecord`` and hands it to the in-flight request, if any"""

    def __init__(self, sink: Callable[[], TelemetrySink | None], hash_deployment: DeploymentHasher) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # base callback constructor accepts untyped kwargs
        self._sink: Final = sink
        self._hash_deployment: Final = hash_deployment

    def _record(self, kwargs: Mapping[str, object], *, succeeded: bool) -> None:
        sink: Final = self._sink()
        if sink is None or _rejected_before_routing(kwargs.get("litellm_params")):
            return
        try:
            logged: Final = _LoggedAttempt.model_validate(kwargs.get("standard_logging_object"))
        except ValidationError:
            verbose_proxy_logger.debug("telemetry: skipping attempt without a standard logging payload")
            return
        observation: Final = observe(
            logged, succeeded=succeeded, messages=kwargs.get("messages"), hash_deployment=self._hash_deployment
        )
        if not observation.litellm_cache_hit:
            sink.record_attempt(observation.attempt)
        request: Final[RequestAccumulator | None] = current_request.get()
        if request is not None:
            request.add(observation)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs, succeeded=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs, succeeded=False)
