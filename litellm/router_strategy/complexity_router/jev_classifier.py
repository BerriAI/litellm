from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, Literal, NamedTuple, Protocol, TypeAlias
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_router_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.internal_call_metadata import (
    effective_turn_off_message_logging,
    forwarded_internal_call_metadata,
    parent_session_kwargs,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.typesafe_passthrough_logging_handler import (
    TypeSafePassthroughLoggingHandler,
)
from litellm.router_strategy.complexity_router.config import DEFAULT_JEV_INSTRUCTIONS as _DEFAULT_JEV_INSTRUCTIONS
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN

JevProbability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
DEFAULT_JEV_INSTRUCTIONS: Final = _DEFAULT_JEV_INSTRUCTIONS


class JevChoiceQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Mapping[str, str]


class JevSystemOneRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: str
    model: str
    questions: Mapping[str, JevChoiceQuestion]


class JevChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    type: Literal["choice"]
    choice: str
    probabilities: Mapping[str, JevProbability]
    confidence: JevProbability


class JevUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)


class JevSystemOneResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    answers: Mapping[str, JevChoiceAnswer]
    usage: JevUsage | None = None


class JevClassifierClient(Protocol):
    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse: ...


class HttpJevClassifierClient:
    def __init__(
        self,
        api_key: str | None,
        api_base: str,
        http_client: AsyncHTTPHandler,
        custom_llm_provider: Literal["typesafe", "laya"] = "typesafe",
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._http_client = http_client
        self._provider = custom_llm_provider

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        start_time: Final = datetime.now(timezone.utc)
        response: Final = await self._http_client.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler has a dynamic post signature
            f"{self._api_base}/v1/systemone",
            json=request.model_dump(mode="json"),
            headers=MappingProxyType(
                {
                    **({"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}),
                    "Content-Type": "application/json",
                }
            ),  # pyright: ignore[reportArgumentType]  # HTTP headers are not mutated by AsyncHTTPHandler
            timeout=timeout_s,
        )
        response.raise_for_status()
        try:
            self._log_response(request, response, request_kwargs, start_time)
        except Exception as exc:  # noqa: BLE001  # logging integrations must not discard a provider verdict
            verbose_router_logger.warning("JEV response logging failed (%s)", type(exc).__name__)
        parsed: Final = TypeAdapter(JevSystemOneResponse).validate_python(response.json())
        return parsed.model_copy(update={"model": request.model}) if self._provider == "laya" else parsed

    def _log_response(
        self,
        request: JevSystemOneRequest,
        response: httpx.Response,
        request_kwargs: Mapping[str, object] | None,
        start_time: datetime,
    ) -> None:
        try:
            body: Final = TypeAdapter(dict[str, object]).validate_json(response.content)
            _ = TypeAdapter(JevUsage | None).validate_python(body.get("usage"))
        except ValidationError:
            return
        end_time: Final = datetime.now(timezone.utc)
        parent: Final = request_kwargs or MappingProxyType({})
        parent_metadata: Final = MappingProxyType(
            {
                key: value
                for field in ("metadata", "litellm_metadata")
                if isinstance(metadata := parent.get(field), Mapping)
                for key, value in TypeAdapter(Mapping[str, object]).validate_python(metadata).items()
            }
        )
        params: Final = {  # mutable-ok: Logging's kwargs and litellm_params require dicts
            "metadata": {  # mutable-ok: Logging enriches metadata in place before dispatching callbacks
                **forwarded_internal_call_metadata(parent_metadata, AUTOROUTER_CLASSIFIER_CALL_ORIGIN),
                INTERNAL_CALL_ORIGIN_METADATA_KEY: AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
            },
            **parent_session_kwargs(request_kwargs),
            "turn_off_message_logging": effective_turn_off_message_logging(request_kwargs),
        }
        logging_obj: Final = Logging(
            model=f"{self._provider}/{request.model}",
            messages=[{"role": "user", "content": request.state}],  # mutable-ok: callbacks require JSON message lists
            stream=False,
            call_type="pass_through_endpoint",
            start_time=start_time,
            litellm_call_id=str(uuid4()),
            function_id="jev_classifier",
            litellm_trace_id=parent_session_kwargs(request_kwargs).get("litellm_trace_id"),
            kwargs=params,
        )
        logging_obj.update_environment_variables(
            model=f"{self._provider}/{request.model}",
            user=parent_user if isinstance(parent_user := parent.get("user"), str) else None,
            optional_params={},  # mutable-ok: Logging's optional_params contract requires a dict
            litellm_params=params,
        )
        normalized: Final = TypeSafePassthroughLoggingHandler.typesafe_passthrough_handler(
            httpx_response=response,
            response_body={**body, "model": request.model} if self._provider == "laya" else body,
            logging_obj=logging_obj,
            url_route=str(response.request.url),
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            request_body=MappingProxyType({"model": request.model}),
            custom_llm_provider=self._provider,
            litellm_params=params,
        )
        success_handlers: Final = logging_obj.dispatch_success_handlers(
            result=normalized["result"],
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            prefer_async_handlers=True,
            **TypeAdapter(dict[str, object]).validate_python(normalized["kwargs"]),
        )
        try:
            GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(success_handlers)
        except BaseException:
            success_handlers.close()
            raise


class JevVerdict(NamedTuple):
    label: str
    probabilities: Mapping[str, float]
    confidence: float
    model: str
    cost: float | None
    provider: str = "typesafe"


class _RegistryPricing(BaseModel):
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0


_REGISTRY_PRICING_ADAPTER: Final = TypeAdapter(_RegistryPricing)


def build_jev_request(
    prompt: str,
    system_prompt: str | None,
    model: str,
    instructions: str,
    criteria: Mapping[str, str],
) -> JevSystemOneRequest:
    state: Final = prompt if system_prompt is None else f"System prompt:\n{system_prompt}\n\nRequest:\n{prompt}"
    question: Final = JevChoiceQuestion(instructions=instructions, criteria=criteria)
    return JevSystemOneRequest(state=state, model=model, questions=MappingProxyType({"tier": question}))


def jev_classifier_cost(
    response: JevSystemOneResponse, configured_model: str, provider: str = "typesafe"
) -> float | None:
    usage: Final = response.usage
    if usage is None:
        return None
    model: Final = response.model or configured_model
    model_key: Final = f"{provider}/{model}"
    if model_key not in litellm.model_cost:  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
        return None
    try:
        pricing: Final = _REGISTRY_PRICING_ADAPTER.validate_python(
            litellm.model_cost[model_key]  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
        )
    except ValidationError:
        return None
    return usage.input_tokens * pricing.input_cost_per_token + usage.output_tokens * pricing.output_cost_per_token
