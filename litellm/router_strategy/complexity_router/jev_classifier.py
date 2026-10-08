from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, Literal, NamedTuple, Protocol, TypeAlias
from uuid import uuid4

import httpx
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

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
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.laya.common_utils import laya_response_model
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.typesafe_passthrough_logging_handler import (
    TypeSafePassthroughLoggingHandler,
)
from litellm.router_strategy.complexity_router.config import DEFAULT_JEV_INSTRUCTIONS as _DEFAULT_JEV_INSTRUCTIONS
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, LlmProviders
from litellm.utils import ProviderConfigManager

JevProbability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
DEFAULT_JEV_INSTRUCTIONS: Final = _DEFAULT_JEV_INSTRUCTIONS
ClassifierProvider: TypeAlias = Literal["typesafe", "laya", "bespoke", "strands_decider", "cloudflare"]
DecisionsClassifierProvider: TypeAlias = Literal["strands_decider", "cloudflare"]
_DECISIONS_PROVIDERS: Final[Mapping[DecisionsClassifierProvider, LlmProviders]] = MappingProxyType(
    {"strands_decider": LlmProviders.STRANDS_DECIDER, "cloudflare": LlmProviders.CLOUDFLARE}
)
_BODY_ADAPTER: Final = TypeAdapter(dict[str, object])


class JevChoiceQuestion(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Mapping[str, str]


class JevSystemOneRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    state: str
    model: str
    questions: Mapping[str, JevChoiceQuestion]


class JevChoiceAnswer(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    type: Literal["choice"]
    choice: str
    probabilities: Mapping[str, JevProbability]
    confidence: JevProbability


class JevUsage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)


class JevSystemOneResponse(LiteLLMBaseModel):
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
        provider: ClassifierProvider = "typesafe",
        provider_config: BaseDecisionsConfig | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._http_client = http_client
        self._provider: ClassifierProvider = provider
        self._provider_config = provider_config

    def _request_url(self, model: str) -> str:
        if self._provider_config is None:
            return f"{self._api_base}/v1/systemone"
        return self._provider_config.get_complete_url(self._api_base, self._provider_config.canonical_model(model))

    def _wire_model(self, model: str) -> str:
        return model if self._provider_config is None else self._provider_config.request_model(model)

    def _response_body(self, response: httpx.Response) -> dict[str, object]:
        body: Final = _BODY_ADAPTER.validate_json(response.content)
        if self._provider_config is None:
            return body
        return _BODY_ADAPTER.validate_python(self._provider_config.unwrap_response(body))

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        start_time: Final = datetime.now(timezone.utc)
        authorization: Final[Mapping[str, str]] = (
            MappingProxyType({"Authorization": f"Bearer {self._api_key}"}) if self._api_key else MappingProxyType({})
        )
        response: Final = await self._http_client.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler has a dynamic post signature
            self._request_url(request.model),
            json={**request.model_dump(mode="json"), "model": self._wire_model(request.model)},
            headers=MappingProxyType({**authorization, "Content-Type": "application/json"}),  # pyright: ignore[reportArgumentType]  # HTTP headers are not mutated by AsyncHTTPHandler
            timeout=timeout_s,
        )
        response.raise_for_status()
        body: Final = self._response_body(response)
        normalized_body: Final = (
            MappingProxyType({**body, "model": laya_response_model(body, request.model)})
            if self._provider == "laya"
            else body
        )
        try:
            self._log_response(request, response, body, request_kwargs, start_time)
        except Exception as exc:  # noqa: BLE001  # logging integrations must not discard a provider verdict
            verbose_router_logger.warning("JEV response logging failed (%s)", type(exc).__name__)
        return TypeAdapter(JevSystemOneResponse).validate_python(normalized_body)

    def _log_response(
        self,
        request: JevSystemOneRequest,
        response: httpx.Response,
        body: Mapping[str, object],
        request_kwargs: Mapping[str, object] | None,
        start_time: datetime,
    ) -> None:
        try:
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
        params: Final = {
            "metadata": {
                **forwarded_internal_call_metadata(parent_metadata, AUTOROUTER_CLASSIFIER_CALL_ORIGIN),
                INTERNAL_CALL_ORIGIN_METADATA_KEY: AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
            },
            **parent_session_kwargs(request_kwargs),
            "turn_off_message_logging": effective_turn_off_message_logging(request_kwargs),
        }
        logging_obj: Final = Logging(
            model=f"{self._provider}/{request.model}",
            messages=[{"role": "user", "content": request.state}],
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
            optional_params={},
            litellm_params=params,
        )
        normalized: Final = TypeSafePassthroughLoggingHandler.typesafe_passthrough_handler(
            httpx_response=response,
            response_body=body,
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
    provider: ClassifierProvider = "typesafe"


class _RegistryPricing(LiteLLMBaseModel):
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


def decisions_classifier_client(
    provider: DecisionsClassifierProvider,
    model: str,
    api_base: str | None,
    api_key: str | None,
    http_client: AsyncHTTPHandler,
) -> HttpJevClassifierClient:
    provider_config: Final = ProviderConfigManager.get_provider_decisions_config(model, _DECISIONS_PROVIDERS[provider])
    if provider_config is None:
        raise ValueError(f"opensource_classifier_config.provider {provider!r} has no Decisions provider config")
    resolved_api_base: Final = provider_config.resolve_api_base(api_base)
    if resolved_api_base is None:
        raise ValueError(
            f"opensource_classifier_config.api_base or {' or '.join(provider_config.api_base_env)} is required for "
            f"provider {provider!r}: {provider_config.missing_api_base_message(provider)}"
        )
    resolved_api_key: Final = api_key if api_base is not None else provider_config.resolve_api_key(api_key)
    if resolved_api_key is None and provider_config.api_key_required:
        raise ValueError(
            f"opensource_classifier_config.api_key or {' or '.join(provider_config.api_key_env)} is required for "
            f"provider {provider!r}"
        )
    return HttpJevClassifierClient(
        api_key=resolved_api_key,
        api_base=resolved_api_base,
        http_client=http_client,
        provider=provider,
        provider_config=provider_config,
    )


def jev_classifier_cost(
    response: JevSystemOneResponse, configured_model: str, provider: ClassifierProvider = "typesafe"
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
