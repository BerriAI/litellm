from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple, Protocol, TypeAlias
from uuid import uuid4

import httpx
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_router_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.classifier_logging import masked_originating_request
from litellm.litellm_core_utils.internal_call_metadata import (
    effective_turn_off_message_logging,
    forwarded_internal_call_metadata,
    parent_session_kwargs,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.databricks.decisions.transformation import DATABRICKS_DECISIONS_CONFIG, is_ai_decide_model
from litellm.llms.laya.common_utils import laya_response_model
from litellm.llms.pass_through.typesafe_logging_handler import TypeSafePassthroughLoggingHandler
from litellm.router_strategy.complexity_router.config import DEFAULT_JEV_INSTRUCTIONS as _DEFAULT_JEV_INSTRUCTIONS
from litellm.types.decisions import DecisionsResponse
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, StandardLoggingRoutingDecision

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.router_api import RoutingContext
    from litellm.router_strategy.complexity_router.config import OpenSourceClassifierConfig
    from litellm.types.router import Deployment

JevProbability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
ClassifierProvider: TypeAlias = Literal["typesafe", "laya", "bespoke", "databricks"]
DEFAULT_JEV_INSTRUCTIONS: Final = _DEFAULT_JEV_INSTRUCTIONS
_WITHOUT_MODEL: Final = MappingProxyType({"model": True})


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


@dataclass(frozen=True, slots=True)
class JevInputTarget:
    model: str
    max_tokens: int | None


@dataclass(frozen=True, slots=True)
class JevBudgetedRequest:
    request: JevSystemOneRequest | None
    telemetry: StandardLoggingRoutingDecision


def jev_input_targets(router: "Router", config: "OpenSourceClassifierConfig") -> tuple[JevInputTarget, ...]:
    from litellm.router_api import RoutingContext
    from litellm.types.router import Deployment

    if config.deployment_name is None:
        provider: Final = "typesafe" if config.provider == "jev" else config.provider
        return (_jev_input_target({"model": f"{provider}/{config.model}"}, None),)
    context: Final = RoutingContext(router)
    deployments: Final = context.registry().deployments(model_name=config.deployment_name) or ()
    return tuple(
        _jev_deployment_input_target(context, Deployment.model_validate(deployment)) for deployment in deployments
    )


def _jev_deployment_input_target(context: "RoutingContext", deployment: "Deployment") -> JevInputTarget:
    params: Final = context.deployment_params({}, deployment)
    explicit_limit: Final = deployment.model_info.model_dump().get("max_input_tokens")
    return _jev_input_target(params, explicit_limit)


def _jev_input_target(params: Mapping[str, object], explicit_limit: object) -> JevInputTarget:
    from litellm.decisions.call import decisions_outbound_headers, get_decisions_provider_config
    from litellm.litellm_core_utils.get_llm_provider_logic import resolve_model_and_provider_for_metadata

    qualified_model: Final = str(params.get("model") or "")
    override: Final = params.get("custom_llm_provider")
    try:
        model, provider = resolve_model_and_provider_for_metadata(
            qualified_model, override if isinstance(override, str) else None
        )
        headers: Final = decisions_outbound_headers(
            TypeAdapter[Mapping[str, str] | None](Mapping[str, str] | None).validate_python(
                params.get("extra_headers")
            ),
            provider,
            params,
        )
        limits: Final = get_decisions_provider_config(model, provider).get_input_limits(model, provider, headers)
        candidates: Final = tuple(
            value
            for value in (limits.max_request_tokens, limits.max_question_tokens, explicit_limit)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        )
        return JevInputTarget(f"{provider}/{model}", min(candidates) if candidates else None)
    except Exception:  # noqa: BLE001  # unavailable metadata must not turn an existing classifier into a request failure
        return JevInputTarget(qualified_model, None)


def _jev_request_count(request: JevSystemOneRequest, models: Sequence[str], budget: int) -> tuple[int, str]:
    from litellm._lazy_imports import get_token_counter_new

    payload: Final = request.model_dump_json(exclude={"model"})
    byte_count: Final = len(payload.encode("utf-8"))
    if byte_count <= budget or litellm.disable_token_counter:
        return byte_count, "utf8_bytes"
    try:
        counts: Final = tuple(get_token_counter_new()(model=model, text=payload) for model in models)
        if counts and all(count > 0 for count in counts):
            return max(counts), "estimated_tokens"
    except Exception:  # noqa: BLE001  # tokenization is best-effort; byte counting remains available
        return byte_count, "utf8_bytes"
    return byte_count, "utf8_bytes"


def fit_jev_request(
    build_request: Callable[[str, str | None], JevSystemOneRequest],
    prompt: str,
    system_prompt: str | None,
    targets: Sequence[JevInputTarget],
    truncate: Callable[[str, int], str],
) -> JevBudgetedRequest:
    known: Final = tuple(target for target in targets if target.max_tokens is not None)
    if not known:
        return JevBudgetedRequest(
            build_request(prompt, system_prompt),
            {"classifier_input_budget_status": "unknown_limits", "classifier_input_truncated": False},
        )
    budget: Final = max(1, int(min(target.max_tokens for target in known if target.max_tokens is not None) * 0.75))
    models: Final = tuple(dict.fromkeys(target.model for target in known))
    initial_cap: Final = max(120, budget * 8)
    bounded_prompt: Final = truncate(prompt, initial_cap)
    bounded_system: Final = truncate(system_prompt, initial_cap) if system_prompt is not None else None
    return _fit_jev_components(
        build_request,
        bounded_prompt,
        bounded_system,
        models,
        budget,
        truncate,
        len(known) != len(targets),
        bounded_prompt != prompt or bounded_system != system_prompt,
    )


def _fit_jev_components(
    build_request: Callable[[str, str | None], JevSystemOneRequest],
    prompt: str,
    system_prompt: str | None,
    models: Sequence[str],
    budget: int,
    truncate: Callable[[str, int], str],
    unknown_limits: bool,
    truncated: bool,
) -> JevBudgetedRequest:
    attempts: Final = len(prompt).bit_length() + len(system_prompt or "").bit_length() + 1
    components: tuple[str, str | None, bool, int] = (  # rebind-ok: bounded iterative fitting state
        prompt,
        system_prompt,
        truncated,
        attempts,
    )
    while True:
        current_prompt, current_system, was_truncated, remaining = components
        request = build_request(current_prompt, current_system)
        count, method = _jev_request_count(request, models, budget)
        telemetry: StandardLoggingRoutingDecision = {
            "classifier_input_budget_tokens": budget,
            "classifier_input_tokens": count,
            "classifier_input_count_method": method,
            "classifier_input_truncated": was_truncated,
            "classifier_input_budget_status": "unknown_limits"
            if unknown_limits
            else "truncated"
            if was_truncated
            else "within_budget",
        }
        if count <= budget:
            return JevBudgetedRequest(request, telemetry)
        if remaining == 0 or (len(current_system or "") <= 123 and len(current_prompt) <= 123):
            return JevBudgetedRequest(None, {**telemetry, "classifier_input_budget_status": "untrimmable_input"})
        components = (
            (current_prompt, truncate(current_system, max(120, len(current_system) // 2)), True, remaining - 1)
            if current_system is not None and len(current_system) > 123
            else (truncate(current_prompt, max(120, len(current_prompt) // 2)), current_system, True, remaining - 1)
        )


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
    classifier_cost: float | None = None
    classifier_provider: str | None = None
    usage: JevUsage | None = None


class JevClassifierClient(Protocol):
    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse: ...


class _DecisionsRouter(Protocol):
    async def adecisions(
        self,
        *,
        model: str,
        state: str,
        questions: Mapping[str, object],
        timeout: float,
        num_retries: int,
        disable_fallbacks: bool,
        metadata: Mapping[str, object],
        proxy_server_request: Mapping[str, object],
        turn_off_message_logging: bool | None,
        litellm_session_id: str = "",
        litellm_trace_id: str = "",
    ) -> object: ...


_PARENT_METADATA: Final = TypeAdapter[Mapping[str, object] | None](Mapping[str, object] | None)


class RouterJevClassifierClient:
    def __init__(self, router: "Router", deployment_name: str) -> None:
        self._router: _DecisionsRouter = router
        self._deployment_name = deployment_name

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        parent: Final = request_kwargs or MappingProxyType({})
        metadata: Final = {
            **forwarded_internal_call_metadata(
                _PARENT_METADATA.validate_python(parent.get("litellm_metadata") or parent.get("metadata")),
                AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
            ),
            INTERNAL_CALL_ORIGIN_METADATA_KEY: AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
        }
        payload: Final = {
            "state": request.state,
            "questions": {key: value.model_dump() for key, value in request.questions.items()},
        }
        response: Final = DecisionsResponse.model_validate(
            await self._router.adecisions(
                model=self._deployment_name,
                state=request.state,
                questions={key: value.model_dump() for key, value in request.questions.items()},
                timeout=timeout_s,
                num_retries=0,
                disable_fallbacks=True,
                metadata=metadata,
                proxy_server_request={
                    "originating_request_masked": masked_originating_request(request_kwargs),
                    "body": {"model": self._deployment_name, **payload},
                },
                turn_off_message_logging=effective_turn_off_message_logging(request_kwargs),
                **parent_session_kwargs(request_kwargs),
            )
        )
        return JevSystemOneResponse.model_validate(
            {
                **response.model_dump(),
                "classifier_cost": response.hidden_params.get("response_cost"),
                "classifier_provider": response.hidden_params.get("custom_llm_provider"),
            }
        )


class HttpJevClassifierClient:
    def __init__(
        self,
        api_key: str | None,
        api_base: str,
        http_client: AsyncHTTPHandler,
        provider: ClassifierProvider = "typesafe",
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._http_client = http_client
        self._provider: ClassifierProvider = provider

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
            json=request.model_dump(mode="json", exclude=self._excluded_fields(request.model)),
            headers=MappingProxyType({**authorization, "Content-Type": "application/json"}),  # pyright: ignore[reportArgumentType]  # HTTP headers are not mutated by AsyncHTTPHandler
            timeout=timeout_s,
        )
        response.raise_for_status()
        body: Final = TypeAdapter(dict[str, object]).validate_json(response.content)
        normalized_body: Final = self._normalized_body(body, request.model)
        try:
            self._log_response(request, response, normalized_body, request_kwargs, start_time)
        except Exception as exc:  # noqa: BLE001  # logging integrations must not discard a provider verdict
            verbose_router_logger.warning("JEV response logging failed (%s)", type(exc).__name__)
        return TypeAdapter(JevSystemOneResponse).validate_python(normalized_body)

    def _excluded_fields(self, model: str) -> Mapping[str, bool] | None:
        return _WITHOUT_MODEL if self._provider == "databricks" and is_ai_decide_model(model) else None

    def _request_url(self, model: str) -> str:
        if self._provider == "databricks":
            return DATABRICKS_DECISIONS_CONFIG.get_complete_url(self._api_base, model)
        return f"{self._api_base}/v1/systemone"

    def _normalized_body(self, body: Mapping[str, object], requested_model: str) -> Mapping[str, object]:
        match self._provider:
            case "laya":
                return MappingProxyType({**body, "model": laya_response_model(body, requested_model)})
            case "databricks":
                return DATABRICKS_DECISIONS_CONFIG.classifier_response(body, requested_model)
            case "typesafe" | "bespoke":
                return body

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
    provider: str = "typesafe"


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
