from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, TypeAlias

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER, normalize_drop_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    ir_to_systemone_response,
    systemone_request_to_ir,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.decisions.transformation import ir_to_openai_response, openai_request_to_ir
from litellm.types.decisions import (
    DecisionQuestion,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsJSON,
    DecisionsRequestBody,
    DecisionsResponse,
    OpenAIDecisionInput,
    OpenAIDecisionQuestion,
    OpenAIDecisionRequestBody,
    OpenAIDecisionResponse,
    UnsupportedDecisionsRequest,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

SystemOneQuestions: TypeAlias = Mapping[str, DecisionQuestion | Mapping[str, object]]
OpenAIDecisionQuestions: TypeAlias = Sequence[OpenAIDecisionQuestion | Mapping[str, object]]
DecisionsRequestFormat: TypeAlias = DecisionsRequestBody | OpenAIDecisionRequestBody

_SYSTEMONE_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_REQUEST_ADAPTER: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(OpenAIDecisionRequestBody)
_SAFETY_IDENTIFIER_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(
    str | None, config=ConfigDict(title="safety_identifier")
)
_HANDLER: Final = BaseLLMHTTPHandler()


@dataclass(frozen=True, slots=True)
class SystemOneRequestFields:
    state: DecisionsJSON | None
    questions: SystemOneQuestions | None

    def validate(self, safety_identifier: str | None) -> DecisionsRequestBody:
        return _SYSTEMONE_REQUEST_ADAPTER.validate_python({"state": self.state, "questions": self.questions})


@dataclass(frozen=True, slots=True)
class OpenAIDecisionRequestFields:
    input: OpenAIDecisionInput | None
    questions: OpenAIDecisionQuestions | None

    def validate(self, safety_identifier: str | None) -> OpenAIDecisionRequestBody:
        return _OPENAI_REQUEST_ADAPTER.validate_python(
            {"input": self.input, "questions": self.questions, "safety_identifier": safety_identifier}
        )


RequestFields: TypeAlias = SystemOneRequestFields | OpenAIDecisionRequestFields


@dataclass(frozen=True, slots=True, repr=False)
class DecisionsCall:
    model: str
    requested_model: str
    custom_llm_provider: str
    provider_config: BaseDecisionsConfig
    request: DecisionsRequestFormat
    ir_request: DecisionsIRRequest
    body: Mapping[str, object] = field(repr=False)
    api_base: str
    api_key: str | None = field(repr=False)
    logging_obj: LiteLLMLoggingObj | None
    headers: Mapping[str, str]
    timeout: float | httpx.Timeout | None
    litellm_params: Mapping[str, object] = field(repr=False)


def supported_decisions_providers() -> tuple[str, ...]:
    return tuple(
        provider.value
        for provider in LlmProviders
        if ProviderConfigManager.get_provider_decisions_config(model="", provider=provider) is not None
    )


def _provider_config(model: str, custom_llm_provider: str) -> BaseDecisionsConfig:
    provider: Final = next((member for member in LlmProviders if member.value == custom_llm_provider), None)
    provider_config: Final = (
        None if provider is None else ProviderConfigManager.get_provider_decisions_config(model, provider)
    )
    if provider_config is None:
        supported: Final = ", ".join(supported_decisions_providers())
        raise litellm.BadRequestError(
            message=f"Unknown Decisions provider '{custom_llm_provider}'. Supported providers: {supported}",
            model=model,
            llm_provider=custom_llm_provider,
        )
    return provider_config


def _ir_request(request: DecisionsRequestFormat, safety_identifier: str | None) -> DecisionsIRRequest:
    match request:
        case DecisionsRequestBody():
            return replace(systemone_request_to_ir(request), safety_identifier=safety_identifier)
        case OpenAIDecisionRequestBody():
            return openai_request_to_ir(request)
        case _:
            assert_never(request)


def _drops_params(kwargs: Mapping[str, object]) -> bool:
    return litellm.drop_params is True or normalize_drop_params(kwargs.get("drop_params")) is True


def _request_safety_identifier(safety_identifier: object, kwargs: Mapping[str, object]) -> str | None:
    if isinstance(safety_identifier, str) or not _drops_params(kwargs):
        return _SAFETY_IDENTIFIER_ADAPTER.validate_python(safety_identifier)
    return None


@dataclass(frozen=True, slots=True)
class _UnsupportedSafetyIdentifier:
    pass


def _provider_ir_request(
    request: DecisionsRequestFormat,
    *,
    safety_identifier: str | None,
    provider_config: BaseDecisionsConfig,
    kwargs: Mapping[str, object],
) -> DecisionsIRRequest | _UnsupportedSafetyIdentifier:
    ir_request: Final = _ir_request(request, safety_identifier)
    if ir_request.safety_identifier is None or provider_config.supports_safety_identifier:
        return ir_request
    if _drops_params(kwargs):
        return replace(ir_request, safety_identifier=None)
    return _UnsupportedSafetyIdentifier()


def prepare_call(
    *,
    model: str,
    fields: RequestFields,
    safety_identifier: str | None,
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
    kwargs: Mapping[str, object],
) -> DecisionsCall:
    upstream_model, provider, dynamic_api_key, dynamic_api_base = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )
    provider_config: Final = _provider_config(upstream_model, provider)
    if not upstream_model:
        raise litellm.BadRequestError(
            message="A model name is required for the Decisions API",
            model=model,
            llm_provider=provider,
        )
    try:
        canonical_model: Final = provider_config.canonical_model(upstream_model)
    except ValueError as error:
        raise litellm.BadRequestError(message=str(error), model=model, llm_provider=provider) from error
    try:
        request_safety_identifier: Final = _request_safety_identifier(safety_identifier, kwargs)
        request: Final = fields.validate(request_safety_identifier)
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=f"Invalid Decisions request: {error}",
            model=model,
            llm_provider=provider,
        ) from error

    resolved_api_base: Final = provider_config.resolve_api_base(dynamic_api_base or api_base)
    if resolved_api_base is None:
        raise litellm.BadRequestError(
            message=provider_config.missing_api_base_message(provider),
            model=model,
            llm_provider=provider,
        )
    resolved_api_key: Final = provider_config.resolve_api_key(dynamic_api_key or api_key)
    if resolved_api_key is None and provider_config.api_key_required:
        raise litellm.AuthenticationError(
            message=f"Missing API key for Decisions provider '{provider}'",
            model=model,
            llm_provider=provider,
        )

    ir_request: Final = _provider_ir_request(
        request,
        safety_identifier=request_safety_identifier,
        provider_config=provider_config,
        kwargs=kwargs,
    )
    if isinstance(ir_request, _UnsupportedSafetyIdentifier):
        raise litellm.UnsupportedParamsError(
            message=(
                f"{provider} does not support parameters: ['safety_identifier'], for model={model}. "
                "To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
                "`litellm_settings:\n drop_params: true`\n"
            ),
            model=model,
            llm_provider=provider,
        )
    body: Final = provider_config.transform_decisions_request(
        model=canonical_model, request=ir_request, custom_llm_provider=provider
    )
    if isinstance(body, UnsupportedDecisionsRequest):
        raise litellm.BadRequestError(
            message=f"Decisions provider '{provider}' cannot serve this request: {body.reason}",
            model=model,
            llm_provider=provider,
        )

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if isinstance(logging_obj, LiteLLMLoggingObj):
        logging_obj.update_from_kwargs(
            kwargs=dict(kwargs),
            model=canonical_model,
            litellm_params={
                "litellm_call_id": kwargs.get("litellm_call_id"),
                "api_base": provider_config.get_complete_url(resolved_api_base, canonical_model),
            },
            custom_llm_provider=provider,
        )
    return DecisionsCall(
        model=canonical_model,
        requested_model=model,
        custom_llm_provider=provider,
        provider_config=provider_config,
        request=request,
        ir_request=ir_request,
        body=body,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        logging_obj=logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None,
        headers=extra_headers or {},
        timeout=timeout,
        litellm_params=kwargs,
    )


def _format_response(response: DecisionsIRResponse, call: DecisionsCall) -> DecisionsResponse | OpenAIDecisionResponse:
    formatted: Final = _formatted_response(response, call)
    provider_cost: Final = call.provider_config.provider_reported_cost(response)
    formatted.set_hidden_params(
        {
            "model": f"{call.custom_llm_provider}/{call.model}",
            "custom_llm_provider": call.custom_llm_provider,
            "provider_response_model": f"{call.custom_llm_provider}/{call.model}",
            **({"additional_headers": {RESPONSE_COST_HEADER: provider_cost}} if provider_cost is not None else {}),
        }
    )
    return formatted


def _formatted_response(
    response: DecisionsIRResponse, call: DecisionsCall
) -> DecisionsResponse | OpenAIDecisionResponse:
    match call.request:
        case DecisionsRequestBody():
            return ir_to_systemone_response(response, call.ir_request)
        case OpenAIDecisionRequestBody():
            return ir_to_openai_response(response, call.ir_request, call.requested_model)
        case _:
            assert_never(call.request)


def _map_upstream_exception(error: Exception, call: DecisionsCall) -> Exception:
    if isinstance(error, BaseLLMException) and error.status_code_is_synthesized:
        provider_label: Final = f"{call.custom_llm_provider[0].upper()}{call.custom_llm_provider[1:]}Exception"
        return litellm.APIConnectionError(
            message=f"{provider_label} - {error.message}",
            llm_provider=call.custom_llm_provider,
            model=f"{call.custom_llm_provider}/{call.model}",
        )
    return litellm.exception_type(
        model=f"{call.custom_llm_provider}/{call.model}",
        custom_llm_provider=call.custom_llm_provider,
        original_exception=error,
    )


async def asend(call: DecisionsCall) -> DecisionsResponse | OpenAIDecisionResponse:
    try:
        response: Final = await _HANDLER.adecisions(
            model=call.model,
            custom_llm_provider=call.custom_llm_provider,
            logging_obj=call.logging_obj,
            provider_config=call.provider_config,
            request=call.ir_request,
            body=call.body,
            api_base=call.api_base,
            api_key=call.api_key,
            headers=call.headers,
            timeout=call.timeout,
            litellm_params=call.litellm_params,
        )
    except Exception as error:
        raise _map_upstream_exception(error, call) from error
    return _format_response(response, call)


def send(call: DecisionsCall) -> DecisionsResponse | OpenAIDecisionResponse:
    try:
        response: Final = _HANDLER.decisions(
            model=call.model,
            custom_llm_provider=call.custom_llm_provider,
            logging_obj=call.logging_obj,
            provider_config=call.provider_config,
            request=call.ir_request,
            body=call.body,
            api_base=call.api_base,
            api_key=call.api_key,
            headers=call.headers,
            timeout=call.timeout,
            litellm_params=call.litellm_params,
        )
    except Exception as error:
        raise _map_upstream_exception(error, call) from error
    return _format_response(response, call)


def reject_other_format(*, model: str, kwargs: Mapping[str, object], field_name: str, other_function: str) -> None:
    if field_name not in kwargs:
        return
    raise litellm.BadRequestError(
        message=f"Requests with '{field_name}' go through litellm.{other_function}",
        model=model,
        llm_provider="",
    )
