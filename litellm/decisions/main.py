from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.decisions.transformation import (
    DecisionsProviderConfig,
    ir_to_systemone_response,
    systemone_request_to_ir,
)
from litellm.llms.cloudflare.decisions.transformation import CLOUDFLARE_DECISIONS_ENDPOINT
from litellm.llms.custom_httpx.http_handler import _get_httpx_client, get_async_httpx_client
from litellm.llms.openai.decisions.transformation import (
    OPENAI_DECISIONS_ENDPOINT,
    ir_to_openai_response,
    openai_request_to_ir,
)
from litellm.llms.openrouter.decisions.transformation import OPENROUTER_DECISIONS_ENDPOINT
from litellm.llms.perplexity.decisions.transformation import PERPLEXITY_DECISIONS_ENDPOINT
from litellm.llms.strands_decider.decisions.transformation import STRANDS_DECIDER_DECISIONS_ENDPOINT
from litellm.llms.typesafe.decisions.transformation import TYPESAFE_DECISIONS_ENDPOINT
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
from litellm.utils import client

DECISIONS_ENDPOINTS: Final[Mapping[str, DecisionsProviderConfig]] = MappingProxyType(
    {
        "perplexity": PERPLEXITY_DECISIONS_ENDPOINT,
        "typesafe": TYPESAFE_DECISIONS_ENDPOINT,
        "openrouter": OPENROUTER_DECISIONS_ENDPOINT,
        "cloudflare": CLOUDFLARE_DECISIONS_ENDPOINT,
        "strands_decider": STRANDS_DECIDER_DECISIONS_ENDPOINT,
        "openai": OPENAI_DECISIONS_ENDPOINT,
    }
)

DecisionsQuestions: TypeAlias = (
    Mapping[str, DecisionQuestion | Mapping[str, object]] | Sequence[OpenAIDecisionQuestion | Mapping[str, object]]
)
DecisionsRequestFormat: TypeAlias = DecisionsRequestBody | OpenAIDecisionRequestBody

_SYSTEMONE_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_REQUEST_ADAPTER: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(OpenAIDecisionRequestBody)
_DECISIONS_PAYLOAD_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedDecisionsRequest:
    config: DecisionsProviderConfig
    provider: str
    upstream_model: str
    url: str
    api_key: str | None = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)
    body: Mapping[str, object] = field(repr=False)
    request: DecisionsRequestFormat = field(repr=False)
    ir_request: DecisionsIRRequest = field(repr=False)


def _resolve_provider_model(model: str, custom_llm_provider: str | None) -> tuple[str, str]:
    provider: Final = model.partition("/")[0] if custom_llm_provider is None else custom_llm_provider
    if provider not in DECISIONS_ENDPOINTS:
        supported: Final = ", ".join(DECISIONS_ENDPOINTS)
        raise litellm.BadRequestError(
            message=f"Unknown Decisions provider '{provider}'. Supported providers: {supported}",
            model=model,
            llm_provider=provider,
        )
    upstream_model: Final = model.removeprefix(f"{provider}/") if model.startswith(f"{provider}/") else model
    if not upstream_model:
        raise litellm.BadRequestError(
            message="A model name is required for the Decisions API",
            model=model,
            llm_provider=provider,
        )
    return provider, upstream_model


def _resolve_api_key(
    *,
    provider: str,
    model: str,
    endpoint: DecisionsProviderConfig,
    api_key: str | None,
) -> str | None:
    if api_key is not None:
        return api_key
    configured_api_key: Final = endpoint.configured_api_key()
    if configured_api_key:
        return configured_api_key
    if not endpoint.api_key_required:
        return None
    raise litellm.AuthenticationError(
        message=f"Missing API key for Decisions provider '{provider}'",
        model=model,
        llm_provider=provider,
    )


def _validate_request(
    *,
    state: DecisionsJSON | None,
    questions: DecisionsQuestions | None,
    decision_input: OpenAIDecisionInput | None,
    safety_identifier: str | None,
) -> DecisionsRequestFormat:
    if decision_input is None:
        return _SYSTEMONE_REQUEST_ADAPTER.validate_python({"state": state, "questions": questions})
    return _OPENAI_REQUEST_ADAPTER.validate_python(
        {"input": decision_input, "questions": questions, "safety_identifier": safety_identifier}
    )


def _ir_request(request: DecisionsRequestFormat) -> DecisionsIRRequest:
    match request:
        case DecisionsRequestBody():
            return systemone_request_to_ir(request)
        case OpenAIDecisionRequestBody():
            return openai_request_to_ir(request)
        case _:
            assert_never(request)


def _prepare_request(
    *,
    model: str,
    state: DecisionsJSON | None,
    questions: DecisionsQuestions | None,
    decision_input: OpenAIDecisionInput | None,
    safety_identifier: str | None,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
) -> _PreparedDecisionsRequest:
    provider, upstream_model = _resolve_provider_model(model, custom_llm_provider)
    if state is not None and decision_input is not None:
        raise litellm.BadRequestError(
            message="Pass either state (System One format) or input (OpenAI format) to the Decisions API, not both",
            model=model,
            llm_provider=provider,
        )
    try:
        validated_request: Final = _validate_request(
            state=state, questions=questions, decision_input=decision_input, safety_identifier=safety_identifier
        )
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=f"Invalid Decisions request: {error}",
            model=model,
            llm_provider=provider,
        ) from error

    endpoint: Final = DECISIONS_ENDPOINTS[provider]
    resolved_api_base: Final = api_base or endpoint.configured_api_base()
    if resolved_api_base is None:
        raise litellm.BadRequestError(
            message=endpoint.missing_api_base_message(provider),
            model=model,
            llm_provider=provider,
        )

    resolved_api_key: Final = _resolve_api_key(
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
    )

    canonical_model: Final = endpoint.canonical_model(upstream_model)
    outbound_headers: Final = MappingProxyType(
        {
            **{
                name: value
                for name, value in (extra_headers or {}).items()
                if name.lower() not in {"authorization", "content-type"}
            },
            **({"Authorization": f"Bearer {resolved_api_key}"} if resolved_api_key is not None else {}),
            "Content-Type": "application/json",
        }
    )
    ir_request: Final = _ir_request(validated_request)
    body: Final = endpoint.request_body(endpoint.request_model(canonical_model), ir_request)
    if isinstance(body, UnsupportedDecisionsRequest):
        raise litellm.BadRequestError(
            message=f"Decisions provider '{provider}' cannot serve this request: {body.reason}",
            model=model,
            llm_provider=provider,
        )
    return _PreparedDecisionsRequest(
        config=endpoint,
        provider=provider,
        upstream_model=canonical_model,
        url=endpoint.endpoint_url(resolved_api_base, canonical_model),
        api_key=resolved_api_key,
        headers=outbound_headers,
        body=MappingProxyType(body),
        request=validated_request,
        ir_request=ir_request,
    )


def _log_request(
    prepared: _PreparedDecisionsRequest,
    kwargs: Mapping[str, object],
) -> LiteLLMLoggingObj | None:
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, LiteLLMLoggingObj):
        return None
    logging_obj.update_from_kwargs(
        kwargs=dict(kwargs),
        model=prepared.upstream_model,
        litellm_params={
            "litellm_call_id": kwargs.get("litellm_call_id"),
            "api_base": prepared.url,
        },
        custom_llm_provider=prepared.provider,
    )
    request_body: Final = dict(prepared.body)
    request_headers: Final = dict(prepared.headers)
    logging_obj.pre_call(
        input=request_body,
        api_key=prepared.api_key,
        model=prepared.upstream_model,
        additional_args={
            "api_base": prepared.url,
            "complete_input_dict": request_body,
            "headers": request_headers,
        },
    )
    return logging_obj


def _format_response(
    response: DecisionsIRResponse, prepared: _PreparedDecisionsRequest, model: str
) -> DecisionsResponse | OpenAIDecisionResponse:
    match prepared.request:
        case DecisionsRequestBody():
            return ir_to_systemone_response(response, prepared.ir_request)
        case OpenAIDecisionRequestBody():
            return ir_to_openai_response(response, prepared.ir_request, model)
        case _:
            assert_never(prepared.request)


def _parse_response(
    response: httpx.Response,
    prepared: _PreparedDecisionsRequest,
    model: str,
) -> DecisionsResponse | OpenAIDecisionResponse:
    response.raise_for_status()
    payload: Final[object] = _DECISIONS_PAYLOAD_ADAPTER.validate_json(response.content)
    result: Final = _format_response(prepared.config.parse_response(payload, prepared.ir_request), prepared, model)
    result._hidden_params.update(
        {
            "model": f"{prepared.provider}/{prepared.upstream_model}",
            "custom_llm_provider": prepared.provider,
            "provider_response_model": f"{prepared.provider}/{prepared.upstream_model}",
        }
    )
    return result


def _map_upstream_exception(error: Exception, prepared: _PreparedDecisionsRequest) -> Exception:
    return litellm.exception_type(
        model=f"{prepared.provider}/{prepared.upstream_model}",
        custom_llm_provider=prepared.provider,
        original_exception=error,
    )


@client
async def adecisions(
    model: str,
    state: DecisionsJSON | None = None,
    questions: DecisionsQuestions | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    input: OpenAIDecisionInput | None = None,
    safety_identifier: str | None = None,
    **kwargs: object,
) -> DecisionsResponse | OpenAIDecisionResponse:
    prepared: Final = _prepare_request(
        model=model,
        state=state,
        questions=questions,
        decision_input=input,
        safety_identifier=safety_identifier,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
    )
    logging_obj: Final = _log_request(prepared, kwargs)
    try:
        handler: Final = get_async_httpx_client(llm_provider=prepared.provider)
        response: Final = await handler.post(
            prepared.url,
            json=dict(prepared.body),
            headers=dict(prepared.headers),
            timeout=timeout,
            logging_obj=logging_obj,
        )
        return _parse_response(response=response, prepared=prepared, model=model)
    except Exception as error:
        raise _map_upstream_exception(error, prepared) from error


@client
def decisions(
    model: str,
    state: DecisionsJSON | None = None,
    questions: DecisionsQuestions | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    input: OpenAIDecisionInput | None = None,
    safety_identifier: str | None = None,
    **kwargs: object,
) -> DecisionsResponse | OpenAIDecisionResponse:
    prepared: Final = _prepare_request(
        model=model,
        state=state,
        questions=questions,
        decision_input=input,
        safety_identifier=safety_identifier,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
    )
    logging_obj: Final = _log_request(prepared, kwargs)
    try:
        handler: Final = _get_httpx_client()
        response: Final = handler.post(
            prepared.url,
            json=dict(prepared.body),
            headers=dict(prepared.headers),
            timeout=timeout,
            logging_obj=logging_obj,
        )
        return _parse_response(response=response, prepared=prepared, model=model)
    except Exception as error:
        raise _map_upstream_exception(error, prepared) from error


__all__ = ["DECISIONS_ENDPOINTS", "adecisions", "decisions"]
