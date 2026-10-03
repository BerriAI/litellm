from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.decisions.transformation import DecisionsProviderConfig
from litellm.llms.cloudflare.decisions.transformation import CLOUDFLARE_DECISIONS_ENDPOINT
from litellm.llms.custom_httpx.http_handler import _get_httpx_client, get_async_httpx_client
from litellm.llms.openrouter.decisions.transformation import OPENROUTER_DECISIONS_ENDPOINT
from litellm.llms.perplexity.decisions.transformation import PERPLEXITY_DECISIONS_ENDPOINT
from litellm.llms.strands_decider.decisions.transformation import STRANDS_DECIDER_DECISIONS_ENDPOINT
from litellm.llms.typesafe.decisions.transformation import TYPESAFE_DECISIONS_ENDPOINT
from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import (
    DecisionQuestion,
    DecisionsJSON,
    DecisionsRequest,
    DecisionsResponse,
)
from litellm.utils import client

DECISIONS_ENDPOINTS: Final[Mapping[str, DecisionsProviderConfig]] = MappingProxyType(
    {
        "perplexity": PERPLEXITY_DECISIONS_ENDPOINT,
        "typesafe": TYPESAFE_DECISIONS_ENDPOINT,
        "openrouter": OPENROUTER_DECISIONS_ENDPOINT,
        "cloudflare": CLOUDFLARE_DECISIONS_ENDPOINT,
        "strands_decider": STRANDS_DECIDER_DECISIONS_ENDPOINT,
    }
)

_DECISIONS_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequest]] = TypeAdapter(DecisionsRequest)
_DECISIONS_PAYLOAD_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)
_DECISIONS_RESPONSE_ADAPTER: Final[TypeAdapter[DecisionsResponse]] = TypeAdapter(DecisionsResponse)


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedDecisionsRequest:
    config: DecisionsProviderConfig
    provider: str
    upstream_model: str
    url: str
    api_key: str | None = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)
    body: Mapping[str, object] = field(repr=False)


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

    server_api_key: Final = next(
        (key for key in (get_secret_str(name) for name in endpoint.api_key_env) if key),
        None,
    )
    if server_api_key is None:
        if not endpoint.api_key_required:
            return None
        raise litellm.AuthenticationError(
            message=f"Missing API key for Decisions provider '{provider}'",
            model=model,
            llm_provider=provider,
        )

    return server_api_key


def _prepare_request(
    *,
    model: str,
    state: DecisionsJSON,
    questions: Mapping[str, DecisionQuestion | Mapping[str, object]],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
) -> _PreparedDecisionsRequest:
    provider, upstream_model = _resolve_provider_model(model, custom_llm_provider)
    try:
        validated_request: Final = _DECISIONS_REQUEST_ADAPTER.validate_python(
            {"model": model, "state": state, "questions": questions}
        )
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=f"Invalid Decisions request: {error}",
            model=model,
            llm_provider=provider,
        ) from error

    endpoint: Final = DECISIONS_ENDPOINTS[provider]
    env_api_base: Final = get_secret_str(endpoint.api_base_env)
    default_api_base: Final = endpoint.default_api_base()
    resolved_api_base: Final = api_base or env_api_base or default_api_base
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
    body: Final = MappingProxyType(
        {
            "model": endpoint.request_model(canonical_model),
            "state": validated_request.state,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in validated_request.questions.items()
            },
        }
    )
    return _PreparedDecisionsRequest(
        config=endpoint,
        provider=provider,
        upstream_model=canonical_model,
        url=endpoint.endpoint_url(resolved_api_base, canonical_model),
        api_key=resolved_api_key,
        headers=outbound_headers,
        body=body,
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


def _parse_response(
    response: httpx.Response,
    prepared: _PreparedDecisionsRequest,
) -> DecisionsResponse:
    response.raise_for_status()
    payload: Final[object] = _DECISIONS_PAYLOAD_ADAPTER.validate_json(response.content)
    result: Final = _DECISIONS_RESPONSE_ADAPTER.validate_python(prepared.config.unwrap_response(payload))
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
    state: DecisionsJSON,
    questions: Mapping[str, DecisionQuestion | Mapping[str, object]],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,
) -> DecisionsResponse:
    prepared: Final = _prepare_request(
        model=model,
        state=state,
        questions=questions,
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
        return _parse_response(response=response, prepared=prepared)
    except Exception as error:
        raise _map_upstream_exception(error, prepared) from error


@client
def decisions(
    model: str,
    state: DecisionsJSON,
    questions: Mapping[str, DecisionQuestion | Mapping[str, object]],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,
) -> DecisionsResponse:
    prepared: Final = _prepare_request(
        model=model,
        state=state,
        questions=questions,
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
        return _parse_response(response=response, prepared=prepared)
    except Exception as error:
        raise _map_upstream_exception(error, prepared) from error


__all__ = ["DECISIONS_ENDPOINTS", "adecisions", "decisions"]
