import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.decisions.call import log_caller_request
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.tool_fields import DropPolicy, drop_policy
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.extraction.transformation import BaseExtractionConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.extraction import (
    EXTRACTION_STATE_REWRITTEN_METADATA_KEY,
    ExtractionAnswer,
    ExtractionBase64Image,
    ExtractionField,
    ExtractionOptions,
    ExtractionRequestBody,
    ExtractionResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders, is_litellm_owned_kwarg
from litellm.utils import (
    ProviderConfigManager,
    client,  # pyright: ignore[reportUnknownVariableType]  # client is an untyped decorator
    get_model_cost_key,
)

ExtractionFields: TypeAlias = Mapping[str, ExtractionField | Mapping[str, object]]
ExtractionImages: TypeAlias = Sequence[str | ExtractionBase64Image | Mapping[str, object]]
ExtractionOptionsParam: TypeAlias = ExtractionOptions | Mapping[str, object]

_REQUEST_ADAPTER: Final[TypeAdapter[ExtractionRequestBody]] = TypeAdapter(ExtractionRequestBody)
_METADATA_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_HANDLER: Final = BaseLLMHTTPHandler()
_LOCAL_KWARGS: Final = frozenset(
    (
        *GenericLiteLLMParams.model_fields,
        "extra_headers",
        "base_url",
        "default_headers",
        "deployment_id",
        "additional_drop_params",
    )
)


@dataclass(frozen=True, slots=True, repr=False)
class _ExtractionCall:
    model: str
    requested_model: str
    custom_llm_provider: str
    provider_config: BaseExtractionConfig
    body: Mapping[str, object]
    api_base: str
    api_key: str | None
    logging_obj: LiteLLMLoggingObj | None
    headers: Mapping[str, str]
    timeout: float | httpx.Timeout | None
    litellm_params: Mapping[str, object]
    client: HTTPHandler | AsyncHTTPHandler | None


def supported_extraction_providers() -> tuple[str, ...]:
    return tuple(
        provider.value
        for provider in LlmProviders
        if ProviderConfigManager.get_provider_extraction_config(model="", provider=provider) is not None
    )


def _provider_config(model: str, custom_llm_provider: str) -> BaseExtractionConfig:
    provider: Final = next((member for member in LlmProviders if member.value == custom_llm_provider), None)
    provider_config: Final = (
        None if provider is None else ProviderConfigManager.get_provider_extraction_config(model, provider)
    )
    if provider_config is None:
        raise litellm.BadRequestError(
            message=(
                f"Unknown Extraction provider '{custom_llm_provider}'. "
                f"Supported providers: {', '.join(supported_extraction_providers())}"
            ),
            model=model,
            llm_provider=custom_llm_provider,
        )
    return provider_config


def _request_fields(
    *,
    state: str | None,
    fields: ExtractionFields | None,
    images: ExtractionImages | None,
    reference_date: str | None,
    options: ExtractionOptionsParam | None,
) -> Mapping[str, object]:
    sent: Final[Mapping[str, object | None]] = MappingProxyType(
        {"state": state, "fields": fields, "images": images, "reference_date": reference_date, "options": options}
    )
    return MappingProxyType({name: value for name, value in sent.items() if value is not None})


def _is_unsupported(name: str, value: object, policy: DropPolicy) -> bool:
    return (
        value is not None
        and value is not False
        and name not in _LOCAL_KWARGS
        and not is_litellm_owned_kwarg(name)
        and not policy.drops(name)
    )


def _unsupported_params(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    policy: Final = drop_policy(kwargs)
    return tuple(sorted(name for name, value in kwargs.items() if _is_unsupported(name, value, policy)))


def _prepare_call(
    *,
    model: str,
    request_fields: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
    kwargs: Mapping[str, object],
) -> _ExtractionCall:
    log_caller_request(kwargs, request_fields)
    upstream_model, provider, dynamic_api_key, dynamic_api_base = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )
    provider_config: Final = _provider_config(model, provider)
    unsupported: Final = _unsupported_params(kwargs)
    if unsupported:
        raise litellm.UnsupportedParamsError(
            message=(
                f"{provider} does not support parameters: {list(unsupported)}, for model={model}. "
                "To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
                "`litellm_settings:\n drop_params: true`\n"
            ),
            model=model,
            llm_provider=provider,
        )
    try:
        request: Final = _REQUEST_ADAPTER.validate_python(request_fields)
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=f"Invalid Extraction request: {error}", model=model, llm_provider=provider
        ) from error
    canonical_model: Final = provider_config.canonical_model(upstream_model)
    resolved_api_base: Final = provider_config.resolve_api_base(dynamic_api_base or api_base)
    if resolved_api_base is None:
        raise litellm.BadRequestError(
            message=provider_config.missing_api_base_message(provider), model=model, llm_provider=provider
        )
    resolved_api_key: Final = provider_config.resolve_api_key(dynamic_api_key or api_key)
    if resolved_api_key is None:
        raise litellm.BadRequestError(
            message=f"Missing API key for Extraction provider '{provider}'", model=model, llm_provider=provider
        )
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    http_client: Final = kwargs.get("client")
    if isinstance(logging_obj, LiteLLMLoggingObj):
        logging_obj.update_from_kwargs(  # pyright: ignore[reportUnknownMemberType]  # Logging takes untyped dicts
            kwargs=dict(kwargs),
            model=canonical_model,
            litellm_params={
                "litellm_call_id": kwargs.get("litellm_call_id"),
                "api_base": provider_config.get_complete_url(resolved_api_base, canonical_model),
            },
            custom_llm_provider=provider,
        )
    return _ExtractionCall(
        model=canonical_model,
        requested_model=model,
        custom_llm_provider=provider,
        provider_config=provider_config,
        body=provider_config.transform_extraction_request(canonical_model, request),
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        logging_obj=logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None,
        headers=MappingProxyType(dict(extra_headers or {})),
        timeout=timeout,
        litellm_params=kwargs,
        client=http_client if isinstance(http_client, HTTPHandler | AsyncHTTPHandler) else None,
    )


def _state_rewritten_by_guardrail(litellm_params: Mapping[str, object]) -> bool:
    metadata: Final = litellm_params.get(get_metadata_variable_name_from_kwargs(litellm_params))
    return (
        isinstance(metadata, Mapping)
        and _METADATA_ADAPTER.validate_python(metadata).get(EXTRACTION_STATE_REWRITTEN_METADATA_KEY) is True
    )


def _without_spans(answers: Mapping[str, ExtractionAnswer]) -> Mapping[str, ExtractionAnswer]:
    return {name: answer.model_copy(update={"span": None, "spans": None}) for name, answer in answers.items()}


def _formatted_response(response: ExtractionResponse, call: _ExtractionCall) -> ExtractionResponse:
    spans_index_hidden_text: Final = response.state_text is None and _state_rewritten_by_guardrail(call.litellm_params)
    formatted: Final = response.model_copy(
        update={
            "model": call.requested_model,
            **({"answers": _without_spans(response.answers)} if spans_index_hidden_text else {}),
        }
    )
    formatted.set_hidden_params(
        {
            "model": f"{call.custom_llm_provider}/{call.model}",
            "custom_llm_provider": call.custom_llm_provider,
            "provider_response_model": (
                get_model_cost_key(f"{call.custom_llm_provider}/{response.model}")
                or f"{call.custom_llm_provider}/{call.model}"
            ),
        }
    )
    return formatted


def _upstream_exception(error: Exception, call: _ExtractionCall) -> Exception:
    if isinstance(error, BaseLLMException) and error.status_code_is_synthesized:
        return litellm.APIConnectionError(
            message=f"{call.custom_llm_provider.capitalize()}Exception - {error.message}",
            llm_provider=call.custom_llm_provider,
            model=f"{call.custom_llm_provider}/{call.model}",
        )
    return litellm.exception_type(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped
        model=f"{call.custom_llm_provider}/{call.model}",
        custom_llm_provider=call.custom_llm_provider,
        original_exception=error,
    )


async def _asend(call: _ExtractionCall) -> ExtractionResponse:
    try:
        response: Final = await _HANDLER.aextraction(
            model=call.model,
            custom_llm_provider=call.custom_llm_provider,
            logging_obj=call.logging_obj,
            provider_config=call.provider_config,
            body=call.body,
            api_base=call.api_base,
            api_key=call.api_key,
            headers=call.headers,
            timeout=call.timeout,
            litellm_params=call.litellm_params,
            client=call.client if isinstance(call.client, AsyncHTTPHandler) else None,
        )
    except Exception as error:
        raise _upstream_exception(error, call) from error
    return _formatted_response(response, call)


def _send(call: _ExtractionCall) -> ExtractionResponse:
    try:
        response: Final = _HANDLER.extraction(
            model=call.model,
            custom_llm_provider=call.custom_llm_provider,
            logging_obj=call.logging_obj,
            provider_config=call.provider_config,
            body=call.body,
            api_base=call.api_base,
            api_key=call.api_key,
            headers=call.headers,
            timeout=call.timeout,
            litellm_params=call.litellm_params,
            client=call.client if isinstance(call.client, HTTPHandler) else None,
        )
    except Exception as error:
        raise _upstream_exception(error, call) from error
    return _formatted_response(response, call)


@client
async def aextraction(
    model: str,
    fields: ExtractionFields,
    state: str | None = None,
    images: ExtractionImages | None = None,
    reference_date: str | None = None,
    options: ExtractionOptionsParam | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,  # kwargs-ok: litellm_params forwarded by @client
) -> ExtractionResponse:
    call: Final = await asyncio.to_thread(
        partial(
            _prepare_call,
            model=model,
            request_fields=_request_fields(
                state=state, fields=fields, images=images, reference_date=reference_date, options=options
            ),
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            kwargs=kwargs,
        )
    )
    return await _asend(call)


@client
def extraction(
    model: str,
    fields: ExtractionFields,
    state: str | None = None,
    images: ExtractionImages | None = None,
    reference_date: str | None = None,
    options: ExtractionOptionsParam | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,  # kwargs-ok: litellm_params forwarded by @client
) -> ExtractionResponse:
    call: Final = _prepare_call(
        model=model,
        request_fields=_request_fields(
            state=state, fields=fields, images=images, reference_date=reference_date, options=options
        ),
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    return _send(call)


__all__ = ["aextraction", "extraction"]
