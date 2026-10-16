"""Hands a bridged Responses request to the native Responses executor, so the bridge never re-enters `aresponses`."""

from collections.abc import Coroutine, Mapping, Sequence
from typing import Final, NoReturn

import httpx
from aiohttp import ClientSession
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.native.responses import execute_responses
from litellm.responses.main import (
    apply_managed_file_id_mapping,
    base_llm_http_handler,
    requested_reasoning,
    resolve_model_provider_for_responses,
    resolve_responses_api_provider_config,
    with_provider_affinity_header,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams

__all__ = ("anative_responses", "has_native_responses_config", "native_logging_obj", "native_responses")

_STRING_KEYED_DICT: Final = TypeAdapter(dict[str, object])
_CLIENT_HEADERS: Final = TypeAdapter(dict[str, str])
_STRINGS: Final = TypeAdapter(tuple[str, ...])


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_keyed_dict(value: object) -> dict[str, object] | None:
    try:
        return _STRING_KEYED_DICT.validate_python(value) if value is not None else None
    except ValidationError:
        return None


def _client_headers(value: object) -> Mapping[str, str] | None:
    try:
        return _CLIENT_HEADERS.validate_python(value) if isinstance(value, dict) else None
    except ValidationError:
        return None


def _shared_session(value: object) -> ClientSession | None:
    return value if isinstance(value, ClientSession) else None


def _http_client(value: object) -> HTTPHandler | AsyncHTTPHandler | None:
    return value if isinstance(value, (HTTPHandler, AsyncHTTPHandler)) else None


def _timeout(value: object) -> float | httpx.Timeout:
    return value if isinstance(value, (float, int, httpx.Timeout)) and value else request_timeout


def _allowed_openai_params(value: object) -> Sequence[str] | None:
    try:
        return _STRINGS.validate_python(value) if value is not None else None
    except ValidationError:
        return None


def _responses_target(
    model: str, forwarded_kwargs: Mapping[str, object], litellm_params: GenericLiteLLMParams
) -> tuple[str, str, BaseResponsesAPIConfig] | None:
    provider_model, custom_llm_provider = resolve_model_provider_for_responses(
        model=model,
        custom_llm_provider=_optional_str(forwarded_kwargs.get("custom_llm_provider")),
        litellm_params=litellm_params,
        local_vars={},
    )
    if custom_llm_provider is None:
        return None
    provider_config: Final = resolve_responses_api_provider_config(
        provider_model, custom_llm_provider, forwarded_kwargs.get("model_info"), litellm_params.api_base
    )
    return None if provider_config is None else (provider_model, custom_llm_provider, provider_config)


def has_native_responses_config(model: str, forwarded_kwargs: Mapping[str, object]) -> bool:
    return _responses_target(model, forwarded_kwargs, GenericLiteLLMParams(**forwarded_kwargs)) is not None


def _native_responses_call(
    request: Mapping[str, object],
    model: str,
    forwarded_kwargs: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
    is_async: bool,
) -> (
    ResponsesAPIResponse
    | BaseResponsesAPIStreamingIterator
    | Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
):
    """Resolve the deployment and hand the translated request to the native Responses executor."""
    litellm_params: Final = GenericLiteLLMParams(**forwarded_kwargs)
    target: Final = _responses_target(model, forwarded_kwargs, litellm_params)
    if target is None:
        raise ValueError(f"No Responses API config for {model}")
    provider_model, custom_llm_provider, provider_config = target
    request_input, request_tools = apply_managed_file_id_mapping(
        input=request["input"],
        tools=request.get("tools"),
        kwargs=dict(forwarded_kwargs),
        local_vars={},
    )
    params: Final = {
        **forwarded_kwargs,
        **request,
        "input": request_input,
        "tools": request_tools,
        "custom_llm_provider": custom_llm_provider,
    }
    reasoning: Final = requested_reasoning(params, provider_model, custom_llm_provider)
    return execute_responses(
        http_handler=base_llm_http_handler,
        model=provider_model,
        input=request_input,
        provider_config=provider_config,
        requested_params=ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            {key: value for key, value in {**params, "reasoning": reasoning}.items() if key != "reasoning_effort"}
        ),
        allowed_openai_params=_allowed_openai_params(forwarded_kwargs.get("allowed_openai_params")),
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        user=_optional_str(request.get("user")),
        extra_headers=with_provider_affinity_header(
            ResponsesAPIRequestUtils.merge_client_forwarded_headers(
                extra_headers=_string_keyed_dict(forwarded_kwargs.get("extra_headers")),
                client_headers=_client_headers(forwarded_kwargs.get("headers")),
            ),
            litellm_params,
            _string_keyed_dict(request.get("metadata")),
            forwarded_kwargs,
            provider_model,
            custom_llm_provider,
        ),
        extra_body=_string_keyed_dict(forwarded_kwargs.get("extra_body")),
        timeout=_timeout(forwarded_kwargs.get("timeout")),
        is_async=is_async,
        stream=request.get("stream") is True,
        client=_http_client(forwarded_kwargs.get("client")),
        shared_session=_shared_session(forwarded_kwargs.get("shared_session")),
        litellm_metadata=_string_keyed_dict(forwarded_kwargs.get("litellm_metadata")) or {},
        request_kwargs=dict(forwarded_kwargs),
    )


def native_logging_obj(
    logging_obj: LiteLLMLoggingObj | None, forwarded_kwargs: Mapping[str, object], request: Mapping[str, object]
) -> LiteLLMLoggingObj | None:
    """The logging object when the native executor can serve the call, None when only `aresponses` can.

    The executor skips the prompt management hooks `aresponses` runs, so a call that needs them re-enters it.
    """
    if logging_obj is None:
        return None
    params: Final = {**forwarded_kwargs, **request}
    if logging_obj.should_run_prompt_management_hooks(
        non_default_params=params, prompt_id=_optional_str(params.get("prompt_id"))
    ):
        return None
    return logging_obj


def _raise_mapped(error: Exception, model: str, forwarded_kwargs: Mapping[str, object]) -> NoReturn:
    raise litellm.exception_type(
        model=model,
        custom_llm_provider=_optional_str(forwarded_kwargs.get("custom_llm_provider")),
        original_exception=error,
        extra_kwargs=dict(forwarded_kwargs),
    ) from error


def native_responses(
    request: Mapping[str, object],
    model: str,
    forwarded_kwargs: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    try:
        response: Final = _native_responses_call(request, model, forwarded_kwargs, logging_obj, is_async=False)
    except Exception as e:  # noqa: BLE001  # every provider failure maps to the litellm exception aresponses would raise
        _raise_mapped(e, model, forwarded_kwargs)
    if isinstance(response, Coroutine):
        raise TypeError("The sync Responses executor returned a coroutine")
    return response


async def anative_responses(
    request: Mapping[str, object],
    model: str,
    forwarded_kwargs: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    try:
        response: Final = _native_responses_call(request, model, forwarded_kwargs, logging_obj, is_async=True)
        return await response if isinstance(response, Coroutine) else response
    except Exception as e:  # noqa: BLE001  # every provider failure maps to the litellm exception aresponses would raise
        _raise_mapped(e, model, forwarded_kwargs)
