"""Sends a Responses API request to a provider that speaks Responses natively."""

from collections.abc import Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.data_residency import infer_openai_data_residency
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from aiohttp import ClientSession

__all__ = ("execute_responses",)


def execute_responses(
    *,
    http_handler: BaseLLMHTTPHandler,
    model: str,
    input: str | ResponseInputParam,
    provider_config: BaseResponsesAPIConfig,
    requested_params: ResponsesAPIOptionalRequestParams,
    allowed_openai_params: Sequence[str] | None,
    custom_llm_provider: str,
    litellm_params: GenericLiteLLMParams,
    logging_obj: LiteLLMLoggingObj,
    user: str | None,
    extra_headers: dict[str, object] | None,
    extra_body: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    is_async: bool,
    stream: bool | None,
    client: HTTPHandler | AsyncHTTPHandler | None,
    shared_session: "ClientSession | None",
    litellm_metadata: dict[str, object] | None,
    request_kwargs: dict[str, object],
) -> (
    ResponsesAPIResponse
    | BaseResponsesAPIStreamingIterator
    | Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
):
    request_params: Final = ResponsesAPIRequestUtils.get_optional_params_responses_api(
        model=model,
        responses_api_provider_config=provider_config,
        response_api_optional_params=requested_params,
        allowed_openai_params=allowed_openai_params,
        drop_params=normalize_drop_params(request_kwargs.get("drop_params")),
    )

    logging_obj.update_from_kwargs(
        kwargs=request_kwargs,
        model=model,
        user=user,
        optional_params=dict(request_params),
        litellm_params={
            **request_params,
            "aresponses": is_async,
            "litellm_call_id": request_kwargs.get("litellm_call_id"),
            "model_info": request_kwargs.get("model_info"),
            "data_residency": infer_openai_data_residency(custom_llm_provider, litellm_params.api_base),
            "metadata": (
                request_kwargs["litellm_metadata"]
                if "litellm_metadata" in request_kwargs
                else request_kwargs.get("metadata")
            ),
            "provider_affinity_header": litellm_params.provider_affinity_header,
        },
        custom_llm_provider=custom_llm_provider,
    )

    response: Final = http_handler.response_api_handler(
        model=model,
        input=ResponsesAPIRequestUtils.restore_encrypted_content_item_ids_in_input(input),
        responses_api_provider_config=provider_config,
        response_api_optional_request_params=request_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        extra_headers=extra_headers,
        extra_body=extra_body,
        timeout=timeout,
        _is_async=is_async,
        client=client,
        fake_stream=provider_config.should_fake_stream(
            model=model, stream=stream, custom_llm_provider=custom_llm_provider
        ),
        litellm_metadata=litellm_metadata,
        shared_session=shared_session,
    )
    if not isinstance(response, ResponsesAPIResponse):
        return response
    stamped: Final = ResponsesAPIRequestUtils.update_responses_api_response_id_with_model_id(
        responses_api_response=response,
        litellm_metadata=litellm_metadata,
        custom_llm_provider=custom_llm_provider,
    )
    stamped.hidden_params["custom_llm_provider"] = custom_llm_provider
    return stamped
