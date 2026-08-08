"""
Handler for transforming responses api requests to litellm.completion requests
"""

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Any, Final

import litellm
from litellm._logging import verbose_logger
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.litellm_core_utils.get_litellm_params import OPTIONAL_KWARGS_KEYS
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse, all_litellm_params

BRIDGE_COMPLETION_KWARGS: Final[frozenset[str]] = (
    frozenset(all_litellm_params)
    | frozenset(OPENAI_CHAT_COMPLETION_PARAMS)
    | OPTIONAL_KWARGS_KEYS
    | frozenset(
        {
            "extra_body",
            "extra_query",
            "drop_params",
            "additional_drop_params",
            "ssl_verify",
            "mock_tool_calls",
            "project_id",
            "space_id",
            "initial_prompt_value",
        }
    )
)


def _filter_bridge_kwargs(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    allowed: Final = BRIDGE_COMPLETION_KWARGS | frozenset(kwargs.get("allowed_openai_params") or ())
    dropped: Final = tuple(k for k in kwargs if k not in allowed)
    if dropped:
        verbose_logger.debug("Responses API to chat completion bridge dropped unsupported params: %s", dropped)
    return MappingProxyType({k: v for k, v in kwargs.items() if k in allowed})


class LiteLLMCompletionTransformationHandler:
    def response_api_handler(
        self,
        model: str,
        input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: str | None = None,
        _is_async: bool = False,
        stream: bool | None = None,
        extra_headers: dict[str, Any] | None = None,
        **kwargs,
    ) -> (
        ResponsesAPIResponse
        | BaseResponsesAPIStreamingIterator
        | Coroutine[Any, Any, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
    ):
        bridge_kwargs: Final = _filter_bridge_kwargs(kwargs)
        litellm_completion_request: Final[dict] = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **bridge_kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                litellm_completion_request=litellm_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **bridge_kwargs,
            )

        completion_args: Final = MappingProxyType(
            {
                **bridge_kwargs,
                **litellm_completion_request,
                "_skip_responses_api_bridge": True,
            }
        )

        litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = litellm.completion(
            **completion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: Final[ResponsesAPIResponse] = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=bridge_kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        previous_response_id: Final[str | None] = responses_api_request.get("previous_response_id")
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )

        acompletion_args: Final = MappingProxyType(
            {
                **kwargs,
                **litellm_completion_request,
                "_skip_responses_api_bridge": True,
            }
        )

        litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = await litellm.acompletion(
            **acompletion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: Final[ResponsesAPIResponse] = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=litellm_completion_request.get("model") or "",
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=litellm_completion_request.get("custom_llm_provider"),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")
