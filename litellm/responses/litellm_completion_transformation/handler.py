"""
Handler for transforming responses api requests to litellm.completion requests
"""

from typing import Any, Coroutine, Dict, Optional, Union

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse


def _resolve_item_references_to_previous_response_id(
    request_input: Union[str, ResponseInputParam, None],
) -> Optional[str]:
    """When a Responses API client sends prior turn items as opaque
    ``item_reference`` entries (``@ai-sdk/openai`` with default ``store: true``
    does this), resolve them to the original ``previous_response_id`` so the
    existing session handler can rebuild conversation history.

    Scans the input list for items of type ``item_reference``, decodes the
    most recent one through the envelope codec, and returns the result in the
    ``resp_<env>`` form that ``async_responses_api_session_handler`` already
    accepts. Returns ``None`` when no decodable references are present.
    """
    if not isinstance(request_input, list):
        return None
    refs = [
        item
        for item in request_input
        if isinstance(item, dict) and item.get("type") == "item_reference"
    ]
    if not refs:
        return None
    for ref in reversed(refs):
        decoded = ResponsesAPIRequestUtils._decode_item_envelope(ref.get("id", ""))
        if decoded:
            return decoded
    return None


def _strip_item_references(
    request_input: Union[str, ResponseInputParam, None],
) -> Union[str, ResponseInputParam, None]:
    """Filter ``item_reference`` items out of the input list so the standard
    message transformer does not see opaque references it cannot inline.
    Non-list inputs pass through unchanged.
    """
    if not isinstance(request_input, list):
        return request_input
    return [
        item
        for item in request_input
        if not (isinstance(item, dict) and item.get("type") == "item_reference")
    ]


class LiteLLMCompletionTransformationHandler:
    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        litellm_completion_request: dict = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                litellm_completion_request=litellm_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **kwargs,
            )

        completion_args = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = litellm.completion(
            **completion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
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
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        # When the caller did not supply ``previous_response_id`` explicitly,
        # try to derive it from any ``item_reference`` entries in the input.
        # Vercel ``@ai-sdk/openai`` with default ``store: true`` sends prior
        # turn items as references instead of inlining them, so without this
        # the bridge would treat each turn as a fresh conversation.
        if not previous_response_id:
            resolved = _resolve_item_references_to_previous_response_id(request_input)
            if resolved:
                previous_response_id = resolved
                request_input = _strip_item_references(request_input)
                litellm_completion_request[
                    "messages"
                ] = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                    input=request_input,
                    responses_api_request=responses_api_request,
                )
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )

        acompletion_args = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = await litellm.acompletion(
            **acompletion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
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
                custom_llm_provider=litellm_completion_request.get(
                    "custom_llm_provider"
                ),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )
