"""
Handler for transforming responses api requests to litellm.completion requests
"""

from typing import Any, Coroutine, Dict, List, Optional, Union

import litellm
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
from litellm.types.utils import ModelResponse


def _strip_empty_text_content_parts(litellm_completion_request: dict) -> dict:
    """
    Strip empty text content parts from messages before calling litellm.completion().

    Some strict OpenAI-compatible models (e.g. Kimi-K2.5, gpt-oss-120b on Azure AI)
    reject messages whose content array contains {"type": "text", "text": ""}.  This
    happens when the Responses API → chat/completions transformation produces a text
    part next to a tool-output part but the text is an empty string.

    Only returns a modified copy of the dict when changes are actually needed.
    """
    messages: Optional[List[Any]] = litellm_completion_request.get("messages")
    if not messages:
        return litellm_completion_request

    modified = False
    new_messages: List[Any] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            filtered = [
                part
                for part in content
                if not (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and part.get("text") == ""
                )
            ]
            if len(filtered) != len(content):
                modified = True
                msg = dict(msg)
                # Avoid sending an empty content list; fall back to empty string.
                msg["content"] = filtered if filtered else ""
        new_messages.append(msg)

    if not modified:
        return litellm_completion_request

    result = dict(litellm_completion_request)
    result["messages"] = new_messages
    return result


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

        litellm_completion_request = _strip_empty_text_content_parts(
            litellm_completion_request
        )

        completion_args = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = litellm.completion(
            **litellm_completion_request,
            **kwargs,
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
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

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
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )

        litellm_completion_request = _strip_empty_text_content_parts(
            litellm_completion_request
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
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")
