"""
Handler for transforming responses api requests to litellm.completion requests
"""

from typing import Any, Coroutine, Dict, List, Optional, Union

import litellm
from litellm.litellm_core_utils.asyncify import run_async_function
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


class LiteLLMCompletionTransformationHandler:

    @staticmethod
    def _prepend_compaction_to_response(
        responses_api_response: ResponsesAPIResponse,
        compaction_item: Dict[str, Any],
    ) -> ResponsesAPIResponse:
        if responses_api_response.output is None:
            responses_api_response.output = []
        responses_api_response.output.insert(0, compaction_item)
        return responses_api_response

    async def _maybe_compact_messages(
        self,
        model: str,
        messages: List[Any],
        context_management: Optional[List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> tuple[Optional[Dict[str, Any]], List[Any]]:
        if not context_management:
            return None, messages

        new_messages, compaction_item = (
            await LiteLLMCompletionResponsesConfig._transform_context_management(
                model=model,
                input=messages,
                context_management=context_management,
                **kwargs,
            )
        )
        if compaction_item is None:
            return None, messages
        return compaction_item, new_messages

    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        context_management: Optional[List[Dict[str, Any]]] = None,
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
                context_management=context_management,
                **kwargs,
            )

        compaction_item: Optional[Dict[str, Any]] = None
        if context_management:
            compact_model = litellm_completion_request.get("model", model)
            compaction_item, compacted_messages = run_async_function(
                self._maybe_compact_messages,
                model=compact_model,
                messages=litellm_completion_request.get("messages", []),
                context_management=context_management,
                **kwargs,
            )
            if compaction_item:
                litellm_completion_request["messages"] = compacted_messages

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
            if compaction_item:
                return self._prepend_compaction_to_response(
                    responses_api_response, compaction_item
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
                compaction_item=compaction_item,
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        context_management: Optional[List[Dict[str, Any]]] = None,
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

        model = litellm_completion_request.get("model", "")

        compaction_item, compacted_messages = await self._maybe_compact_messages(
            model=model,
            messages=litellm_completion_request.get("messages", []),
            context_management=context_management,
            **kwargs,
        )
        if compaction_item:
            litellm_completion_request["messages"] = compacted_messages

        acompletion_args = {**kwargs, **litellm_completion_request}
        acompletion_args.pop("context_management", None)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = await litellm.acompletion(**acompletion_args)

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )
            if compaction_item:
                return self._prepend_compaction_to_response(
                    responses_api_response, compaction_item
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
                compaction_item=compaction_item,
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )
