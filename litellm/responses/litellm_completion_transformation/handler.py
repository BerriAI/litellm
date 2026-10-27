"""
Handler for transforming responses api requests to litellm.completion requests
"""

from collections.abc import Coroutine, Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.responses.additional_tools import hoist_additional_tools
from litellm.responses.litellm_completion_transformation.compaction_items import holds_compaction_summary
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    with_compaction_items_as_params,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse

_REQUEST_KWARGS_ADAPTER: Final = TypeAdapter(dict[str, object])


class LiteLLMCompletionTransformationHandler:
    def response_api_handler(
        self,
        previous_response_client_ids: tuple[str, ...] = (),
        /,
        *,
        model: str,
        input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: str | None = None,
        _is_async: bool = False,
        stream: bool | None = None,
        extra_headers: Mapping[str, object] | None = None,
        **kwargs,
    ) -> (
        ResponsesAPIResponse
        | BaseResponsesAPIStreamingIterator
        | Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
    ):
        hoisted: Final = hoist_additional_tools(input, responses_api_request.get("tools"))
        bridged_input: Final = hoisted.input
        bridged_request: Final[ResponsesAPIOptionalRequestParams] = (
            {**responses_api_request, "tools": list(hoisted.tools)} if hoisted.hoisted else responses_api_request
        )
        litellm_completion_request: Final = _REQUEST_KWARGS_ADAPTER.validate_python(
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=bridged_input,
                responses_api_request=bridged_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                previous_response_client_ids,
                litellm_completion_request=litellm_completion_request,
                request_input=bridged_input,
                responses_api_request=bridged_request,
                **kwargs,
            )

        previous_response_id: Final = bridged_request.get("previous_response_id")
        completion_request: Final = _REQUEST_KWARGS_ADAPTER.validate_python(
            run_async_function(
                LiteLLMCompletionResponsesConfig.async_responses_api_session_handler,
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
                instructions=bridged_request.get("instructions"),
                request_kwargs=_REQUEST_KWARGS_ADAPTER.validate_python(kwargs),
                previous_response_client_ids=previous_response_client_ids,
                input_replaces_history=holds_compaction_summary(with_compaction_items_as_params(bridged_input)),
            )
            if previous_response_id
            else litellm_completion_request
        )
        completion_args: Final = _REQUEST_KWARGS_ADAPTER.validate_python(
            {**kwargs, **completion_request, "_skip_responses_api_bridge": True}
        )

        litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = litellm.completion(
            **completion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: Final[ResponsesAPIResponse] = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=bridged_input,
                    responses_api_request=bridged_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=bridged_input,
                responses_api_request=bridged_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

    async def async_response_api_handler(
        self,
        previous_response_client_ids: tuple[str, ...] = (),
        /,
        *,
        litellm_completion_request: dict,
        request_input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        previous_response_id: Final = responses_api_request.get("previous_response_id")
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
                instructions=responses_api_request.get("instructions"),
                request_kwargs=_REQUEST_KWARGS_ADAPTER.validate_python(kwargs),
                previous_response_client_ids=previous_response_client_ids,
                input_replaces_history=holds_compaction_summary(with_compaction_items_as_params(request_input)),
            )

        acompletion_args: Final = _REQUEST_KWARGS_ADAPTER.validate_python(
            {**kwargs, **litellm_completion_request, "_skip_responses_api_bridge": True}
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
