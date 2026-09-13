"""
Handler for transforming responses api requests to litellm.completion requests
"""

from collections.abc import Coroutine, Mapping
from typing import Final

import httpx

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    MockResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse


class LiteLLMCompletionTransformationHandler:
    @staticmethod
    def _maybe_wrap_as_fake_stream(
        responses_api_response: ResponsesAPIResponse,
        model: str,
        custom_llm_provider: str | None,
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse | MockResponsesAPIStreamingIterator:
        """
        An interceptor (e.g. websearch interception) can force stream=False so its
        agentic loop runs on the non-streaming path. When the caller originally asked
        for streaming, rebuild a synthetic responses stream from the final response.
        """
        if not kwargs.get("_websearch_interception_converted_stream"):
            return responses_api_response
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
        from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if not isinstance(logging_obj, LiteLLMLoggingObj):
            return responses_api_response

        raw_response: Final = httpx.Response(status_code=200, json=responses_api_response.model_dump())
        litellm_metadata: Final = kwargs.get("litellm_metadata")
        return MockResponsesAPIStreamingIterator(
            response=raw_response,
            model=model,
            responses_api_provider_config=OpenAIResponsesAPIConfig(),
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata if isinstance(litellm_metadata, dict) else None,
            custom_llm_provider=custom_llm_provider,
        )

    def response_api_handler(
        self,
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
        litellm_completion_request: Final[dict] = (
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

        completion_args: Final = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)
        completion_args["_skip_responses_api_bridge"] = True

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

            return self._maybe_wrap_as_fake_stream(
                responses_api_response=responses_api_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
            )

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

        acompletion_args: Final = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(litellm_completion_request)
        acompletion_args["_skip_responses_api_bridge"] = True

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

            return self._maybe_wrap_as_fake_stream(
                responses_api_response=responses_api_response,
                model=litellm_completion_request.get("model") or "",
                custom_llm_provider=litellm_completion_request.get("custom_llm_provider"),
                kwargs=kwargs,
            )

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
