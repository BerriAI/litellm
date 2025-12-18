"""
Handler for transforming Interactions API requests to LiteLLM Responses API requests.

This follows the same pattern as LiteLLMCompletionTransformationHandler that transforms
Responses API -> Chat Completion API.
"""

from typing import Any, AsyncIterator, Coroutine, Dict, Iterator, Optional, Union

import litellm
from litellm.interactions.litellm_responses_transformation.streaming_iterator import (
    LiteLLMResponsesStreamingIterator,
)
from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.interactions import (
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse


class LiteLLMResponsesTransformationHandler:
    """
    Handler that transforms Interactions API requests to Responses API requests.
    
    This enables the Interactions API to work with any provider that supports
    the Responses API (or the Chat Completion API via the responses bridge).
    """

    def interactions_api_handler(
        self,
        model: Optional[str],
        agent: Optional[str],
        input: Optional[InteractionInput],
        interactions_api_request: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        InteractionsAPIResponse,
        Iterator[InteractionsAPIStreamingResponse],
        Coroutine[
            Any,
            Any,
            Union[
                InteractionsAPIResponse,
                AsyncIterator[InteractionsAPIStreamingResponse],
            ],
        ],
    ]:
        """
        Handle an Interactions API request by transforming it to a Responses API request.
        
        Args:
            model: The model name
            agent: The agent name (used as model if model not specified)
            input: The input content
            interactions_api_request: Optional parameters for the request
            custom_llm_provider: Override the LLM provider
            _is_async: Whether to use async execution
            stream: Whether to stream the response
            extra_headers: Additional headers
            **kwargs: Additional arguments passed to the Responses API
            
        Returns:
            InteractionsAPIResponse or streaming iterator
        """
        # Transform request from Interactions API format to Responses API format
        responses_api_request: Dict[str, Any] = (
            LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
                model=model,
                agent=agent,
                input=input,
                interactions_api_request=interactions_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_interactions_api_handler(
                responses_api_request=responses_api_request,
                request_input=input,
                interactions_api_request=interactions_api_request,
                **kwargs,
            )

        # Make synchronous call to Responses API
        responses_api_response = litellm.responses(
            **responses_api_request,
            **kwargs,
        )

        # Handle non-streaming response
        if isinstance(responses_api_response, ResponsesAPIResponse):
            interactions_api_response: InteractionsAPIResponse = (
                LiteLLMResponsesInteractionsConfig.transform_responses_api_response_to_interactions_api_response(
                    responses_api_response=responses_api_response,
                    request_input=input,
                    interactions_api_request=interactions_api_request,
                )
            )
            return interactions_api_response

        # Handle streaming response
        elif isinstance(responses_api_response, BaseResponsesAPIStreamingIterator):
            return LiteLLMResponsesStreamingIterator(
                model=model or agent or "",
                responses_stream_iterator=responses_api_response,
                request_input=input,
                interactions_api_request=interactions_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )

        # Handle other iterator types (CustomStreamWrapper used internally)
        elif hasattr(responses_api_response, '__iter__') or hasattr(responses_api_response, '__aiter__'):
            return LiteLLMResponsesStreamingIterator(
                model=model or agent or "",
                responses_stream_iterator=responses_api_response,
                request_input=input,
                interactions_api_request=interactions_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )

        # Fallback - should not reach here
        raise ValueError(f"Unexpected response type from Responses API: {type(responses_api_response)}")

    async def async_interactions_api_handler(
        self,
        responses_api_request: Dict[str, Any],
        request_input: Optional[InteractionInput],
        interactions_api_request: InteractionsAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]:
        """
        Async handler for Interactions API requests.
        """
        # Make async call to Responses API
        responses_api_response = await litellm.aresponses(
            **responses_api_request,
            **kwargs,
        )

        # Handle non-streaming response
        if isinstance(responses_api_response, ResponsesAPIResponse):
            interactions_api_response: InteractionsAPIResponse = (
                LiteLLMResponsesInteractionsConfig.transform_responses_api_response_to_interactions_api_response(
                    responses_api_response=responses_api_response,
                    request_input=request_input,
                    interactions_api_request=interactions_api_request,
                )
            )
            return interactions_api_response

        # Handle streaming response
        elif isinstance(responses_api_response, BaseResponsesAPIStreamingIterator):
            return LiteLLMResponsesStreamingIterator(
                model=responses_api_request.get("model", ""),
                responses_stream_iterator=responses_api_response,
                request_input=request_input,
                interactions_api_request=interactions_api_request,
                custom_llm_provider=responses_api_request.get("custom_llm_provider"),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )

        # Handle other async iterator types
        elif hasattr(responses_api_response, '__aiter__'):
            return LiteLLMResponsesStreamingIterator(
                model=responses_api_request.get("model", ""),
                responses_stream_iterator=responses_api_response,
                request_input=request_input,
                interactions_api_request=interactions_api_request,
                custom_llm_provider=responses_api_request.get("custom_llm_provider"),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )

        # Fallback
        raise ValueError(f"Unexpected response type from Responses API: {type(responses_api_response)}")
