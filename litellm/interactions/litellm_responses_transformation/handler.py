"""
Handler for transforming interactions API requests to litellm.responses requests.
"""

from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from typing import Any, Final

import litellm
from litellm.interactions.litellm_responses_transformation.streaming_iterator import (
    LiteLLMResponsesInteractionsStreamingIterator,
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


class LiteLLMResponsesInteractionsHandler:
    """Handler for bridging Interactions API to Responses API via litellm.responses()."""

    def interactions_api_handler(
        self,
        model: str,
        input: InteractionInput | None,
        optional_params: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: str | None = None,
        _is_async: bool = False,
        stream: bool | None = None,
        **kwargs,
    ) -> (
        InteractionsAPIResponse
        | Iterator[InteractionsAPIStreamingResponse]
        | Coroutine[object, object, InteractionsAPIResponse | AsyncIterator[InteractionsAPIStreamingResponse]]
    ):
        """
        Handle Interactions API request by calling litellm.responses().

        Args:
            model: The model to use
            input: The input content
            optional_params: Optional parameters for the request
            custom_llm_provider: Override LLM provider
            _is_async: Whether this is an async call
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            InteractionsAPIResponse or streaming iterator
        """
        # Transform interactions request to responses request
        responses_request: Final = (
            LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
                model=model,
                input=input,
                optional_params=optional_params,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_interactions_api_handler(
                responses_request=responses_request,
                model=model,
                input=input,
                optional_params=optional_params,
                **kwargs,
            )

        # Call litellm.responses()
        # Note: litellm.responses() returns Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        # but the type checker may see it as a coroutine in some contexts
        responses_fn: Final[Callable[..., ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]] = vars(litellm)[
            "responses"
        ]
        responses_response: Final = responses_fn(
            **responses_request,
        )

        # Handle streaming response
        if isinstance(responses_response, BaseResponsesAPIStreamingIterator):
            return LiteLLMResponsesInteractionsStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=responses_response,
                request_input=input,
                optional_params=optional_params,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )

        # At this point, responses_response must be ResponsesAPIResponse (not streaming)
        responses_api_response: Final = responses_response

        # Transform responses response to interactions response
        return LiteLLMResponsesInteractionsConfig.transform_responses_response_to_interactions_response(
            responses_response=responses_api_response,
            model=model,
        )

    async def async_interactions_api_handler(
        self,
        responses_request: dict[str, Any],
        model: str,
        input: InteractionInput | None,
        optional_params: InteractionsAPIOptionalRequestParams,
        **kwargs,
    ) -> InteractionsAPIResponse | AsyncIterator[InteractionsAPIStreamingResponse]:
        """Async handler for interactions API requests."""
        # Call litellm.aresponses()
        # Note: litellm.aresponses() returns Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        aresponses_fn: Final[
            Callable[..., Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]]
        ] = vars(litellm)["aresponses"]
        responses_response: Final = await aresponses_fn(
            **responses_request,
        )

        # Handle streaming response
        if isinstance(responses_response, BaseResponsesAPIStreamingIterator):
            return LiteLLMResponsesInteractionsStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=responses_response,
                request_input=input,
                optional_params=optional_params,
                custom_llm_provider=responses_request.get("custom_llm_provider"),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )

        # At this point, responses_response must be ResponsesAPIResponse (not streaming)
        responses_api_response: Final = responses_response

        # Transform responses response to interactions response
        return LiteLLMResponsesInteractionsConfig.transform_responses_response_to_interactions_response(
            responses_response=responses_api_response,
            model=model,
        )
