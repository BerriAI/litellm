from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union, cast

import litellm
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse

from .transformation import GoogleGenAIAdapter

# Initialize adapter
GOOGLE_GENAI_ADAPTER = GoogleGenAIAdapter()


class GenerateContentToCompletionHandler:
    """Handler for transforming generate_content calls to completion format when provider config is None"""

    @staticmethod
    def _prepare_completion_kwargs(
        model: str,
        contents: Union[List[Dict[str, Any]], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        litellm_params: Optional[GenericLiteLLMParams] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare kwargs for litellm.completion/acompletion"""

        # Transform generate_content request to completion format
        completion_request = (
            GOOGLE_GENAI_ADAPTER.translate_generate_content_to_completion(
                model=model,
                contents=contents,
                config=config,
                litellm_params=litellm_params,
                **(extra_kwargs or {}),
            )
        )

        completion_kwargs: Dict[str, Any] = dict(completion_request)

        # feed metadata for custom callback
        if extra_kwargs is not None and "metadata" in extra_kwargs:
            completion_kwargs["metadata"] = extra_kwargs["metadata"]

        if stream:
            completion_kwargs["stream"] = stream

        return completion_kwargs

    @staticmethod
    async def async_generate_content_handler(
        model: str,
        contents: Union[List[Dict[str, Any]], Dict[str, Any]],
        litellm_params: GenericLiteLLMParams,
        config: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncIterator[bytes]]:
        """Handle generate_content call asynchronously using completion adapter"""

        completion_kwargs = (
            GenerateContentToCompletionHandler._prepare_completion_kwargs(
                model=model,
                contents=contents,
                config=config,
                stream=stream,
                litellm_params=litellm_params,
                extra_kwargs=kwargs,
            )
        )

        try:
            completion_response = await litellm.acompletion(**completion_kwargs)

            if stream:
                # Transform streaming completion response to generate_content format
                transformed_stream = (
                    GOOGLE_GENAI_ADAPTER.translate_completion_output_params_streaming(
                        completion_response
                    )
                )
                if transformed_stream is not None:
                    return transformed_stream
                raise ValueError("Failed to transform streaming response")
            else:
                # Transform completion response back to generate_content format
                generate_content_response = (
                    GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                        cast(ModelResponse, completion_response)
                    )
                )
                return generate_content_response

        except Exception as e:
            raise ValueError(
                f"Error calling litellm.acompletion for generate_content: {str(e)}"
            )

    @staticmethod
    def generate_content_handler(
        model: str,
        contents: Union[List[Dict[str, Any]], Dict[str, Any]],
        litellm_params: GenericLiteLLMParams,
        config: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        Dict[str, Any],
        AsyncIterator[bytes],
        Coroutine[Any, Any, Union[Dict[str, Any], AsyncIterator[bytes]]],
    ]:
        """Handle generate_content call using completion adapter"""

        if _is_async:
            return GenerateContentToCompletionHandler.async_generate_content_handler(
                model=model,
                contents=contents,
                config=config,
                stream=stream,
                litellm_params=litellm_params,
                **kwargs,
            )

        completion_kwargs = (
            GenerateContentToCompletionHandler._prepare_completion_kwargs(
                model=model,
                contents=contents,
                config=config,
                stream=stream,
                litellm_params=litellm_params,
                extra_kwargs=kwargs,
            )
        )

        try:
            completion_response = litellm.completion(**completion_kwargs)

            if stream:
                # Transform streaming completion response to generate_content format
                transformed_stream = (
                    GOOGLE_GENAI_ADAPTER.translate_completion_output_params_streaming(
                        completion_response
                    )
                )
                if transformed_stream is not None:
                    return transformed_stream
                raise ValueError("Failed to transform streaming response")
            else:
                # Transform completion response back to generate_content format
                generate_content_response = (
                    GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                        cast(ModelResponse, completion_response)
                    )
                )
                return generate_content_response

        except Exception as e:
            raise ValueError(
                f"Error calling litellm.completion for generate_content: {str(e)}"
            )
