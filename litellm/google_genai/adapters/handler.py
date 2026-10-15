from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Final, cast

import litellm
from litellm.types.google_genai.adapters import GenerateContentCompletionKwargs
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse

from .transformation import GoogleGenAIAdapter

# Initialize adapter
GOOGLE_GENAI_ADAPTER: Final = GoogleGenAIAdapter()


class GenerateContentToCompletionHandler:
    """Handler for transforming generate_content calls to completion format when provider config is None"""

    @staticmethod
    def _prepare_completion_kwargs(
        model: str,
        contents: list[dict[str, object]] | dict[str, object],
        config: dict[str, object] | None = None,
        stream: bool = False,
        litellm_params: GenericLiteLLMParams | None = None,
        extra_kwargs: Mapping[str, object] | None = None,
    ) -> GenerateContentCompletionKwargs:
        """Prepare kwargs for litellm.completion/acompletion"""

        # Transform generate_content request to completion format
        completion_request: Final = GOOGLE_GENAI_ADAPTER.translate_generate_content_to_completion(
            model=model,
            contents=contents,
            config=config,
            litellm_params=litellm_params,
            **(extra_kwargs or {}),
        )

        completion_kwargs: Final = dict(completion_request)

        # Forward extra_kwargs that should be passed to completion call
        if extra_kwargs is not None:
            # Forward metadata for custom callback
            if "metadata" in extra_kwargs:
                completion_kwargs["metadata"] = extra_kwargs["metadata"]
            # Forward extra_headers for providers that require custom headers (e.g., github_copilot)
            if "extra_headers" in extra_kwargs:
                completion_kwargs["extra_headers"] = extra_kwargs["extra_headers"]
            if "proxy_server_request" in extra_kwargs:
                completion_kwargs["proxy_server_request"] = extra_kwargs["proxy_server_request"]

        if stream:
            completion_kwargs["stream"] = stream

        return GenerateContentCompletionKwargs(**completion_kwargs)

    @staticmethod
    async def async_generate_content_handler(
        model: str,
        contents: list[dict[str, object]] | dict[str, object],
        litellm_params: GenericLiteLLMParams,
        config: dict[str, object] | None = None,
        stream: bool = False,
        **kwargs: object,
    ) -> dict[str, object] | AsyncIterator[bytes]:
        """Handle generate_content call asynchronously using completion adapter"""

        completion_kwargs: Final = GenerateContentToCompletionHandler._prepare_completion_kwargs(
            model=model,
            contents=contents,
            config=config,
            stream=stream,
            litellm_params=litellm_params,
            extra_kwargs=kwargs,
        )

        completion_response: Final = await litellm.acompletion(**completion_kwargs)
        if not stream or not hasattr(completion_response, "__aiter__"):
            return GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                cast(ModelResponse, completion_response)
            )
        transformed_stream: Final = GOOGLE_GENAI_ADAPTER.translate_completion_output_params_streaming(
            completion_response
        )
        if transformed_stream is None:
            raise ValueError("Failed to transform streaming response")
        return transformed_stream

    @staticmethod
    def generate_content_handler(
        model: str,
        contents: list[dict[str, object]] | dict[str, object],
        litellm_params: GenericLiteLLMParams,
        config: dict[str, object] | None = None,
        stream: bool = False,
        _is_async: bool = False,
        **kwargs: object,
    ) -> dict[str, object] | AsyncIterator[bytes] | Coroutine[None, None, dict[str, object] | AsyncIterator[bytes]]:
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

        completion_kwargs: Final = GenerateContentToCompletionHandler._prepare_completion_kwargs(
            model=model,
            contents=contents,
            config=config,
            stream=stream,
            litellm_params=litellm_params,
            extra_kwargs=kwargs,
        )

        completion_response: Final = litellm.completion(**completion_kwargs)
        if not stream or not hasattr(completion_response, "__iter__"):
            return GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                cast(ModelResponse, completion_response)
            )
        transformed_stream: Final = GOOGLE_GENAI_ADAPTER.translate_completion_output_params_streaming(
            completion_response
        )
        if transformed_stream is None:
            raise ValueError("Failed to transform streaming response")
        return transformed_stream
