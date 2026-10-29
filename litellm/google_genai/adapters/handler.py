from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Final, cast

from openai.types.chat import ChatCompletionStreamOptionsParam

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.google_genai.adapters import GenerateContentCompletionKwargs
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse

from .transformation import GoogleGenAIAdapter

# Initialize adapter
GOOGLE_GENAI_ADAPTER: Final = GoogleGenAIAdapter()


_FORWARDED_REQUEST_KEYS: Final = (
    "metadata",
    "litellm_metadata",
    "proxy_server_request",
    "extra_headers",
    "litellm_logging_obj",
    "litellm_call_id",
    "litellm_session_id",
    "no-log",
)


def _adapter_stream_options(requested: object) -> ChatCompletionStreamOptionsParam | None:
    match requested:
        case {"include_usage": True}:
            return {"include_usage": True}
        case Mapping():
            return None
        case _:
            return {"include_usage": True}


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

        request_kwargs: Final = extra_kwargs or {}
        completion_request: Final = GOOGLE_GENAI_ADAPTER.translate_generate_content_to_completion(
            model=model,
            contents=contents,
            config=config,
            litellm_params=litellm_params,
            **request_kwargs,
        )
        stream_options: Final = _adapter_stream_options(request_kwargs.get("stream_options")) if stream else None
        logging_obj: Final = request_kwargs.get("litellm_logging_obj")
        if stream and isinstance(logging_obj, LiteLLMLoggingObj):
            logging_obj.stream_options = stream_options
        completion_kwargs: Final = dict(
            completion_request,
            **{key: request_kwargs[key] for key in _FORWARDED_REQUEST_KEYS if key in request_kwargs},
            **({"stream": True} if stream else {}),
            **({"stream_options": stream_options} if stream_options is not None else {}),
        )

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

        if stream:
            # Check if completion_response is actually a stream or a ModelResponse
            # This can happen in error cases or when stream is not properly supported
            if not hasattr(completion_response, "__aiter__"):
                # If it's not a stream, treat it as a regular response
                generate_content_response = GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                    cast(ModelResponse, completion_response)
                )
                return generate_content_response
            else:
                # Transform streaming completion response to generate_content format
                transformed_stream: Final = GOOGLE_GENAI_ADAPTER.translate_completion_output_params_streaming(
                    completion_response
                )
                if transformed_stream is not None:
                    return transformed_stream
                raise ValueError("Failed to transform streaming response")
        else:
            # Transform completion response back to generate_content format
            generate_content_response = GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                cast(ModelResponse, completion_response)
            )
            return generate_content_response

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

        if stream:
            # Check if completion_response is actually a stream or a ModelResponse
            # This can happen in error cases or when stream is not properly supported
            if not hasattr(completion_response, "__iter__"):
                # If it's not a stream, treat it as a regular response
                generate_content_response = GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                    cast(ModelResponse, completion_response)
                )
                return generate_content_response
            else:
                # Transform streaming completion response to generate_content format
                transformed_stream: Final = GOOGLE_GENAI_ADAPTER.translate_completion_output_params_streaming(
                    completion_response
                )
                if transformed_stream is not None:
                    return transformed_stream
                raise ValueError("Failed to transform streaming response")
        else:
            # Transform completion response back to generate_content format
            generate_content_response = GOOGLE_GENAI_ADAPTER.translate_completion_to_generate_content(
                cast(ModelResponse, completion_response)
            )
            return generate_content_response
