# What is this?
## Handler file for a Custom Chat LLM

"""
- completion
- acompletion
- streaming
- async_streaming
"""

from typing import Any, AsyncIterator, Callable, Iterator, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import ImageResponse, ModelResponse, EmbeddingResponse

from .base import BaseLLM


class CustomLLMError(Exception):  # use this for all your exceptions
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class CustomLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def image_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ImageResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: Optional[
            str
        ],  # dynamically set api_key - https://docs.litellm.ai/docs/set_keys#api_key
        api_base: Optional[
            str
        ],  # dynamically set api_base - https://docs.litellm.ai/docs/set_keys#api_base
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        litellm_params=None,
    ) -> EmbeddingResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def aembedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        litellm_params=None,
    ) -> EmbeddingResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")


def custom_chat_llm_router(
    async_fn: bool, stream: Optional[bool], custom_llm: CustomLLM
):
    """
    Routes call to CustomLLM completion/acompletion/streaming/astreaming functions, based on call type

    Validates if response is in expected format
    """
    if async_fn:
        if stream:
            return custom_llm.astreaming
        return custom_llm.acompletion
    if stream:
        return custom_llm.streaming
    return custom_llm.completion
