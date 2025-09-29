# What is this?
## Handler file for a Custom Chat LLM

"""
- completion
- acompletion
- streaming
- async_streaming
"""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Optional,
    Type,
    Union,
)

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import EmbeddingResponse, ImageResponse, ModelResponse

from .base import BaseLLM

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper


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
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
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
    ) -> Union[
        Coroutine[Any, Any, Union[ModelResponse, "CustomStreamWrapper"]],
        Union[ModelResponse, "CustomStreamWrapper"],
    ]:
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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")



def register_custom_provider(
    provider: str,
    handler: Union["CustomLLM", Type["CustomLLM"]],
) -> "CustomLLM":
    """Register a custom provider so Router can resolve it.

    Accepts either an instantiated handler or the handler class. Ensures the
    provider appears in ``litellm.custom_provider_map`` and updates
    ``provider_list`` / ``_custom_providers`` so validation succeeds.
    """

    # Deferred imports keep this helper usable during package initialization.
    import litellm  # type: ignore

    if isinstance(handler, type):
        instance: CustomLLM = handler()
    else:
        instance = handler

    # Ensure globals exist even during early import cycles.
    current_map = getattr(litellm, "custom_provider_map", None)
    if not isinstance(current_map, list):
        current_map = []

    # Replace existing entry for this provider if present.
    filtered = [item for item in current_map if item.get("provider") != provider]
    filtered.append({"provider": provider, "custom_handler": instance})
    litellm.custom_provider_map = filtered  # type: ignore[attr-defined]

    if not hasattr(litellm, "_custom_providers") or not isinstance(
        getattr(litellm, "_custom_providers"), list
    ):
        litellm._custom_providers = []  # type: ignore[attr-defined]
    if provider not in litellm._custom_providers:  # type: ignore[attr-defined]
        litellm._custom_providers.append(provider)  # type: ignore[attr-defined]

    provider_list = getattr(litellm, "provider_list", None)
    if isinstance(provider_list, list):
        if provider not in provider_list:
            provider_list.append(provider)
    else:
        litellm.provider_list = [provider]  # type: ignore[attr-defined]

    # Best-effort: update any additional bookkeeping helpers.
    try:
        from litellm.utils import custom_llm_setup

        custom_llm_setup()
    except Exception:
        pass

    return instance

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
