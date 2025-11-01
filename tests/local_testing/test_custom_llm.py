# What is this?
## Unit tests for the CustomLLM class


import asyncio
import os
import sys
import time
import traceback

import openai
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Optional,
    Union,
)
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from dotenv import load_dotenv

import litellm
from litellm import (
    ChatCompletionDeltaChunk,
    ChatCompletionUsageBlock,
    CustomLLM,
    GenericStreamingChunk,
    ModelResponse,
    acompletion,
    completion,
    get_llm_provider,
    image_generation,
)
from litellm.utils import ModelResponseIterator
from litellm.types.utils import ImageResponse, ImageObject, EmbeddingResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


class CustomModelResponseIterator:
    def __init__(self, streaming_response: Union[Iterator, AsyncIterator]):
        self.streaming_response = streaming_response

    def chunk_parser(self, chunk: Any) -> GenericStreamingChunk:
        return GenericStreamingChunk(
            text="hello world",
            tool_use=None,
            is_finished=True,
            finish_reason="stop",
            usage=ChatCompletionUsageBlock(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
            index=0,
        )

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self) -> GenericStreamingChunk:
        try:
            chunk: Any = self.streaming_response.__next__()  # type: ignore
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.chunk_parser(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()  # type: ignore
        return self.streaming_response

    async def __anext__(self) -> GenericStreamingChunk:
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.chunk_parser(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")


class MyCustomLLM(CustomLLM):
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable[..., Any],
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, openai.Timeout]] = None,
        client: Optional[litellm.HTTPHandler] = None,
    ) -> ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable[..., Any],
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, openai.Timeout]] = None,
        client: Optional[litellm.AsyncHTTPHandler] = None,
    ) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable[..., Any],
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, openai.Timeout]] = None,
        client: Optional[litellm.HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": "Hello world",
            "tool_use": None,
            "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        }

        completion_stream = ModelResponseIterator(
            model_response=generic_streaming_chunk  # type: ignore
        )
        custom_iterator = CustomModelResponseIterator(
            streaming_response=completion_stream
        )
        return custom_iterator

    async def astreaming(  # type: ignore
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable[..., Any],
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, openai.Timeout]] = None,
        client: Optional[litellm.AsyncHTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": "Hello world",
            "tool_use": None,
            "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        }

        yield generic_streaming_chunk  # type: ignore

    def image_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout=None,
        client: Optional[HTTPHandler] = None,
    ):
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url="https://example.com/image.png")],
            response_ms=1000,
        )

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout=None,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url="https://example.com/image.png")],
            response_ms=1000,
        )

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
        model_response.model = model

        model_response.data = [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": i,
            }
            for i, _ in enumerate(input)
        ]

        return model_response

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
        model_response.model = model

        model_response.data = [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": i,
            }
            for i, _ in enumerate(input)
        ]

        return model_response


def test_get_llm_provider():
    """"""
    from litellm.utils import custom_llm_setup

    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]

    custom_llm_setup()

    model, provider, _, _ = get_llm_provider(model="custom_llm/my-fake-model")

    assert provider == "custom_llm"


def test_simple_completion():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = completion(
        model="custom_llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert resp.choices[0].message.content == "Hi!"


@pytest.mark.asyncio
async def test_simple_acompletion():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = await acompletion(
        model="custom_llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert resp.choices[0].message.content == "Hi!"


def test_simple_completion_streaming():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = completion(
        model="custom_llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
        stream=True,
    )

    for chunk in resp:
        print(chunk)
        if chunk.choices[0].finish_reason is None:
            assert isinstance(chunk.choices[0].delta.content, str)
        else:
            assert chunk.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_simple_completion_async_streaming():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = await litellm.acompletion(
        model="custom_llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
        stream=True,
    )

    async for chunk in resp:
        print(chunk)
        if chunk.choices[0].finish_reason is None:
            assert isinstance(chunk.choices[0].delta.content, str)
        else:
            assert chunk.choices[0].finish_reason == "stop"


def test_simple_image_generation():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = image_generation(
        model="custom_llm/my-fake-model",
        prompt="Hello world",
    )

    print(resp)


@pytest.mark.asyncio
async def test_simple_image_generation_async():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = await litellm.aimage_generation(
        model="custom_llm/my-fake-model",
        prompt="Hello world",
    )

    print(resp)


@pytest.mark.asyncio
async def test_image_generation_async_additional_params():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]

    with patch.object(
        my_custom_llm, "aimage_generation", new=AsyncMock()
    ) as mock_client:
        try:
            resp = await litellm.aimage_generation(
                model="custom_llm/my-fake-model",
                prompt="Hello world",
                api_key="my-api-key",
                api_base="my-api-base",
                my_custom_param="my-custom-param",
            )

            print(resp)
        except Exception as e:
            print(e)

        mock_client.assert_awaited_once()

        mock_client.call_args.kwargs["api_key"] == "my-api-key"
        mock_client.call_args.kwargs["api_base"] == "my-api-base"
        mock_client.call_args.kwargs["optional_params"] == {
            "my_custom_param": "my-custom-param"
        }


def test_get_supported_openai_params():

    class MyCustomLLM(CustomLLM):

        # This is what `get_supported_openai_params` should be returning:
        def get_supported_openai_params(self, model: str) -> list[str]:
            return [
                "tools",
                "tool_choice",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "typical_p",
                "stop",
                "seed",
                "response_format",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "repeat_penalty",
                "tfs_z",
                "mirostat_mode",
                "mirostat_tau",
                "mirostat_eta",
                "logit_bias",
            ]

        def completion(self, *args, **kwargs) -> litellm.ModelResponse:
            return litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world"}],
                mock_response="Hi!",
            )  # type: ignore

    my_custom_llm = MyCustomLLM()

    litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
        {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
    ]

    resp = completion(
        model="my-custom-llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert resp.choices[0].message.content == "Hi!"

    # Get supported openai params
    from litellm import get_supported_openai_params

    response = get_supported_openai_params(model="my-custom-llm/my-fake-model")
    assert response is not None

def test_simple_embedding():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = litellm.embedding(
        model="custom_llm/my-fake-model",
        input=["good morning from litellm", "good night from litellm"]
    )

    assert resp.data[1] == {
        "object": "embedding",
        "embedding": [0.1, 0.2, 0.3],
        "index": 1,
    }

@pytest.mark.asyncio
async def test_simple_aembedding():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = await litellm.aembedding(
        model="custom_llm/my-fake-model",
        input=["good morning from litellm", "good night from litellm"]
    )

    assert resp.data[1] == {
        "object": "embedding",
        "embedding": [0.1, 0.2, 0.3],
        "index": 1,
    }
