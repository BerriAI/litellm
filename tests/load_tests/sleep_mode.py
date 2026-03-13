"""
CustomLLM sleep model for memory profiling and load testing.

Simulates real provider latency and configurable response payload sizes.
Used with k6 or pytest load tests to reproduce production traffic patterns.

Usage in litellm config:
    model_list:
      - model_name: "sleep_model"
        litellm_params:
          model: "openai/sleep"

    Register via:
        import litellm
        from tests.load_tests.sleep_mode import SleepModel
        litellm.custom_provider_map = [
            {"provider": "openai", "custom_handler": SleepModel()}
        ]
"""

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)


def _build_response(
    model_response: ModelResponse,
    model: str,
    output_size: int,
) -> ModelResponse:
    """Build a ModelResponse with a configurable output payload size."""
    # Generate output content of the requested size
    output_content = "x" * output_size

    model_response.choices = [
        Choices(
            message=Message(role="assistant", content=output_content),
            index=0,
            finish_reason="stop",
        )
    ]
    model_response.model = model
    model_response.id = f"chatcmpl-sleep-{uuid.uuid4().hex[:12]}"
    model_response.usage = Usage(
        prompt_tokens=max(1, output_size // 4),
        completion_tokens=max(1, output_size // 4),
        total_tokens=max(2, output_size // 2),
    )
    return model_response


class SleepModel(CustomLLM):
    """A CustomLLM that sleeps for a configurable duration and returns a
    configurable-size payload.

    Accepts these keys in ``optional_params``:
    - ``sleep_time`` (float): seconds to sleep (default 0.01)
    - ``output_size`` (int): characters in the response content (default 100)
    - ``error_rate`` (float): probability [0, 1] of raising an error (default 0)
    """

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
        sleep_time = optional_params.get("sleep_time", 0.01)
        output_size = optional_params.get("output_size", 100)

        time.sleep(sleep_time)

        return _build_response(model_response, model, output_size)

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
        sleep_time = optional_params.get("sleep_time", 0.01)
        output_size = optional_params.get("output_size", 100)

        await asyncio.sleep(sleep_time)

        return _build_response(model_response, model, output_size)

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
        sleep_time = optional_params.get("sleep_time", 0.01)
        output_size = optional_params.get("output_size", 100)
        content = "x" * output_size

        time.sleep(sleep_time)

        yield GenericStreamingChunk(
            text=content,
            is_finished=True,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": max(1, output_size // 4)},
        )

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
        sleep_time = optional_params.get("sleep_time", 0.01)
        output_size = optional_params.get("output_size", 100)
        content = "x" * output_size

        await asyncio.sleep(sleep_time)

        yield GenericStreamingChunk(
            text=content,
            is_finished=True,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": max(1, output_size // 4)},
        )
