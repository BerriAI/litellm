import asyncio
import copy
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from typing import Any, Dict


import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.abspath("../.."))

from litellm.router_utils.fallback_event_handlers import (
    run_async_fallback,
    log_success_fallback_event,
    log_failure_fallback_event,
)


# Helper function to create a Router instance
def create_test_router():
    return Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}],
    )


router: Router = create_test_router()


@pytest.mark.parametrize(
    "original_function",
    [router._acompletion, router._atext_completion, router._aembedding],
)
@pytest.mark.asyncio
async def test_run_async_fallback(original_function):
    """
    Basic test - given a list of fallback models, run the original function with the fallback models
    """
    litellm.set_verbose = True
    fallback_model_group = ["gpt-4"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    request_kwargs = {
        "mock_response": "hello this is a test for run_async_fallback",
        "metadata": {"previous_models": ["gpt-3.5-turbo"]},
    }

    if original_function == router._aembedding:
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif original_function == router._atext_completion:
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif original_function == router._acompletion:
        request_kwargs["messages"] = [{"role": "user", "content": "Hello, world!"}]

    result = await run_async_fallback(
        litellm_router=router,
        original_function=original_function,
        num_retries=1,
        fallback_model_group=fallback_model_group,
        original_model_group=original_model_group,
        original_exception=original_exception,
        max_fallbacks=5,
        fallback_depth=0,
        **request_kwargs
    )

    assert result is not None

    if original_function == router._acompletion:
        assert isinstance(result, litellm.ModelResponse)
    elif original_function == router._atext_completion:
        assert isinstance(result, litellm.TextCompletionResponse)
    elif original_function == router._aembedding:
        assert isinstance(result, litellm.EmbeddingResponse)


class CustomTestLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.success_fallback_events = []
        self.failure_fallback_events = []

    async def log_success_fallback_event(
        self, original_model_group, kwargs, original_exception
    ):
        print(
            "in log_success_fallback_event for original_model_group: ",
            original_model_group,
        )
        self.success_fallback_events.append(
            (original_model_group, kwargs, original_exception)
        )

    async def log_failure_fallback_event(
        self, original_model_group, kwargs, original_exception
    ):
        print(
            "in log_failure_fallback_event for original_model_group: ",
            original_model_group,
        )
        self.failure_fallback_events.append(
            (original_model_group, kwargs, original_exception)
        )


@pytest.mark.asyncio
async def test_log_success_fallback_event():
    """
    Tests that successful fallback events are logged correctly
    """
    original_model_group = "gpt-3.5-turbo"
    kwargs = {"messages": [{"role": "user", "content": "Hello, world!"}]}
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    logger = CustomTestLogger()
    litellm.callbacks = [logger]

    # This test mainly checks if the function runs without errors
    await log_success_fallback_event(original_model_group, kwargs, original_exception)

    await asyncio.sleep(0.5)
    assert len(logger.success_fallback_events) == 1
    assert len(logger.failure_fallback_events) == 0
    assert logger.success_fallback_events[0] == (
        original_model_group,
        kwargs,
        original_exception,
    )


@pytest.mark.asyncio
async def test_log_failure_fallback_event():
    """
    Tests that failed fallback events are logged correctly
    """
    original_model_group = "gpt-3.5-turbo"
    kwargs = {"messages": [{"role": "user", "content": "Hello, world!"}]}
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    logger = CustomTestLogger()
    litellm.callbacks = [logger]

    # This test mainly checks if the function runs without errors
    await log_failure_fallback_event(original_model_group, kwargs, original_exception)

    await asyncio.sleep(0.5)

    assert len(logger.failure_fallback_events) == 1
    assert len(logger.success_fallback_events) == 0
    assert logger.failure_fallback_events[0] == (
        original_model_group,
        kwargs,
        original_exception,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_function", [router._acompletion, router._atext_completion]
)
async def test_failed_fallbacks_raise_most_recent_exception(original_function):
    """
    Tests that if all fallbacks fail, the most recent occuring exception is raised

    meaning the exception from the last fallback model is raised
    """
    fallback_model_group = ["gpt-4"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    request_kwargs: Dict[str, Any] = {
        "metadata": {"previous_models": ["gpt-3.5-turbo"]}
    }

    if original_function == router._aembedding:
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif original_function == router._atext_completion:
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif original_function == router._acompletion:
        request_kwargs["messages"] = [{"role": "user", "content": "Hello, world!"}]

    with pytest.raises(litellm.exceptions.RateLimitError):
        await run_async_fallback(
            litellm_router=router,
            original_function=original_function,
            num_retries=1,
            fallback_model_group=fallback_model_group,
            original_model_group=original_model_group,
            original_exception=original_exception,
            mock_response="litellm.RateLimitError",
            max_fallbacks=5,
            fallback_depth=0,
            **request_kwargs
        )


router_2 = Router(
    model_list=[
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": "very-fake-key",
            },
        },
        {
            "model_name": "fake-openai-endpoint-2",
            "litellm_params": {
                "model": "openai/fake-openai-endpoint-2",
                "api_key": "working-key-since-this-is-fake-endpoint",
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
            },
        },
    ],
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_function", [router_2._acompletion, router_2._atext_completion]
)
async def test_multiple_fallbacks(original_function):
    """
    Tests that if multiple fallbacks passed:
    - fallback 1 = bad configured deployment / failing endpoint
    - fallback 2 = working deployment / working endpoint

    Assert that:
    - a success response is received from the working endpoint (fallback 2)
    """
    fallback_model_group = ["gpt-4", "fake-openai-endpoint-2"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = Exception("Simulated error")

    request_kwargs: Dict[str, Any] = {
        "metadata": {"previous_models": ["gpt-3.5-turbo"]}
    }

    if original_function == router_2._aembedding:
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif original_function == router_2._atext_completion:
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif original_function == router_2._acompletion:
        request_kwargs["messages"] = [{"role": "user", "content": "Hello, world!"}]

    result = await run_async_fallback(
        litellm_router=router_2,
        original_function=original_function,
        num_retries=1,
        fallback_model_group=fallback_model_group,
        original_model_group=original_model_group,
        original_exception=original_exception,
        max_fallbacks=5,
        fallback_depth=0,
        **request_kwargs
    )

    print(result)

    print(result._hidden_params)

    assert (
        result._hidden_params["api_base"]
        == "https://exampleopenaiendpoint-production.up.railway.app/"
    )


@pytest.mark.asyncio
async def test_fallback_request_object_not_mutated_for_vision_models():
    """
    Tests that when a call to a vision model fails and falls back to another,
    the `messages` object in the request is not mutated.
    """
    model_list = [
        {
            "model_name": "openai-vision-fail",
            "litellm_params": {
                "model": "openai/gpt-4-vision-preview",
                "api_key": "bad-key",
            },
        },
        {
            "model_name": "gemini-vision-success",
            "litellm_params": {
                "model": "gemini/gemini-pro-vision",
                "api_key": os.getenv("GEMINI_API_KEY"),
            },
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"openai-vision-fail": ["gemini-vision-success"]}],
        num_retries=0,
    )

    original_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    messages_for_call = copy.deepcopy(original_messages)

    with patch('litellm.router.litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:

        captured_messages_on_fallback = None

        async def side_effect(*args, **kwargs):
            nonlocal captured_messages_on_fallback
            model = kwargs.get('model')
            messages_arg = kwargs.get('messages') 

            if "openai" in model:
                if messages_arg and isinstance(messages_arg, list):
                    for message in messages_arg:
                        if message.get('role') == 'user' and isinstance(message.get('content'), list):
                            for content_item in message['content']:
                                if content_item.get('type') == 'image_url':
                                    # Simulate adding a provider-specific key or modifying structure
                                    content_item['image_url']['openai_processed_field'] = True
                                    break 
                
                raise litellm.exceptions.AuthenticationError(message="Invalid API key", llm_provider="openai", model=model)
            elif "gemini" in model:
                captured_messages_on_fallback = copy.deepcopy(messages_arg)
                return litellm.ModelResponse(
                    id="chatcmpl-123",
                    choices=[litellm.Choices(finish_reason="stop", index=0, message=litellm.Message(content="hello", role="assistant"))],
                    model="gemini/gemini-pro-vision",
                )

        mock_acompletion.side_effect = side_effect

        response = await router.acompletion(
            model="openai-vision-fail",
            messages=messages_for_call,
        )

        assert response is not None
        assert captured_messages_on_fallback is not None
        assert captured_messages_on_fallback == original_messages
        assert messages_for_call == original_messages
