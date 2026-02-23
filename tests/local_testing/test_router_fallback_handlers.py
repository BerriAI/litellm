import asyncio
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
