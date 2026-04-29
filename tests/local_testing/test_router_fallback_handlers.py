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
from typing import Any, Dict, List

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


def create_test_router_2():
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


@pytest.mark.parametrize(
    "function_name",
    ["_acompletion", "_atext_completion", "_aembedding"],
)
@pytest.mark.asyncio
async def test_run_async_fallback(function_name):
    """
    Basic test - given a list of fallback models, run the original function with the fallback models
    """
    router = create_test_router()
    original_function = getattr(router, function_name)

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

    if function_name == "_aembedding":
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif function_name == "_atext_completion":
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif function_name == "_acompletion":
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
        **request_kwargs,
    )

    assert result is not None

    if function_name == "_acompletion":
        assert isinstance(result, litellm.ModelResponse)
    elif function_name == "_atext_completion":
        assert isinstance(result, litellm.TextCompletionResponse)
    elif function_name == "_aembedding":
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
@pytest.mark.parametrize("function_name", ["_acompletion", "_atext_completion"])
async def test_failed_fallbacks_raise_most_recent_exception(function_name):
    """
    Tests that if all fallbacks fail, the most recent occuring exception is raised

    meaning the exception from the last fallback model is raised
    """
    router = create_test_router()
    original_function = getattr(router, function_name)

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

    if function_name == "_aembedding":
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif function_name == "_atext_completion":
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif function_name == "_acompletion":
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
            **request_kwargs,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("function_name", ["_acompletion", "_atext_completion"])
async def test_multiple_fallbacks(function_name):
    """
    Tests that if multiple fallbacks passed:
    - fallback 1 = bad configured deployment / failing endpoint
    - fallback 2 = working deployment / working endpoint

    Assert that:
    - a success response is received from the working endpoint (fallback 2)
    """
    router_2 = create_test_router_2()
    original_function = getattr(router_2, function_name)

    fallback_model_group = ["gpt-4", "fake-openai-endpoint-2"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = Exception("Simulated error")

    request_kwargs: Dict[str, Any] = {
        "metadata": {"previous_models": ["gpt-3.5-turbo"]}
    }

    if function_name == "_aembedding":
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif function_name == "_atext_completion":
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif function_name == "_acompletion":
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
        **request_kwargs,
    )

    print(result)

    print(result._hidden_params)

    assert (
        result._hidden_params["api_base"]
        == "https://exampleopenaiendpoint-production.up.railway.app/"
    )


@pytest.mark.asyncio
async def test_fallback_kwargs_not_mutated():
    """
    Verify that each fallback attempt receives a fresh copy of kwargs.

    Regression test for https://github.com/BerriAI/litellm/issues/24764:
    When a provider handler mutates kwargs (e.g. Bedrock pops `tools`),
    subsequent fallback attempts should still see the original parameters.

    We verify this by patching `safe_deep_copy` with a wrapper that tracks
    calls, ensuring it is invoked once per fallback iteration.  If the
    `safe_deep_copy` call is removed, only a shallow reference is passed and
    `safe_deep_copy` is never called — causing this test to fail.
    """
    from litellm.litellm_core_utils.core_helpers import (
        safe_deep_copy as _real_safe_deep_copy,
    )

    router = Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {
                    "model": "openai/primary",
                    "api_key": "fake-key",
                    "api_base": "http://localhost:1/",
                },
            },
            {
                "model_name": "fallback-a",
                "litellm_params": {
                    "model": "openai/fallback-a",
                    "api_key": "fake-key",
                    "api_base": "http://localhost:2/",
                },
            },
            {
                "model_name": "fallback-b",
                "litellm_params": {
                    "model": "openai/fallback-b",
                    "api_key": "fake-key",
                    "api_base": "http://localhost:3/",
                },
            },
        ],
    )

    async def mock_async_function_with_fallbacks(*args, **kwargs):
        """Always fail so the fallback loop keeps going."""
        raise litellm.exceptions.ServiceUnavailableError(
            message="simulated timeout",
            model="test",
            llm_provider="openai",
        )

    router.async_function_with_fallbacks = mock_async_function_with_fallbacks

    original_tools = [
        {
            "type": "function",
            "function": {
                "name": "classify",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    original_tool_choice = {
        "type": "function",
        "function": {"name": "classify"},
    }

    request_kwargs: Dict[str, Any] = {
        "messages": [{"role": "user", "content": "test"}],
        "tools": original_tools,
        "tool_choice": original_tool_choice,
        "stream": True,
        "metadata": {},
    }

    with patch(
        "litellm.router_utils.fallback_event_handlers.safe_deep_copy",
        wraps=_real_safe_deep_copy,
    ) as mock_sdc:
        with pytest.raises(Exception):
            await run_async_fallback(
                litellm_router=router,
                original_function=router._acompletion,
                num_retries=0,
                fallback_model_group=["fallback-a", "fallback-b"],
                original_model_group="primary-model",
                original_exception=Exception("primary failed"),
                max_fallbacks=5,
                fallback_depth=0,
                **request_kwargs,
            )

        # safe_deep_copy must be called once per fallback model group iteration.
        # If the deep-copy is ever removed, this assertion will fail.
        assert mock_sdc.call_count == 2, (
            f"Expected safe_deep_copy to be called once per fallback attempt (2), "
            f"but was called {mock_sdc.call_count} times"
        )
