import os
from typing import Any

import pytest

from litellm import Router
from litellm.router_utils.fallback_event_handlers import (
    run_async_fallback,
)
from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE

# Helper function to create a Router instance

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
                    "api_base": FAKE_OPENAI_API_BASE,
                },
            },
        ],
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

    request_kwargs: dict[str, Any] = {
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

    assert result._hidden_params["api_base"] == FAKE_OPENAI_API_BASE
