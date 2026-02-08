import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router


@pytest.mark.asyncio
@pytest.mark.skip(reason="Local e2e test - requires API keys and live LLM endpoints")
async def test_router_auto_router():
    """
    Simple e2e test to validate we get an llm response from the auto router.

    Uses tier-based routing (no embedding model needed).
    The auto-router will classify messages using regex patterns and route
    to the appropriate tier model.
    """
    import litellm
    litellm._turn_on_debug()

    router = Router(
    model_list=[
            {
                "model_name": "litellm-gpt-4.1",
                "litellm_params": {
                    "model": "gpt-4.1",
                },
                "model_info": {"id": "openai-id"},
            },

            {
                "model_name": "litellm-claude-35",
                "litellm_params": {
                    "model": "claude-sonnet-4-5-20250929",
                },
                "model_info": {"id": "claude-id"},
            },
            {
                "model_name": "auto_router1",
                "litellm_params": {
                    "model": "auto_router/auto_router_1",
                    "auto_router_default_model": "gpt-4o-mini",
                },
            },
        ],
    )

    # Coding tasks route to MID tier
    response = await router.acompletion(
        model="auto_router1",
        messages=[{"role": "user", "content": "write a Python function to sort a list"}],
    )
    print(response)
    print("response._hidden_params", response._hidden_params)

    # Simple greetings route to LOW tier
    response = await router.acompletion(
        model="auto_router1",
        messages=[{"role": "user", "content": "hello"}],
    )
    print("response._hidden_params", response._hidden_params)
