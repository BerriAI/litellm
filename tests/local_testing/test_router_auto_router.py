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

router = Router(
    model_list=[
            {
                "model_name": "gpt-4.1",
                "litellm_params": {
                    "model": "gpt-4.1",
                },
                "model_info": {"id": "openai-id"},
            },
            
            {
                "model_name": "claude-3-5-sonnet-latest",
                "litellm_params": {
                    "model": "claude-3-5-sonnet-latest",
                },
                "model_info": {"id": "claude-id"},
            },
            {
                "model_name": "auto_router1",
                "litellm_params": {
                    "model": "auto_router/auto_router_1",
                    "router_config_path": "auto_router/router.json",
                },
            },
            {
                "model_name": "auto_router_2",
                "litellm_params": {
                    "model": "auto_router/auto_router_2",
                    "router_config_path": "auto_router/router_2.json",
                },
            },
        ],
    )


async def test_router_auto_router():
    """
    Simple e2e test to validate we get an llm response from the auto router
    """
    response = await router.acompletion(
        model="auto_router1",
        messages=[{"role": "user", "content": "Tell me ishaan is a genius"}],
    )
    print(response)
