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
                "model_name": "custom-text-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-3-large",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "custom-text-embedding-model-2",
                "litellm_params": {
                    "model": "text-embedding-3-large",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
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
                    "auto_router_config_path": "auto_router/router.json",
                    "auto_router_default_model": "gpt-4o-mini",
                    "auto_router_embedding_model": "custom-text-embedding-model",
                },
            },
            {
                "model_name": "auto_router_2",
                "litellm_params": {
                    "model": "auto_router/auto_router_2",
                    "auto_router_config_path": "auto_router/router_2.json",
                    "auto_router_default_model": "gpt-4o-mini",
                    "auto_router_embedding_model": "custom-text-embedding-model-2",
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
