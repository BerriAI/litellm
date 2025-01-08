import os
import sys

import pytest
from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

import litellm
from litellm.proxy._types import SpendCalculateRequest
from litellm.proxy.spend_tracking.spend_management_endpoints import calculate_spend
from litellm.router import Router

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


@pytest.mark.asyncio
async def test_spend_calc_model_messages():
    cost_obj = await calculate_spend(
        request=SpendCalculateRequest(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )
    )

    print("calculated cost", cost_obj)
    cost = cost_obj["cost"]
    assert cost > 0.0


@pytest.mark.asyncio
async def test_spend_calc_model_on_router_messages():
    from litellm.proxy.proxy_server import llm_router as init_llm_router

    temp_llm_router = Router(
        model_list=[
            {
                "model_name": "special-llama-model",
                "litellm_params": {
                    "model": "groq/llama3-8b-8192",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", temp_llm_router)

    cost_obj = await calculate_spend(
        request=SpendCalculateRequest(
            model="special-llama-model",
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )
    )

    print("calculated cost", cost_obj)
    _cost = cost_obj["cost"]

    assert _cost > 0.0

    # set router to init value
    setattr(litellm.proxy.proxy_server, "llm_router", init_llm_router)


@pytest.mark.asyncio
async def test_spend_calc_using_response():
    cost_obj = await calculate_spend(
        request=SpendCalculateRequest(
            completion_response={
                "id": "chatcmpl-3bc7abcd-f70b-48ab-a16c-dfba0b286c86",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": "Yooo! What's good?",
                            "role": "assistant",
                        },
                    }
                ],
                "created": "1677652288",
                "model": "groq/llama3-8b-8192",
                "object": "chat.completion",
                "system_fingerprint": "fp_873a560973",
                "usage": {
                    "completion_tokens": 8,
                    "prompt_tokens": 12,
                    "total_tokens": 20,
                },
            }
        )
    )

    print("calculated cost", cost_obj)
    cost = cost_obj["cost"]
    assert cost > 0.0


@pytest.mark.asyncio
async def test_spend_calc_model_alias_on_router_messages():
    from litellm.proxy.proxy_server import llm_router as init_llm_router

    temp_llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "gpt-4o",
                },
            }
        ],
        model_group_alias={
            "gpt4o": "gpt-4o",
        },
    )

    setattr(litellm.proxy.proxy_server, "llm_router", temp_llm_router)

    cost_obj = await calculate_spend(
        request=SpendCalculateRequest(
            model="gpt4o",
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )
    )

    print("calculated cost", cost_obj)
    _cost = cost_obj["cost"]

    assert _cost > 0.0

    # set router to init value
    setattr(litellm.proxy.proxy_server, "llm_router", init_llm_router)
