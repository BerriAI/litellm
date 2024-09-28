#### What this tests ####
# This tests litellm router with batch completion

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

import httpx
from dotenv import load_dotenv

import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params, ModelInfo

load_dotenv()


@pytest.mark.parametrize("mode", ["all_responses", "fastest_response"])
@pytest.mark.asyncio
async def test_batch_completion_multiple_models(mode):
    litellm.set_verbose = True

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "groq-llama",
                "litellm_params": {
                    "model": "groq/llama3-8b-8192",
                },
            },
        ]
    )

    if mode == "all_responses":
        response = await router.abatch_completion(
            models=["gpt-3.5-turbo", "groq-llama"],
            messages=[
                {"role": "user", "content": "is litellm becoming a better product ?"}
            ],
            max_tokens=15,
        )

        print(response)
        assert len(response) == 2

        models_in_responses = []
        print(f"response: {response}")
        for individual_response in response:
            _model = individual_response["model"]
            models_in_responses.append(_model)

        # assert both models are different
        assert models_in_responses[0] != models_in_responses[1]
    elif mode == "fastest_response":
        from openai.types.chat.chat_completion import ChatCompletion

        response = await router.abatch_completion_fastest_response(
            model="gpt-3.5-turbo, groq-llama",
            messages=[
                {"role": "user", "content": "is litellm becoming a better product ?"}
            ],
            max_tokens=15,
        )

        ChatCompletion.model_validate(response.model_dump(), strict=True)


@pytest.mark.asyncio
async def test_batch_completion_fastest_response_unit_test():
    """
    Unit test to confirm fastest response will always return the response which arrives earliest.

    2 models -> 1 is cached, the other is a real llm api call => assert cached response always returned
    """
    litellm.set_verbose = True

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "This is a fake response",
                },
                "model_info": {"id": "2"},
            },
        ]
    )

    response = await router.abatch_completion_fastest_response(
        model="gpt-4, gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "is litellm becoming a better product ?"}
        ],
        max_tokens=500,
    )

    assert response._hidden_params["model_id"] == "2"
    assert response.choices[0].message.content == "This is a fake response"
    print(f"response: {response}")


@pytest.mark.asyncio
async def test_batch_completion_fastest_response_streaming():
    litellm.set_verbose = True

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "groq-llama",
                "litellm_params": {
                    "model": "groq/llama3-8b-8192",
                },
            },
        ]
    )

    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    response = await router.abatch_completion_fastest_response(
        model="gpt-3.5-turbo, groq-llama",
        messages=[
            {"role": "user", "content": "is litellm becoming a better product ?"}
        ],
        max_tokens=15,
        stream=True,
    )

    async for chunk in response:
        ChatCompletionChunk.model_validate(chunk.model_dump(), strict=True)


@pytest.mark.asyncio
async def test_batch_completion_multiple_models_multiple_messages():
    litellm.set_verbose = True

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "groq-llama",
                "litellm_params": {
                    "model": "groq/llama3-8b-8192",
                },
            },
        ]
    )

    response = await router.abatch_completion(
        models=["gpt-3.5-turbo", "groq-llama"],
        messages=[
            [{"role": "user", "content": "is litellm becoming a better product ?"}],
            [{"role": "user", "content": "who is this"}],
        ],
        max_tokens=15,
    )

    print("response from batches =", response)
    assert len(response) == 2
    assert len(response[0]) == 2
    assert isinstance(response[0][0], litellm.ModelResponse)

    # models_in_responses = []
    # for individual_response in response:
    #     _model = individual_response["model"]
    #     models_in_responses.append(_model)

    # # assert both models are different
    # assert models_in_responses[0] != models_in_responses[1]
