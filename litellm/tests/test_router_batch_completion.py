#### What this tests ####
# This tests litellm router with batch completion

import sys, os, time, openai
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params, ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
import os, httpx

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
        for individual_response in response:
            _model = individual_response["model"]
            models_in_responses.append(_model)

        # assert both models are different
        assert models_in_responses[0] != models_in_responses[1]
    elif mode == "fastest_response":
        from openai.types.chat.chat_completion import ChatCompletion

        response = await router.abatch_completion_fastest_response(
            models=["gpt-3.5-turbo", "groq-llama"],
            messages=[
                {"role": "user", "content": "is litellm becoming a better product ?"}
            ],
            max_tokens=15,
        )

        ChatCompletion.model_validate(response.model_dump(), strict=True)
