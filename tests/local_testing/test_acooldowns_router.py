#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import asyncio
import os
import time
import pytest

import concurrent

from dotenv import load_dotenv

import litellm

from litellm import Router

load_dotenv()


# test_multiple_deployments_sync()


# Assuming litellm, router, and executor are defined somewhere in your code


# test_multiple_deployments_parallel()
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cooldown_same_model_name(sync_mode):
    # users could have the same model with different api_base
    # example
    # azure/chatgpt, api_base: 1234
    # azure/chatgpt, api_base: 1235
    # if 1234 fails, it should only cooldown 1234 and then try with 1235
    litellm.set_verbose = False
    try:
        print("testing cooldown same model name")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": "bad-key",
                    "tpm": 90,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "tpm": 1,
                },
            },
        ]

        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="simple-shuffle",
            set_verbose=True,
            num_retries=3,
            allowed_fails=0,
        )  # type: ignore

        if sync_mode:
            response = router.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hello this request will pass"}],
            )
            print(router.model_list)
            model_ids = []
            for model in router.model_list:
                model_ids.append(model["model_info"]["id"])
            print("\n litellm model ids ", model_ids)

            # example litellm_model_names ['azure/gpt-4.1-mini-ModelID-64321', 'azure/gpt-4.1-mini-ModelID-63960']
            assert (
                model_ids[0] != model_ids[1]
            )  # ensure both models have a uuid added, and they have different names

            print("\ngot response\n", response)
        else:
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hello this request will pass"}],
            )
            print(router.model_list)
            model_ids = []
            for model in router.model_list:
                model_ids.append(model["model_info"]["id"])
            print("\n litellm model ids ", model_ids)

            # example litellm_model_names ['azure/gpt-4.1-mini-ModelID-64321', 'azure/gpt-4.1-mini-ModelID-63960']
            assert (
                model_ids[0] != model_ids[1]
            )  # ensure both models have a uuid added, and they have different names

            print("\ngot response\n", response)
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_cooldown_same_model_name()
