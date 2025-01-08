#### What this tests ####
# This tests if the router timeout error handling during fallbacks

import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import patch, MagicMock, AsyncMock
import os

from dotenv import load_dotenv

import litellm
from litellm import Router

load_dotenv()


def test_router_timeouts():
    # Model list for OpenAI and Anthropic models
    model_list = [
        {
            "model_name": "openai-gpt-4",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": "os.environ/AZURE_API_KEY",
                "api_base": "os.environ/AZURE_API_BASE",
                "api_version": "os.environ/AZURE_API_VERSION",
            },
            "tpm": 80000,
        },
        {
            "model_name": "anthropic-claude-3-5-haiku-20241022",
            "litellm_params": {
                "model": "claude-3-5-haiku-20241022",
                "api_key": "os.environ/ANTHROPIC_API_KEY",
                "mock_response": "hello world",
            },
            "tpm": 20000,
        },
    ]

    fallbacks_list = [
        {"openai-gpt-4": ["anthropic-claude-3-5-haiku-20241022"]},
    ]

    # Configure router
    router = Router(
        model_list=model_list,
        fallbacks=fallbacks_list,
        routing_strategy="usage-based-routing",
        debug_level="INFO",
        set_verbose=True,
        redis_host=os.getenv("REDIS_HOST"),
        redis_password=os.getenv("REDIS_PASSWORD"),
        redis_port=int(os.getenv("REDIS_PORT")),
        timeout=10,
        num_retries=0,
    )

    print("***** TPM SETTINGS *****")
    for model_object in model_list:
        print(f"{model_object['model_name']}: {model_object['tpm']} TPM")

    # Sample list of questions
    questions_list = [
        {"content": "Tell me a very long joke.", "modality": "voice"},
    ]

    total_tokens_used = 0

    # Process each question
    for question in questions_list:
        messages = [{"content": question["content"], "role": "user"}]

        prompt_tokens = litellm.token_counter(text=question["content"], model="gpt-4")
        print("prompt_tokens = ", prompt_tokens)

        response = router.completion(
            model="openai-gpt-4", messages=messages, timeout=5, num_retries=0
        )

        total_tokens_used += response.usage.total_tokens

        print("Response:", response)
        print("********** TOKENS USED SO FAR = ", total_tokens_used)


@pytest.mark.asyncio
async def test_router_timeouts_bedrock():
    import uuid

    import openai

    # Model list for OpenAI and Anthropic models
    _model_list = [
        {
            "model_name": "bedrock",
            "litellm_params": {
                "model": "bedrock/anthropic.claude-instant-v1",
                "timeout": 0.00001,
            },
            "tpm": 80000,
        },
    ]

    # Configure router
    router = Router(
        model_list=_model_list,
        routing_strategy="usage-based-routing",
        debug_level="DEBUG",
        set_verbose=True,
        num_retries=0,
    )

    litellm.set_verbose = True
    try:
        response = await router.acompletion(
            model="bedrock",
            messages=[{"role": "user", "content": f"hello, who are u {uuid.uuid4()}"}],
        )
        print(response)
        pytest.fail("Did not raise error `openai.APITimeoutError`")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


@pytest.mark.parametrize(
    "num_retries, expected_call_count",
    [(0, 1), (1, 2), (2, 3), (3, 4)],
)
def test_router_timeout_with_retries_anthropic_model(num_retries, expected_call_count):
    """
    If request hits custom timeout, ensure it's retried.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    import time

    litellm.num_retries = num_retries
    litellm.request_timeout = 0.000001

    router = Router(
        model_list=[
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "anthropic/claude-3-haiku-20240307",
                },
            }
        ],
    )

    custom_client = HTTPHandler()

    with patch.object(custom_client, "post", new=MagicMock()) as mock_client:
        try:

            def delayed_response(*args, **kwargs):
                time.sleep(0.01)  # Exceeds the 0.000001 timeout
                raise TimeoutError("Request timed out.")

            mock_client.side_effect = delayed_response

            router.completion(
                model="claude-3-haiku",
                messages=[{"role": "user", "content": "hello, who are u"}],
                client=custom_client,
            )
        except litellm.Timeout:
            pass

        assert mock_client.call_count == expected_call_count
