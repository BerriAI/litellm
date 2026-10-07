#### What this tests ####
# This tests if the router timeout error handling during fallbacks

import os

import pytest
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
                "model": "azure/gpt-4.1-mini",
                "api_key": "os.environ/AZURE_AI_API_KEY",
                "api_base": "os.environ/AZURE_AI_API_BASE",
                "api_version": "os.environ/AZURE_API_VERSION",
            },
            "tpm": 80000,
        },
        {
            "model_name": "anthropic-claude-haiku-4-5",
            "litellm_params": {
                "model": "claude-haiku-4-5",
                "api_key": "os.environ/ANTHROPIC_API_KEY",
                "mock_response": "hello world",
            },
            "tpm": 20000,
        },
    ]

    fallbacks_list = [
        {"openai-gpt-4": ["anthropic-claude-haiku-4-5"]},
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
    import openai

    from litellm._uuid import uuid

    # Model list for OpenAI and Anthropic models
    _model_list = [
        {
            "model_name": "bedrock",
            "litellm_params": {
                "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
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
    "model",
    [
        "llama3",
        "bedrock-anthropic",
    ],
)
def test_router_stream_timeout(model):
    import os

    import litellm
    from litellm.router import AllowedFailsPolicy, RetryPolicy, Router

    litellm.set_verbose = True

    model_list = [
        {
            "model_name": "llama3",
            "litellm_params": {
                "model": "watsonx/meta-llama/llama-3-1-8b-instruct",
                "api_base": os.getenv("WATSONX_URL_US_SOUTH"),
                "api_key": os.getenv("WATSONX_API_KEY"),
                "project_id": os.getenv("WATSONX_PROJECT_ID_US_SOUTH"),
                "timeout": 0.01,
                "stream_timeout": 0.0000001,
            },
        },
        {
            "model_name": "bedrock-anthropic",
            "litellm_params": {
                "model": "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
                "timeout": 0.01,
                "stream_timeout": 0.0000001,
            },
        },
        {
            "model_name": "llama3-fallback",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ]

    # Initialize router with retry and timeout settings
    router = Router(
        model_list=model_list,
        fallbacks=[
            {"llama3": ["llama3-fallback"]},
            {"bedrock-anthropic": ["llama3-fallback"]},
        ],
        routing_strategy="latency-based-routing",  # 👈 set routing strategy
        retry_policy=RetryPolicy(
            TimeoutErrorRetries=1,  # Number of retries for timeout errors
            RateLimitErrorRetries=3,
            BadRequestErrorRetries=2,
        ),
        allowed_fails_policy=AllowedFailsPolicy(
            TimeoutErrorAllowedFails=2,  # Number of timeouts allowed before cooldown
            RateLimitErrorAllowedFails=2,
        ),
        cooldown_time=120,  # Cooldown time in seconds,
        set_verbose=True,
        routing_strategy_args={"lowest_latency_buffer": 0.5},
    )

    print("this fall back does NOT work:")
    response = router.completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "write a 100 word story about a cat"},
        ],
        temperature=0.6,
        max_tokens=500,
        stream=True,
    )

    t = 0
    for chunk in response:
        assert "llama" not in chunk.model
        chunk_text = chunk.choices[0].delta.content or ""
        print(chunk_text)
        t += 1
        if t > 10:
            break
