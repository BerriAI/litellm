#### What this tests ####
#    This tests if ahealth_check() actually works

import os
import sys

import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio

import litellm


@pytest.mark.asyncio
async def test_azure_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "azure/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hey, how's it going?"}],
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        }
    )
    print(f"response: {response}")

    assert "x-ratelimit-remaining-tokens" in response
    return response


# asyncio.run(test_azure_health_check())


@pytest.mark.asyncio
async def test_text_completion_health_check():
    response = await litellm.ahealth_check(
        model_params={"model": "gpt-3.5-turbo-instruct"},
        mode="completion",
        prompt="What's the weather in SF?",
    )
    print(f"response: {response}")
    return response


@pytest.mark.asyncio
async def test_azure_embedding_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "azure/text-embedding-ada-002",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        },
        input=["test for litellm"],
        mode="embedding",
    )
    print(f"response: {response}")

    assert "x-ratelimit-remaining-tokens" in response
    return response


@pytest.mark.asyncio
async def test_openai_img_gen_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "dall-e-3",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        mode="image_generation",
        prompt="cute baby sea otter",
    )
    print(f"response: {response}")

    assert isinstance(response, dict) and "error" not in response
    return response


# asyncio.run(test_openai_img_gen_health_check())


@pytest.mark.asyncio
async def test_azure_img_gen_health_check():
    """
    Test Azure image generation health check with retry logic for transient errors.
    Azure sometimes returns internal server errors which are transient and not something we can control.
    """
    litellm._turn_on_debug()
    max_retries = 3
    retry_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        response = await litellm.ahealth_check(
            model_params={
                "model": "azure/dall-e-3",
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_key": os.getenv("AZURE_API_KEY"),
            },
            mode="image_generation",
            prompt="cute baby sea otter",
        )
        
        # Check if response is successful (no error)
        if isinstance(response, dict) and "error" not in response:
            return response
        
        # Check if error is a transient Azure internal server error
        error_str = str(response.get("error", "")).lower()
        is_transient_error = (
            "internalservererror" in error_str
            or "internal server error" in error_str
            or "internalfailure" in error_str
            or "internal failure" in error_str
        )
        
        # If it's the last attempt or not a transient error, fail the test
        if attempt == max_retries - 1 or not is_transient_error:
            assert isinstance(response, dict) and "error" not in response, f"Health check failed: {response.get('error', 'Unknown error')}"
            return response
        
        # Wait before retrying with exponential backoff
        await asyncio.sleep(retry_delay)
        retry_delay *= 2  # Exponential backoff
    
    # Should not reach here, but just in case
    assert False, "Health check failed after all retries"


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_sagemaker_embedding_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "sagemaker/berri-benchmarking-gpt-j-6b-fp16",
            "messages": [{"role": "user", "content": "Hey, how's it going?"}],
        },
        mode="embedding",
        input=["test from litellm"],
    )
    print(f"response: {response}")

    assert isinstance(response, dict)
    return response


# asyncio.run(test_sagemaker_embedding_health_check())


@pytest.mark.asyncio
async def test_groq_health_check():
    """
    This should not fail

    ensure that provider wildcard model passes health check
    """
    litellm.set_verbose = True
    response = await litellm.ahealth_check(
        model_params={
            "api_key": os.environ.get("GROQ_API_KEY"),
            "model": "groq/*",
            "messages": [{"role": "user", "content": "What's 1 + 1?"}],
        },
        mode=None,
        prompt="What's 1 + 1?",
        input=["test from litellm"],
    )
    print(f"response: {response}")
    assert response == {}

    return response


@pytest.mark.asyncio
async def test_cohere_rerank_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "cohere/rerank-english-v3.0",
            "api_key": os.getenv("COHERE_API_KEY"),
        },
        mode="rerank",
        prompt="Hey, how's it going",
    )

    assert "error" not in response

    print(response)


@pytest.mark.asyncio
async def test_audio_speech_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "openai/tts-1",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        mode="audio_speech",
        prompt="Hey",
    )

    assert "error" not in response

    print(response)


@pytest.mark.asyncio
async def test_audio_speech_health_check_with_another_voice():
    response = await litellm.ahealth_check(
        model_params={
            "model": "openai/tts-1",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "health_check_voice": "en-US-JennyNeural",
        },
        mode="audio_speech",
        prompt="Hey",
    )

    assert "error" not in response

    print(response)


@pytest.mark.asyncio
async def test_audio_transcription_health_check():
    litellm.set_verbose = True
    response = await litellm.ahealth_check(
        model_params={
            "model": "openai/whisper-1",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        mode="audio_transcription",
    )

    print(f"response: {response}")

    assert "error" not in response

    print(response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["azure/gpt-4o-realtime-preview", "openai/gpt-4o-realtime-preview"]
)
async def test_async_realtime_health_check(model, mocker):
    """
    Test Health Check with Valid models passes

    """
    mock_websocket = AsyncMock()
    mock_connect = AsyncMock().__aenter__.return_value = mock_websocket
    mocker.patch("websockets.connect", return_value=mock_connect)

    litellm.set_verbose = True
    model_params = {
        "model": model,
    }
    if model == "azure/gpt-4o-realtime-preview":
        model_params["api_base"] = os.getenv("AZURE_REALTIME_API_BASE")
        model_params["api_key"] = os.getenv("AZURE_REALTIME_API_KEY")
        model_params["api_version"] = os.getenv("AZURE_REALTIME_API_VERSION")
    response = await litellm.ahealth_check(
        model_params=model_params,
        mode="realtime",
    )
    print(response)
    assert response == {}


def test_update_litellm_params_for_health_check():
    """
    Test if _update_litellm_params_for_health_check correctly:
    1. Updates messages with a random message
    2. Updates model name when health_check_model is provided
    3. Updates voice when health_check_voice is provided for audio_speech mode
    """
    from litellm.proxy.health_check import _update_litellm_params_for_health_check

    # Test with health_check_model
    model_info = {"health_check_model": "gpt-3.5-turbo"}
    litellm_params = {
        "model": "gpt-4",
        "api_key": "fake_key",
    }

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert "messages" in updated_params
    assert isinstance(updated_params["messages"], list)
    assert updated_params["model"] == "gpt-3.5-turbo"

    # Test without health_check_model
    model_info = {}
    litellm_params = {
        "model": "gpt-4",
        "api_key": "fake_key",
    }

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert "messages" in updated_params
    assert isinstance(updated_params["messages"], list)
    assert updated_params["model"] == "gpt-4"

    # Test with health_check_voice for audio_speech mode
    model_info = {"mode": "audio_speech", "health_check_voice": "en-US-JennyNeural"}
    litellm_params = {
        "model": "gpt-4",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert "voice" in updated_params
    assert updated_params["voice"] == "en-US-JennyNeural"

    # Test without health_check_voice for audio_speech mode
    model_info = {"mode": "audio_speech"}
    litellm_params = {
        "model": "gpt-4",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert "voice" in updated_params
    assert updated_params["voice"] == "alloy"

    # Test with health_check_voice for non-audio_speech mode
    model_info = {"mode": "chat", "health_check_voice": "en-US-JennyNeural"}
    litellm_params = {
        "model": "gpt-4",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert "voice" not in updated_params

    # Test with Bedrock model with region routing - should strip bedrock/ and region/ prefix
    # Issue #15807: Fixes health checks sending "region/model" as model ID to AWS
    model_info = {}
    litellm_params = {
        "model": "bedrock/us-gov-west-1/anthropic.claude-3-7-sonnet-20250219-v1:0",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "anthropic.claude-3-7-sonnet-20250219-v1:0"

    # Test with Bedrock cross-region inference profile - should preserve the inference profile prefix
    # AWS requires inference profile IDs like "us.anthropic.claude..." for cross-region routing
    litellm_params = {
        "model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "us.anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Test with Bedrock model without region routing - should just strip bedrock/ prefix
    litellm_params = {
        "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Test that non-Bedrock models are not affected by Bedrock-specific logic
    litellm_params = {
        "model": "openai/gpt-4",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "openai/gpt-4"  # Should remain unchanged

    # Test ALL cross-region inference profile prefixes (CRIS)
    cris_prefixes = ["us.", "eu.", "apac.", "jp.", "au.", "us-gov.", "global."]
    for prefix in cris_prefixes:
        litellm_params = {
            "model": f"bedrock/{prefix}anthropic.claude-3-haiku-20240307-v1:0",
            "api_key": "fake_key",
        }
        updated_params = _update_litellm_params_for_health_check(
            model_info, litellm_params
        )
        assert (
            updated_params["model"] == f"{prefix}anthropic.claude-3-haiku-20240307-v1:0"
        ), f"Failed to preserve CRIS prefix: {prefix}"

    # Test regional + CRIS combination - region should be stripped, CRIS preserved
    litellm_params = {
        "model": "bedrock/us-east-2/us.anthropic.claude-3-haiku-20240307-v1:0",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "us.anthropic.claude-3-haiku-20240307-v1:0"

    # Test GovCloud regions
    litellm_params = {
        "model": "bedrock/us-gov-east-1/anthropic.claude-instant-v1",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "anthropic.claude-instant-v1"

    # Test imported models with handler prefixes - handlers should be preserved
    litellm_params = {
        "model": "bedrock/llama/arn:aws:bedrock:us-east-1:123:imported-model/abc",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert (
        updated_params["model"]
        == "llama/arn:aws:bedrock:us-east-1:123:imported-model/abc"
    )

    litellm_params = {
        "model": "bedrock/deepseek_r1/arn:aws:bedrock:us-west-2:456:imported-model/xyz",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert (
        updated_params["model"]
        == "deepseek_r1/arn:aws:bedrock:us-west-2:456:imported-model/xyz"
    )

    # Test route specifications - routes should be preserved
    litellm_params = {
        "model": "bedrock/converse/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert (
        updated_params["model"]
        == "converse/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    litellm_params = {
        "model": "bedrock/invoke/us-west-2/anthropic.claude-instant-v1",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert updated_params["model"] == "invoke/anthropic.claude-instant-v1"

    # Test ARN formats - should be preserved
    litellm_params = {
        "model": "bedrock/arn:aws:bedrock:eu-central-1:000:application-inference-profile/abc",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert (
        updated_params["model"]
        == "arn:aws:bedrock:eu-central-1:000:application-inference-profile/abc"
    )

    # Test edge case: region + handler + ARN
    litellm_params = {
        "model": "bedrock/us-west-2/llama/arn:aws:bedrock:us-east-1:123:imported-model/abc",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert (
        updated_params["model"]
        == "llama/arn:aws:bedrock:us-east-1:123:imported-model/abc"
    )

    # Test edge case: route + region + CRIS
    litellm_params = {
        "model": "bedrock/converse/us-west-2/eu.anthropic.claude-3-sonnet-20240229-v1:0",
        "api_key": "fake_key",
    }
    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    assert (
        updated_params["model"] == "converse/eu.anthropic.claude-3-sonnet-20240229-v1:0"
    )


@pytest.mark.asyncio
async def test_perform_health_check_with_health_check_model():
    """
    Test if _perform_health_check correctly uses `health_check_model` when model=`openai/*`:
    1. Verifies that health_check_model overrides the original model when model=`openai/*`
    2. Ensures the health check is performed with the override model
    """
    from litellm.proxy.health_check import _perform_health_check

    # Mock model list with health_check_model specified
    model_list = [
        {
            "litellm_params": {"model": "openai/*", "api_key": "fake-key"},
            "model_info": {
                "mode": "chat",
                "health_check_model": "openai/gpt-4o-mini",  # Override model for health check
            },
        }
    ]

    # Track which model is actually used in the health check
    health_check_calls = []

    async def mock_health_check(litellm_params, **kwargs):
        health_check_calls.append(litellm_params["model"])
        return {"status": "healthy"}

    with patch("litellm.ahealth_check", side_effect=mock_health_check):
        healthy_endpoints, unhealthy_endpoints = await _perform_health_check(model_list)
        print("health check calls: ", health_check_calls)

        # Verify the health check used the override model
        assert health_check_calls[0] == "openai/gpt-4o-mini"
        # Verify the result still shows the original model
        print("healthy endpoints: ", healthy_endpoints)
        assert healthy_endpoints[0]["model"] == "openai/gpt-4o-mini"
        assert len(healthy_endpoints) == 1
        assert len(unhealthy_endpoints) == 0


@pytest.mark.asyncio
async def test_health_check_bad_model():
    from litellm.proxy.health_check import _perform_health_check
    import time

    model_list = [
        {
            "model_name": "openai-gpt-4o",
            "litellm_params": {
                "api_key": "sk-1234",
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app",
                "model": "openai/my-fake-openai-endpoint",
                "mock_timeout": True,
                "timeout": 60,
            },
            "model_info": {
                "id": "ca27ca2eeea2f9e38bb274ead831948a26621a3738d06f1797253f0e6c4278c0",
                "db_model": False,
                "health_check_timeout": 1,
            },
        },
    ]
    details = None
    healthy_endpoints, unhealthy_endpoints = await _perform_health_check(
        model_list, details
    )
    print(f"healthy_endpoints: {healthy_endpoints}")
    print(f"unhealthy_endpoints: {unhealthy_endpoints}")

    # Track which model is actually used in the health check
    health_check_calls = []

    async def mock_health_check(litellm_params, **kwargs):
        health_check_calls.append(litellm_params["model"])
        await asyncio.sleep(10)
        return {"status": "healthy"}

    with patch(
        "litellm.ahealth_check", side_effect=mock_health_check
    ) as mock_health_check:
        start_time = time.time()
        healthy_endpoints, unhealthy_endpoints = await _perform_health_check(model_list)
        end_time = time.time()
        print("health check calls: ", health_check_calls)
        assert len(healthy_endpoints) == 0
        assert len(unhealthy_endpoints) == 1
        assert (
            end_time - start_time < 2
        ), "Health check took longer than health_check_timeout"


@pytest.mark.asyncio
async def test_ahealth_check_ocr():
    litellm._turn_on_debug()
    response = await litellm.ahealth_check(
        model_params={
            "model": "mistral/mistral-ocr-latest",
            "api_key": os.getenv("MISTRAL_API_KEY"),
        },
        mode="ocr",
    )
    print(response)
    return response


@pytest.mark.asyncio
async def test_image_generation_health_check_prompt(monkeypatch):
    """Health checks should respect default and environment-configured prompts."""

    import importlib
    import litellm.constants as litellm_constants
    import litellm.proxy.health_check as health_check

    def reload_modules():
        reloaded_constants = importlib.reload(litellm_constants)
        reloaded_health_check = importlib.reload(health_check)
        return reloaded_constants, reloaded_health_check

    async def run_health_check(health_check_module):
        health_check_calls = []

        async def mock_health_check(litellm_params, mode=None, prompt=None, input=None):
            health_check_calls.append(
                {
                    "mode": mode,
                    "prompt": prompt,
                    "model": litellm_params.get("model"),
                }
            )
            return {"status": "healthy"}

        model_list = [
            {
                "litellm_params": {"model": "dall-e-3", "api_key": "fake-key"},
                "model_info": {
                    "mode": "image_generation",
                },
            }
        ]

        with patch(
            "litellm.proxy.health_check.litellm.ahealth_check",
            side_effect=mock_health_check,
        ):
            await health_check_module._perform_health_check(model_list)

        return health_check_calls

    # Default prompt is used when env var is unset
    monkeypatch.delenv("DEFAULT_HEALTH_CHECK_PROMPT", raising=False)
    litellm_constants, health_check = reload_modules()
    health_check_calls = await run_health_check(health_check)

    assert len(health_check_calls) == 1
    assert (
        health_check_calls[0]["prompt"] == litellm_constants.DEFAULT_HEALTH_CHECK_PROMPT
    )

    # Environment override should change the prompt without code changes
    override_prompt = "environment override prompt"
    monkeypatch.setenv("DEFAULT_HEALTH_CHECK_PROMPT", override_prompt)
    litellm_constants, health_check = reload_modules()
    health_check_calls = await run_health_check(health_check)

    assert len(health_check_calls) == 1
    assert health_check_calls[0]["prompt"] == override_prompt
