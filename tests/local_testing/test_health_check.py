#### What this tests ####
#    This tests if ahealth_check() actually works

import os
import sys
import traceback

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
            "model": "azure/chatgpt-v-2",
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
            "model": "azure/azure-embedding-model",
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


async def test_azure_img_gen_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "azure/",
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": "2023-06-01-preview",
        },
        mode="image_generation",
        prompt="cute baby sea otter",
    )

    assert isinstance(response, dict) and "error" not in response
    return response


# asyncio.run(test_azure_img_gen_health_check())


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
async def test_audio_transcription_health_check():
    litellm.set_verbose = True
    response = await litellm.ahealth_check(
        model_params={
            "model": "openai/whisper-1",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        mode="audio_transcription",
    )

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
