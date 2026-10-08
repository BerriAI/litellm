#### What this tests ####
#    This tests if ahealth_check() actually works

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

import litellm


@pytest.mark.asyncio
async def test_azure_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "azure/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hey, how's it going?"}],
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_version": os.getenv("AZURE_AI_API_VERSION"),
        }
    )
    print(f"response: {response}")

    assert "x-ratelimit-remaining-tokens" in response
    return response


# asyncio.run(test_azure_health_check())




@pytest.mark.asyncio
async def test_azure_embedding_health_check():
    response = await litellm.ahealth_check(
        model_params={
            "model": "azure/text-embedding-ada-002",
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_version": os.getenv("AZURE_AI_API_VERSION"),
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
            "model": "gpt-image-1",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        mode="image_generation",
        prompt="cute baby sea otter",
    )
    print(f"response: {response}")

    assert isinstance(response, dict) and "error" not in response
    return response


# asyncio.run(test_openai_img_gen_health_check())






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
async def test_health_check_bad_model():
    import time

    from litellm.proxy.health_check import _perform_health_check

    model_list = [
        {
            "model_name": "openai-gpt-4o",
            "litellm_params": {
                "api_key": "sk-9876",
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
    healthy_endpoints, unhealthy_endpoints, _ = await _perform_health_check(
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
        healthy_endpoints, unhealthy_endpoints, _ = await _perform_health_check(
            model_list
        )
        end_time = time.time()
        print("health check calls: ", health_check_calls)
        assert len(healthy_endpoints) == 0
        assert len(unhealthy_endpoints) == 1
        assert (
            end_time - start_time < 2
        ), "Health check took longer than health_check_timeout"


@pytest.mark.asyncio
async def test_health_check_respects_concurrency_limit():
    from litellm.proxy.health_check import _perform_health_check

    model_list = [
        {"litellm_params": {"model": f"openai/gpt-4o-mini-{i}", "api_key": "fake-key"}}
        for i in range(6)
    ]

    active = 0
    max_active = 0

    async def mock_health_check(litellm_params, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return {"status": "healthy"}

    with patch("litellm.ahealth_check", side_effect=mock_health_check):
        await _perform_health_check(model_list, max_concurrency=2)

    assert max_active <= 2


@pytest.mark.asyncio
async def test_health_check_creates_only_bounded_initial_tasks():
    from litellm.proxy.health_check import _perform_health_check

    model_list = [
        {"litellm_params": {"model": f"openai/gpt-4o-mini-{i}", "api_key": "fake-key"}}
        for i in range(10)
    ]
    release_event = asyncio.Event()
    create_task_call_count = 0
    real_create_task = asyncio.create_task

    async def mock_health_check(litellm_params, **kwargs):
        await release_event.wait()
        return {"status": "healthy"}

    def tracked_create_task(coro):
        nonlocal create_task_call_count
        create_task_call_count += 1
        return real_create_task(coro)

    with (
        patch("litellm.ahealth_check", side_effect=mock_health_check),
        patch(
            "litellm.proxy.health_check.asyncio.create_task",
            side_effect=tracked_create_task,
        ),
    ):
        perform_task = real_create_task(
            _perform_health_check(model_list, max_concurrency=2)
        )
        await asyncio.sleep(0.05)
        assert create_task_call_count == 2
        release_event.set()
        await perform_task


@pytest.mark.asyncio
async def test_timeout_does_not_cancel_other_health_checks():
    from litellm.proxy.health_check import _perform_health_check

    model_list = [
        {
            "litellm_params": {"model": "openai/slow-model", "api_key": "fake-key"},
            "model_info": {"health_check_timeout": 0.05},
        },
        {
            "litellm_params": {"model": "openai/fast-model", "api_key": "fake-key"},
            "model_info": {"health_check_timeout": 1},
        },
    ]

    async def mock_health_check(litellm_params, **kwargs):
        if litellm_params["model"] == "openai/slow-model":
            await asyncio.sleep(0.2)
            return {"status": "healthy"}
        await asyncio.sleep(0.01)
        return {"status": "healthy"}

    with patch("litellm.ahealth_check", side_effect=mock_health_check):
        healthy_endpoints, unhealthy_endpoints, _ = await _perform_health_check(
            model_list, max_concurrency=1
        )

    healthy_models = {endpoint["model"] for endpoint in healthy_endpoints}
    unhealthy_models = {endpoint["model"] for endpoint in unhealthy_endpoints}

    assert "openai/fast-model" in healthy_models
    assert "openai/slow-model" in unhealthy_models
