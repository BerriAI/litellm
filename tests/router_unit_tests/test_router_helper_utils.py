import sys
import os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "dall-e-3",
            "litellm_params": {
                "model": "dall-e-3",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ]


def test_validate_fallbacks(model_list):
    router = Router(model_list=model_list, fallbacks=[{"gpt-4o": "gpt-3.5-turbo"}])
    router.validate_fallbacks(fallback_param=[{"gpt-4o": "gpt-3.5-turbo"}])


def test_routing_strategy_init(model_list):
    """Test if all routing strategies are initialized correctly"""
    from litellm.types.router import RoutingStrategy

    router = Router(model_list=model_list)
    for strategy in RoutingStrategy._member_names_:
        router.routing_strategy_init(
            routing_strategy=strategy, routing_strategy_args={}
        )


def test_print_deployment(model_list):
    """Test if the api key is masked correctly"""

    router = Router(model_list=model_list)
    deployment = {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    }
    printed_deployment = router.print_deployment(deployment)
    assert 10 * "*" in printed_deployment["litellm_params"]["api_key"]


def test_completion(model_list):
    """Test if the completion function is working correctly"""
    router = Router(model_list=model_list)
    response = router._completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=6, delay=1)
@pytest.mark.asyncio
async def test_image_generation(model_list, sync_mode):
    """Test if the underlying '_image_generation' function is working correctly"""
    from litellm.types.utils import ImageResponse

    router = Router(model_list=model_list)
    if sync_mode:
        response = router._image_generation(
            model="dall-e-3",
            prompt="A cute baby sea otter",
        )
    else:
        response = await router._aimage_generation(
            model="dall-e-3",
            prompt="A cute baby sea otter",
        )

    ImageResponse.model_validate(response)


@pytest.mark.asyncio
async def test_router_acompletion_util(model_list):
    """Test if the underlying '_acompletion' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router._acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_abatch_completion_one_model_multiple_requests_util(model_list):
    """Test if the 'abatch_completion_one_model_multiple_requests' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router.abatch_completion_one_model_multiple_requests(
        model="gpt-3.5-turbo",
        messages=[
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "Hello, how are you?"}],
        ],
        mock_response="I'm fine, thank you!",
    )
    print(response)
    assert response[0]["choices"][0]["message"]["content"] == "I'm fine, thank you!"
    assert response[1]["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_schedule_acompletion(model_list):
    """Test if the 'schedule_acompletion' function is working correctly"""
    router = Router(model_list=model_list)
    response = await router.schedule_acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
        priority=1,
    )
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_arealtime(model_list):
    """Test if the '_arealtime' function is working correctly"""
    import litellm

    router = Router(model_list=model_list)
    with patch.object(litellm, "_arealtime", AsyncMock()) as mock_arealtime:
        mock_arealtime.return_value = "I'm fine, thank you!"
        await router._arealtime(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        mock_arealtime.assert_awaited_once()


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_router_function_with_fallbacks(model_list, sync_mode):
    """Test if the router 'async_function_with_fallbacks' + 'function_with_fallbacks' are working correctly"""
    router = Router(model_list=model_list)
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "mock_response": "I'm fine, thank you!",
        "num_retries": 0,
    }
    if sync_mode:
        response = router.function_with_fallbacks(
            original_function=router._completion,
            **data,
        )
    else:
        response = await router.async_function_with_fallbacks(
            original_function=router._acompletion,
            **data,
        )
    assert response.choices[0].message.content == "I'm fine, thank you!"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_router_function_with_retries(model_list, sync_mode):
    """Test if the router 'async_function_with_retries' + 'function_with_retries' are working correctly"""
    router = Router(model_list=model_list)
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "mock_response": "I'm fine, thank you!",
        "num_retries": 0,
    }
    if sync_mode:
        response = router.function_with_retries(
            original_function=router._completion,
            **data,
        )
    else:
        response = await router.async_function_with_retries(
            original_function=router._acompletion,
            **data,
        )
    assert response.choices[0].message.content == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_make_call(model_list):
    """Test if the router 'make_call' function is working correctly"""

    ## ACOMPLETION
    router = Router(model_list=model_list)
    response = await router.make_call(
        original_function=router._acompletion,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="I'm fine, thank you!",
    )
    assert response.choices[0].message.content == "I'm fine, thank you!"

    ## ATEXT_COMPLETION
    response = await router.make_call(
        original_function=router._atext_completion,
        model="gpt-3.5-turbo",
        prompt="Hello, how are you?",
        mock_response="I'm fine, thank you!",
    )
    assert response.choices[0].text == "I'm fine, thank you!"

    ## AEMBEDDING
    response = await router.make_call(
        original_function=router._aembedding,
        model="gpt-3.5-turbo",
        input="Hello, how are you?",
        mock_response=[0.1, 0.2, 0.3],
    )
    assert response.data[0].embedding == [0.1, 0.2, 0.3]

    ## AIMAGE_GENERATION
    response = await router.make_call(
        original_function=router._aimage_generation,
        model="dall-e-3",
        prompt="A cute baby sea otter",
        mock_response="https://example.com/image.png",
    )
    assert response.data[0].url == "https://example.com/image.png"
