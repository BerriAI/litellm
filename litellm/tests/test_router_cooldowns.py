#### What this tests ####
#    This tests calling router with fallback models

import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger


@pytest.mark.asyncio
async def test_cooldown_badrequest_error():
    """
    Test 1. It SHOULD NOT cooldown a deployment on a BadRequestError
    """

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
        debug_level="DEBUG",
        set_verbose=True,
        cooldown_time=300,
        num_retries=0,
        allowed_fails=0,
    )

    # Act & Assert
    try:

        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "gm"}],
            bad_param=200,
        )
    except:
        pass

    await asyncio.sleep(3)  # wait for deployment to get cooled-down

    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "gm"}],
        mock_response="hello",
    )

    assert response is not None

    print(response)


@pytest.mark.asyncio
async def test_dynamic_cooldowns():
    """
    Assert kwargs for completion/embedding have 'cooldown_time' as a litellm_param
    """
    # litellm.set_verbose = True
    tmp_mock = MagicMock()

    litellm.failure_callback = [tmp_mock]

    router = Router(
        model_list=[
            {
                "model_name": "my-fake-model",
                "litellm_params": {
                    "model": "openai/gpt-1",
                    "api_key": "my-key",
                    "mock_response": Exception("this is an error"),
                },
            }
        ],
        cooldown_time=60,
    )

    try:
        _ = router.completion(
            model="my-fake-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            cooldown_time=0,
            num_retries=0,
        )
    except Exception:
        pass

    tmp_mock.assert_called_once()

    print(tmp_mock.call_count)

    assert "cooldown_time" in tmp_mock.call_args[0][0]["litellm_params"]
    assert tmp_mock.call_args[0][0]["litellm_params"]["cooldown_time"] == 0
