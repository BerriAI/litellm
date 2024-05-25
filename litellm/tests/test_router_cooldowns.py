#### What this tests ####
#    This tests calling router with fallback models

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
import openai, httpx


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
