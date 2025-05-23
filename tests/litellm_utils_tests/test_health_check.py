import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.health_check import perform_health_check
from litellm.router import Router

@pytest.mark.asyncio
async def test_health_check_all_models():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                },
            },
            {
                "model_name": "openai/gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                },
            },
            {
                "model_name": "openai/bad-model",
                "litellm_params": {
                    "model": "openai/bad-model",
                    "api_key": "bad-key",
                },
            },
        ],

    )

    health_check_response = await perform_health_check(router.model_list)

    print("health check response=", health_check_response)




    pass
