# What is this?
## This tests the llm guard integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.enterprise.enterprise_hooks.openai_moderation import (
    _ENTERPRISE_OpenAI_Moderation,
)
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache

### UNIT TESTS FOR OpenAI Moderation ###


@pytest.mark.asyncio
async def test_openai_moderation_error_raising(monkeypatch):
    """
    Tests to see OpenAI Moderation raises an error for a flagged response
    """
    from unittest.mock import AsyncMock, MagicMock
    from litellm.types.llms.openai import OpenAIModerationResponse
    
    litellm.openai_moderations_model_name = "text-moderation-latest"
    openai_mod = _ENTERPRISE_OpenAI_Moderation()
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    from litellm.proxy.proxy_server import llm_router

    llm_router = litellm.Router(
        model_list=[
            {
                "model_name": "text-moderation-latest",
                "litellm_params": {
                    "model": "text-moderation-latest",
                    "api_key": os.environ.get("OPENAI_API_KEY", "fake-key"),
                },
            }
        ]
    )

    # Mock the amoderation call to return a flagged response
    mock_response = MagicMock(spec=OpenAIModerationResponse)
    mock_response.results = [MagicMock(flagged=True)]
    
    async def mock_amoderation(*args, **kwargs):
        return mock_response
    
    llm_router.amoderation = mock_amoderation

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    try:
        await openai_mod.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "fuck off you're the worst",
                    }
                ]
            },
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
        pytest.fail(f"Should have failed")
    except Exception as e:
        print("Got exception: ", e)
        assert "Violated content safety policy" in str(e)
        pass
