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
from litellm.proxy.enterprise.enterprise_hooks.llm_guard import _ENTERPRISE_LLMGuard
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache

### UNIT TESTS FOR LLM GUARD ###


#   Test if PII masking works with input A
@pytest.mark.asyncio
async def test_llm_guard_valid_response():
    """
    Tests to see llm guard raises an error for a flagged response
    """
    input_a_anonymizer_results = {
        "sanitized_prompt": "hello world",
        "is_valid": True,
        "scanners": {"Regex": 0.0},
    }
    llm_guard = _ENTERPRISE_LLMGuard(
        mock_testing=True, mock_redacted_text=input_a_anonymizer_results
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    try:
        await llm_guard.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
                    }
                ]
            },
        )
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


#   Test if PII masking works with input B (also test if the response != A's response)
@pytest.mark.asyncio
async def test_llm_guard_error_raising():
    """
    Tests to see llm guard raises an error for a flagged response
    """

    input_b_anonymizer_results = {
        "sanitized_prompt": "hello world",
        "is_valid": False,
        "scanners": {"Regex": 0.0},
    }
    llm_guard = _ENTERPRISE_LLMGuard(
        mock_testing=True, mock_redacted_text=input_b_anonymizer_results
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    try:
        await llm_guard.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
                    }
                ]
            },
        )
        pytest.fail(f"Should have failed - {str(e)}")
    except Exception as e:
        pass
