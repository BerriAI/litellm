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
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache

### UNIT TESTS FOR LLM GUARD ###


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
    _api_key = hash_token("sk-12345")
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
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


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
    _api_key = hash_token("sk-12345")
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
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
        pytest.fail(f"Should have failed - {str(e)}")
    except Exception as e:
        pass


def test_llm_guard_key_specific_mode():
    """
    Tests to see if llm guard 'key-specific' permissions work
    """
    litellm.llm_guard_mode = "key-specific"

    llm_guard = _ENTERPRISE_LLMGuard(mock_testing=True)

    _api_key = "sk-12345"
    # NOT ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
    )

    request_data = {}
    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == False

    # ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, permissions={"enable_llm_guard_check": True}
    )

    request_data = {}

    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == True


def test_llm_guard_request_specific_mode():
    """
    Tests to see if llm guard 'request-specific' permissions work
    """
    litellm.llm_guard_mode = "request-specific"

    llm_guard = _ENTERPRISE_LLMGuard(mock_testing=True)

    _api_key = "sk-12345"
    # NOT ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
    )

    request_data = {}

    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == False

    # ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, permissions={"enable_llm_guard_check": True}
    )

    request_data = {"metadata": {"permissions": {"enable_llm_guard_check": True}}}

    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == True
