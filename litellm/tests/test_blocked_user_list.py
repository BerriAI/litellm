# What is this?
## This tests the blocked user pre call hook for the proxy server


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
from litellm.proxy.enterprise.enterprise_hooks.blocked_user_list import (
    _ENTERPRISE_BlockedUserList,
)
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache


@pytest.mark.asyncio
async def test_block_user_check():
    """
    - Set a blocked user as a litellm module value
    - Test to see if a call with that user id is made, an error is raised
    - Test to see if a call without that user is passes
    """
    litellm.blocked_user_list = ["user_id_1"]

    blocked_user_obj = _ENTERPRISE_BlockedUserList()

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    ## Case 1: blocked user id passed
    try:
        await blocked_user_obj.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            call_type="completion",
            data={"user_id": "user_id_1"},
        )
        pytest.fail(f"Expected call to fail")
    except Exception as e:
        pass

    ## Case 2: normal user id passed
    try:
        await blocked_user_obj.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            call_type="completion",
            data={"user_id": "user_id_2"},
        )
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")
