"""
Tests litellm pre_call_utils
"""

import os
import sys
import traceback
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.proxy_server import ProxyConfig, chat_completion

load_dotenv()
import io
import os
import time

import pytest

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


@pytest.mark.parametrize("tier", ["free", "paid"])
@pytest.mark.asyncio()
async def test_adding_key_tier_to_request_metadata(tier):
    """
    Tests if we can add tier: free/paid from key metadata to the request metadata
    """
    data = {}

    api_route = APIRoute(path="/chat/completions", endpoint=chat_completion)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "route": api_route,
            "path": api_route.path,
            "headers": [],
        }
    )
    new_data = await add_litellm_data_to_request(
        data=data,
        request=request,
        user_api_key_dict=UserAPIKeyAuth(metadata={"tier": tier}),
        proxy_config=ProxyConfig(),
    )

    print("new_data", new_data)

    assert new_data["metadata"]["tier"] == tier
