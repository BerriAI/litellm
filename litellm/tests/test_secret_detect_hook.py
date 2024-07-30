# What is this?
## This tests the llm guard integration

import asyncio
import os
import random

# What is this?
## Unit test for presidio pii masking
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from fastapi import Request, Response
from starlette.datastructures import URL

import litellm
from litellm import Router, mock_completion
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.enterprise.enterprise_hooks.secret_detection import (
    _ENTERPRISE_SecretDetection,
)
from litellm.proxy.proxy_server import chat_completion
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.router import Router

### UNIT TESTS FOR OpenAI Moderation ###


@pytest.mark.asyncio
async def test_basic_secret_detection_chat():
    """
    Tests to see if secret detection hook will mask api keys


    It should mask the following API_KEY = 'sk_1234567890abcdef' and  OPENAI_API_KEY = 'sk_1234567890abcdef'
    """
    secret_instance = _ENTERPRISE_SecretDetection()
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    from litellm.proxy.proxy_server import llm_router

    test_data = {
        "messages": [
            {
                "role": "user",
                "content": "Hey, how's it going, API_KEY = 'sk_1234567890abcdef'",
            },
            {
                "role": "assistant",
                "content": "Hello! I'm doing well. How can I assist you today?",
            },
            {
                "role": "user",
                "content": "this is my OPENAI_API_KEY = 'sk_1234567890abcdef'",
            },
            {
                "role": "user",
                "content": "My hi API Key is sk-Pc4nlxVoMz41290028TbMCxx, does it seem to be in the correct format?",
            },
            {"role": "user", "content": "i think it is +1 412-555-5555"},
        ],
        "model": "gpt-3.5-turbo",
    }

    await secret_instance.async_pre_call_hook(
        cache=local_cache,
        data=test_data,
        user_api_key_dict=user_api_key_dict,
        call_type="completion",
    )
    print(
        "test data after running pre_call_hook: Expect all API Keys to be masked",
        test_data,
    )

    assert test_data == {
        "messages": [
            {"role": "user", "content": "Hey, how's it going, API_KEY = '[REDACTED]'"},
            {
                "role": "assistant",
                "content": "Hello! I'm doing well. How can I assist you today?",
            },
            {"role": "user", "content": "this is my OPENAI_API_KEY = '[REDACTED]'"},
            {
                "role": "user",
                "content": "My hi API Key is [REDACTED], does it seem to be in the correct format?",
            },
            {"role": "user", "content": "i think it is +1 412-555-5555"},
        ],
        "model": "gpt-3.5-turbo",
    }, "Expect all API Keys to be masked"


@pytest.mark.asyncio
async def test_basic_secret_detection_text_completion():
    """
    Tests to see if secret detection hook will mask api keys


    It should mask the following API_KEY = 'sk_1234567890abcdef' and  OPENAI_API_KEY = 'sk_1234567890abcdef'
    """
    secret_instance = _ENTERPRISE_SecretDetection()
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    from litellm.proxy.proxy_server import llm_router

    test_data = {
        "prompt": "Hey, how's it going, API_KEY = 'sk_1234567890abcdef', my OPENAI_API_KEY = 'sk_1234567890abcdef' and i want to know what is the weather",
        "model": "gpt-3.5-turbo",
    }

    await secret_instance.async_pre_call_hook(
        cache=local_cache,
        data=test_data,
        user_api_key_dict=user_api_key_dict,
        call_type="completion",
    )

    test_data == {
        "prompt": "Hey, how's it going, API_KEY = '[REDACTED]', my OPENAI_API_KEY = '[REDACTED]' and i want to know what is the weather",
        "model": "gpt-3.5-turbo",
    }
    print(
        "test data after running pre_call_hook: Expect all API Keys to be masked",
        test_data,
    )


@pytest.mark.asyncio
async def test_basic_secret_detection_embeddings():
    """
    Tests to see if secret detection hook will mask api keys


    It should mask the following API_KEY = 'sk_1234567890abcdef' and  OPENAI_API_KEY = 'sk_1234567890abcdef'
    """
    secret_instance = _ENTERPRISE_SecretDetection()
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    from litellm.proxy.proxy_server import llm_router

    test_data = {
        "input": "Hey, how's it going, API_KEY = 'sk_1234567890abcdef', my OPENAI_API_KEY = 'sk_1234567890abcdef' and i want to know what is the weather",
        "model": "gpt-3.5-turbo",
    }

    await secret_instance.async_pre_call_hook(
        cache=local_cache,
        data=test_data,
        user_api_key_dict=user_api_key_dict,
        call_type="embedding",
    )

    assert test_data == {
        "input": "Hey, how's it going, API_KEY = '[REDACTED]', my OPENAI_API_KEY = '[REDACTED]' and i want to know what is the weather",
        "model": "gpt-3.5-turbo",
    }
    print(
        "test data after running pre_call_hook: Expect all API Keys to be masked",
        test_data,
    )


@pytest.mark.asyncio
async def test_basic_secret_detection_embeddings_list():
    """
    Tests to see if secret detection hook will mask api keys


    It should mask the following API_KEY = 'sk_1234567890abcdef' and  OPENAI_API_KEY = 'sk_1234567890abcdef'
    """
    secret_instance = _ENTERPRISE_SecretDetection()
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    from litellm.proxy.proxy_server import llm_router

    test_data = {
        "input": [
            "hey",
            "how's it going, API_KEY = 'sk_1234567890abcdef'",
            "my OPENAI_API_KEY = 'sk_1234567890abcdef' and i want to know what is the weather",
        ],
        "model": "gpt-3.5-turbo",
    }

    await secret_instance.async_pre_call_hook(
        cache=local_cache,
        data=test_data,
        user_api_key_dict=user_api_key_dict,
        call_type="embedding",
    )

    print(
        "test data after running pre_call_hook: Expect all API Keys to be masked",
        test_data,
    )
    assert test_data == {
        "input": [
            "hey",
            "how's it going, API_KEY = '[REDACTED]'",
            "my OPENAI_API_KEY = '[REDACTED]' and i want to know what is the weather",
        ],
        "model": "gpt-3.5-turbo",
    }


class testLogger(CustomLogger):

    def __init__(self):
        self.logged_message = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success")

        self.logged_message = kwargs.get("messages")


router = Router(
    model_list=[
        {
            "model_name": "fake-model",
            "litellm_params": {
                "model": "openai/fake",
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                "api_key": "sk-12345",
            },
        }
    ]
)


@pytest.mark.asyncio
async def test_chat_completion_request_with_redaction():
    """
    IMPORTANT Enterprise Test - Do not delete it:
    Makes a /chat/completions request on LiteLLM Proxy

    Ensures that the secret is redacted EVEN on the callback
    """
    from litellm.proxy import proxy_server

    setattr(proxy_server, "llm_router", router)
    _test_logger = testLogger()
    litellm.callbacks = [_ENTERPRISE_SecretDetection(), _test_logger]
    litellm.set_verbose = True

    # Prepare the query string
    query_params = "param1=value1&param2=value2"

    # Create the Request object with query parameters
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
            "query_string": query_params.encode(),
        }
    )

    request._url = URL(url="/chat/completions")

    async def return_body():
        return b'{"model": "fake-model", "messages": [{"role": "user", "content": "Hello here is my OPENAI_API_KEY = sk-12345"}]}'

    request.body = return_body

    response = await chat_completion(
        request=request,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-12345",
            token="hashed_sk-12345",
        ),
        fastapi_response=Response(),
    )

    await asyncio.sleep(3)

    print("Info in callback after running request=", _test_logger.logged_message)

    assert _test_logger.logged_message == [
        {"role": "user", "content": "Hello here is my OPENAI_API_KEY = [REDACTED]"}
    ]
    pass
