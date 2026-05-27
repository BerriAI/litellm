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
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm_enterprise.enterprise_callbacks.secret_detection import (
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
    litellm._turn_on_debug()

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


@pytest.mark.asyncio
async def test_pem_private_key_full_block_redaction_in_message():
    """LIT-3292: detect-secrets PrivateKeyDetector returns only the
    `-----BEGIN ... PRIVATE KEY-----` header line as the secret value, so a
    naive `str.replace(secret_value, "[REDACTED]")` leaves the base64 body
    and `-----END ... PRIVATE KEY-----` footer in the message. The guardrail
    must strip the whole PEM block, not just the header line.
    """
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "fakebase64bodyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        "fakebase64bodyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
        "-----END RSA PRIVATE KEY-----"
    )
    secret_instance = _ENTERPRISE_SecretDetection(
        guardrail_name="hide_secrets", event_hook="pre_call"
    )
    data = {
        "messages": [
            {"role": "user", "content": f"please review this key:\n{pem}\nthanks"}
        ]
    }
    await secret_instance.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        call_type="completion",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-fake"),
    )
    content = data["messages"][0]["content"]
    assert "[REDACTED]" in content
    # Full block must be gone: no header, no body, no footer line.
    assert "-----BEGIN" not in content
    assert "-----END" not in content
    assert "fakebase64bodyA" not in content
    assert "fakebase64bodyB" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    [
        "RSA PRIVATE KEY",
        "EC PRIVATE KEY",
        "DSA PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "PRIVATE KEY",
        "ENCRYPTED PRIVATE KEY",
    ],
)
async def test_pem_private_key_full_block_redaction_all_pem_variants(header):
    """LIT-3292: all PEM private-key flavours (RSA / EC / DSA / OPENSSH /
    PKCS#8 / encrypted PKCS#8) must be redacted as a whole block."""
    pem = (
        f"-----BEGIN {header}-----\n"
        "fakebase64bodyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        f"-----END {header}-----"
    )
    secret_instance = _ENTERPRISE_SecretDetection(
        guardrail_name="hide_secrets", event_hook="pre_call"
    )
    data = {"messages": [{"role": "user", "content": pem}]}
    await secret_instance.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        call_type="completion",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-fake"),
    )
    content = data["messages"][0]["content"]
    assert content.strip() == "[REDACTED]", content


@pytest.mark.asyncio
async def test_pem_private_key_full_block_redaction_in_prompt_str():
    """The PEM full-block sweep must also cover the legacy `data['prompt']`
    string branch (completion-style requests)."""
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "fakebase64bodyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n"
        "-----END PRIVATE KEY-----"
    )
    secret_instance = _ENTERPRISE_SecretDetection(
        guardrail_name="hide_secrets", event_hook="pre_call"
    )
    data = {"prompt": f"key follows: {pem}"}
    await secret_instance.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        call_type="completion",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-fake"),
    )
    assert "fakebase64body" not in data["prompt"]
    assert "[REDACTED]" in data["prompt"]


@pytest.mark.asyncio
async def test_pem_private_key_full_block_redaction_in_prompt_list():
    """The PEM full-block sweep must also cover the `data['prompt']` list
    branch (some legacy / multi-prompt completion shapes)."""
    pem = (
        "-----BEGIN EC PRIVATE KEY-----\n"
        "fakebase64bodyEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n"
        "-----END EC PRIVATE KEY-----"
    )
    secret_instance = _ENTERPRISE_SecretDetection(
        guardrail_name="hide_secrets", event_hook="pre_call"
    )
    data = {"prompt": ["clean string", f"key follows: {pem}"]}
    await secret_instance.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        call_type="completion",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-fake"),
    )
    assert "fakebase64body" not in data["prompt"][1]
    assert "[REDACTED]" in data["prompt"][1]
    assert data["prompt"][0] == "clean string"


def test_redact_pem_blocks_helper_is_pure_function():
    """Direct unit test of the helper so a regression in the regex is caught
    even if the call-site integration is mocked away."""
    from litellm_enterprise.enterprise_callbacks.secret_detection import (
        _redact_pem_blocks,
    )

    # Single block
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "ABCDEF\nGHIJKL\n"
        "-----END RSA PRIVATE KEY-----"
    )
    assert _redact_pem_blocks(pem) == "[REDACTED]"
    # Two blocks in one message
    two = pem + "\nmiddle\n" + pem
    assert _redact_pem_blocks(two) == "[REDACTED]\nmiddle\n[REDACTED]"
    # No PEM -> untouched
    assert _redact_pem_blocks("hello world") == "hello world"
