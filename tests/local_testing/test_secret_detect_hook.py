# What is this?
## This tests the llm guard integration

import asyncio
import random

# What is this?
## Unit test for presidio pii masking
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import pytest
from fastapi import Request, Response
from starlette.datastructures import URL

import litellm
from litellm import Router, mock_completion
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
)
from litellm.proxy.proxy_server import chat_completion

from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE

### UNIT TESTS FOR OpenAI Moderation ###


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
                "api_base": FAKE_OPENAI_API_BASE,
                "api_key": "sk-98765",
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
    litellm.turn_on_debug()

    # Prepare the query string
    query_params = "param1=value1&param2=value2"

    # Create the Request object with query parameters
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": query_params.encode(),
        }
    )

    request._url = URL(url="/chat/completions")

    async def return_body():
        return b'{"model": "fake-model", "messages": [{"role": "user", "content": "Hello here is my OPENAI_API_KEY = sk-98765"}]}'

    request.body = return_body

    response = await chat_completion(
        request=request,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-98765",
            token="hashed_sk-98765",
        ),
        fastapi_response=Response(),
    )

    await asyncio.sleep(3)

    print("Info in callback after running request=", _test_logger.logged_message)

    assert _test_logger.logged_message == [
        {"role": "user", "content": "Hello here is my OPENAI_API_KEY = [REDACTED]"}
    ]
    pass
