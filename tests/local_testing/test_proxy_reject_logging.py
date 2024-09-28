# What is this?
## Unit test that rejected requests are also logged as failures

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
from typing import Literal

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
from litellm.proxy.proxy_server import (
    Depends,
    HTTPException,
    chat_completion,
    completion,
    embeddings,
)
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.router import Router


class testLogger(CustomLogger):

    def __init__(self):
        self.reaches_sync_failure_event = False
        self.reaches_async_failure_event = False

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ):
        raise HTTPException(
            status_code=429, detail={"error": "Max parallel request limit reached"}
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.reaches_async_failure_event = True

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.reaches_sync_failure_event = True


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


@pytest.mark.parametrize(
    "route, body",
    [
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello here is my OPENAI_API_KEY = sk-12345",
                    }
                ],
            },
        ),
        ("/v1/completions", {"model": "fake-model", "prompt": "ping"}),
        (
            "/v1/embeddings",
            {
                "input": "The food was delicious and the waiter...",
                "model": "text-embedding-ada-002",
                "encoding_format": "float",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_chat_completion_request_with_redaction(route, body):
    """
    IMPORTANT Enterprise Test - Do not delete it:
    Makes a /chat/completions request on LiteLLM Proxy

    Ensures that the secret is redacted EVEN on the callback
    """
    from litellm.proxy import proxy_server

    setattr(proxy_server, "llm_router", router)
    _test_logger = testLogger()
    litellm.callbacks = [_test_logger]
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

    request._url = URL(url=route)

    async def return_body():
        import json

        return json.dumps(body).encode()

    request.body = return_body

    try:
        if route == "/v1/chat/completions":
            response = await chat_completion(
                request=request,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-12345", token="hashed_sk-12345", rpm_limit=0
                ),
                fastapi_response=Response(),
            )
        elif route == "/v1/completions":
            response = await completion(
                request=request,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-12345", token="hashed_sk-12345", rpm_limit=0
                ),
                fastapi_response=Response(),
            )
        elif route == "/v1/embeddings":
            response = await embeddings(
                request=request,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-12345", token="hashed_sk-12345", rpm_limit=0
                ),
                fastapi_response=Response(),
            )
    except:
        pass
    await asyncio.sleep(3)

    assert _test_logger.reaches_async_failure_event is True

    assert _test_logger.reaches_sync_failure_event is True
