import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Optional
from fastapi import Request
import pytest
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request

class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.logged_kwargs: Optional[dict] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async log success event kwargs", json.dumps(kwargs, indent=4, default=str))
        self.logged_kwargs = kwargs

@pytest.mark.asyncio
async def test_assistants_passthrough_logging():
    test_custom_logger = TestCustomLogger()
    litellm._async_success_callback = [test_custom_logger]

    TARGET_URL = "https://api.openai.com/v1/assistants"
    REQUEST_BODY = {
        "instructions": "You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
        "name": "Math Tutor",
        "tools": [{"type": "code_interpreter"}],
        "model": "gpt-4o"
    }
    TARGET_METHOD = "POST"

    result = await pass_through_request(
        request=Request(
            scope={
                "type": "http",
                "method": TARGET_METHOD,
                "path": "/v1/assistants",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"authorization", f"Bearer {os.getenv('OPENAI_API_KEY')}".encode()),
                    (b"openai-beta", b"assistants=v2")
                ]
            },
        ),
        target=TARGET_URL,
        custom_headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "assistants=v2"
        },
        user_api_key_dict=UserAPIKeyAuth(
            api_key="test",
            user_id="test",
            team_id="test",
            end_user_id="test",
        ),
        custom_body=REQUEST_BODY,
        forward_headers=False,
        merge_query_params=False,
    )

    print("got result", result)
    print("result status code", result.status_code)
    print("result content", result.body)
    
    await asyncio.sleep(1)

    assert test_custom_logger.logged_kwargs is not None
    passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = test_custom_logger.logged_kwargs["passthrough_logging_payload"]
    assert passthrough_logging_payload is not None
    assert passthrough_logging_payload["url"] == TARGET_URL
    assert passthrough_logging_payload["request_body"] == REQUEST_BODY

    # assert that the response body content matches the response body content
    client_facing_response_body = json.loads(result.body)
    assert passthrough_logging_payload["response_body"]  == client_facing_response_body

    # assert that the request method is correct
    assert passthrough_logging_payload["request_method"] == TARGET_METHOD

@pytest.mark.asyncio
async def test_threads_passthrough_logging():
    test_custom_logger = TestCustomLogger()
    litellm._async_success_callback = [test_custom_logger]

    TARGET_URL = "https://api.openai.com/v1/threads"
    REQUEST_BODY = {} 
    TARGET_METHOD = "POST"

    result = await pass_through_request(
        request=Request(
            scope={
                "type": "http",
                "method": TARGET_METHOD,
                "path": "/v1/threads",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"authorization", f"Bearer {os.getenv('OPENAI_API_KEY')}".encode()),
                    (b"openai-beta", b"assistants=v2")
                ]
            },
        ),
        target=TARGET_URL,
        custom_headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "assistants=v2"
        },
        user_api_key_dict=UserAPIKeyAuth(
            api_key="test",
            user_id="test",
            team_id="test",
            end_user_id="test",
        ),
        custom_body=REQUEST_BODY,
        forward_headers=False,
        merge_query_params=False,
    )

    print("got result", result)
    print("result status code", result.status_code)
    print("result content", result.body)
    
    await asyncio.sleep(1)

    assert test_custom_logger.logged_kwargs is not None
    passthrough_logging_payload = test_custom_logger.logged_kwargs["passthrough_logging_payload"]
    assert passthrough_logging_payload is not None
    
    # Fix for TypedDict access errors
    assert passthrough_logging_payload.get("url") == TARGET_URL
    assert passthrough_logging_payload.get("request_body") == REQUEST_BODY

    # Fix for json.loads error with potential memoryview
    response_body = result.body
    client_facing_response_body = json.loads(response_body)
    
    assert passthrough_logging_payload.get("response_body") == client_facing_response_body
    assert passthrough_logging_payload.get("request_method") == TARGET_METHOD



