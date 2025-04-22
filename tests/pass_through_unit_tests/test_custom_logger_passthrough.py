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
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request

class TestCustomLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async log success event kwargs", json.dumps(kwargs, indent=4, default=str))
        pass

@pytest.mark.asyncio
async def test_assistants_passthrough_logging():
    test_custom_logger = TestCustomLogger()
    litellm._async_success_callback = [test_custom_logger]

    result = await pass_through_request(
        request=Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": "/v1/assistants",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"authorization", f"Bearer {os.getenv('OPENAI_API_KEY')}".encode()),
                    (b"openai-beta", b"assistants=v2")
                ]
            },
        ),
        target="https://api.openai.com/v1/assistants",
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
        custom_body={
            "instructions": "You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
            "name": "Math Tutor",
            "tools": [{"type": "code_interpreter"}],
            "model": "gpt-4o"
        },
        forward_headers=False,
        merge_query_params=False,
    )

    print("got result", result)
    print("result status code", result.status_code)
    print("result content", result.body)
    


    await asyncio.sleep(5)

