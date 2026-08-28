import json
import os
from typing import Optional
from fastapi import Request
import pytest
import asyncio


import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.logged_kwargs: Optional[dict] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(
            "in async log success event kwargs",
            json.dumps(kwargs, indent=4, default=str),
        )
        self.logged_kwargs = kwargs


@pytest.mark.asyncio
async def test_untracked_openai_route_passthrough_logging():
    """Keep this on a route `_is_supported_openai_endpoint` does not claim, or the
    OpenAI-specific handler takes over and the generic payload stops being exercised."""
    test_custom_logger = TestCustomLogger()
    litellm._async_success_callback = [test_custom_logger]

    TARGET_URL = "https://api.openai.com/v1/moderations"
    REQUEST_BODY = {
        "model": "omni-moderation-latest",
        "input": "I want to bake a cake for my friend's birthday.",
    }
    TARGET_METHOD = "POST"

    result = await pass_through_request(
        request=Request(
            scope={
                "type": "http",
                "method": TARGET_METHOD,
                "path": "/v1/moderations",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (
                        b"authorization",
                        f"Bearer {os.getenv('OPENAI_API_KEY')}".encode(),
                    ),
                ],
            },
        ),
        target=TARGET_URL,
        custom_headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
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

    assert result.status_code == 200

    await asyncio.sleep(1)

    assert test_custom_logger.logged_kwargs is not None
    passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (
        test_custom_logger.logged_kwargs["passthrough_logging_payload"]
    )
    assert passthrough_logging_payload is not None
    assert passthrough_logging_payload["url"] == TARGET_URL
    assert passthrough_logging_payload["request_body"] == REQUEST_BODY
    assert passthrough_logging_payload["request_method"] == TARGET_METHOD

    client_facing_response_body = json.loads(result.body)
    assert client_facing_response_body["results"]
    assert passthrough_logging_payload["response_body"] == client_facing_response_body
