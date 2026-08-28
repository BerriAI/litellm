import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


UPSTREAM_RESPONSE_BODY = {
    "id": "modr-abc123",
    "model": "omni-moderation-latest",
    "results": [
        {
            "flagged": False,
            "categories": {"violence": False},
            "category_scores": {"violence": 1.2e-06},
        }
    ],
}


@pytest.fixture
def upstream():
    received: dict = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0) or 0))
            received["path"] = self.path
            received["body"] = json.loads(body or b"{}")
            payload = json.dumps(UPSTREAM_RESPONSE_BODY).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_untracked_openai_route_passthrough_logging(upstream):
    """Keep this on a route `_is_supported_openai_endpoint` does not claim, or the
    OpenAI-specific handler takes over and the generic payload stops being exercised."""
    base_url, upstream_received = upstream

    test_custom_logger = TestCustomLogger()
    litellm._async_success_callback = [test_custom_logger]

    TARGET_URL = f"{base_url}/v1/moderations"
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
                    (b"authorization", b"Bearer sk-test-passthrough"),
                ],
            },
        ),
        target=TARGET_URL,
        custom_headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-test-passthrough",
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

    assert upstream_received.get("path") == "/v1/moderations"
    assert upstream_received.get("body") == REQUEST_BODY
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
    assert client_facing_response_body == UPSTREAM_RESPONSE_BODY
    assert passthrough_logging_payload["response_body"] == client_facing_response_body
