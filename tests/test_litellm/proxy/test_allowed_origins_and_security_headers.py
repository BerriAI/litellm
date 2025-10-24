import asyncio
import json
import os
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))


request_test_data = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}]
}

security_headers = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}

example_chat_completion_result = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}


def mock_patch_acompletion():
    mock_obj = mock.AsyncMock(return_value=example_chat_completion_result)
    mock_obj.__name__ = "acompletion"
    return mock.patch(
        "litellm.acompletion",
        new_callable=lambda: mock_obj,
    )

@pytest.fixture(scope="function")
def client_with_selected_origins_and_security_headers():
    os.environ["ALLOWED_ORIGINS"] = "https://docs.litellm.ai,https://docs.litellm.com"
    os.environ["SECURITY_HEADERS"] = json.dumps(security_headers)
    from litellm.proxy.proxy_server import cleanup_router_config_variables
    cleanup_router_config_variables()
    mock_completion = mock.AsyncMock(return_value=example_chat_completion_result)
    mock_completion.__name__ = "acompletion"
    with mock.patch(
            "litellm.acompletion",
            new_callable=lambda: mock_completion,
    ) as patched_completion, \
            mock.patch(
                "litellm.proxy.route_llm_request.route_request",
                new_callable=lambda: mock.AsyncMock(return_value=example_chat_completion_result)
            ):
        from litellm.proxy.proxy_server import app, initialize
        filepath = os.path.dirname(os.path.abspath(__file__))
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        print(f"config_fp: {config_fp}")
        asyncio.run(initialize(config=config_fp, debug=True))
        client = TestClient(app)
        yield client, patched_completion


class TestAllowedOrigins:
    """Test suite for Allowed Origins and Security Headers."""

    def test_cors_specific_origin_allowed(self, client_with_selected_origins_and_security_headers):
        client, mock_acompletion = client_with_selected_origins_and_security_headers
        response = client.post(
            "/v1/chat/completions",
            json=request_test_data,
            headers={"origin": "https://docs.litellm.ai"}
        )
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://docs.litellm.ai"

    def test_cors_specific_origin_blocked(self, client_with_selected_origins_and_security_headers):
        client, mock_acompletion = client_with_selected_origins_and_security_headers
        response = client.post(
            "/v1/chat/completions",
            json=request_test_data,
            headers={"origin": "https://malicious.com"}
        )
        assert response.status_code == 403
        response_data = response.json()
        assert "CORS policy violation" in response_data["error"]["message"]
        assert "https://malicious.com" in response_data["error"]["message"]

    def test_cors_no_origin_header(self, client_with_selected_origins_and_security_headers):
        client, mock_acompletion = client_with_selected_origins_and_security_headers
        response = client.post("/v1/chat/completions", json=request_test_data)
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_security_headers_applied(self, client_with_selected_origins_and_security_headers):
        client, mock_acompletion = client_with_selected_origins_and_security_headers
        response = client.post(
            "/v1/chat/completions",
            json=request_test_data,
            headers={"origin": "https://docs.litellm.ai"}
        )
        assert response.status_code == 200
        for header, value in security_headers.items():
            assert response.headers.get(header) == value