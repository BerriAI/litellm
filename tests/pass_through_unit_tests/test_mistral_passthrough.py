from unittest.mock import patch

import fastapi
import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints


@pytest.mark.asyncio
async def test_mistral_passthrough_accepts_multipart_without_json_parsing():
    boundary = "----litellm-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "ocr\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="document.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
        "%PDF-1.4 test\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    async def receive():
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/mistral/v1/files",
            "headers": [
                (
                    b"content-type",
                    f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                )
            ],
            "query_string": b"",
        },
        receive=receive,
    )

    captured_kwargs = {}

    async def fake_endpoint(request, fastapi_response, user_api_key_dict):
        return {"ok": True}

    def fake_create_pass_through_route(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_endpoint

    user_api_key_dict = UserAPIKeyAuth(token="test-key")

    with (
        patch.object(
            llm_passthrough_endpoints.passthrough_endpoint_router,
            "get_credentials",
            return_value="mistral-test-key",
        ),
        patch.object(
            llm_passthrough_endpoints,
            "create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
    ):
        response = await llm_passthrough_endpoints.mistral_proxy_route(
            endpoint="v1/files",
            request=request,
            fastapi_response=fastapi.Response(),
            user_api_key_dict=user_api_key_dict,
        )

    assert response == {"ok": True}
    assert captured_kwargs["is_streaming_request"] is False
    assert captured_kwargs["custom_headers"] == {
        "Authorization": "Bearer mistral-test-key"
    }
