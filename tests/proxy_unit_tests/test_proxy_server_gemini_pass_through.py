import os

import pytest


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="Requires GEMINI_API_KEY or GOOGLE_API_KEY.",
)
@pytest.mark.asyncio
async def test_gemini_pass_through_endpoint():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        Request,
        Response,
        gemini_proxy_route,
    )

    body = b"""
        {
            "contents": [{
                "parts":[{
                "text": "The quick brown fox jumps over the lazy dog."
                }]
                }]
        }
        """

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gemini/v1beta/models/gemini-2.5-flash:countTokens",
        "query_string": b"key=sk-1234",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }

    async def async_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        scope=scope,
        receive=async_receive,
    )

    await gemini_proxy_route(
        endpoint="v1beta/models/gemini-2.5-flash:countTokens?key=sk-1234",
        request=request,
        fastapi_response=Response(),
    )

