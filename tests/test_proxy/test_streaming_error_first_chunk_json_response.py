import json

import pytest

from litellm.proxy.common_request_processing import create_response


@pytest.mark.asyncio
async def test_create_response_returns_json_when_first_chunk_error_code_is_none():
    async def gen():
        # Simulate a provider returning an SSE-formatted error as the first chunk,
        # but with `error.code` set to null/None.
        payload = {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }
        yield f"data: {json.dumps(payload)}\n\n"

    resp = await create_response(
        generator=gen(),
        media_type="text/event-stream",
        headers={},
        default_status_code=200,
    )

    # When the first chunk is an error, LiteLLM should not return an SSE response.
    # It should return a JSON error response.
    assert resp.__class__.__name__ == "JSONResponse"
    # When error.code is missing/None, LiteLLM should still treat this as an error and return 500.
    assert resp.status_code == 500
