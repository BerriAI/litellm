import json
from io import BytesIO
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.openai.common_utils import OpenAIError

_UPSTREAM_ERROR_BODY: Final = '{"error":{"message":"image is not a valid PNG","type":"invalid_request_error"}}'


def _upstream_rejection(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        400,
        content=_UPSTREAM_ERROR_BODY,
        headers={"x-request-id": "req_variation_400", "content-type": "application/json"},
    )


_VARIATION_KWARGS: Final = {
    "custom_llm_provider": "openai",
    "api_key": "transport-only",
    "api_base": "https://upstream.example/v1",
    "max_retries": 0,
}


async def _request_variation(is_async: bool) -> None:
    if is_async:
        await litellm.aimage_variation(image=BytesIO(b"image-bytes"), **_VARIATION_KWARGS)
        return
    litellm.image_variation(image=BytesIO(b"image-bytes"), **_VARIATION_KWARGS)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))
async def test_rejected_image_variation_carries_the_upstream_error_response(
    is_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = httpx.MockTransport(_upstream_rejection)
    monkeypatch.setattr(litellm, "client_session", httpx.Client(transport=transport))
    monkeypatch.setattr(litellm, "aclient_session", httpx.AsyncClient(transport=transport))

    with pytest.raises((litellm.BadRequestError, OpenAIError)) as raised:
        await _request_variation(is_async)

    assert raised.value.response.text == _UPSTREAM_ERROR_BODY
    assert raised.value.response.json() == json.loads(_UPSTREAM_ERROR_BODY)
    assert raised.value.response.headers["x-request-id"] == "req_variation_400"
