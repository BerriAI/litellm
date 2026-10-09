"""
Regression tests for https://github.com/BerriAI/litellm/issues/42185

/v1/images/edits with no `image` part raised a raw TypeError out of aimage_edit,
which the proxy surfaced as a 500. A 500 tells OpenAI-SDK clients to retry a
request that can never succeed, so a missing image has to read as a 400.
"""

import httpx
import pytest

import litellm

PNG_BYTES: bytes = b"\x89PNG\r\n\x1a\nfakepng"


@pytest.mark.asyncio
async def test_aimage_edit_without_image_is_a_bad_request_not_a_type_error():
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.aimage_edit(
            model="gemini/gemini-3.1-flash-image",
            prompt="make it blue",
            api_key="fake-key-never-used",
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_aimage_edit_still_reaches_the_provider_when_an_image_is_supplied(monkeypatch):
    sent: dict = {}

    async def fake_post(self, *args, **kwargs):
        sent["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}]},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", fake_post)

    await litellm.aimage_edit(
        model="gemini/gemini-3.1-flash-image",
        prompt="make it blue",
        image=PNG_BYTES,
        api_key="fake-key-never-used",
    )

    parts = sent["json"]["contents"][0]["parts"]
    assert any("inlineData" in part for part in parts)
