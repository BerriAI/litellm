import base64
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def async_only_image_fetch(monkeypatch):
    from litellm.litellm_core_utils.prompt_templates import factory, image_handling
    from litellm.llms.gemini.chat import transformation as gemini_chat_transformation

    fetch = SimpleNamespace(
        fetched=[],
        base64_png=base64.b64encode(ONE_PIXEL_PNG).decode(),
        data_url="data:image/png;base64," + base64.b64encode(ONE_PIXEL_PNG).decode(),
    )

    def forbid_sync_fetch(client, url, **kwargs):
        raise litellm.ImageFetchError(f"sync image fetch ran on the event loop: {url}")

    async def serve_png(client, url, **kwargs):
        fetch.fetched.append(url)
        return httpx.Response(
            200,
            content=ONE_PIXEL_PNG,
            headers={"content-type": "image/png"},
            request=httpx.Request("GET", url),
        )

    def forbid_sync_convert(url, *args, **kwargs):
        if url.startswith(("http://", "https://")):
            raise litellm.ImageFetchError(f"sync convert_url_to_base64 ran on the request path: {url}")
        return url

    monkeypatch.setattr(image_handling, "safe_get", forbid_sync_fetch)
    monkeypatch.setattr(image_handling, "async_safe_get", serve_png)
    for module in (image_handling, factory, gemini_chat_transformation):
        monkeypatch.setattr(module, "convert_url_to_base64", forbid_sync_convert)
    return fetch


async def test_bedrock_mantle_claude_async_completion_inlines_remote_images_off_the_event_loop(async_only_image_fetch):
    image_url = f"http://img.example/{uuid.uuid4()}.png"
    captured = {}

    def handle(request):
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "us.anthropic.claude-sonnet-5",
                "content": [{"type": "text", "text": "Green"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model="bedrock/mantle/us.anthropic.claude-sonnet-5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="fake-secret",
        aws_region_name="us-east-1",
        client=client,
    )

    assert response.choices[0].message.content == "Green"
    assert async_only_image_fetch.fetched == [image_url]
    assert image_url not in captured["body"]
    assert async_only_image_fetch.base64_png in captured["body"]


async def test_bedrock_mantle_claude_async_completion_inlines_document_url_sources_off_the_event_loop(async_only_image_fetch):
    pdf_url = f"http://docs.example/{uuid.uuid4()}.pdf"
    captured = {}

    def handle(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "us.anthropic.claude-sonnet-5",
                "content": [{"type": "text", "text": "A lease"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model="bedrock/mantle/us.anthropic.claude-sonnet-5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this document?"},
                    {"type": "document", "source": {"type": "url", "url": pdf_url}},
                ],
            }
        ],
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="fake-secret",
        aws_region_name="us-east-1",
        client=client,
    )

    assert response.choices[0].message.content == "A lease"
    assert async_only_image_fetch.fetched == [pdf_url]
    assert {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": async_only_image_fetch.base64_png},
    } in captured["body"]["messages"][0]["content"]
