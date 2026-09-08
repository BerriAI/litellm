import json
import uuid

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


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
