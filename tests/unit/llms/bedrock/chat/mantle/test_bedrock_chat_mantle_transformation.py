import base64
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


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


RISK_ANSWER = {"tier": 2, "summary": "ok"}
MANTLE_CLAUDE_ROUTES = ("bedrock_mantle/anthropic.claude-scripted-9", "bedrock/mantle/anthropic.claude-scripted-9")


def _json_schema_stream_kwargs(model):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Assess the risk"}],
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Risk",
                "schema": {
                    "type": "object",
                    "properties": {"tier": {"type": "integer"}, "summary": {"type": "string"}},
                    "required": ["tier", "summary"],
                    "additionalProperties": False,
                },
            },
        },
        "aws_access_key_id": "AKIAEXAMPLE",
        "aws_secret_access_key": "fake-secret",
        "aws_region_name": "us-east-1",
    }


def _wrapped_json_tool_stream(request):
    wrapper = json.dumps({"$PARAMETER_NAME": RISK_ANSWER})
    events = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "anthropic.claude-scripted-9",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "json_tool_call", "input": {}},
        },
        *(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": wrapper[start : start + 20]},
            }
            for start in range(0, len(wrapper), 20)
        ),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode(),
    )


def _streamed_answer(chunks):
    deltas = tuple(chunk.choices[0].delta for chunk in chunks)
    return {
        "content": json.loads("".join(delta.content or "" for delta in deltas)),
        "tool_call_deltas": sum(1 for delta in deltas if delta.tool_calls),
        "finish_reasons": tuple(chunk.choices[0].finish_reason for chunk in chunks if chunk.choices[0].finish_reason),
    }


BARE_STREAMED_ANSWER = {"content": RISK_ANSWER, "tool_call_deltas": 0, "finish_reasons": ("stop",)}


@pytest.mark.parametrize("model", MANTLE_CLAUDE_ROUTES)
def test_mantle_claude_completion_streams_a_wrapped_json_schema_answer_as_the_bare_object(model):
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_wrapped_json_tool_stream)))

    chunks = tuple(litellm.completion(**_json_schema_stream_kwargs(model), client=client))

    assert _streamed_answer(chunks) == BARE_STREAMED_ANSWER


@pytest.mark.parametrize("model", MANTLE_CLAUDE_ROUTES)
async def test_mantle_claude_async_completion_streams_a_wrapped_json_schema_answer_as_the_bare_object(model):
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_wrapped_json_tool_stream))

    stream = await litellm.acompletion(**_json_schema_stream_kwargs(model), client=client)
    chunks = tuple([chunk async for chunk in stream])

    assert _streamed_answer(chunks) == BARE_STREAMED_ANSWER
