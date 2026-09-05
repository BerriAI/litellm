import json

import httpx
import pytest
import respx

import litellm


@pytest.fixture(autouse=True)
def httpx_transport(monkeypatch):
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


@pytest.mark.asyncio
async def test_standardcompute_completion_preserves_tools_and_maps_token_limit():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "standardcompute",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-read",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    with respx.mock as transport:
        request = transport.post("https://api.stdcmpt.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response)
        )
        result = await litellm.acompletion(
            model="standardcompute/standardcompute",
            api_key="test-key",
            messages=[{"role": "user", "content": "Read README.md"}],
            tools=tools,
            max_completion_tokens=64,
        )
    sent = json.loads(request.calls.last.request.content)
    assert sent["model"] == "standardcompute"
    assert sent["tools"] == tools
    assert sent["max_tokens"] == 64
    assert "max_completion_tokens" not in sent
    assert request.calls.last.request.headers["authorization"] == "Bearer test-key"
    assert result.choices[0].message.tool_calls[0].function.name == "read_file"


@pytest.mark.asyncio
async def test_standardcompute_environment_key_and_api_base_override(monkeypatch):
    monkeypatch.setenv("STANDARDCOMPUTE_API_KEY", "environment-test-key")
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "standardcompute",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Connected"},
            }
        ],
    }
    with respx.mock as transport:
        request = transport.post("https://example.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response)
        )
        result = await litellm.acompletion(
            model="standardcompute/standardcompute",
            api_base="https://example.test/v1",
            messages=[{"role": "user", "content": "Connection check"}],
        )
    assert (
        request.calls.last.request.headers["authorization"]
        == "Bearer environment-test-key"
    )
    assert result.choices[0].message.content == "Connected"


@pytest.mark.asyncio
async def test_standardcompute_stream_consumes_chat_completion_events():
    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "standardcompute",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Connected"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "standardcompute",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    events = (
        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )
    with respx.mock as transport:
        request = transport.post("https://api.stdcmpt.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=events, headers={"content-type": "text/event-stream"}
            )
        )
        stream = await litellm.acompletion(
            model="standardcompute/standardcompute",
            api_key="test-key",
            messages=[{"role": "user", "content": "Connection check"}],
            stream=True,
        )
        received = [chunk async for chunk in stream]
    assert json.loads(request.calls.last.request.content)["stream"] is True
    assert (
        "".join(
            chunk.choices[0].delta.content or "" for chunk in received if chunk.choices
        )
        == "Connected"
    )
    assert any(
        chunk.choices and chunk.choices[0].finish_reason == "stop" for chunk in received
    )
