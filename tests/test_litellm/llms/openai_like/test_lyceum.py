import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

BASE: Final = "https://api.lyceum.technology/openai/v1"
MODEL: Final = "lyceum/z-ai/glm-5.3-flash"
REMOTE_MODEL: Final = "z-ai/glm-5.3-flash"
KEY: Final = "lyceum-test-key"


@pytest.fixture(autouse=True)
def lyceum_environment(monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None) -> None:
    monkeypatch.setenv("LYCEUM_API_KEY", KEY)
    monkeypatch.delenv("LYCEUM_API_BASE", raising=False)


@pytest.mark.parametrize("remote_model", [REMOTE_MODEL, "lyceum/simple"])
def test_provider_resolution(remote_model: str) -> None:
    assert get_llm_provider(model=f"lyceum/{remote_model}") == (
        remote_model,
        "lyceum",
        KEY,
        BASE,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_override_precedence(monkeypatch: pytest.MonkeyPatch, explicit: bool) -> None:
    monkeypatch.setenv("LYCEUM_API_BASE", "https://environment.example/v1")
    assert get_llm_provider(
        model=MODEL,
        api_base="https://explicit.example/v1" if explicit else None,
        api_key="explicit-key" if explicit else None,
    ) == (
        REMOTE_MODEL,
        "lyceum",
        "explicit-key" if explicit else KEY,
        "https://explicit.example/v1" if explicit else "https://environment.example/v1",
    )


def completion_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-lyceum-test",
        "object": "chat.completion",
        "created": 1,
        "model": REMOTE_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    }


def streaming_response() -> bytes:
    chunks: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}},
    )
    return (
        "".join(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-lyceum-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": REMOTE_MODEL,
                    **chunk,
                }
            )
            + "\n\n"
            for chunk in chunks
        )
        + "data: [DONE]\n\n"
    ).encode()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("override", [False, True])
async def test_chat_wire_contract(
    monkeypatch: pytest.MonkeyPatch, is_async: bool, stream: bool, override: bool
) -> None:
    expected_base: Final = "https://override.example/v1" if override else BASE
    if override:
        monkeypatch.setenv("LYCEUM_API_BASE", expected_base)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{expected_base}/chat/completions"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        body: Final = json.loads(request.content)
        assert body["model"] == REMOTE_MODEL
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["max_tokens"] == 64
        assert body.get("stream", False) is stream
        if stream:
            assert body["stream_options"] == {"include_usage": True}
            return httpx.Response(200, content=streaming_response(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=completion_response())

    with httpx.Client(transport=httpx.MockTransport(respond)) as sync_client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
            monkeypatch.setattr(litellm, "client_session", sync_client)
            monkeypatch.setattr(litellm, "aclient_session", async_client)
            kwargs: Final = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 64,
                "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
            }
            response: Final = await litellm.acompletion(**kwargs) if is_async else litellm.completion(**kwargs)
            if stream:
                chunks: Final = [chunk async for chunk in response] if is_async else list(response)
                assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "Hello"
                assert any(chunk.choices and chunk.choices[0].finish_reason == "stop" for chunk in chunks)
                assert next(chunk.usage.total_tokens for chunk in chunks if getattr(chunk, "usage", None)) == 10
            else:
                assert response.choices[0].message.content == "Hello"
                assert response.usage.prompt_tokens == 8
                assert response.usage.completion_tokens == 2
                assert response.usage.total_tokens == 10


@pytest.mark.parametrize(
    ("status", "message", "error_type", "exception"),
    [
        (401, "not authenticated", "unauthorized_error", litellm.AuthenticationError),
        (404, "model not found", "not_found_error", litellm.NotFoundError),
        (429, "inference capacity exhausted", "rate_limit_error", litellm.RateLimitError),
        (400, "inference request rejected", "invalid_request_error", litellm.BadRequestError),
    ],
)
def test_http_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, message: str, error_type: str, exception: type[Exception]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE}/chat/completions"
        return httpx.Response(status, json={"error": {"message": message, "type": error_type, "param": None}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "client_session", client)
        with pytest.raises(exception, match=message):
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                num_retries=0,
            )


def test_tools_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    tools: Final = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}
    ]
    tool_calls: Final = [
        {"id": "call-weather", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert body["tools"] == tools
        assert body["tool_choice"] == "required"
        if body["messages"][-1]["role"] == "tool":
            assert body["messages"][-1]["tool_call_id"] == "call-weather"
            assert body["messages"][-1]["content"] == "Sunny"
            assert body["messages"][-2]["tool_calls"] == tool_calls
            return httpx.Response(200, json=completion_response())
        return httpx.Response(
            200,
            json={
                **completion_response(),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "client_session", client)
        response: Final = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Weather?"}],
            tools=tools,
            tool_choice="required",
        )
        followup: Final = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Weather?"},
                response.choices[0].message.model_dump(exclude_none=True),
                {"role": "tool", "tool_call_id": "call-weather", "content": "Sunny"},
            ],
            tools=tools,
            tool_choice="required",
        )
    assert followup.choices[0].message.content == "Hello"
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert response.choices[0].message.tool_calls[0].function.arguments == "{}"
    assert response.usage.prompt_tokens_details.cached_tokens == 4


def test_pricing_uses_lyceum_metadata() -> None:
    metadata: Final = litellm.model_cost[MODEL]
    input_cost, output_cost = litellm.cost_per_token(model=MODEL, prompt_tokens=8, completion_tokens=2)
    assert input_cost == pytest.approx(8 * metadata["input_cost_per_token"])
    assert output_cost == pytest.approx(2 * metadata["output_cost_per_token"])


def test_midstream_error_is_not_silently_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content=b'data: {"error":{"message":"inference service temporarily unavailable","type":"server_error","param":null}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "client_session", client)
        with pytest.raises(MidStreamFallbackError, match="inference service temporarily unavailable"):
            list(litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True))
