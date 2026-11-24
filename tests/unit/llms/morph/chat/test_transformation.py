import json
from typing import Final

import pytest
import respx

import litellm


@pytest.mark.parametrize("cached_tokens", [0, 80])
def test_morph_forwards_chat_options_and_prices_cached_input(local_model_cost_map: None, cached_tokens: int) -> None:
    messages: Final = [{"role": "user", "content": "Return a JSON greeting"}]
    response_format: Final = {"type": "json_object"}
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.morphllm.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "morph-dsv41flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"greeting":"hi"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        )
        response: Final = litellm.completion(
            model="morph/morph-dsv41flash",
            api_key="fake-test-key",
            messages=messages,
            response_format=response_format,
            reasoning_effort="none",
            max_completion_tokens=64,
            prompt_cache_key="example-conversation",
            extra_headers={"x-session-id": "example-conversation"},
        )
    request: Final = route.calls.last.request
    assert request.headers["x-session-id"] == "example-conversation"
    assert json.loads(request.content) == {
        "model": "morph-dsv41flash",
        "messages": messages,
        "response_format": response_format,
        "reasoning_effort": "none",
        "max_tokens": 64,
        "prompt_cache_key": "example-conversation",
    }
    assert response.usage.prompt_tokens_details.cached_tokens == cached_tokens
    rates: Final = litellm.model_cost["morph/morph-dsv41flash"]
    expected: Final = (
        (100 - cached_tokens) * rates["input_cost_per_token"]
        + cached_tokens * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_morph_preserves_tool_calls_and_stream_usage() -> None:
    tool: Final = {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    tool_call: Final = {
        "index": 0,
        "id": "call-test",
        "type": "function",
        "function": {"name": "get_temperature", "arguments": '{"city":"Paris"}'},
    }
    chunks: Final = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "morph-dsv41flash",
            "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [tool_call]}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "morph-dsv41flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "morph-dsv41flash",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        },
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.morphllm.com/v1/chat/completions").respond(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        )
        response: Final = litellm.stream_chunk_builder(
            list(
                litellm.completion(
                    model="morph/morph-dsv41flash",
                    api_key="fake-test-key",
                    messages=[{"role": "user", "content": "Get Paris temperature"}],
                    tools=[tool],
                    tool_choice="required",
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
        )
    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == [tool]
    assert body["tool_choice"] == "required"
    assert body["stream_options"] == {"include_usage": True}
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"city":"Paris"}'
    assert response.usage.prompt_tokens_details.cached_tokens == 80


@pytest.mark.parametrize("model", ["morph-v3-fast", "morph-v3-large"])
def test_morph_apply_keeps_rejecting_chat_only_options(model: str) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model="morph/" + model,
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Apply an edit"}],
            response_format={"type": "json_object"},
        )
