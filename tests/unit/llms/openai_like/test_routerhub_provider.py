import json
from math import isclose
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse, Usage


def test_routerhub_resolves_environment_key_and_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTERHUB_API_KEY", "fake-environment-key")
    assert get_llm_provider(model="routerhub/glm-5.3-abliterated") == (
        "glm-5.3-abliterated",
        "routerhub",
        "fake-environment-key",
        "https://routerhub.si/v1",
    )
    assert get_llm_provider(
        model="routerhub/glm-5.3-abliterated", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("glm-5.3-abliterated", "routerhub", "fake-explicit-key", "https://gateway.example/v1")


@pytest.mark.parametrize("model", ("glm-5.3-abliterated", "glm-5.3-flash-abliterated"))
def test_routerhub_chat_routes_and_bills_reported_cached_usage(model: str, local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://routerhub.si/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "total_tokens": 105,
                        "prompt_tokens_details": {"cached_tokens": 80},
                    },
                },
            ),
        )
        response: Final = litellm.completion(
            model=f"routerhub/{model}",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
        )
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(request.content) == {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 32,
    }
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "hello"
    usage: Final = getattr(response, "usage", None)
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 5
    rates: Final = litellm.get_model_info(f"routerhub/{model}")
    input_rate: Final = rates.get("input_cost_per_token")
    output_rate: Final = rates.get("output_cost_per_token")
    cache_rate: Final = rates.get("cache_read_input_token_cost")
    assert input_rate is not None
    assert output_rate is not None
    assert cache_rate is not None
    expected: Final = 20 * input_rate + 80 * cache_rate + 5 * output_rate
    assert isclose(litellm.completion_cost(completion_response=response), expected, rel_tol=1e-9, abs_tol=1e-12)


def test_routerhub_stream_preserves_text_finish_reason_and_usage() -> None:
    chunks: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5.3-abliterated",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5.3-abliterated",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5.3-abliterated",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        },
    )
    sse: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    with respx.mock() as upstream:
        route: Final = upstream.post("https://routerhub.si/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
        )
        stream: Final = litellm.completion(
            model="routerhub/glm-5.3-abliterated",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert isinstance(stream, CustomStreamWrapper)
        received: Final = tuple(stream)
    assert json.loads(route.calls.last.request.content) == {
        "model": "glm-5.3-abliterated",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert "".join(chunk.choices[0].delta.content or "" for chunk in received if chunk.choices) == "hello"
    assert tuple(chunk.choices[0].finish_reason for chunk in received if chunk.choices)[-1] == "stop"
    usage: Final = getattr(received[-1], "usage", None)
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens == 4
    assert usage.completion_tokens == 1


@pytest.mark.parametrize(
    ("status", "error"),
    ((401, litellm.AuthenticationError), (404, litellm.NotFoundError), (429, litellm.RateLimitError)),
)
def test_routerhub_preserves_upstream_error_status(
    status: int, error: type[litellm.AuthenticationError | litellm.NotFoundError | litellm.RateLimitError]
) -> None:
    with respx.mock() as upstream:
        upstream.post("https://routerhub.si/v1/chat/completions").mock(
            return_value=httpx.Response(
                status, json={"error": {"message": "Request rejected", "type": "api_error", "code": "rejected"}}
            )
        )
        with pytest.raises(error) as caught:
            litellm.completion(
                model="routerhub/glm-5.3-abliterated",
                api_key="fake-test-key",
                messages=[{"role": "user", "content": "Say hello"}],
                num_retries=0,
                max_retries=0,
            )
    assert caught.value.status_code == status
