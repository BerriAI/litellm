import json
from typing import Final

import pytest
import respx

import litellm

_CHAT_URL: Final = "https://api.abstract.inc/v1/chat/completions"
_LANES: Final = ("glm-5.3:superfast", "deepseek-v4-flash:superfast", "kimi-k3:superfast")


def _completion(model: str, text: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-abstract",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


def _sse(*chunks: dict[str, object]) -> bytes:
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()


def _chunk(delta: dict[str, object], finish_reason: str | None = None, **extra: object) -> dict[str, object]:
    return {
        "id": "chatcmpl-abstract",
        "object": "chat.completion.chunk",
        "created": 1_790_000_000,
        "model": "glm-5.3:superfast",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


def test_abstract_provider_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ABSTRACT_API_KEY", "abstract-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="abstract/glm-5.3:superfast",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert (model, provider, api_key, api_base) == (
        "glm-5.3:superfast",
        "abstract",
        "abstract-env-key",
        "https://api.abstract.inc/v1",
    )


@pytest.mark.parametrize("lane", _LANES)
def test_abstract_completion_is_sent_to_chat_completions(lane: str) -> None:
    with respx.mock(assert_all_called=False) as upstream:
        responses: Final = upstream.post("https://api.abstract.inc/v1/responses").respond(404)
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_completion(lane, "Hello from Abstract"))
        response: Final = litellm.completion(
            model=f"abstract/{lane}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="abstract-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert responses.call_count == 0
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer abstract-test-key"
    assert body["model"] == lane
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Abstract"


def test_abstract_streaming_completion_is_sent_to_chat_completions() -> None:
    stream: Final = _sse(
        _chunk({"role": "assistant", "content": "Hello "}),
        _chunk({"content": "from Abstract"}),
        _chunk({}, "stop"),
        {**_chunk({}), "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}},
    )
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, content=stream, headers={"content-type": "text/event-stream"}
        )
        chunks: Final = tuple(
            litellm.completion(
                model="abstract/glm-5.3:superfast",
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="abstract-test-key",
                stream=True,
                stream_options={"include_usage": True},
            )
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "Hello from Abstract"
    assert chunks[-1].usage.total_tokens == 7


@pytest.mark.parametrize("lane", _LANES)
def test_abstract_lane_cost_is_priced_from_its_cost_map_row(lane: str) -> None:
    row: Final = litellm.model_cost[f"abstract/{lane}"]
    cost: Final = litellm.completion_cost(
        completion_response=litellm.ModelResponse(
            model=lane,
            usage=litellm.Usage(prompt_tokens=1_000, completion_tokens=500, total_tokens=1_500),
        ),
        model=f"abstract/{lane}",
    )

    assert row["mode"] == "chat"
    assert row["supported_endpoints"] == ["/v1/chat/completions"]
    assert cost == pytest.approx(1_000 * row["input_cost_per_token"] + 500 * row["output_cost_per_token"])
