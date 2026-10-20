import json
from typing import Final

import httpx
import pytest
import respx
from openai import AsyncOpenAI
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import ModelResponse


@pytest.mark.parametrize("model", ("gemma-4-31b", "deepseek-v4.1-flash"))
def test_hopper_resolves_model_and_environment_credentials(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPER_API_KEY", "hopper-test-key")
    result: Final = get_llm_provider(model=f"hopper/{model}")

    assert result == (model, "hopper", "hopper-test-key", "https://api.withhopper.com/v1"), (
        "Hopper endpoint and model IDs verified at https://withhopper.com/mcp on 2026-10-19"
    )


def test_hopper_explicit_credentials_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPER_API_KEY", "environment-test-key")
    result: Final = get_llm_provider(
        model="hopper/gemma-4-31b", api_key="explicit-test-key", api_base="https://hopper.example/v1"
    )

    assert result == ("gemma-4-31b", "hopper", "explicit-test-key", "https://hopper.example/v1")


@pytest.mark.parametrize("model", ("gemma-4-31b", "deepseek-v4.1-flash"))
def test_hopper_completion_preserves_model_and_bills_cached_tokens(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPER_API_KEY", "hopper-test-key")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.withhopper.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-hopper-test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        )
        response: Final = litellm.completion(
            model=f"hopper/{model}", messages=[{"role": "user", "content": "Hello"}], max_tokens=32
        )

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer hopper-test-key"
    assert TypeAdapter(JsonValue).validate_json(route.calls.last.request.content) == {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
    }
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hello"
    row: Final = litellm.model_cost[f"hopper/{model}"]
    assert litellm.completion_cost(completion_response=response, model=f"hopper/{model}") == pytest.approx(
        200 * row["input_cost_per_token"] + 800 * row["cache_read_input_token_cost"] + 20 * row["output_cost_per_token"]
    )


@pytest.mark.parametrize("model", ("gemma-4-31b", "deepseek-v4.1-flash"))
@pytest.mark.asyncio
async def test_hopper_stream_keeps_content_and_finish_reason(model: str) -> None:
    chunk: Final = {
        "id": "chatcmpl-hopper-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
    }
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.withhopper.com/v1/chat/completions").respond(
            200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )
        async with AsyncOpenAI(
            api_key="hopper-test-key",
            base_url="https://api.withhopper.com/v1",
            http_client=httpx.AsyncClient(),
        ) as client:
            stream: Final = await litellm.acompletion(
                model=f"hopper/{model}",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="hopper-test-key",
                stream=True,
                client=client,
            )
            chunks: Final = [part async for part in stream]

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer hopper-test-key"
    assert TypeAdapter(JsonValue).validate_json(route.calls.last.request.content) == {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    assert "".join(part.choices[0].delta.content or "" for part in chunks) == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_hopper_gemma_forwards_tools() -> None:
    tools: Final = [{"type": "function", "function": {"name": "get_balance", "parameters": {"type": "object"}}}]
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.withhopper.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-hopper-tools",
                "object": "chat.completion",
                "created": 1,
                "model": "gemma-4-31b",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
            },
        )
        litellm.completion(
            model="hopper/gemma-4-31b",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="hopper-test-key",
            tools=tools,
        )

    assert route.call_count == 1
    assert TypeAdapter(JsonValue).validate_json(route.calls.last.request.content) == {
        "model": "gemma-4-31b",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": tools,
    }
