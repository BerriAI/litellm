import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
from openai import AsyncOpenAI

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

BASE: Final = "https://inference-api.nousresearch.com/v1"
MODEL: Final = "google/gemini-4-hydrogen"
CHAT: Final = {
    "id": "chatcmpl-nous",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": MODEL,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
MODELS: Final = tuple(name for name in litellm.model_cost if name.startswith("nousresearch/"))


def test_unlisted_model_forwards_tools() -> None:
    model: Final = "nousresearch/lit-test/unlisted-chat"
    tools: Final = [{"type": "function", "function": {"name": "status", "parameters": {"type": "object"}}}]
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/chat/completions").respond(json=CHAT)
        response: Final = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            tools=tools,
            tool_choice="auto",
            api_key="nous-key",
        )
    assert json.loads(route.calls.last.request.content)["tools"] == tools
    assert json.loads(route.calls.last.request.content)["tool_choice"] == "auto"
    assert response.choices[0].message.content == "Hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["responses", "messages"])
async def test_chat_only_model_translates_native_requests(endpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model: Final = "nousresearch/lit-test/chat-only"
    litellm.register_model(
        {model: {"litellm_provider": "nousresearch", "mode": "chat", "supported_endpoints": ["/v1/chat/completions"]}}
    )
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/chat/completions").respond(json=CHAT)
        if endpoint == "responses":
            response: Final = await litellm.aresponses(model=model, input="Hello", api_key="nous-key")
            assert response.output[0].content[0].text == "Hello"
        else:
            message: Final = await litellm.anthropic.messages.acreate(
                model=model, messages=[{"role": "user", "content": "Hello"}], max_tokens=32, api_key="nous-key"
            )
            assert message["content"] == [{"type": "text", "text": "Hello"}]
    assert json.loads(route.calls.last.request.content)["model"] == "lit-test/chat-only"


def test_bundled_model_listing_includes_chat_and_embedding_models() -> None:
    assert set(litellm.get_valid_models(custom_llm_provider="nousresearch", check_provider_endpoint=False)) == set(
        MODELS
    )
    assert set(MODELS) <= set(litellm.model_list)


@pytest.mark.parametrize("prefixed", [True, False])
def test_resolves_nested_model_and_environment_credentials(monkeypatch: pytest.MonkeyPatch, prefixed: bool) -> None:
    monkeypatch.setenv("NOUSRESEARCH_API_KEY", "nous-key")
    monkeypatch.delenv("NOUSRESEARCH_API_BASE", raising=False)
    result: Final = get_llm_provider(
        model=f"nousresearch/{MODEL}" if prefixed else MODEL, api_base=None if prefixed else BASE
    )
    assert result == (MODEL, "nousresearch", "nous-key", BASE)


def test_explicit_credentials_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOUSRESEARCH_API_KEY", "env-key")
    monkeypatch.setenv("NOUSRESEARCH_API_BASE", "https://env.example/v1")
    assert get_llm_provider(model=f"nousresearch/{MODEL}")[2:] == ("env-key", "https://env.example/v1")
    assert get_llm_provider(
        model=f"nousresearch/{MODEL}", api_key="explicit-key", api_base="https://explicit.example/v1"
    ) == (MODEL, "nousresearch", "explicit-key", "https://explicit.example/v1")


def test_chat_preserves_model_and_maps_completion_limit() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/chat/completions").respond(json=CHAT)
        response: Final = litellm.completion(
            model=f"nousresearch/{MODEL}",
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=32,
            temperature=0.5,
            api_key="nous-key",
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer nous-key"
    assert json.loads(route.calls.last.request.content) == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "temperature": 0.5,
    }
    assert response.choices[0].message.content == "Hello"
    assert response.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_async_chat_uses_injected_http_client() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/chat/completions").respond(json=CHAT)
        async with httpx.AsyncClient() as http_client:
            client: Final = AsyncOpenAI(api_key="nous-key", base_url=BASE, http_client=http_client)
            response: Final = await litellm.acompletion(
                model=f"nousresearch/{MODEL}", messages=[{"role": "user", "content": "Hello"}], client=client
            )
    assert json.loads(route.calls.last.request.content)["model"] == MODEL
    assert route.calls.last.request.headers["authorization"] == "Bearer nous-key"
    assert response.choices[0].message.content == "Hello"


def test_streaming_chat_returns_content_and_finish_reason() -> None:
    event: Final = {
        "id": "chatcmpl-nous",
        "object": "chat.completion.chunk",
        "created": CHAT["created"],
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/chat/completions").respond(
            headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        )
        stream: Final = litellm.completion(
            model=f"nousresearch/{MODEL}",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            api_key="nous-key",
        )
        chunks: Final = tuple(stream)
    assert json.loads(route.calls.last.request.content)["stream"] is True
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_text_completion_routes_to_completions() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/completions").respond(
            json={
                "id": "cmpl-nous",
                "object": "text_completion",
                "created": CHAT["created"],
                "model": MODEL,
                "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop", "logprobs": None}],
                "usage": CHAT["usage"],
            }
        )
        response: Final = litellm.text_completion(model=f"nousresearch/{MODEL}", prompt="Hi", api_key="nous-key")
    assert json.loads(route.calls.last.request.content)["model"] == MODEL
    assert response.choices[0].text == "Hello"


def test_embedding_keeps_vendor_prefix_and_returns_vectors() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/embeddings").respond(
            json={
                "object": "list",
                "model": "cohere/embed-5-fast",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )
        response: Final = litellm.embedding(
            model="nousresearch/cohere/embed-5-fast", input=["Hello"], api_key="nous-key"
        )
    assert json.loads(route.calls.last.request.content)["model"] == "cohere/embed-5-fast"
    assert route.calls.last.request.headers["authorization"] == "Bearer nous-key"
    assert response.data[0]["embedding"] == [0.25, 0.75]


def test_responses_uses_native_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/responses").respond(
            json={
                "id": "resp-nous",
                "object": "response",
                "created_at": CHAT["created"],
                "model": MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg-nous",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            }
        )
        response: Final = litellm.responses(model=f"nousresearch/{MODEL}", input="Hello", api_key="nous-key")
    assert route.calls.last.request.headers["authorization"] == "Bearer nous-key"
    assert json.loads(route.calls.last.request.content)["model"] == MODEL
    assert response.output[0].content[0].text == "Hello"


@pytest.mark.asyncio
async def test_messages_uses_native_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{BASE}/messages").respond(
            json={
                "id": "msg-nous",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            }
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"nousresearch/{MODEL}",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=32,
            api_key="nous-key",
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer nous-key"
    assert json.loads(route.calls.last.request.content) == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": False,
    }
    assert response["content"] == [{"type": "text", "text": "Hello"}]


@pytest.mark.parametrize("model", MODELS)
def test_cost_calculation_uses_nous_rates_including_cached_input(model: str) -> None:
    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(model=model, prompt_tokens=1000, completion_tokens=100)
    cached_cost, _ = cost_per_token(model=model, prompt_tokens=1000, completion_tokens=0, cache_read_input_tokens=400)
    assert prompt_cost == pytest.approx(1000 * entry["input_cost_per_token"])
    assert completion_cost == pytest.approx(100 * entry["output_cost_per_token"])
    assert cached_cost == pytest.approx(
        600 * entry["input_cost_per_token"]
        + 400 * entry.get("cache_read_input_token_cost", entry["input_cost_per_token"])
    )
    assert litellm.get_model_info(model)["litellm_provider"] == "nousresearch"


def test_provider_is_available_in_add_model_form() -> None:
    path: Final = Path(litellm.__file__).parent / "proxy/public_endpoints/provider_create_fields.json"
    provider: Final = next(row for row in json.loads(path.read_text()) if row["litellm_provider"] == "nousresearch")
    assert provider["default_model_placeholder"] in MODELS
    assert {field["key"]: field["required"] for field in provider["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
