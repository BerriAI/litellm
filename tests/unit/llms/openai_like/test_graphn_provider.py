import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages

GRAPHN_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("graphn/")))


def test_graphn_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHN_API_KEY", "fake-environment-key")
    assert get_llm_provider(model="graphn/glm-5.3") == (
        "glm-5.3",
        "graphn",
        "fake-environment-key",
        "https://model.graphn.ai/v1",
    )
    assert get_llm_provider(
        model="graphn/glm-5.3", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("glm-5.3", "graphn", "fake-explicit-key", "https://gateway.example/v1")


def test_graphn_api_base_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHN_API_KEY", "fake-environment-key")
    monkeypatch.setenv("GRAPHN_API_BASE", "https://graphn.internal.example/v1")
    assert get_llm_provider(model="graphn/glm-5.3")[3] == "https://graphn.internal.example/v1"


def test_graphn_chat_forwards_reasoning_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://model.graphn.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-5.3",
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
        )
        response: Final = litellm.completion(
            model="graphn/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "glm-5.3"
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["graphn/glm-5.3"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_graphn_uses_native_responses_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://model.graphn.ai/v1/responses").respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "deepseek-v4.1-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="graphn/deepseek-v4.1-flash",
            api_key="fake-test-key",
            input="Say hello",
            store=False,
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(route.calls.last.request.content)["model"] == "deepseek-v4.1-flash"
    assert response.output[0].content[0].text == "hello"


@pytest.mark.asyncio
async def test_graphn_uses_native_messages_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://model.graphn.ai/v1/messages").respond(
            200,
            json={
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="graphn/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="fake-test-key",
        )
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(request.content)["model"] == "deepseek-v4.1-flash"
    assert response["content"][0]["text"] == "hello"


def test_graphn_backup_registry_mirrors_cost_map() -> None:
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("graphn/")}
    assert tuple(sorted(entries)) == GRAPHN_MODELS
    assert entries == {name: backup[name] for name in entries}


@pytest.mark.parametrize("model", GRAPHN_MODELS)
def test_graphn_model_limits_and_cache_pricing_are_consistent(model: str) -> None:
    info: Final = litellm.get_model_info(model)
    assert info["max_tokens"] == info["max_output_tokens"] <= info["max_input_tokens"]
    assert info["supports_prompt_caching"] is ("cache_read_input_token_cost" in litellm.model_cost[model])
    assert litellm.supports_vision(model) is info["supports_vision"]


def test_graphn_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    assert GRAPHN_MODELS
    messages: Final = [{"role": "user", "content": "hello " * 500}]
    budget: Final = 100
    for model in GRAPHN_MODELS:
        trimmed: Final = trim_messages(
            messages,
            model=model,
            trim_ratio=budget / litellm.model_cost[model]["max_input_tokens"],
        )
        assert litellm.token_counter(model=model, messages=messages) > budget
        assert litellm.token_counter(model=model, messages=trimmed) <= budget
