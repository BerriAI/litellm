import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.lyceum.technology/openai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_EMBEDDINGS_URL: Final = f"{_DEFAULT_API_BASE}/embeddings"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_lyceum",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "z-ai/glm-5.3-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Lyceum", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}
_EMBEDDING: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "qwen/qwen3-embedding-8b",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": True,
    "embeddings": True,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}
LYCEUM_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("lyceum/")))
LYCEUM_CHAT_MODELS: Final = tuple(name for name in LYCEUM_MODELS if litellm.model_cost[name]["mode"] == "chat")
LYCEUM_EMBEDDING_MODELS: Final = tuple(name for name in LYCEUM_MODELS if litellm.model_cost[name]["mode"] == "embedding")


def test_lyceum_is_a_registered_provider():
    assert litellm.LlmProviders.LYCEUM.value == "lyceum"
    assert "lyceum" in litellm.provider_list
    assert "lyceum" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints
    assert LYCEUM_CHAT_MODELS
    assert LYCEUM_EMBEDDING_MODELS
    assert set(litellm.models_by_provider["lyceum"]) == set(LYCEUM_MODELS)


def test_lyceum_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LYCEUM_API_KEY", "lyceum-test-key")
    monkeypatch.delenv("LYCEUM_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="lyceum/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "z-ai/glm-5.3-flash"
    assert provider == "lyceum"
    assert api_key == "lyceum-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_lyceum_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LYCEUM_API_KEY", "lyceum-env-key")
    monkeypatch.setenv("LYCEUM_API_BASE", "https://lyceum.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="lyceum/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://lyceum.internal.example/v1",
        api_key="lyceum-explicit-key",
    )

    assert provider == "lyceum"
    assert api_key == "lyceum-explicit-key"
    assert api_base == "https://lyceum.internal.example/v1"


def test_lyceum_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LYCEUM_API_KEY", "lyceum-env-key")
    monkeypatch.setenv("LYCEUM_API_BASE", "https://lyceum.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="lyceum/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "lyceum"
    assert api_base == "https://lyceum.env.example/v1"


def test_lyceum_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LYCEUM_API_KEY", "lyceum-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "z-ai/glm-5.3-flash"
    assert provider == "lyceum"
    assert api_key == "lyceum-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", LYCEUM_CHAT_MODELS)
def test_lyceum_chat_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_function_calling,
        supports_prompt_caching,
        supports_reasoning,
        supports_tool_choice,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="lyceum",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="lyceum",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert cached_prompt_cost < prompt_cost
    assert model_info["max_output_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "lyceum"
    assert supports_function_calling(model) is True
    assert supports_tool_choice(model) is True
    assert supports_prompt_caching(model) is True
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_vision(model) is entry["supports_vision"]


@pytest.mark.parametrize("model", LYCEUM_EMBEDDING_MODELS)
def test_lyceum_embedding_model_cost(model: str):
    from litellm.cost_calculator import cost_per_token

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        custom_llm_provider="lyceum",
    )

    assert entry["mode"] == "embedding"
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert prompt_cost > 0
    assert completion_cost == 0


@pytest.mark.parametrize("model", ("lyceum/z-ai/glm-5.3", "lyceum/z-ai/glm-5.3-instant"))
def test_lyceum_instant_variant_turns_reasoning_off(model: str):
    from litellm.utils import supports_reasoning

    assert supports_reasoning(model) is model.endswith("glm-5.3")


@pytest.mark.usefixtures("local_model_cost_map")
def test_lyceum_wildcard_expands_to_every_lyceum_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("lyceum/*")) == set(LYCEUM_MODELS)


def test_lyceum_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    lyceum_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("lyceum/")}

    assert lyceum_entries
    assert tuple(sorted(lyceum_entries)) == LYCEUM_MODELS
    assert lyceum_entries == {name: backup[name] for name in lyceum_entries}


def test_lyceum_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    lyceum: Final = next(provider for provider in providers if provider["litellm_provider"] == "lyceum")

    assert lyceum["provider"] == "LYCEUM"
    assert lyceum["provider_display_name"] == "Lyceum"
    assert lyceum["default_model_placeholder"] in LYCEUM_MODELS
    assert {field["key"]: field["required"] for field in lyceum["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_lyceum_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["lyceum"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["lyceum"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_lyceum_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="lyceum/z-ai/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort="high",
            api_key="lyceum-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer lyceum-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "high"
    assert response.choices[0].message.content == "Hello from Lyceum"
    assert response.choices[0].message.reasoning_content == "Say hi"
    assert response.usage.prompt_tokens_details.cached_tokens == 4


def test_lyceum_completion_cost_uses_the_cost_map_entry():
    model: Final = "lyceum/z-ai/glm-5.3-flash"
    entry: Final = litellm.model_cost[model]
    with respx.mock() as upstream:
        upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="lyceum-test-key",
        )

    expected: Final = (
        6 * entry["input_cost_per_token"]
        + 4 * entry["cache_read_input_token_cost"]
        + 3 * entry["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_lyceum_embedding_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_EMBEDDINGS_URL).respond(200, json=_EMBEDDING)
        response: Final = litellm.embedding(
            model="lyceum/qwen/qwen3-embedding-8b",
            input=["Say hello"],
            api_key="lyceum-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _EMBEDDINGS_URL
    assert request.headers["authorization"] == "Bearer lyceum-test-key"
    assert body["model"] == "qwen/qwen3-embedding-8b"
    assert body["input"] == ["Say hello"]
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_lyceum_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="lyceum/z-ai/glm-5.3-flash",
            input="Say hello",
            api_key="lyceum-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer lyceum-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Lyceum"


@pytest.mark.asyncio
async def test_lyceum_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="lyceum/z-ai/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="lyceum-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer lyceum-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Lyceum" for block in response["content"])
