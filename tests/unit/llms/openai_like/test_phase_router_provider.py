import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://router.phase.xyz/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_EMBEDDINGS_URL: Final = f"{_DEFAULT_API_BASE}/embeddings"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_phase_router",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Phase Router"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": 0.000001},
}
_EMBEDDING_RESPONSE: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "bge-m3",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
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
PHASE_ROUTER_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("phase_router/")))


def test_phase_router_is_a_registered_provider():
    assert litellm.LlmProviders.PHASE_ROUTER.value == "phase_router"
    assert "phase_router" in litellm.provider_list
    assert "phase_router" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_phase_router_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PHASE_API_KEY", "phk_test")
    monkeypatch.delenv("PHASE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="phase_router/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "auto"
    assert provider == "phase_router"
    assert api_key == "phk_test"
    assert api_base == _DEFAULT_API_BASE


def test_phase_router_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PHASE_API_KEY", "phk_env")
    monkeypatch.setenv("PHASE_API_BASE", "https://phase.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="phase_router/auto",
        custom_llm_provider=None,
        api_base="https://phase.internal.example/v1",
        api_key="phk_explicit",
    )

    assert provider == "phase_router"
    assert api_key == "phk_explicit"
    assert api_base == "https://phase.internal.example/v1"


def test_phase_router_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PHASE_API_KEY", "phk_env")
    monkeypatch.setenv("PHASE_API_BASE", "https://phase.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="phase_router/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "phase_router"
    assert api_base == "https://phase.env.example/v1"


def test_phase_router_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PHASE_API_KEY", "phk_env")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-oss-120b",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "gpt-oss-120b"
    assert provider == "phase_router"
    assert api_key == "phk_env"
    assert api_base == _DEFAULT_API_BASE


def test_phase_router_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    phase_router = next(provider for provider in providers if provider["litellm_provider"] == "phase_router")

    assert phase_router["provider"] == "PHASE_ROUTER"
    assert phase_router["provider_display_name"] == "Phase Router"
    assert phase_router["default_model_placeholder"] in PHASE_ROUTER_MODELS
    assert {field["key"]: field["required"] for field in phase_router["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


@pytest.mark.parametrize("model", PHASE_ROUTER_MODELS)
def test_phase_router_model_cost(model: str):
    from litellm.cost_calculator import cost_per_token

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="phase_router",
    )
    model_info: Final = litellm.get_model_info(model)

    assert entry["input_cost_per_token"] > 0
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert model_info["litellm_provider"] == "phase_router"
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]


def test_phase_router_catalog_rates_are_converted_from_micro_usd_per_million():
    # Phase Router publishes phase.input_per_m / phase.output_per_m in micro-USD per 1M tokens:
    # gpt-oss-120b lists 38500 / 176800, i.e. $0.0385 / $0.1768 per 1M tokens.
    entry: Final = litellm.model_cost["phase_router/gpt-oss-120b"]

    assert entry["input_cost_per_token"] == pytest.approx(38_500 / 1e12)
    assert entry["output_cost_per_token"] == pytest.approx(176_800 / 1e12)


def test_phase_router_routing_classes_are_not_priced():
    for routing_class in ("auto", "eco", "premium", "agentic"):
        assert f"phase_router/{routing_class}" not in litellm.model_cost


@pytest.mark.usefixtures("local_model_cost_map")
def test_phase_router_wildcard_expands_to_every_phase_router_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("phase_router/*")) == set(PHASE_ROUTER_MODELS)


def test_phase_router_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("phase_router/")}

    assert entries
    assert tuple(sorted(entries)) == PHASE_ROUTER_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_phase_router_supported_endpoints():
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    for path in (backup_path, root_path):
        endpoints = json.loads(path.read_text())["providers"]["phase_router"]["endpoints"]
        assert {key: endpoints[key] for key in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_phase_router_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="phase_router/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="phk_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer phk_test"
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Phase Router"


def test_phase_router_embeddings_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_EMBEDDINGS_URL).respond(200, json=_EMBEDDING_RESPONSE)
        response: Final = litellm.embedding(
            model="phase_router/bge-m3",
            input=["Say hello"],
            api_key="phk_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer phk_test"
    assert body["model"] == "bge-m3"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_phase_router_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="phase_router/gpt-oss-120b",
            input="Say hello",
            api_key="phk_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer phk_test"
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Phase Router"


@pytest.mark.asyncio
async def test_phase_router_anthropic_messages_request_is_bridged_to_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="phase_router/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="phk_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer phk_test"
    assert body["model"] == "gpt-oss-120b"
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Phase Router"
