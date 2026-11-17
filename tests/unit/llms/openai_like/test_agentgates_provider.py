import json
from functools import partial
from pathlib import Path
from typing import Final, Literal

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.agentgates.ai/api/inference/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_agentgates",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Agentgates", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": True,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}
AGENTGATES_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("agentgates/")))


def test_agentgates_is_a_registered_provider():
    assert litellm.LlmProviders.AGENTGATES.value == "agentgates"
    assert "agentgates" in litellm.provider_list
    assert "agentgates" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_agentgates_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENTGATES_API_KEY", "ag_test_key")
    monkeypatch.delenv("AGENTGATES_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="agentgates/gpt-oss-120b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-oss-120b"
    assert provider == "agentgates"
    assert api_key == "ag_test_key"
    assert api_base == _DEFAULT_API_BASE


def test_agentgates_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENTGATES_API_KEY", "ag_env_key")
    monkeypatch.setenv("AGENTGATES_API_BASE", "https://agentgates.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="agentgates/gpt-oss-120b",
        custom_llm_provider=None,
        api_base="https://agentgates.internal.example/v1",
        api_key="ag_explicit_key",
    )

    assert provider == "agentgates"
    assert api_key == "ag_explicit_key"
    assert api_base == "https://agentgates.internal.example/v1"


def test_agentgates_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENTGATES_API_KEY", "ag_env_key")
    monkeypatch.setenv("AGENTGATES_API_BASE", "https://agentgates.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="agentgates/gpt-oss-120b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "agentgates"
    assert api_base == "https://agentgates.env.example/v1"


def test_agentgates_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENTGATES_API_KEY", "ag_env_key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-oss-120b",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "gpt-oss-120b"
    assert provider == "agentgates"
    assert api_key == "ag_env_key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", AGENTGATES_MODELS)
def test_agentgates_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_function_calling,
        supports_reasoning,
        supports_tool_choice,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="agentgates",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]
    assert model_info["litellm_provider"] == "agentgates"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_tool_choice(model) is entry["supports_tool_choice"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_agentgates_wildcard_expands_to_every_agentgates_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("agentgates/*")) == set(AGENTGATES_MODELS)


def test_agentgates_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    agentgates_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("agentgates/")}

    assert agentgates_entries
    assert tuple(sorted(agentgates_entries)) == AGENTGATES_MODELS
    assert agentgates_entries == {name: backup[name] for name in agentgates_entries}


def test_agentgates_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    agentgates: Final = next(provider for provider in providers if provider["litellm_provider"] == "agentgates")

    assert agentgates["provider"] == "AGENTGATES"
    assert agentgates["provider_display_name"] == "Agentgates"
    assert agentgates["default_model_placeholder"] in AGENTGATES_MODELS
    assert {field["key"]: field["required"] for field in agentgates["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_agentgates_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["agentgates"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["agentgates"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("reasoning_effort", ["low", "max"])
def test_agentgates_chat_completion_request(reasoning_effort: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="agentgates/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort=reasoning_effort,
            api_key="ag_test_key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer ag_test_key"
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == reasoning_effort
    assert response.choices[0].message.content == "Hello from Agentgates"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_agentgates_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="agentgates/gpt-oss-120b",
            input="Say hello",
            api_key="ag_test_key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer ag_test_key"
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Agentgates"


@pytest.mark.asyncio
async def test_agentgates_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="agentgates/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="ag_test_key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer ag_test_key"
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Agentgates" for block in response["content"])


@pytest.mark.parametrize("provider_api", ("chat", "responses", "messages"))
def test_missing_agentgates_key_does_not_use_other_provider_credentials(
    provider_api: Literal["chat", "responses", "messages"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENTGATES_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setattr(litellm, "openai_key", "unrelated-openai-key")
    call: Final = {
        "chat": partial(
            litellm.completion, model="agentgates/gpt-oss-120b", messages=[{"role": "user", "content": "Say hello"}]
        ),
        "responses": partial(litellm.responses, model="agentgates/gpt-oss-120b", input="Say hello"),
        "messages": partial(
            litellm.anthropic.messages.create,
            model="agentgates/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
        ),
    }[provider_api]
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="AGENTGATES_API_KEY"):
            call()

    assert route.call_count == 0


def test_agentgates_env_key_is_sent_without_the_openai_organization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTGATES_API_KEY", "ag_env_key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "unrelated-openai-org")
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(model="agentgates/gpt-oss-120b", messages=[{"role": "user", "content": "Say hello"}])

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer ag_env_key"
    assert "openai-organization" not in request.headers
