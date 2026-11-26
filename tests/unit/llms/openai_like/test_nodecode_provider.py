import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.nodecode.ai/v1"
_MODEL: Final = "nodecode/nodecode-cyber-v1"
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
NODECODE_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("nodecode/")))


def test_nodecode_is_a_registered_provider():
    assert litellm.LlmProviders.NODECODE.value == "nodecode"
    assert "nodecode" in litellm.provider_list
    assert "nodecode" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_nodecode_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NODECODE_API_KEY", "sk-nc-test-key")
    monkeypatch.delenv("NODECODE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "nodecode-cyber-v1"
    assert provider == "nodecode"
    assert api_key == "sk-nc-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_nodecode_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NODECODE_API_KEY", "sk-nc-env-key")
    monkeypatch.setenv("NODECODE_API_BASE", "https://nodecode.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base="https://nodecode.internal.example/v1",
        api_key="sk-nc-explicit-key",
    )

    assert provider == "nodecode"
    assert api_key == "sk-nc-explicit-key"
    assert api_base == "https://nodecode.internal.example/v1"


def test_nodecode_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NODECODE_API_KEY", "sk-nc-env-key")
    monkeypatch.setenv("NODECODE_API_BASE", "https://nodecode.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "nodecode"
    assert api_base == "https://nodecode.env.example/v1"


def test_nodecode_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NODECODE_API_KEY", "sk-nc-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="nodecode-cyber-v1",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "nodecode-cyber-v1"
    assert provider == "nodecode"
    assert api_key == "sk-nc-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", NODECODE_MODELS)
def test_nodecode_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_function_calling,
        supports_prompt_caching,
        supports_reasoning,
        supports_response_schema,
        supports_tool_choice,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="nodecode",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="nodecode",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "nodecode"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_tool_choice(model) is entry["supports_tool_choice"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_prompt_caching(model) is entry["supports_prompt_caching"]
    assert supports_response_schema(model) is entry["supports_response_schema"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_nodecode_wildcard_expands_to_every_nodecode_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("nodecode/*")) == set(NODECODE_MODELS)


def test_nodecode_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("nodecode/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("nodecode/a-model-not-in-the-cost-map") is True


def test_nodecode_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    nodecode_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("nodecode/")}

    assert nodecode_entries
    assert tuple(sorted(nodecode_entries)) == NODECODE_MODELS
    assert nodecode_entries == {name: backup[name] for name in nodecode_entries}


def test_nodecode_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    nodecode: Final = next(provider for provider in providers if provider["litellm_provider"] == "nodecode")

    assert nodecode["provider"] == "NODECODE"
    assert nodecode["provider_display_name"] == "Nodecode"
    assert nodecode["default_model_placeholder"] in NODECODE_MODELS
    assert {field["key"]: field["required"] for field in nodecode["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_nodecode_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["nodecode"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["nodecode"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("reasoning_effort", ["none", "max"])
def test_nodecode_chat_completion_request(reasoning_effort: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_nodecode",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "nodecode-cyber-v1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from Nodecode",
                            "reasoning_content": "Say hi",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort=reasoning_effort,
            api_key="sk-nc-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_DEFAULT_API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-nc-test-key"
    assert body["model"] == "nodecode-cyber-v1"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == reasoning_effort
    assert response.choices[0].message.content == "Hello from Nodecode"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_nodecode_responses_request_goes_to_native_responses():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_nodecode",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "nodecode-cyber-v1",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_nodecode",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Nodecode", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=_MODEL,
            input="Say hello",
            api_key="sk-nc-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_DEFAULT_API_BASE}/responses"
    assert request.headers["authorization"] == "Bearer sk-nc-test-key"
    assert body["model"] == "nodecode-cyber-v1"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Nodecode"


@pytest.mark.asyncio
async def test_nodecode_anthropic_messages_request_goes_to_native_messages(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_nodecode",
                "type": "message",
                "role": "assistant",
                "model": "nodecode-cyber-v1",
                "content": [{"type": "text", "text": "Hello from Nodecode"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-nc-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_DEFAULT_API_BASE}/messages"
    assert request.headers["authorization"] == "Bearer sk-nc-test-key"
    assert body["model"] == "nodecode-cyber-v1"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Nodecode"
