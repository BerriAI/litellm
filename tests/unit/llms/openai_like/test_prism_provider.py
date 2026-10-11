import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_prism_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("PRISM_API_KEY", "prism-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="prism/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-flash"
    assert provider == "prism"
    assert api_key == "prism-test-key"
    assert api_base == "https://api.prisminference.com/v1"


def test_prism_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("PRISM_API_KEY", "prism-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="prism/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base="https://prism.internal.example/v1",
        api_key="prism-explicit-key",
    )

    assert provider == "prism"
    assert api_key == "prism-explicit-key"
    assert api_base == "https://prism.internal.example/v1"


PRISM_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("prism/")))


@pytest.mark.parametrize("model", PRISM_MODELS)
def test_prism_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="prism",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "prism"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_native_streaming"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    assert litellm.supports_vision(model) is model_info["supports_vision"]


def test_prism_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    prism_entries = {name: entry for name, entry in cost_map.items() if name.startswith("prism/")}

    assert tuple(sorted(prism_entries)) == PRISM_MODELS
    assert prism_entries
    assert all("supports_vision" in entry for entry in prism_entries.values())
    assert prism_entries == {name: backup[name] for name in prism_entries}


def test_prism_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    prism = next(provider for provider in providers if provider["litellm_provider"] == "prism")

    assert prism["provider"] == "PRISM"
    assert prism["provider_display_name"] == "Prism"
    assert prism["default_model_placeholder"] == "prism/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in prism["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_prism_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["prism"]["endpoints"] == {
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
        "a2a": False,
    }


def test_prism_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.prisminference.com/v1/responses").respond(
            200,
            json={
                "id": "resp_prism",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "deepseek-v4-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_prism",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Prism", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="prism/deepseek-v4-flash",
            input="Say hello",
            api_key="prism-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.prisminference.com/v1/responses"
    assert request.headers["authorization"] == "Bearer prism-test-key"
    assert body["model"] == "deepseek-v4-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Prism"


@pytest.mark.asyncio
async def test_prism_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.prisminference.com/v1/messages").respond(
            200,
            json={
                "id": "msg_prism",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [{"type": "text", "text": "Hello from Prism"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="prism/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="prism-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.prisminference.com/v1/messages"
    assert request.headers["authorization"] == "Bearer prism-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Prism"
