"""
Tests for aiand provider configuration and integration.
"""

import json
from pathlib import Path
from typing import Final, get_args

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.types.llms.openai import REASONING_EFFORT


def test_aiand_provider_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("AIAND_API_KEY", "aiand-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="aiand/deepseek-ai/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-ai/deepseek-v4.1-flash"
    assert provider == "aiand"
    assert api_key == "aiand-test-key"
    assert api_base == "https://api.aiand.com/v1"


def test_aiand_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("AIAND_API_KEY", "aiand-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="aiand/deepseek-ai/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://aiand.internal.example/v1",
        api_key="aiand-explicit-key",
    )

    assert provider == "aiand"
    assert api_key == "aiand-explicit-key"
    assert api_base == "https://aiand.internal.example/v1"


def test_aiand_url_autodetection(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("AIAND_API_KEY", "aiand-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://api.aiand.com/v1",
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "aiand"
    assert api_key == "aiand-test-key"
    assert api_base == "https://api.aiand.com/v1"


AIAND_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("aiand/")))


@pytest.mark.parametrize("model", AIAND_MODELS)
def test_aiand_model_cost_and_capabilities(model: str) -> None:
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="aiand",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "aiand"
    assert model_info["mode"] == "chat"
    assert type(model_info["supports_function_calling"]) is bool
    assert type(model_info["supports_native_streaming"]) is bool
    assert type(model_info["supports_reasoning"]) is bool
    assert type(model_info["supports_response_schema"]) is bool
    assert litellm.supports_vision(model) is model_info["supports_vision"]


def test_aiand_reasoning_effort_levels_are_valid() -> None:
    known_efforts: Final = frozenset(get_args(REASONING_EFFORT))
    for model in AIAND_MODELS:
        model_info = litellm.get_model_info(model)
        levels = model_info.get("reasoning_effort_levels", [])
        assert set(levels) <= known_efforts, f"{model} declares unknown reasoning efforts"
        if levels:
            assert model_info["supports_reasoning"] is True, (
                f"{model} declares reasoning efforts without supports_reasoning"
            )


def test_aiand_backup_registry_mirrors_cost_map() -> None:
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    aiand_entries = {name: entry for name, entry in cost_map.items() if name.startswith("aiand/")}

    assert tuple(sorted(aiand_entries)) == AIAND_MODELS
    assert aiand_entries
    assert all("supports_vision" in entry for entry in aiand_entries.values())
    assert aiand_entries == {name: backup[name] for name in aiand_entries}


def test_aiand_models_listed_by_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = Path(litellm.__file__).parent
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    aiand_keys = {name for name in backup if name.startswith("aiand/")}
    assert aiand_keys

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(litellm, "models_by_provider", dict(litellm.models_by_provider))
    litellm.add_known_models()

    assert "aiand" in litellm.models_by_provider
    assert set(litellm.models_by_provider["aiand"]) == aiand_keys
    assert set(litellm.get_valid_models(custom_llm_provider="aiand")) == aiand_keys


def test_aiand_is_available_in_add_model_form() -> None:
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    aiand = next(provider for provider in providers if provider["litellm_provider"] == "aiand")

    assert aiand["provider"] == "AIAND"
    assert aiand["provider_display_name"] == "ai&"
    assert aiand["default_model_placeholder"] == "aiand/deepseek-ai/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in aiand["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_aiand_supported_endpoints() -> None:
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["aiand"]["endpoints"] == {
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
        "interactions": False,
    }


def test_aiand_registered_for_text_completion() -> None:
    assert "aiand" in litellm.openai_text_completion_compatible_providers


def test_aiand_provider_declares_completions_endpoint() -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider_config: Final = JSONProviderRegistry.get("aiand")

    assert provider_config is not None
    assert "/v1/completions" in provider_config.supported_endpoints


def test_aiand_chat_completion_request() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.aiand.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_aiand",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "deepseek-ai/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from aiand"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="aiand/deepseek-ai/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="aiand-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.aiand.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer aiand-test-key"
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from aiand"


def test_aiand_responses_request() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.aiand.com/v1/responses").respond(
            200,
            json={
                "id": "resp_aiand",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "deepseek-ai/deepseek-v4.1-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_aiand",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from aiand", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="aiand/deepseek-ai/deepseek-v4.1-flash",
            input="Say hello",
            api_key="aiand-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.aiand.com/v1/responses"
    assert request.headers["authorization"] == "Bearer aiand-test-key"
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from aiand"


def test_aiand_text_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAND_API_KEY", "aiand-test-key")

    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.aiand.com/v1/completions").respond(
            200,
            json={
                "id": "cmpl_aiand",
                "object": "text_completion",
                "created": 1_789_550_000,
                "model": "deepseek-ai/deepseek-v4.1-flash",
                "choices": [
                    {
                        "text": "Hello from aiand",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            },
        )
        response: Final = litellm.text_completion(
            model="aiand/deepseek-ai/deepseek-v4.1-flash",
            prompt="Say hello",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.aiand.com/v1/completions"
    assert request.headers["authorization"] == "Bearer aiand-test-key"
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert body["prompt"] == "Say hello"
    assert response.object == "text_completion"
    assert response.choices[0].text == "Hello from aiand"


@pytest.mark.asyncio
async def test_aiand_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.aiand.com/v1/messages").respond(
            200,
            json={
                "id": "msg_aiand",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-ai/deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "Hello from aiand"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="aiand/deepseek-ai/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="aiand-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.aiand.com/v1/messages"
    assert request.headers["authorization"] == "Bearer aiand-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from aiand"
