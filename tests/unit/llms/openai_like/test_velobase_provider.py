import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

VELOBASE_API_BASE: Final = "https://velobase.io/api/v1"
VELOBASE_MODEL_ID: Final = "deepseek/deepseek-v4.1-flash"
VELOBASE_MODEL: Final = f"velobase/{VELOBASE_MODEL_ID}"
VELOBASE_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("velobase/")))


def _chat_completion_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_velobase",
        "object": "chat.completion",
        "created": 1_794_000_000,
        "model": VELOBASE_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Velobase"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def test_velobase_is_a_registered_provider():
    assert litellm.LlmProviders.VELOBASE.value == "velobase"
    assert "velobase" in litellm.provider_list
    assert "velobase" in litellm.constants.openai_compatible_providers
    assert VELOBASE_API_BASE in litellm.constants.openai_compatible_endpoints


def test_velobase_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VELOBASE_API_KEY", "velobase-env-key")
    monkeypatch.delenv("VELOBASE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=VELOBASE_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == VELOBASE_MODEL_ID
    assert provider == "velobase"
    assert api_key == "velobase-env-key"
    assert api_base == VELOBASE_API_BASE


def test_velobase_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VELOBASE_API_KEY", "velobase-env-key")
    monkeypatch.setenv("VELOBASE_API_BASE", "https://velobase.env.example/api/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="velobase/anthropic/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base="https://velobase.internal.example/api/v1",
        api_key="velobase-explicit-key",
    )

    assert provider == "velobase"
    assert api_key == "velobase-explicit-key"
    assert api_base == "https://velobase.internal.example/api/v1"


def test_velobase_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VELOBASE_API_KEY", "velobase-env-key")
    monkeypatch.setenv("VELOBASE_API_BASE", "https://velobase.env.example/api/v1")

    _, provider, _, api_base = get_llm_provider(
        model=VELOBASE_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "velobase"
    assert api_base == "https://velobase.env.example/api/v1"


def test_velobase_cost_map_rows_only_list_registered_endpoints():
    provider: Final = JSONProviderRegistry.get("velobase")
    assert provider is not None
    registered: Final = set(provider.supported_endpoints)

    assert registered == {"/v1/chat/completions", "/v1/responses", "/v1/messages"}
    assert VELOBASE_MODELS
    assert all(set(litellm.model_cost[name]["supported_endpoints"]) <= registered for name in VELOBASE_MODELS)


@pytest.mark.parametrize("model", VELOBASE_MODELS)
def test_velobase_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_reasoning, supports_vision

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000,
        completion_tokens=1_000,
        custom_llm_provider="velobase",
    )
    model_info: Final = litellm.get_model_info(model)

    assert entry["litellm_provider"] == "velobase"
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000)
    assert entry["cache_read_input_token_cost"] < entry["input_cost_per_token"]
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_vision(model) is entry["supports_vision"]


def test_velobase_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("velobase/")}

    assert tuple(sorted(entries)) == VELOBASE_MODELS
    assert entries == {name: backup[name] for name in entries}


@pytest.mark.usefixtures("local_model_cost_map")
def test_velobase_wildcard_expands_to_every_velobase_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("velobase/*")) == set(VELOBASE_MODELS)


def test_velobase_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    velobase: Final = next(provider for provider in providers if provider["litellm_provider"] == "velobase")

    assert velobase["provider"] == "VELOBASE"
    assert velobase["provider_display_name"] == "Velobase"
    assert velobase["default_model_placeholder"] in VELOBASE_MODELS
    assert {field["key"]: field["required"] for field in velobase["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_velobase_supported_endpoints_matrix():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["velobase"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["velobase"]["endpoints"]

    assert backup == root
    assert root["chat_completions"] and root["responses"] and root["messages"]
    assert not root["embeddings"] and not root["image_generations"] and not root["audio_transcriptions"]


def test_velobase_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{VELOBASE_API_BASE}/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        response: Final = litellm.completion(
            model=VELOBASE_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=16,
            api_key="velobase-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer velobase-test-key"
    assert body["model"] == VELOBASE_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 16
    assert response.choices[0].message.content == "Hello from Velobase"


def test_velobase_chat_completion_reads_key_and_base_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VELOBASE_API_KEY", "velobase-env-key")
    monkeypatch.setenv("VELOBASE_API_BASE", "https://velobase.internal.example/api/v1")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://velobase.internal.example/api/v1/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        litellm.completion(model=VELOBASE_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer velobase-env-key"


def test_velobase_chat_completion_cost_bills_cached_tokens_at_cache_read_rate():
    row: Final = litellm.model_cost[VELOBASE_MODEL]
    with respx.mock() as upstream:
        upstream.post(f"{VELOBASE_API_BASE}/chat/completions").respond(
            200,
            json=_chat_completion_body(
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 800},
                }
            ),
        )
        response: Final = litellm.completion(
            model=VELOBASE_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="velobase-test-key",
        )

    assert litellm.completion_cost(completion_response=response, model=VELOBASE_MODEL) == pytest.approx(
        200 * row["input_cost_per_token"] + 800 * row["cache_read_input_token_cost"] + 20 * row["output_cost_per_token"]
    )


def test_velobase_responses_request_is_sent_natively():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{VELOBASE_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_velobase",
                "object": "response",
                "created_at": 1_794_000_000,
                "model": VELOBASE_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_velobase",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Velobase", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=VELOBASE_MODEL,
            input="Say hello",
            api_key="velobase-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer velobase-test-key"
    assert body["model"] == VELOBASE_MODEL_ID
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Velobase"


@pytest.mark.asyncio
async def test_velobase_anthropic_messages_request_is_sent_natively():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{VELOBASE_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_velobase",
                "type": "message",
                "role": "assistant",
                "model": "anthropic/claude-haiku-5.5",
                "content": [{"type": "text", "text": "Hello from Velobase"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="velobase/anthropic/claude-haiku-5.5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="velobase-test-key",
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer velobase-test-key"
    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Velobase"
