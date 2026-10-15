import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

ENGY_API_BASE: Final = "https://api.engy.ai/v1"
ENGY_MODEL: Final = "engy/kimi-k3"
ENGY_MODEL_ID: Final = "kimi-k3"


def test_engy_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ENGY_API_KEY", "engy-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=ENGY_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == ENGY_MODEL_ID
    assert provider == "engy"
    assert api_key == "engy-env-key"
    assert api_base == ENGY_API_BASE


def test_engy_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ENGY_API_KEY", "engy-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=ENGY_MODEL,
        custom_llm_provider=None,
        api_base="https://engy.internal.example/v1",
        api_key="engy-explicit-key",
    )

    assert provider == "engy"
    assert api_key == "engy-explicit-key"
    assert api_base == "https://engy.internal.example/v1"


def test_engy_provider_reads_base_from_env(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ENGY_API_KEY", "engy-env-key")
    monkeypatch.setenv("ENGY_API_BASE", "https://engy.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=ENGY_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "engy"
    assert api_base == "https://engy.internal.example/v1"


def test_engy_provider_resolves_from_api_base():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    _, provider, _, api_base = get_llm_provider(
        model=ENGY_MODEL_ID,
        custom_llm_provider=None,
        api_base=ENGY_API_BASE,
        api_key="engy-explicit-key",
    )

    assert provider == "engy"
    assert api_base == ENGY_API_BASE


def test_engy_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    engy: Final = next(provider for provider in providers if provider["litellm_provider"] == "engy")

    assert engy["provider"] == "ENGY"
    assert engy["provider_display_name"] == "Engy"
    assert engy["default_model_placeholder"].startswith("engy/")
    assert {field["key"]: field["required"] for field in engy["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_engy_endpoint_support_matches_providers_json():
    matrix_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    endpoints: Final = json.loads(matrix_path.read_text())["providers"]["engy"]["endpoints"]
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = set(providers_json["engy"]["supported_endpoints"])

    supported_from_matrix: Final = {
        "/v1/chat/completions": endpoints["chat_completions"],
        "/v1/responses": endpoints["responses"],
        "/v1/messages": endpoints["messages"],
    }
    assert {endpoint for endpoint, supported in supported_from_matrix.items() if supported} == declared


def test_engy_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ENGY_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_engy",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": ENGY_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Engy"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=ENGY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="engy-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{ENGY_API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer engy-test-key"
    assert body["model"] == ENGY_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Engy"


def test_engy_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ENGY_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_engy",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": ENGY_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_engy",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Engy", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=ENGY_MODEL,
            input="Say hello",
            api_key="engy-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{ENGY_API_BASE}/responses"
    assert request.headers["authorization"] == "Bearer engy-test-key"
    assert body["model"] == ENGY_MODEL_ID
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Engy"


@pytest.mark.asyncio
async def test_engy_anthropic_messages_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ENGY_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_engy",
                "type": "message",
                "role": "assistant",
                "model": ENGY_MODEL_ID,
                "content": [{"type": "text", "text": "Hello from Engy"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=ENGY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="engy-test-key",
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{ENGY_API_BASE}/messages"
    assert request.headers["authorization"] == "Bearer engy-test-key"
    assert body["model"] == ENGY_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Engy"


def test_engy_chat_completion_cost_bills_cached_tokens_at_cache_read_rate():
    row: Final = litellm.model_cost[ENGY_MODEL]
    with respx.mock() as upstream:
        upstream.post(f"{ENGY_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_engy",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": ENGY_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
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
            model=ENGY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="engy-test-key",
        )

    assert row["cache_read_input_token_cost"] < row["input_cost_per_token"]
    assert litellm.completion_cost(completion_response=response, model=ENGY_MODEL) == pytest.approx(
        200 * row["input_cost_per_token"] + 800 * row["cache_read_input_token_cost"] + 20 * row["output_cost_per_token"]
    )


def test_engy_models_are_registered_in_the_cost_map():
    catalog: Final = {name for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "engy"}
    assert len(catalog) == 8
    assert all(name.startswith("engy/") for name in catalog)

    info: Final = litellm.get_model_info(model=ENGY_MODEL)
    assert info["litellm_provider"] == "engy"
    assert info["mode"] == "chat"


def test_engy_cost_map_backup_mirrors_main():
    main_map: Final = json.loads(
        (Path(litellm.__file__).parent.parent / "model_prices_and_context_window.json").read_text()
    )
    backup_map: Final = json.loads(
        (Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json").read_text()
    )
    main_keys: Final = {key for key in main_map if key.startswith("engy/")}
    backup_keys: Final = {key for key in backup_map if key.startswith("engy/")}
    assert main_keys and main_keys == backup_keys
    for key in main_keys:
        assert backup_map[key] == main_map[key]


def test_engy_cost_map_rows_declare_same_endpoints_as_providers_json():
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = providers_json["engy"]["supported_endpoints"]
    cost_map_paths: Final = (
        Path(litellm.__file__).parent.parent / "model_prices_and_context_window.json",
        Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json",
    )
    for cost_map_path in cost_map_paths:
        cost_map: Final = json.loads(cost_map_path.read_text())
        engy_rows: Final = {key: row for key, row in cost_map.items() if key.startswith("engy/")}
        assert engy_rows
        for row in engy_rows.values():
            assert row["supported_endpoints"] == declared
