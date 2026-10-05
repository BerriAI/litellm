import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

CORALBRICKS_MODEL: Final = "coralbricks/glm-5.3-flash-fp4"
CORALBRICKS_MODEL_ID: Final = "glm-5.3-flash-fp4"


def test_coralbricks_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORALBRICKS_API_KEY", "coralbricks-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="coralbricks/glm-5.3-fp4",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3-fp4"
    assert provider == "coralbricks"
    assert api_key == "coralbricks-test-key"
    assert api_base == "https://inference.coralbricks.ai/v1"


def test_coralbricks_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORALBRICKS_API_KEY", "coralbricks-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="coralbricks/glm-5.3-fp4",
        custom_llm_provider=None,
        api_base="https://coralbricks.internal.example/v1",
        api_key="coralbricks-explicit-key",
    )

    assert provider == "coralbricks"
    assert api_key == "coralbricks-explicit-key"
    assert api_base == "https://coralbricks.internal.example/v1"


def test_coralbricks_is_available_in_add_model_form():
    fields_path: Final = (
        Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    )
    providers: Final = json.loads(fields_path.read_text())
    coralbricks: Final = next(
        provider for provider in providers if provider["litellm_provider"] == "coralbricks"
    )

    assert coralbricks["provider"] == "CORALBRICKS"
    assert coralbricks["provider_display_name"] == "CoralBricks"
    assert coralbricks["default_model_placeholder"] == "coralbricks/glm-5.3-fp4"
    assert {field["key"]: field["required"] for field in coralbricks["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_coralbricks_endpoint_support_matches_providers_json():
    matrix_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    endpoints: Final = json.loads(matrix_path.read_text())["providers"]["coralbricks"]["endpoints"]
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = set(providers_json["coralbricks"]["supported_endpoints"])

    supported_from_matrix: Final = {
        "/v1/chat/completions": endpoints["chat_completions"],
        "/v1/responses": endpoints["responses"],
        "/v1/messages": endpoints["messages"],
    }
    assert {endpoint for endpoint, supported in supported_from_matrix.items() if supported} == declared


def test_coralbricks_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inference.coralbricks.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_coralbricks",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": CORALBRICKS_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from CoralBricks"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=CORALBRICKS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="coralbricks-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://inference.coralbricks.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer coralbricks-test-key"
    assert body["model"] == CORALBRICKS_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from CoralBricks"


def test_coralbricks_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inference.coralbricks.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_coralbricks",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": CORALBRICKS_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_coralbricks",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Hello from CoralBricks", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=CORALBRICKS_MODEL,
            input="Say hello",
            api_key="coralbricks-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://inference.coralbricks.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer coralbricks-test-key"
    assert body["model"] == CORALBRICKS_MODEL_ID
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from CoralBricks"


@pytest.mark.asyncio
async def test_coralbricks_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inference.coralbricks.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_coralbricks",
                "type": "message",
                "role": "assistant",
                "model": CORALBRICKS_MODEL_ID,
                "content": [{"type": "text", "text": "Hello from CoralBricks"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=CORALBRICKS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="coralbricks-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://inference.coralbricks.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer coralbricks-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == CORALBRICKS_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from CoralBricks"


def _coralbricks_row() -> dict:
    model_cost: Final = litellm.model_cost
    row: Final = model_cost[CORALBRICKS_MODEL]
    assert row["litellm_provider"] == "coralbricks"
    return dict(row)


def _expected_cost(
    row: dict,
    uncached_input_tokens: int,
    cached_read_tokens: int,
    cache_write_tokens: int,
    billed_output_tokens: int,
) -> float:
    return (
        uncached_input_tokens * row["input_cost_per_token"]
        + cached_read_tokens * row["cache_read_input_token_cost"]
        + cache_write_tokens * row["cache_creation_input_token_cost"]
        + billed_output_tokens * row["output_cost_per_token"]
    )


def test_coralbricks_chat_completion_cost_includes_cache_write_and_free_cached_read():
    row: Final = _coralbricks_row()
    with respx.mock() as upstream:
        upstream.post("https://inference.coralbricks.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_coralbricks",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": CORALBRICKS_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4357,
                    "completion_tokens": 20,
                    "total_tokens": 4377,
                    "prompt_tokens_details": {
                        "cached_tokens": 4352,
                        "cache_write_tokens": 5,
                        "billable_cache_write_tokens": 5,
                        "cache_write_blocks": 3,
                    },
                    "completion_tokens_details": {"reasoning_tokens": 21},
                },
            },
        )
        response: Final = litellm.completion(
            model=CORALBRICKS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="coralbricks-test-key",
        )

    cost: Final = litellm.completion_cost(completion_response=response, model=CORALBRICKS_MODEL)
    assert cost == pytest.approx(
        _expected_cost(
            row,
            uncached_input_tokens=4357 - 4352 - 5,
            cached_read_tokens=4352,
            cache_write_tokens=5,
            billed_output_tokens=21,
        )
    )


def test_coralbricks_responses_cost_includes_cache_write_and_free_cached_read():
    row: Final = _coralbricks_row()
    with respx.mock() as upstream:
        upstream.post("https://inference.coralbricks.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_coralbricks",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": CORALBRICKS_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_coralbricks",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": 4357,
                    "output_tokens": 20,
                    "total_tokens": 4377,
                    "input_tokens_details": {
                        "cached_tokens": 4352,
                        "cache_write_tokens": 5,
                        "billable_cache_write_tokens": 5,
                    },
                    "output_tokens_details": {"reasoning_tokens": 23},
                },
            },
        )
        response: Final = litellm.responses(
            model=CORALBRICKS_MODEL,
            input="Say hello",
            api_key="coralbricks-test-key",
        )

    cost: Final = litellm.completion_cost(completion_response=response, model=CORALBRICKS_MODEL)
    assert cost == pytest.approx(
        _expected_cost(
            row,
            uncached_input_tokens=4357 - 4352 - 5,
            cached_read_tokens=4352,
            cache_write_tokens=5,
            billed_output_tokens=23,
        )
    )


@pytest.mark.asyncio
async def test_coralbricks_anthropic_messages_cost_includes_cache_write_and_free_cached_read(
    monkeypatch: pytest.MonkeyPatch,
):
    row: Final = _coralbricks_row()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        upstream.post("https://inference.coralbricks.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_coralbricks",
                "type": "message",
                "role": "assistant",
                "model": CORALBRICKS_MODEL_ID,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_read_input_tokens": 4352,
                    "cache_creation_input_tokens": 5,
                    "output_tokens": 20,
                },
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=CORALBRICKS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="coralbricks-test-key",
        )

    cost: Final = litellm.completion_cost(completion_response=response, model=CORALBRICKS_MODEL)
    assert cost == pytest.approx(
        _expected_cost(
            row,
            uncached_input_tokens=0,
            cached_read_tokens=4352,
            cache_write_tokens=5,
            billed_output_tokens=20,
        )
    )


def test_coralbricks_cost_map_backup_mirrors_main():
    main_map: Final = json.loads(
        (Path(litellm.__file__).parent.parent / "model_prices_and_context_window.json").read_text()
    )
    backup_map: Final = litellm.model_cost
    for key in (key for key in main_map if key.startswith("coralbricks/")):
        assert backup_map[key] == main_map[key]


def test_coralbricks_cost_map_rows_declare_same_endpoints_as_providers_json():
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = providers_json["coralbricks"]["supported_endpoints"]
    for key, row in litellm.model_cost.items():
        if key.startswith("coralbricks/"):
            assert row["supported_endpoints"] == declared
