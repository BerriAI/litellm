import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

CORALBRICKS_MODEL: Final = "coralbricks/deepseek-v4.1-flash-fast"
CORALBRICKS_MODEL_ID: Final = "deepseek-v4.1-flash-fast"


def test_coralbricks_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORALBRICKS_API_KEY", "coralbricks-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="coralbricks/glm-5.3-fast",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3-fast"
    assert provider == "coralbricks"
    assert api_key == "coralbricks-test-key"
    assert api_base == "https://inference.coralbricks.ai/v1"


def test_coralbricks_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORALBRICKS_API_KEY", "coralbricks-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="coralbricks/glm-5.3-fast",
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
    assert coralbricks["default_model_placeholder"] == "coralbricks/glm-5.3-fast"
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


@dataclass(frozen=True)
class CoralBricksRates:
    input: float
    output: float
    cache_write: float
    cache_read: float


def _coralbricks_rates() -> CoralBricksRates:
    row: Final = litellm.model_cost[CORALBRICKS_MODEL]
    assert row["litellm_provider"] == "coralbricks"
    return CoralBricksRates(
        input=float(row["input_cost_per_token"]),
        output=float(row["output_cost_per_token"]),
        cache_write=float(row["cache_creation_input_token_cost"]),
        cache_read=float(row["cache_read_input_token_cost"]),
    )


def _expected_cost(
    rates: CoralBricksRates,
    uncached_input_tokens: int,
    cached_read_tokens: int,
    cache_write_tokens: int,
    billed_output_tokens: int,
) -> float:
    return (
        uncached_input_tokens * rates.input
        + cached_read_tokens * rates.cache_read
        + cache_write_tokens * rates.cache_write
        + billed_output_tokens * rates.output
    )


def test_coralbricks_chat_completion_cost_includes_cache_write_and_free_cached_read():
    rates: Final = _coralbricks_rates()
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
                    "completion_tokens_details": {"reasoning_tokens": 12},
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
            rates,
            uncached_input_tokens=4357 - 4352 - 5,
            cached_read_tokens=4352,
            cache_write_tokens=5,
            billed_output_tokens=20,
        )
    )


def test_coralbricks_responses_cost_includes_cache_write_and_free_cached_read():
    rates: Final = _coralbricks_rates()
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
                    "output_tokens_details": {"reasoning_tokens": 12},
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
            rates,
            uncached_input_tokens=4357 - 4352 - 5,
            cached_read_tokens=4352,
            cache_write_tokens=5,
            billed_output_tokens=20,
        )
    )


@pytest.mark.asyncio
async def test_coralbricks_anthropic_messages_cost_includes_cache_write_and_free_cached_read(
    monkeypatch: pytest.MonkeyPatch,
):
    rates: Final = _coralbricks_rates()
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
            rates,
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
    backup_map: Final = json.loads(
        (Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json").read_text()
    )
    main_keys: Final = {key for key in main_map if key.startswith("coralbricks/")}
    backup_keys: Final = {key for key in backup_map if key.startswith("coralbricks/")}
    assert main_keys and main_keys == backup_keys
    for key in main_keys:
        assert backup_map[key] == main_map[key]


def test_coralbricks_cost_map_rows_declare_same_endpoints_as_providers_json():
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = providers_json["coralbricks"]["supported_endpoints"]
    cost_map_paths: Final = (
        Path(litellm.__file__).parent.parent / "model_prices_and_context_window.json",
        Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json",
    )
    for cost_map_path in cost_map_paths:
        cost_map: Final = json.loads(cost_map_path.read_text())
        coralbricks_rows: Final = {
            key: row for key, row in cost_map.items() if key.startswith("coralbricks/")
        }
        assert coralbricks_rows
        for row in coralbricks_rows.values():
            assert row["supported_endpoints"] == declared


def _billed_chat_cost(model_id: str) -> float:
    with respx.mock() as upstream:
        upstream.post("https://inference.coralbricks.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_coralbricks",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 4400,
                    "completion_tokens": 20,
                    "total_tokens": 4420,
                    "prompt_tokens_details": {
                        "cached_tokens": 4352,
                        "cache_write_tokens": 5,
                        "billable_cache_write_tokens": 5,
                    },
                },
            },
        )
        response: Final = litellm.completion(
            model=f"coralbricks/{model_id}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="coralbricks-test-key",
        )
    return litellm.completion_cost(completion_response=response, model=f"coralbricks/{model_id}")


@pytest.mark.parametrize(
    ("deprecated_slug", "current_slug"),
    (("glm-5.3-fp4", "glm-5.3-fast"), ("deepseek-v4.1-flash-fast-fp4", "deepseek-v4.1-flash-fast")),
)
def test_coralbricks_deprecated_slug_bills_like_its_current_slug(deprecated_slug: str, current_slug: str):
    current_cost: Final = _billed_chat_cost(current_slug)
    assert current_cost > 0
    assert _billed_chat_cost(deprecated_slug) == pytest.approx(current_cost)


def test_coralbricks_streaming_chat_cost_includes_cache_write_and_free_cached_read():
    rates: Final = _coralbricks_rates()
    usage_chunk: Final = {
        "id": "chatcmpl_coralbricks",
        "object": "chat.completion.chunk",
        "created": 1_789_550_000,
        "model": CORALBRICKS_MODEL_ID,
        "choices": [],
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
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }
    stream_body: Final = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl_coralbricks",
                "object": "chat.completion.chunk",
                "created": 1_789_550_000,
                "model": CORALBRICKS_MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}],
            }
        )
        + "\n\n"
        + "data: "
        + json.dumps(
            {
                "id": "chatcmpl_coralbricks",
                "object": "chat.completion.chunk",
                "created": 1_789_550_000,
                "model": CORALBRICKS_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n"
        + "data: "
        + json.dumps(usage_chunk)
        + "\n\n"
        + "data: [DONE]\n\n"
    )
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inference.coralbricks.ai/v1/chat/completions").respond(
            200, headers={"content-type": "text/event-stream"}, content=stream_body
        )
        response: Final = litellm.completion(
            model=CORALBRICKS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="coralbricks-test-key",
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: Final = list(response)

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert body["stream"] is True

    built: Final = litellm.stream_chunk_builder(
        chunks, messages=[{"role": "user", "content": "Say hello"}]
    )
    assert built.usage.prompt_tokens_details.cached_tokens == 4352
    cost: Final = litellm.completion_cost(completion_response=built, model=CORALBRICKS_MODEL)
    assert cost == pytest.approx(
        _expected_cost(
            rates,
            uncached_input_tokens=0,
            cached_read_tokens=4352,
            cache_write_tokens=5,
            billed_output_tokens=20,
        )
    )
