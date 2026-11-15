import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

TADLY_MODEL: Final = "tadly/qwen3.5"
TADLY_MODEL_ID: Final = "qwen3.5"


def test_tadly_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("TADLY_API_KEY", "tadly-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="tadly/qwen3.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen3.5"
    assert provider == "tadly"
    assert api_key == "tadly-test-key"
    assert api_base == "https://api.tadly.dev/v1"


def test_tadly_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("TADLY_API_KEY", "tadly-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="tadly/qwen3.5",
        custom_llm_provider=None,
        api_base="https://tadly.internal.example/v1",
        api_key="tadly-explicit-key",
    )

    assert provider == "tadly"
    assert api_key == "tadly-explicit-key"
    assert api_base == "https://tadly.internal.example/v1"


def test_tadly_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    tadly: Final = next(provider for provider in providers if provider["litellm_provider"] == "tadly")

    assert tadly["provider"] == "TADLY"
    assert tadly["provider_display_name"] == "Tadly"
    assert tadly["default_model_placeholder"] == "tadly/qwen3.5"
    assert {field["key"]: field["required"] for field in tadly["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_tadly_endpoint_support_matches_providers_json():
    matrix_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    endpoints: Final = json.loads(matrix_path.read_text())["providers"]["tadly"]["endpoints"]
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = set(providers_json["tadly"]["supported_endpoints"])

    supported_from_matrix: Final = {
        "/v1/chat/completions": endpoints["chat_completions"],
        "/v1/responses": endpoints["responses"],
        "/v1/messages": endpoints["messages"],
    }
    assert {endpoint for endpoint, supported in supported_from_matrix.items() if supported} == declared


def test_tadly_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.tadly.dev/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_tadly",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": TADLY_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Tadly"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=TADLY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="tadly-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.tadly.dev/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer tadly-test-key"
    assert body["model"] == TADLY_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Tadly"


def test_tadly_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.tadly.dev/v1/responses").respond(
            200,
            json={
                "id": "resp_tadly",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": TADLY_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_tadly",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Tadly", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=TADLY_MODEL,
            input="Say hello",
            api_key="tadly-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.tadly.dev/v1/responses"
    assert request.headers["authorization"] == "Bearer tadly-test-key"
    assert body["model"] == TADLY_MODEL_ID
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Tadly"


@pytest.mark.asyncio
async def test_tadly_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.tadly.dev/v1/messages").respond(
            200,
            json={
                "id": "msg_tadly",
                "type": "message",
                "role": "assistant",
                "model": TADLY_MODEL_ID,
                "content": [{"type": "text", "text": "Hello from Tadly"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=TADLY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="tadly-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.tadly.dev/v1/messages"
    assert request.headers["authorization"] == "Bearer tadly-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == TADLY_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Tadly"


def test_tadly_cost_per_token_uses_the_cost_map_rates():
    rates: Final = litellm.model_cost[TADLY_MODEL]
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=TADLY_MODEL,
        prompt_tokens=1_000,
        completion_tokens=500,
    )

    assert rates["litellm_provider"] == "tadly"
    assert prompt_cost == pytest.approx(1_000 * rates["input_cost_per_token"])
    assert completion_cost == pytest.approx(500 * rates["output_cost_per_token"])
