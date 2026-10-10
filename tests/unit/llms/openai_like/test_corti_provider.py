import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_corti",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "corti-s1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Corti"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_corti_is_a_registered_provider():
    assert litellm.LlmProviders.CORTI.value == "corti"
    assert "corti" in litellm.provider_list
    assert "corti" in litellm.constants.openai_compatible_providers


def test_corti_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORTI_API_KEY", "corti-test-key")
    monkeypatch.delenv("CORTI_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="corti/corti-s1",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "corti-s1"
    assert provider == "corti"
    assert api_key == "corti-test-key"
    assert api_base == "https://ai.eu.corti.app/v1"


def test_corti_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORTI_API_KEY", "corti-env-key")
    monkeypatch.setenv("CORTI_API_BASE", "https://corti.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="corti/corti-s1",
        custom_llm_provider=None,
        api_base="https://ai.us.corti.app/v1",
        api_key="corti-explicit-key",
    )

    assert provider == "corti"
    assert api_key == "corti-explicit-key"
    assert api_base == "https://ai.us.corti.app/v1"


def test_corti_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORTI_API_KEY", "corti-env-key")
    monkeypatch.setenv("CORTI_API_BASE", "https://ai.us.corti.app/v1")

    _, provider, _, api_base = get_llm_provider(
        model="corti/corti-s1",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "corti"
    assert api_base == "https://ai.us.corti.app/v1"


def test_corti_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORTI_API_KEY", "corti-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="corti-s1",
        custom_llm_provider=None,
        api_base="https://ai.eu.corti.app/v1",
        api_key=None,
    )

    assert model == "corti-s1"
    assert provider == "corti"
    assert api_key == "corti-env-key"
    assert api_base == "https://ai.eu.corti.app/v1"


def test_corti_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    corti: Final = next(provider for provider in providers if provider["litellm_provider"] == "corti")

    assert corti["provider"] == "CORTI"
    assert corti["provider_display_name"] == "Corti"
    assert corti["default_model_placeholder"] == "corti/corti-s1"
    assert {field["key"]: field["required"] for field in corti["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_corti_supported_endpoints():
    expected: Final = {
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
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["corti"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["corti"]["endpoints"] == expected


def test_corti_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://ai.eu.corti.app/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="corti/corti-s1",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="corti-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://ai.eu.corti.app/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer corti-test-key"
    assert body["model"] == "corti-s1"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Corti"


def test_corti_chat_completion_maps_unsupported_params():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://ai.eu.corti.app/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="corti/corti-s1",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="corti-test-key",
            max_completion_tokens=64,
            temperature=1.5,
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body
    assert body["temperature"] == 1.0


@pytest.mark.parametrize("model", ["corti-s1", "corti-s1-instant", "corti-s1-mini", "corti-s1-mini-instant"])
def test_corti_chat_completion_cost_comes_from_the_price_map(model: str):
    with respx.mock() as upstream:
        upstream.post("https://ai.eu.corti.app/v1/chat/completions").respond(
            200, json={**_CHAT_COMPLETION, "model": model}
        )
        response: Final = litellm.completion(
            model=f"corti/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="corti-test-key",
        )

    row: Final = litellm.model_cost[f"corti/{model}"]
    expected: Final = 4 * row["input_cost_per_token"] + 3 * row["output_cost_per_token"]
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)
    assert expected > 0


def test_corti_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://ai.eu.corti.app/v1/responses").respond(
            200,
            json={
                "id": "resp_corti",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "corti-s1",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_corti",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Corti", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="corti/corti-s1",
            input="Say hello",
            api_key="corti-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://ai.eu.corti.app/v1/responses"
    assert request.headers["authorization"] == "Bearer corti-test-key"
    assert body["model"] == "corti-s1"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Corti"


@pytest.mark.asyncio
async def test_corti_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://ai.eu.corti.app/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="corti/corti-s1",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="corti-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer corti-test-key"
    assert body["model"] == "corti-s1"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Corti"
