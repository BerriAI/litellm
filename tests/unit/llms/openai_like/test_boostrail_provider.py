import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_BASE: Final = "https://api.boostrail.com/v1"

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_boostrail",
    "object": "chat.completion",
    "created": 1_791_000_000,
    "model": "deepseek-v4.1-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from BoostRail"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_boostrail_is_a_registered_provider():
    assert litellm.LlmProviders.BOOSTRAIL.value == "boostrail"
    assert "boostrail" in litellm.provider_list
    assert "boostrail" in litellm.constants.openai_compatible_providers


def test_boostrail_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOOSTRAIL_API_KEY", "boostrail-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="boostrail/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "boostrail"
    assert api_key == "boostrail-test-key"
    assert api_base == _BASE


def test_boostrail_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOOSTRAIL_API_KEY", "boostrail-env-key")
    monkeypatch.setenv("BOOSTRAIL_API_BASE", "https://boostrail.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="boostrail/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://boostrail.internal.example/v1",
        api_key="boostrail-explicit-key",
    )

    assert provider == "boostrail"
    assert api_key == "boostrail-explicit-key"
    assert api_base == "https://boostrail.internal.example/v1"


def test_boostrail_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOOSTRAIL_API_KEY", "boostrail-env-key")
    monkeypatch.setenv("BOOSTRAIL_API_BASE", "https://boostrail.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="boostrail/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "boostrail"
    assert api_base == "https://boostrail.env.example/v1"


def test_boostrail_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOOSTRAIL_API_KEY", "boostrail-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=_BASE,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "boostrail"
    assert api_key == "boostrail-env-key"
    assert api_base == _BASE


def test_boostrail_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    boostrail: Final = next(provider for provider in providers if provider["litellm_provider"] == "boostrail")

    assert boostrail["provider"] == "BOOSTRAIL"
    assert boostrail["provider_display_name"] == "BoostRail"
    assert boostrail["default_model_placeholder"] == "boostrail/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in boostrail["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_boostrail_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["boostrail"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["boostrail"]["endpoints"] == expected


def test_boostrail_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="boostrail/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="boostrail-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer boostrail-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from BoostRail"


def test_boostrail_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/responses").respond(
            200,
            json={
                "id": "resp_boostrail",
                "object": "response",
                "created_at": 1_791_000_000,
                "model": "deepseek-v4.1-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_boostrail",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from BoostRail", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="boostrail/deepseek-v4.1-flash",
            input="Say hello",
            api_key="boostrail-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_BASE}/responses"
    assert request.headers["authorization"] == "Bearer boostrail-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from BoostRail"


@pytest.mark.asyncio
async def test_boostrail_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/messages").respond(
            200,
            json={
                "id": "msg_boostrail",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "Hello from BoostRail"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="boostrail/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="boostrail-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_BASE}/messages"
    assert request.headers["authorization"] == "Bearer boostrail-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from BoostRail"
