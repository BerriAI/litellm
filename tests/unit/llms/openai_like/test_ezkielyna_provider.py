import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_ezkielyna",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-v4.1-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Ezkielyna"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_ezkielyna_is_a_registered_provider():
    assert litellm.LlmProviders.EZKIELYNA.value == "ezkielyna"
    assert "ezkielyna" in litellm.provider_list
    assert "ezkielyna" in litellm.constants.openai_compatible_providers


def test_ezkielyna_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("EZKIELYNA_API_KEY", "ezkielyna-test-key")
    monkeypatch.delenv("EZKIELYNA_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="ezkielyna/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "ezkielyna"
    assert api_key == "ezkielyna-test-key"
    assert api_base == "https://api.ezkielyna.store/v1"


def test_ezkielyna_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("EZKIELYNA_API_KEY", "ezkielyna-env-key")
    monkeypatch.setenv("EZKIELYNA_API_BASE", "https://ezkielyna.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="ezkielyna/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://ezkielyna.internal.example/v1",
        api_key="ezkielyna-explicit-key",
    )

    assert provider == "ezkielyna"
    assert api_key == "ezkielyna-explicit-key"
    assert api_base == "https://ezkielyna.internal.example/v1"


def test_ezkielyna_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("EZKIELYNA_API_KEY", "ezkielyna-env-key")
    monkeypatch.setenv("EZKIELYNA_API_BASE", "https://ezkielyna.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="ezkielyna/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "ezkielyna"
    assert api_base == "https://ezkielyna.env.example/v1"


def test_ezkielyna_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("EZKIELYNA_API_KEY", "ezkielyna-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://api.ezkielyna.store/v1",
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "ezkielyna"
    assert api_key == "ezkielyna-env-key"
    assert api_base == "https://api.ezkielyna.store/v1"


def test_ezkielyna_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    ezkielyna = next(provider for provider in providers if provider["litellm_provider"] == "ezkielyna")

    assert ezkielyna["provider"] == "EZKIELYNA"
    assert ezkielyna["provider_display_name"] == "Ezkielyna"
    assert ezkielyna["default_model_placeholder"] == "ezkielyna/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in ezkielyna["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_ezkielyna_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.ezkielyna.store/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="ezkielyna/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="ezkielyna-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.ezkielyna.store/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer ezkielyna-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Ezkielyna"


def test_ezkielyna_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.ezkielyna.store/v1/responses").respond(
            200,
            json={
                "id": "resp_ezkielyna",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "deepseek-v4.1-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_ezkielyna",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Ezkielyna", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="ezkielyna/deepseek-v4.1-flash",
            input="Say hello",
            api_key="ezkielyna-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.ezkielyna.store/v1/responses"
    assert request.headers["authorization"] == "Bearer ezkielyna-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Ezkielyna"


@pytest.mark.asyncio
async def test_ezkielyna_anthropic_messages_request_is_bridged_to_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.ezkielyna.store/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="ezkielyna/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="ezkielyna-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer ezkielyna-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Ezkielyna"
