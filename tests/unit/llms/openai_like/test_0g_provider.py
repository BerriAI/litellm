import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm._version import version as litellm_version
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://router-api.0g.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_0g",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "glm-5.3-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from 0G", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
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
}


def test_0g_is_a_registered_provider():
    assert litellm.LlmProviders.ZEROG.value == "0g"
    assert "0g" in litellm.provider_list
    assert "0g" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_0g_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OG_API_KEY", "0g-test-key")
    monkeypatch.delenv("OG_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="0g/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3-flash"
    assert provider == "0g"
    assert api_key == "0g-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_0g_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OG_API_KEY", "0g-env-key")
    monkeypatch.setenv("OG_API_BASE", "https://0g.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="0g/glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://0g.internal.example/v1",
        api_key="0g-explicit-key",
    )

    assert provider == "0g"
    assert api_key == "0g-explicit-key"
    assert api_base == "https://0g.internal.example/v1"


def test_0g_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OG_API_KEY", "0g-env-key")
    monkeypatch.setenv("OG_API_BASE", "https://0g.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="0g/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "0g"
    assert api_base == "https://0g.env.example/v1"


def test_0g_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OG_API_KEY", "0g-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="glm-5.3-flash",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "glm-5.3-flash"
    assert provider == "0g"
    assert api_key == "0g-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_0g_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("0g/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("0g/a-model-not-in-the-cost-map") is True


def test_0g_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    zerog: Final = next(provider for provider in providers if provider["litellm_provider"] == "0g")

    assert zerog["provider"] == "ZEROG"
    assert zerog["provider_display_name"] == "0G Private Computer"
    assert zerog["default_model_placeholder"] == "0g/glm-5.3-flash"
    assert {field["key"]: field["required"] for field in zerog["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_0g_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["0g"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["0g"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("reasoning_effort", ["low", "high"])
def test_0g_chat_completion_request(reasoning_effort: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="0g/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort=reasoning_effort,
            api_key="0g-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer 0g-test-key"
    assert request.headers["user-agent"].startswith(f"litellm/{litellm_version} Python/")
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == reasoning_effort
    assert response.choices[0].message.content == "Hello from 0G"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_0g_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="0g/glm-5.3-flash",
            input="Say hello",
            api_key="0g-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer 0g-test-key"
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from 0G"


@pytest.mark.asyncio
async def test_0g_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="0g/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="0g-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer 0g-test-key"
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from 0G" for block in response["content"])


_MISSING_KEY_CALLS: Final = {
    "chat": lambda: litellm.completion(model="0g/glm-5.3-flash", messages=[{"role": "user", "content": "hi"}]),
    "text_completion": lambda: litellm.text_completion(model="0g/glm-5.3-flash", prompt="hi"),
    "responses": lambda: litellm.responses(model="0g/glm-5.3-flash", input="hi"),
}


@pytest.mark.parametrize("openai_key", [None, "sk-openai-not-for-0g"])
@pytest.mark.parametrize("call", list(_MISSING_KEY_CALLS))
def test_0g_missing_key_never_falls_back_to_openai_key(
    call: str, openai_key: str | None, monkeypatch: pytest.MonkeyPatch
):
    # Without OG_API_KEY, the OpenAI-compatible path would fall back to OPENAI_API_KEY and
    # send that unrelated credential to 0G. The call must fail before any request instead,
    # and name OG_API_KEY ("0G_API_KEY", the slug upper-cased, is not a valid variable name).
    monkeypatch.delenv("OG_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)
    if openai_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route(host="router-api.0g.ai").respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError) as excinfo:
            _MISSING_KEY_CALLS[call]()

    assert route.call_count == 0
    assert "OG_API_KEY" in str(excinfo.value)
    assert "0G_API_KEY" not in str(excinfo.value)
