import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_opper",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "claude-sonnet-4-6",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Opper"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_opper_is_a_registered_provider():
    assert litellm.LlmProviders.OPPER.value == "opper"
    assert "opper" in litellm.provider_list
    assert "opper" in litellm.constants.openai_compatible_providers


def test_opper_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPPER_API_KEY", "opper-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="opper/claude-sonnet-4-6",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-4-6"
    assert provider == "opper"
    assert api_key == "opper-test-key"
    assert api_base == "https://api.opper.ai/v3/compat"


def test_opper_pinned_route_keeps_its_provider_segment(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPPER_API_KEY", "opper-test-key")

    model, provider, _, _ = get_llm_provider(
        model="opper/anthropic/claude-sonnet-4-6",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "anthropic/claude-sonnet-4-6"
    assert provider == "opper"


def test_opper_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPPER_API_KEY", "opper-env-key")
    monkeypatch.setenv("OPPER_API_BASE", "https://opper.env.example/v3/compat")

    _, provider, api_key, api_base = get_llm_provider(
        model="opper/claude-sonnet-4-6",
        custom_llm_provider=None,
        api_base="https://opper.internal.example/v3/compat",
        api_key="opper-explicit-key",
    )

    assert provider == "opper"
    assert api_key == "opper-explicit-key"
    assert api_base == "https://opper.internal.example/v3/compat"


def test_opper_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPPER_API_KEY", "opper-env-key")
    monkeypatch.setenv("OPPER_API_BASE", "https://opper.env.example/v3/compat")

    _, provider, _, api_base = get_llm_provider(
        model="opper/claude-sonnet-4-6",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "opper"
    assert api_base == "https://opper.env.example/v3/compat"


def test_opper_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPPER_API_KEY", "opper-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="claude-sonnet-4-6",
        custom_llm_provider=None,
        api_base="https://api.opper.ai/v3/compat",
        api_key=None,
    )

    assert model == "claude-sonnet-4-6"
    assert provider == "opper"
    assert api_key == "opper-env-key"
    assert api_base == "https://api.opper.ai/v3/compat"


def test_opper_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    opper = next(provider for provider in providers if provider["litellm_provider"] == "opper")

    assert opper["provider"] == "OPPER"
    assert opper["provider_display_name"] == "Opper"
    assert opper["default_model_placeholder"] == "opper/claude-sonnet-4-6"
    assert {field["key"]: field["required"] for field in opper["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_opper_supported_endpoints():
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
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["opper"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["opper"]["endpoints"] == expected


def test_opper_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.opper.ai/v3/compat/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="opper/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="opper-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.opper.ai/v3/compat/chat/completions"
    assert request.headers["authorization"] == "Bearer opper-test-key"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Opper"


def test_opper_chat_completion_sends_pinned_route_unchanged():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.opper.ai/v3/compat/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model="opper/anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="opper-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "anthropic/claude-sonnet-4-6"


def test_opper_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.opper.ai/v3/compat/responses").respond(
            200,
            json={
                "id": "resp_opper",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "claude-sonnet-4-6",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_opper",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Opper", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="opper/claude-sonnet-4-6",
            input="Say hello",
            api_key="opper-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.opper.ai/v3/compat/responses"
    assert request.headers["authorization"] == "Bearer opper-test-key"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Opper"


@pytest.mark.asyncio
async def test_opper_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.opper.ai/v3/compat/v1/messages").respond(
            200,
            json={
                "id": "msg_opper",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Hello from Opper"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="opper/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="opper-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.opper.ai/v3/compat/v1/messages"
    assert request.headers["authorization"] == "Bearer opper-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Opper"
