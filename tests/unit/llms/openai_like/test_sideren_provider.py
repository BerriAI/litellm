import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_MODEL: Final = "@sdn/deepseek-v4.1-flash-1017"
_CHAT_URL: Final = "https://api.sideren.io/v1/chat/completions"
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_sideren",
    "object": "chat.completion",
    "created": 1_792_000_000,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Sideren"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def _sent_chat_request(model: str, **params: object) -> tuple[str, str, dict[str, object]]:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model=f"sideren/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sdn-test-key",
            **params,
        )
    request: Final = route.calls.last.request
    return str(request.url), request.headers["authorization"], json.loads(request.content)


def test_sideren_is_a_registered_provider():
    assert litellm.LlmProviders.SIDEREN.value == "sideren"
    assert "sideren" in litellm.provider_list
    assert "sideren" in litellm.constants.openai_compatible_providers


def test_sideren_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SIDEREN_API_KEY", "sdn-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=f"sideren/{_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == _MODEL
    assert provider == "sideren"
    assert api_key == "sdn-test-key"
    assert api_base == "https://api.sideren.io/v1"


def test_sideren_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SIDEREN_API_KEY", "sdn-env-key")
    monkeypatch.setenv("SIDEREN_API_BASE", "https://sideren.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"sideren/{_MODEL}",
        custom_llm_provider=None,
        api_base="https://sideren.internal.example/v1",
        api_key="sdn-explicit-key",
    )

    assert provider == "sideren"
    assert api_key == "sdn-explicit-key"
    assert api_base == "https://sideren.internal.example/v1"


def test_sideren_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SIDEREN_API_KEY", "sdn-env-key")
    monkeypatch.setenv("SIDEREN_API_BASE", "https://sideren.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=f"sideren/{_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "sideren"
    assert api_base == "https://sideren.env.example/v1"


def test_sideren_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SIDEREN_API_KEY", "sdn-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base="https://api.sideren.io/v1",
        api_key=None,
    )

    assert model == _MODEL
    assert provider == "sideren"
    assert api_key == "sdn-env-key"
    assert api_base == "https://api.sideren.io/v1"


def test_sideren_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    sideren = next(provider for provider in providers if provider["litellm_provider"] == "sideren")

    assert sideren["provider"] == "SIDEREN"
    assert sideren["provider_display_name"] == "Sideren"
    assert sideren["default_model_placeholder"] == f"sideren/{_MODEL}"
    assert {field["key"]: field["required"] for field in sideren["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_sideren_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["sideren"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["sideren"]["endpoints"] == expected


def test_chat_request_reaches_sideren_with_the_public_model_id():
    url, authorization, body = _sent_chat_request(_MODEL)

    assert url == _CHAT_URL
    assert authorization == "Bearer sdn-test-key"
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]


def test_tools_reach_sideren_for_a_model_without_a_cost_map_row():
    _, _, body = _sent_chat_request("@sdn/glm-5.3", tools=[_WEATHER_TOOL])

    assert body["tools"] == [_WEATHER_TOOL]


def test_responses_request_goes_to_sideren_natively():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sideren.io/v1/responses").respond(
            200,
            json={
                "id": "resp_sideren",
                "object": "response",
                "created_at": 1_792_000_000,
                "model": _MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_sideren",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Sideren", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(model=f"sideren/{_MODEL}", input="Say hello", api_key="sdn-test-key")

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer sdn-test-key"
    assert body["model"] == _MODEL
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Sideren"


@pytest.mark.asyncio
async def test_anthropic_messages_request_goes_to_sideren_natively(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sideren.io/v1/messages").respond(
            200,
            json={
                "id": "msg_sideren",
                "type": "message",
                "role": "assistant",
                "model": _MODEL,
                "content": [{"type": "text", "text": "Hello from Sideren"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"sideren/{_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sdn-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer sdn-test-key"
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Sideren"
