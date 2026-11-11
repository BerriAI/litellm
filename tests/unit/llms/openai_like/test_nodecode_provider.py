import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

NODECODE_BASE_URL: Final = "https://api.nodecode.ai/v1"
NODECODE_MODEL: Final = "nodecode/nodecode-cyber-v1"


def test_nodecode_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NODECODE_API_KEY", "sk-nc-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=NODECODE_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "nodecode-cyber-v1"
    assert provider == "nodecode"
    assert api_key == "sk-nc-test-key"
    assert api_base == NODECODE_BASE_URL


def test_nodecode_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NODECODE_API_KEY", "sk-nc-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=NODECODE_MODEL,
        custom_llm_provider=None,
        api_base="https://nodecode.internal.example/v1",
        api_key="sk-nc-explicit-key",
    )

    assert provider == "nodecode"
    assert api_key == "sk-nc-explicit-key"
    assert api_base == "https://nodecode.internal.example/v1"


def test_nodecode_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    nodecode = next(provider for provider in providers if provider["litellm_provider"] == "nodecode")

    assert nodecode["provider"] == "NODECODE"
    assert nodecode["provider_display_name"] == "Nodecode"
    assert nodecode["default_model_placeholder"] == NODECODE_MODEL
    assert {field["key"]: field["required"] for field in nodecode["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_nodecode_cost_map_row(local_model_cost_map: None):
    info = litellm.get_model_info(NODECODE_MODEL)

    assert info["litellm_provider"] == "nodecode"
    assert info["max_input_tokens"] == 1_048_576
    assert info["max_output_tokens"] == 131_072
    assert info["input_cost_per_token"] == pytest.approx(0.33e-6)
    assert info["cache_read_input_token_cost"] == pytest.approx(0.009e-6)
    assert info["output_cost_per_token"] == pytest.approx(1.32e-6)
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True


def test_nodecode_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NODECODE_BASE_URL}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_nodecode",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "nodecode-cyber-v1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Nodecode"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=NODECODE_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sk-nc-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NODECODE_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-nc-test-key"
    assert body["model"] == "nodecode-cyber-v1"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Nodecode"


def test_nodecode_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NODECODE_BASE_URL}/responses").respond(
            200,
            json={
                "id": "resp_nodecode",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "nodecode-cyber-v1",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_nodecode",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Nodecode", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=NODECODE_MODEL,
            input="Say hello",
            api_key="sk-nc-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NODECODE_BASE_URL}/responses"
    assert request.headers["authorization"] == "Bearer sk-nc-test-key"
    assert body["model"] == "nodecode-cyber-v1"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Nodecode"


@pytest.mark.asyncio
async def test_nodecode_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NODECODE_BASE_URL}/messages").respond(
            200,
            json={
                "id": "msg_nodecode",
                "type": "message",
                "role": "assistant",
                "model": "nodecode-cyber-v1",
                "content": [{"type": "text", "text": "Hello from Nodecode"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=NODECODE_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-nc-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NODECODE_BASE_URL}/messages"
    assert request.headers["authorization"] == "Bearer sk-nc-test-key"
    assert body["model"] == "nodecode-cyber-v1"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Nodecode"
