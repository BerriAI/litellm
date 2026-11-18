import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

INFER_MODEL: Final = "claude-opus-5-5:low-cost"


def test_infer_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INFER_API_KEY", "infer-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=f"infer/{INFER_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == INFER_MODEL
    assert provider == "infer"
    assert api_key == "infer-test-key"
    assert api_base == "https://infer.flow7.org/v1"


def test_infer_provider_reads_api_base_from_env(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INFER_API_KEY", "infer-env-key")
    monkeypatch.setenv("INFER_API_BASE", "https://infer.internal.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"infer/{INFER_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "infer"
    assert api_key == "infer-env-key"
    assert api_base == "https://infer.internal.example/v1"


def test_infer_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INFER_API_KEY", "infer-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"infer/{INFER_MODEL}",
        custom_llm_provider=None,
        api_base="https://infer.internal.example/v1",
        api_key="infer-explicit-key",
    )

    assert provider == "infer"
    assert api_key == "infer-explicit-key"
    assert api_base == "https://infer.internal.example/v1"


def test_infer_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    infer: Final = next(provider for provider in providers if provider["litellm_provider"] == "infer")

    assert infer["provider"] == "INFER"
    assert infer["provider_display_name"] == "Infer by Flow7"
    assert infer["default_model_placeholder"] == f"infer/{INFER_MODEL}"
    assert {field["key"]: field["required"] for field in infer["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_infer_missing_key_rejects_before_openai_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(url__regex=r"https://infer\.flow7\.org/v1/.*").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="INFER_API_KEY"):
            litellm.completion(model=f"infer/{INFER_MODEL}", messages=[{"role": "user", "content": "Hello"}])
    assert route.call_count == 0


def test_infer_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://infer.flow7.org/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_infer",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": INFER_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Infer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=f"infer/{INFER_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="infer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://infer.flow7.org/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer infer-test-key"
    assert body["model"] == INFER_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Infer"


def test_infer_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://infer.flow7.org/v1/responses").respond(
            200,
            json={
                "id": "resp_infer",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": INFER_MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_infer",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Infer", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=f"infer/{INFER_MODEL}",
            input="Say hello",
            api_key="infer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://infer.flow7.org/v1/responses"
    assert request.headers["authorization"] == "Bearer infer-test-key"
    assert body["model"] == INFER_MODEL
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Infer"


@pytest.mark.asyncio
async def test_infer_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://infer.flow7.org/v1/messages").respond(
            200,
            json={
                "id": "msg_infer",
                "type": "message",
                "role": "assistant",
                "model": INFER_MODEL,
                "content": [{"type": "text", "text": "Hello from Infer"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"infer/{INFER_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="infer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://infer.flow7.org/v1/messages"
    assert request.headers["authorization"] == "Bearer infer-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == INFER_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Infer"
