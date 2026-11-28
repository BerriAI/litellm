import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_martllm_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MARTLLM_API_KEY", "martllm-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="martllm/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek/deepseek-v4.1-flash"
    assert provider == "martllm"
    assert api_key == "martllm-test-key"
    assert api_base == "https://api.martllm.com/v1"


def test_martllm_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MARTLLM_API_KEY", "martllm-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="martllm/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://martllm.internal.example/v1",
        api_key="martllm-explicit-key",
    )

    assert provider == "martllm"
    assert api_key == "martllm-explicit-key"
    assert api_base == "https://martllm.internal.example/v1"


def test_martllm_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    martllm = next(provider for provider in providers if provider["litellm_provider"] == "martllm")

    assert martllm["provider"] == "MARTLLM"
    assert martllm["provider_display_name"] == "MartLLM"
    assert martllm["default_model_placeholder"] == "martllm/deepseek/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in martllm["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_martllm_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["martllm"]["endpoints"] == {
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


@pytest.mark.parametrize("reasoning_effort", [None, "none", "high"])
def test_martllm_chat_completion_request(reasoning_effort: str | None):
    effort_kwargs: Final = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.martllm.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_martllm",
                "object": "chat.completion",
                "created": 1_793_330_000,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from MartLLM"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="martllm/deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="martllm-test-key",
            **effort_kwargs,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.martllm.com/v1/chat/completions"
    assert body.get("reasoning_effort") == reasoning_effort
    assert request.headers["authorization"] == "Bearer martllm-test-key"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from MartLLM"


@pytest.mark.asyncio
async def test_martllm_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.martllm.com/v1/messages").respond(
            200,
            json={
                "id": "msg_martllm",
                "type": "message",
                "role": "assistant",
                "model": "deepseek/deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "Hello from MartLLM"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="martllm/deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="martllm-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.martllm.com/v1/messages"
    assert request.headers["authorization"] == "Bearer martllm-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from MartLLM"


def test_martllm_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.martllm.com/v1/responses").respond(
            200,
            json={
                "id": "resp_martllm",
                "object": "response",
                "created_at": 1_795_600_000,
                "model": "deepseek/deepseek-v4.1-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_martllm",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from MartLLM", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="martllm/deepseek/deepseek-v4.1-flash",
            input="Say hello",
            api_key="martllm-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.martllm.com/v1/responses"
    assert request.headers["authorization"] == "Bearer martllm-test-key"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from MartLLM"


def test_martllm_never_borrows_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MARTLLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    monkeypatch.setattr(litellm, "openai_key", "openai-module-key")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://api.martllm.com/v1/chat/completions").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="MARTLLM_API_KEY"):
            litellm.completion(
                model="martllm/deepseek/deepseek-v4.1-flash",
                messages=[{"role": "user", "content": "Say hello"}],
            )

    assert route.call_count == 0
