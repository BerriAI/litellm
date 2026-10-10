import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_dahl_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("DAHL_API_KEY", "dahl-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="dahl/MiniMaxAI/MiniMax-M2.7",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "MiniMaxAI/MiniMax-M2.7"
    assert provider == "dahl"
    assert api_key == "dahl-test-key"
    assert api_base == "https://inference.dahl.global/v1"


def test_dahl_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("DAHL_API_KEY", "dahl-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="dahl/MiniMaxAI/MiniMax-M2.7",
        custom_llm_provider=None,
        api_base="https://dahl.internal.example/v1",
        api_key="dahl-explicit-key",
    )

    assert provider == "dahl"
    assert api_key == "dahl-explicit-key"
    assert api_base == "https://dahl.internal.example/v1"


def test_dahl_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    dahl: Final = next(provider for provider in providers if provider["litellm_provider"] == "dahl")

    assert dahl["provider"] == "DAHL"
    assert dahl["provider_display_name"] == "Gonka DAHL"
    assert dahl["default_model_placeholder"] == "dahl/MiniMaxAI/MiniMax-M2.7"
    assert {field["key"]: field["required"] for field in dahl["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_dahl_supported_endpoints():
    matrix_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers: Final = json.loads(matrix_path.read_text())["providers"]

    assert providers["dahl"]["endpoints"] == {
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


def test_dahl_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inference.dahl.global/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_dahl",
                "object": "chat.completion",
                "created": 1_791_450_000,
                "model": "MiniMaxAI/MiniMax-M2.7",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from DAHL"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="dahl/MiniMaxAI/MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="dahl-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://inference.dahl.global/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer dahl-test-key"
    assert body["model"] == "MiniMaxAI/MiniMax-M2.7"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from DAHL"


def test_dahl_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inference.dahl.global/v1/responses").respond(
            200,
            json={
                "id": "resp_dahl",
                "object": "response",
                "created_at": 1_791_450_000,
                "model": "MiniMaxAI/MiniMax-M2.7",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_dahl",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from DAHL", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="dahl/MiniMaxAI/MiniMax-M2.7",
            input="Say hello",
            api_key="dahl-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://inference.dahl.global/v1/responses"
    assert request.headers["authorization"] == "Bearer dahl-test-key"
    assert body["model"] == "MiniMaxAI/MiniMax-M2.7"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from DAHL"
