import json
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_BUTLER_BASE_URL: Final = "https://api.butlercompute.com/v1"


def test_butler_chat_models_route_through_the_json_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUTLER_API_KEY", "environment-key")
    monkeypatch.delenv("BUTLER_API_BASE", raising=False)

    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BUTLER_BASE_URL}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-req_01",
                "object": "chat.completion",
                "created": 1_793_940_103,
                "model": "cicero",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Butler"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        )
        response: Final = litellm.completion(
            model="butler/cicero",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=16,
        )

    request: Final = route.calls.last.request
    assert litellm.get_llm_provider("butler/cicero") == ("cicero", "butler", "environment-key", _BUTLER_BASE_URL)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer environment-key"
    assert json.loads(request.content) == {
        "model": "cicero",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_completion_tokens": 16,
    }
    assert response.choices[0].message.content == "Hello from Butler"


def test_butler_api_base_from_the_environment_replaces_the_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUTLER_API_BASE", "https://gateway.example/v1")

    assert litellm.get_llm_provider("butler/cicero", api_key="caller-key") == (
        "cicero",
        "butler",
        "caller-key",
        "https://gateway.example/v1",
    )


def test_butler_responses_requests_go_to_butler_natively() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BUTLER_BASE_URL}/responses").respond(
            200,
            json={
                "id": "resp_req_01",
                "object": "response",
                "created_at": 1_793_941_697,
                "model": "cicero",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_req_01",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Butler", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            },
        )
        response: Final = litellm.responses(model="butler/cicero", input="Say hello", api_key="caller-key")

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer caller-key"
    assert (body["model"], body["input"]) == ("cicero", "Say hello")
    assert response.output[0].content[0].text == "Hello from Butler"


@pytest.mark.asyncio
async def test_butler_anthropic_messages_requests_go_to_butler_natively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BUTLER_BASE_URL}/messages").respond(
            200,
            json={
                "id": "msg_req_01",
                "type": "message",
                "role": "assistant",
                "model": "cicero",
                "content": [{"type": "text", "text": "Hello from Butler"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="butler/cicero",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="caller-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer caller-key"
    assert (body["model"], body["messages"]) == ("cicero", [{"role": "user", "content": "Say hello"}])
    assert response["content"][0]["text"] == "Hello from Butler"
