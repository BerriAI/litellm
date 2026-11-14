import json
from typing import Final

import httpx
import pytest
import respx
from openai import OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

AXFORGE_API_BASE: Final = "https://api.axforge.ai/v1"


class _RecordingAxForge:
    def __init__(self) -> None:
        self.requests: tuple[httpx.Request, ...] = ()

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-axforge",
                "object": "chat.completion",
                "created": 1234567890,
                "model": json.loads(request.content)["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    def client(self) -> OpenAI:
        return OpenAI(
            api_key="axforge-test-key",
            base_url=AXFORGE_API_BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )


def test_prefixed_model_reaches_axforge_chat_completions_without_the_prefix():
    axforge: Final = _RecordingAxForge()

    with axforge.client() as client:
        litellm.completion(
            model="axforge/glm-5.3-flash",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    assert len(axforge.requests) == 1
    assert str(axforge.requests[0].url) == f"{AXFORGE_API_BASE}/chat/completions"
    assert json.loads(axforge.requests[0].content) == {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_responses_request_reaches_axforge_responses_natively():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{AXFORGE_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_axforge",
                "object": "response",
                "created_at": 1_792_000_000,
                "model": "glm-5.3-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_axforge",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            },
        )
        response: Final = litellm.responses(model="axforge/glm-5.3-flash", input="hello", api_key="axforge-test-key")

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "glm-5.3-flash"
    assert body["input"] == "hello"
    assert response.output[0].content[0].text == "hi"


def test_axforge_key_and_base_come_from_axforge_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AXFORGE_API_KEY", "axforge-env-key")
    monkeypatch.delenv("AXFORGE_API_BASE", raising=False)

    assert get_llm_provider(model="axforge/glm-5.3-flash") == (
        "glm-5.3-flash",
        "axforge",
        "axforge-env-key",
        AXFORGE_API_BASE,
    )

    monkeypatch.setenv("AXFORGE_API_BASE", "https://axforge.internal.example/v1")

    assert get_llm_provider(model="axforge/glm-5.3-flash")[3] == "https://axforge.internal.example/v1"
