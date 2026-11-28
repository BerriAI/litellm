import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.custom_logger import CustomLogger

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_wiro",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "openai/gpt-5-4-nano",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Wiro"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 11, "total_tokens": 23, "cost": 0.00001615},
}
_MESSAGES_USAGE: Final = {"input_tokens": 12, "output_tokens": 11, "cost": 0.00001615, "cost_usd": 0.00001615}
_MESSAGE: Final = {
    "id": "msg_wiro",
    "type": "message",
    "role": "assistant",
    "model": "openai/gpt-5-4-nano",
    "content": [{"type": "text", "text": "Hello from Wiro"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": _MESSAGES_USAGE,
}


def _sse(*events: Mapping[str, object]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


_MESSAGE_STREAM: Final = _sse(
    {
        "type": "message_start",
        "message": {**_MESSAGE, "content": [], "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0}},
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello from Wiro"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": _MESSAGES_USAGE},
    {"type": "message_stop"},
)


class _CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.cost: Final[asyncio.Future[object]] = asyncio.get_running_loop().create_future()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        if not self.cost.done():
            self.cost.set_result(kwargs.get("response_cost"))


def test_wiro_is_a_registered_provider():
    assert litellm.LlmProviders.WIRO.value == "wiro"
    assert "wiro" in litellm.provider_list
    assert "wiro" in litellm.constants.openai_compatible_providers


def test_wiro_provider_resolution_keeps_the_owner_in_the_model_id(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WIRO_API_KEY", "wiro-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="wiro/openai/gpt-5-4-nano",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "openai/gpt-5-4-nano"
    assert provider == "wiro"
    assert api_key == "wiro-test-key"
    assert api_base == "https://llm.wiro.ai/v1"


def test_wiro_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WIRO_API_KEY", "wiro-env-key")
    monkeypatch.setenv("WIRO_API_BASE", "https://wiro.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="wiro/openai/gpt-5-4-nano",
        custom_llm_provider=None,
        api_base="https://wiro.internal.example/v1",
        api_key="wiro-explicit-key",
    )

    assert provider == "wiro"
    assert api_key == "wiro-explicit-key"
    assert api_base == "https://wiro.internal.example/v1"


def test_wiro_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WIRO_API_KEY", "wiro-env-key")
    monkeypatch.setenv("WIRO_API_BASE", "https://wiro.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="wiro/openai/gpt-5-4-nano",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "wiro"
    assert api_base == "https://wiro.env.example/v1"


def test_wiro_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("WIRO_API_KEY", "wiro-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="bytedance/seed-v2-mini",
        custom_llm_provider=None,
        api_base="https://llm.wiro.ai/v1",
        api_key=None,
    )

    assert model == "bytedance/seed-v2-mini"
    assert provider == "wiro"
    assert api_key == "wiro-env-key"
    assert api_base == "https://llm.wiro.ai/v1"


def test_wiro_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    wiro = next(provider for provider in providers if provider["litellm_provider"] == "wiro")

    assert wiro["provider"] == "WIRO"
    assert wiro["provider_display_name"] == "Wiro AI"
    assert wiro["default_model_placeholder"] == "wiro/openai/gpt-5-4-nano"
    assert {field["key"]: field["required"] for field in wiro["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_wiro_chat_completion_request_bills_the_cost_wiro_reports():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://llm.wiro.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="wiro/openai/gpt-5-4-nano",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="wiro-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://llm.wiro.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer wiro-test-key"
    assert body["model"] == "openai/gpt-5-4-nano"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Wiro"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(_CHAT_COMPLETION["usage"]["cost"])


def test_wiro_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://llm.wiro.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_wiro",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "openai/gpt-5-4-nano",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_wiro",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Wiro", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 12, "total_tokens": 24},
            },
        )
        response: Final = litellm.responses(
            model="wiro/openai/gpt-5-4-nano",
            input="Say hello",
            api_key="wiro-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://llm.wiro.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer wiro-test-key"
    assert body["model"] == "openai/gpt-5-4-nano"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Wiro"


@pytest.mark.asyncio
async def test_wiro_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://llm.wiro.ai/v1/messages").respond(200, json=_MESSAGE)
        response: Final = await litellm.anthropic.messages.acreate(
            model="wiro/openai/gpt-5-4-nano",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="wiro-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://llm.wiro.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer wiro-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "openai/gpt-5-4-nano"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Wiro"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
async def test_wiro_anthropic_messages_bill_the_cost_wiro_reports(monkeypatch: pytest.MonkeyPatch, stream: bool):
    recorder: Final = _CostRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    reply: Final = (
        httpx.Response(200, headers={"content-type": "text/event-stream"}, text=_MESSAGE_STREAM)
        if stream
        else httpx.Response(200, json=_MESSAGE)
    )
    with respx.mock() as upstream:
        upstream.post("https://llm.wiro.ai/v1/messages").mock(return_value=reply)
        response: Final = await litellm.anthropic.messages.acreate(
            model="wiro/openai/gpt-5-4-nano",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="wiro-test-key",
            stream=stream,
        )
        if stream:
            assert b"Hello from Wiro" in b"".join([chunk async for chunk in response])

    assert await asyncio.wait_for(recorder.cost, timeout=5) == pytest.approx(_MESSAGES_USAGE["cost"])
