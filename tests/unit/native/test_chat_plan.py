from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import JSONSchemaValidationError

API_BASE: Final = "https://bridge-parity.invalid/v1"
SCHEMA: Final = {
    "type": "object",
    "properties": {"b": {"type": "integer"}},
    "required": ["b"],
    "additionalProperties": False,
}
OFF_SCHEMA_TEXT: Final = '{"b": 2, "extra": true}'
_CHAT_REPLY: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": OFF_SCHEMA_TEXT}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _router(litellm_params: dict[str, object], model_info: dict[str, object] | None = None) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"api_base": API_BASE, "api_key": "k", **litellm_params},
                **({"model_info": model_info} if model_info is not None else {}),
            }
        ],
        num_retries=0,
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize(
    ("litellm_params", "model_info"),
    [
        ({"model": "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"}, None),
        ({"model": "openai/gpt-4.1", "use_chat_completions_api": True}, None),
    ],
)
@pytest.mark.asyncio
async def test_a_messages_request_bridged_to_chat_validates_its_json_schema(
    litellm_params: dict[str, object],
    model_info: dict[str, object] | None,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "enable_json_schema_validation", True)
    respx_mock.post(f"{API_BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_CHAT_REPLY))

    with pytest.raises(JSONSchemaValidationError):
        await _router(litellm_params, model_info).aanthropic_messages(
            model="m",
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
            output_format={"type": "json_schema", "schema": SCHEMA},
        )


@pytest.mark.asyncio
async def test_a_responses_request_bridged_to_chat_validates_its_json_schema(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "enable_json_schema_validation", True)
    respx_mock.post(f"{API_BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_CHAT_REPLY))

    with pytest.raises(JSONSchemaValidationError):
        await _router({"model": "deepseek/deepseek-reasoner"}, {"mode": "responses"}).aresponses(
            model="m",
            input="hi",
            text={"format": {"type": "json_schema", "name": "x", "strict": True, "schema": SCHEMA}},
        )


@pytest.mark.asyncio
async def test_a_provider_failure_on_the_messages_to_chat_bridge_is_mapped(respx_mock: respx.MockRouter):
    respx_mock.post(f"{API_BASE}/chat/completions").mock(return_value=httpx.Response(200, content=b"not json"))

    with pytest.raises(litellm.exceptions.APIConnectionError):
        await _router({"model": "groq/llama-3.1-8b-instant"}, {"mode": "responses"}).aanthropic_messages(
            model="m", max_tokens=32, messages=[{"role": "user", "content": "hi"}]
        )
