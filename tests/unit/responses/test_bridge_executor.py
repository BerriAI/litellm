import json
from typing import Final

import httpx
import pytest
import respx

import litellm

API_BASE: Final = "https://bridge-executor.invalid/v1"
_CHAT_REPLY: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize(
    ("surface", "litellm_params", "model_info", "upstream_model"),
    [
        ("messages", {"model": "openai/gpt-4.1", "use_chat_completions_api": True}, None, "gpt-4.1"),
        ("messages", {"model": "openai/gpt-4.1", "use_chat_completions_api": True}, {"mode": "responses"}, "gpt-4.1"),
        ("messages", {"model": "openai/chat_completions/o3-pro"}, None, "o3-pro"),
        ("responses", {"model": "openai/chat_completions/o3-pro"}, None, "o3-pro"),
    ],
)
@pytest.mark.asyncio
async def test_a_deployment_asking_for_chat_completions_is_sent_to_chat_without_the_prefix(
    surface: str,
    litellm_params: dict[str, object],
    model_info: dict[str, object] | None,
    upstream_model: str,
    respx_mock: respx.MockRouter,
):
    chat: Final = respx_mock.post(f"{API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_REPLY)
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"api_base": API_BASE, "api_key": "k", **litellm_params},
                **({"model_info": model_info} if model_info is not None else {}),
            }
        ],
        num_retries=0,
    )

    if surface == "messages":
        await router.aanthropic_messages(model="m", max_tokens=32, messages=[{"role": "user", "content": "hi"}])
    else:
        await router.aresponses(model="m", input="hi")

    assert chat.call_count == 1
    assert json.loads(chat.calls.last.request.content)["model"] == upstream_model
