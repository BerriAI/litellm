import datetime
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.pass_through.adapters.handler import LiteLLMMessagesToCompletionTransformationHandler
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

API_BASE: Final = "http://chat-upstream.test/v1"
CHAT_RESPONSE: Final = {
    "id": "chatcmpl-bridge",
    "object": "chat.completion",
    "created": 1,
    "model": "served-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "from upstream"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class _DeploymentHookCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.pre_call = 0
        self.post_call = 0

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: object) -> None:
        self.pre_call += 1

    async def async_post_call_success_deployment_hook(
        self, request_data: Mapping[str, object], response: object, call_type: object
    ) -> None:
        self.post_call += 1


_PromptResult: TypeAlias = tuple[str, list[AllMessageValues], dict[str, object]]


class _TemplatePromptManager(CustomPromptManagement):
    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        non_default_params: Mapping[str, object],
        prompt_id: str | None,
        prompt_variables: Mapping[str, object] | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: Logging,
        prompt_spec: PromptSpec | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> _PromptResult:
        return model, [{"role": "user", "content": "from the template"}], {**non_default_params}


@pytest.fixture
def upstream(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post(f"{API_BASE}/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))


async def _ask(extra: Mapping[str, object] = MappingProxyType({})) -> object:
    return await litellm.anthropic.messages.acreate(
        model="hosted_vllm/served-model",
        api_base=API_BASE,
        api_key="sk-chat",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )


@pytest.mark.asyncio
async def test_bridge_runs_deployment_hooks_once_per_request(upstream: respx.Route, monkeypatch: pytest.MonkeyPatch):
    counter: Final = _DeploymentHookCounter()
    monkeypatch.setattr(litellm, "callbacks", [counter])

    response: Final = await _ask()

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "from upstream"}]
    assert upstream.call_count == 1
    assert (counter.pre_call, counter.post_call) == (1, 1)


@pytest.mark.asyncio
async def test_bridge_streams_the_upstream_reply(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    chunk: Final = '{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"streamed"},"finish_reason":null}]}'
    done: Final = '{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
    respx_mock.post(f"{API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=f"data: {chunk}\n\ndata: {done}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )
    )

    stream: Final = await _ask({"stream": True})
    body: Final = b"".join([event async for event in stream]).decode()

    assert "event: message_start" in body
    assert '"text":"streamed"' in body.replace(" ", "")


@pytest.mark.asyncio
async def test_bridge_applies_the_prompt_template_named_by_prompt_id(
    upstream: respx.Route, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "callbacks", [_TemplatePromptManager()])

    await _ask({"prompt_id": "bridge_template"})

    assert "from the template" in upstream.calls.last.request.content.decode()


@pytest.mark.asyncio
async def test_bridge_maps_an_upstream_429_to_rate_limit_error(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{API_BASE}/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "slow down", "type": "rate_limit"}})
    )

    with pytest.raises(litellm.RateLimitError):
        await _ask()


def test_sync_bridge_answers_from_one_upstream_call(upstream: respx.Route):
    logging_obj: Final = Logging(
        model="hosted_vllm/served-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(datetime.timezone.utc),
        litellm_call_id="6825beef-0000-4000-8000-000000000005",
        function_id="1234",
    )

    response: Final = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        model="hosted_vllm/served-model",
        api_base=API_BASE,
        api_key="sk-chat",
        litellm_logging_obj=logging_obj,
    )

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "from upstream"}]
    assert upstream.call_count == 1


@pytest.mark.asyncio
async def test_bridge_runs_prompt_management_triggered_without_prompt_id(
    upstream: respx.Route, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "callbacks", [_TemplatePromptManager()])

    await _ask({"vector_store_ids": ["vs_bridge"]})

    assert "from the template" in upstream.calls.last.request.content.decode()


@pytest.mark.asyncio
async def test_bridge_keeps_the_litellm_system_prompt(upstream: respx.Route):
    await _ask({"litellm_system_prompt": "answer like a pirate"})

    assert "answer like a pirate" in upstream.calls.last.request.content.decode()


@pytest.mark.asyncio
async def test_bridge_goes_through_a_wrapped_acompletion(upstream: respx.Route, monkeypatch: pytest.MonkeyPatch):
    unwrapped: Final = litellm.acompletion
    seen: Final[list[object]] = []  # mutable-ok: records the calls the wrapper sees

    async def traced_acompletion(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("model"))
        return await unwrapped(*args, **kwargs)

    monkeypatch.setattr(litellm, "acompletion", traced_acompletion)

    response: Final = await _ask()

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "from upstream"}]
    assert seen == ["hosted_vllm/served-model"]
    assert upstream.call_count == 1
