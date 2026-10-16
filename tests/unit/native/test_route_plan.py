import asyncio
import json
from collections.abc import AsyncIterable, Callable, Coroutine, Mapping, Sequence
from typing import Final, cast

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.pass_through.messages.transformation import AnthropicMessagesConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig
from litellm.messages.dispatch import anthropic_messages
from litellm.native.route_plan import (
    NativeChat,
    NativeMessages,
    NativeResponses,
    RoutePlan,
    plan_chat_route,
    plan_messages_route,
    plan_responses_route,
)

MESSAGES_PASSTHROUGH: dict[str, object] = {"supported_endpoints": ["/v1/messages"]}


def _plan(
    model: str,
    provider: str,
    *,
    model_info: object = None,
    compaction_requested: bool = False,
    force_chat_completions: bool = False,
    thinking_enabled: bool = False,
) -> RoutePlan:
    return plan_messages_route(
        requested_model=f"{provider}/{model}",
        resolved_model=model,
        custom_llm_provider=provider,
        model_info=model_info,
        compaction_requested=compaction_requested,
        force_chat_completions=force_chat_completions,
        thinking_enabled=thinking_enabled,
    )


def test_a_provider_that_speaks_messages_gets_its_own_messages_config():
    plan = _plan("claude-sonnet-4-5", "anthropic")

    assert isinstance(plan, NativeMessages)
    assert isinstance(plan.config, AnthropicMessagesConfig)


@pytest.mark.parametrize(
    ("model", "provider", "kwargs", "expected"),
    [
        ("gpt-5.5", "openai", {}, NativeResponses()),
        ("gpt-5.5", "openai", {"force_chat_completions": True}, NativeChat()),
        ("gpt-5.5", "openai", {"compaction_requested": True}, NativeChat()),
        ("my-model", "hosted_vllm", {}, NativeChat()),
        ("gpt-5.5", "openai", {"force_chat_completions": True, "thinking_enabled": True}, NativeResponses()),
        ("gpt-5.5", "openai", {"compaction_requested": True, "thinking_enabled": True}, NativeChat()),
        ("my-model", "hosted_vllm", {"force_chat_completions": True, "thinking_enabled": True}, NativeChat()),
        ("lit-unmapped-model", "openai", {"force_chat_completions": True, "thinking_enabled": True}, NativeResponses()),
    ],
)
def test_messages_without_a_native_config_bridge_to_responses_only_for_openai(
    model: str, provider: str, kwargs: dict[str, bool], expected: RoutePlan
):
    assert _plan(model, provider, **kwargs) == expected


_MODEL_COST: Final = TypeAdapter(dict[str, object])


def _isolate_model_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", _MODEL_COST.validate_python(litellm.model_cost))


def test_forced_chat_keeps_thinking_on_chat_for_a_model_that_cannot_reason(monkeypatch: pytest.MonkeyPatch):
    _isolate_model_cost(monkeypatch)
    litellm.register_model({"openai/lit-plain-model": {"litellm_provider": "openai", "supports_reasoning": False}})

    plan = _plan("lit-plain-model", "openai", force_chat_completions=True, thinking_enabled=True)

    assert plan == NativeChat()


@pytest.mark.parametrize("cache_control_ttl", [True, False])
def test_a_deployment_that_opts_into_messages_passthrough_skips_translation(cache_control_ttl: bool):
    plan = _plan("my-model", "hosted_vllm", model_info={**MESSAGES_PASSTHROUGH, "cache_control_ttl": cache_control_ttl})

    assert isinstance(plan, NativeMessages)
    assert isinstance(plan.config, OpenAILikeAnthropicMessagesConfig)
    assert plan.config.supports_cache_control_ttl() is cache_control_ttl


@pytest.mark.parametrize(
    ("has_config", "use_chat_completions_api", "expected"),
    [
        (True, False, NativeResponses()),
        (True, True, NativeChat()),
        (False, False, NativeChat()),
        (False, True, NativeChat()),
    ],
)
def test_responses_reach_chat_without_a_provider_config_or_when_asked(
    has_config: bool, use_chat_completions_api: bool, expected: RoutePlan
):
    config = OpenAIResponsesAPIConfig() if has_config else None

    assert plan_responses_route(config, use_chat_completions_api) == expected


@pytest.mark.parametrize(
    ("has_config", "expected"),
    [
        (True, NativeResponses()),
        (False, NativeChat()),
    ],
)
def test_chat_bridges_to_responses_only_when_the_provider_serves_it(has_config: bool, expected: RoutePlan):
    assert plan_chat_route(OpenAIResponsesAPIConfig() if has_config else None) == expected


CHAT_BASE: Final = "http://chat-upstream.test/v1"
RESPONSES_BASE: Final = "http://responses-upstream.test/v1"
CHAT_MODEL: Final = "hosted_vllm/served-model"
RESPONSES_ONLY_MODEL: Final = "openai/bridge-once-model"
USER_TURN: Final = ({"role": "user", "content": "hi"},)

_CHAT_REPLY: Final[Mapping[str, object]] = {
    "id": "chatcmpl-once",
    "object": "chat.completion",
    "created": 1,
    "model": "served-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "once"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}
_CHAT_CHUNKS: Final[tuple[Mapping[str, object], ...]] = (
    {
        "id": "chatcmpl-once",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "served-model",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "once"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-once",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "served-model",
        "choices": [{"index": 0, "delta": dict[str, object](), "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    },
)
_OUTPUT_MESSAGE: Final[Mapping[str, object]] = {
    "type": "message",
    "id": "msg_once",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "once", "annotations": list[object]()}],
}
_RESPONSES_REPLY: Final[Mapping[str, object]] = {
    "id": "resp_once",
    "object": "response",
    "created_at": 1,
    "model": "bridge-once-model",
    "status": "completed",
    "output": [_OUTPUT_MESSAGE],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": list[object](),
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}
_RESPONSES_EVENTS: Final[tuple[Mapping[str, object], ...]] = (
    {"type": "response.created", "sequence_number": 0, "response": {**_RESPONSES_REPLY, "status": "in_progress"}},
    {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": _OUTPUT_MESSAGE},
    {
        "type": "response.output_text.delta",
        "sequence_number": 2,
        "item_id": "msg_once",
        "output_index": 0,
        "content_index": 0,
        "delta": "once",
    },
    {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": _OUTPUT_MESSAGE},
    {"type": "response.completed", "sequence_number": 4, "response": _RESPONSES_REPLY},
)


def _sse(events: Sequence[Mapping[str, object]], *, done: bool) -> bytes:
    frames: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return (frames + ("data: [DONE]\n\n" if done else "")).encode()


class _CallCounter(CustomLogger):
    pre_call: int = 0
    success: int = 0

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: object) -> None:
        self.pre_call += 1

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.success += 1


async def _messages(model: str, api_base: str, stream: bool) -> object:
    return await anthropic_messages(
        model=model, api_base=api_base, api_key="sk-x", max_tokens=10, messages=list(USER_TURN), stream=stream
    )


async def _messages_to_chat(stream: bool) -> object:
    return await _messages(CHAT_MODEL, CHAT_BASE, stream)


async def _messages_to_responses(stream: bool) -> object:
    return await _messages("openai/gpt-5.6-luna", RESPONSES_BASE, stream)


async def _chat_to_responses(stream: bool) -> object:
    return await litellm.acompletion(
        model=RESPONSES_ONLY_MODEL, api_base=RESPONSES_BASE, api_key="sk-x", messages=list(USER_TURN), stream=stream
    )


async def _responses_to_chat(stream: bool) -> object:
    return await litellm.aresponses(
        model="deepseek/served-model", api_base=CHAT_BASE, api_key="sk-x", input="hi", stream=stream
    )


async def _drained(reply: object) -> None:
    if isinstance(reply, AsyncIterable):
        async for _ in cast(AsyncIterable[object], reply):  # cast-ok: every bridged stream is drained the same way
            pass


async def _settled(counter: _CallCounter) -> _CallCounter:
    for _ in range(200):
        if counter.success:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    return counter


_BRIDGES: Final = (
    pytest.param(_messages_to_chat, "chat", id="messages_to_chat"),
    pytest.param(_messages_to_responses, "responses", id="messages_to_responses"),
    pytest.param(_chat_to_responses, "responses", id="chat_to_responses"),
    pytest.param(_responses_to_chat, "chat", id="responses_to_chat"),
)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("bridge", "upstream_api"), _BRIDGES)
@pytest.mark.asyncio
async def test_a_bridged_request_runs_hooks_logs_and_calls_the_provider_once(
    bridge: Callable[[bool], Coroutine[object, object, object]],
    upstream_api: str,
    stream: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
):
    counter: Final = _CallCounter()
    _isolate_model_cost(monkeypatch)
    monkeypatch.setattr(litellm, "callbacks", [counter])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    litellm.register_model(
        {RESPONSES_ONLY_MODEL: {"litellm_provider": "openai", "mode": "responses", "max_tokens": 100}}
    )
    upstream: Final = (
        respx_mock.post(f"{CHAT_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(_CHAT_CHUNKS, done=True)
            )
            if stream
            else httpx.Response(200, json=_CHAT_REPLY)
        )
        if upstream_api == "chat"
        else respx_mock.post(f"{RESPONSES_BASE}/responses").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(_RESPONSES_EVENTS, done=False)
            )
            if stream
            else httpx.Response(200, json=_RESPONSES_REPLY)
        )
    )

    await _drained(await bridge(stream))
    settled: Final = await _settled(counter)

    assert upstream.call_count == 1
    assert (settled.pre_call, settled.success) == (1, 1)
