import asyncio
import inspect
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Final, Literal, NamedTuple, TypedDict

from typing_extensions import ReadOnly

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router

_PRIMARY: Final = "https://hooks-primary.openai.azure.com"
_FALLBACK: Final = "https://hooks-fallback.openai.azure.com"
_API_VERSION: Final = "2024-10-21"
_MESSAGES: Final = [{"role": "user", "content": "Hi - i'm openai"}]
_USAGE: Final = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
_COMPLETION: Final = {
    "id": "chatcmpl-hooks",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4.1-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": _USAGE,
}
_EMBEDDING_VECTOR: Final = [0.1, 0.2, 0.3]
_EMBEDDING: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": _EMBEDDING_VECTOR}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}
_STREAM: Final = (
    "".join(
        f"data: {json.dumps(chunk)}\n\n"
        for chunk in (
            {
                "id": "chatcmpl-hooks",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-hooks",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
            },
        )
    )
    + "data: [DONE]\n\n"
)
_AUTH_ERROR: Final = httpx.Response(
    401,
    json={
        "error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}
    },
)

_OUR_MODEL_GROUPS: Final = frozenset({"hooks-group", "primary-group", "fallback-group"})

_State = Literal[
    "sync_pre_api_call",
    "post_api_call",
    "async_stream",
    "sync_success",
    "async_success",
    "sync_failure",
    "async_failure",
]


class _HookEvent(NamedTuple):
    state: _State
    model: object
    kwargs: Mapping[str, object]
    response: object


def _router_context_problems(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    litellm_params: Final = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return ("litellm_params",)
    metadata: Final = litellm_params.get("metadata")
    model_info: Final = litellm_params.get("model_info")
    checks: Final = {
        "metadata": isinstance(metadata, dict),
        "model_group": isinstance(metadata, dict) and isinstance(metadata.get("model_group"), str),
        "deployment": isinstance(metadata, dict) and isinstance(metadata.get("deployment"), str),
        "model_info": isinstance(model_info, dict),
        "model_info id": isinstance(model_info, dict) and isinstance(model_info.get("id"), str),
        "proxy_server_request": isinstance(litellm_params.get("proxy_server_request"), (str, type(None))),
        "preset_cache_key": isinstance(litellm_params.get("preset_cache_key"), (str, type(None))),
        "stream_response": isinstance(litellm_params.get("stream_response"), dict),
    }
    return tuple(name for name, ok in checks.items() if not ok)


def _request_problems(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    checks: Final = {
        "model": isinstance(kwargs.get("model"), str),
        "messages": isinstance(kwargs.get("messages"), list),
        "optional_params": isinstance(kwargs.get("optional_params"), dict),
        "start_time": isinstance(kwargs.get("start_time"), (datetime, type(None))),
        "stream": isinstance(kwargs.get("stream"), bool),
        "user": isinstance(kwargs.get("user"), (str, type(None))),
    }
    return (*(name for name, ok in checks.items() if not ok), *_router_context_problems(kwargs))


def _call_detail_problems(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    original_response: Final = kwargs.get("original_response")
    checks: Final = {
        "input": isinstance(kwargs.get("input"), (list, dict, str)),
        "api_key": isinstance(kwargs.get("api_key"), (str, type(None))),
        "original_response": isinstance(original_response, (str, litellm.CustomStreamWrapper, type(None)))
        or inspect.iscoroutine(original_response)
        or inspect.isasyncgen(original_response),
        "additional_args": isinstance(kwargs.get("additional_args"), (dict, type(None))),
        "log_event_type": isinstance(kwargs.get("log_event_type"), str),
    }
    return tuple(name for name, ok in checks.items() if not ok)


def _is_from_this_test(kwargs: Mapping[str, object]) -> bool:
    litellm_params: Final = kwargs.get("litellm_params")
    metadata: Final = litellm_params.get("metadata") if isinstance(litellm_params, dict) else None
    return isinstance(metadata, dict) and metadata.get("model_group") in _OUR_MODEL_GROUPS


class _HookRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: Final[list[_HookEvent]] = []
        self.errors: Final[list[str]] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.waiters: Final[list[tuple[Callable[[Sequence[_State]], bool], asyncio.Event]]] = []

    @property
    def states(self) -> list[_State]:
        return [event.state for event in self.events]

    def _record(
        self, state: _State, model: object, kwargs: Mapping[str, object], response: object, problems: Sequence[str]
    ) -> None:
        if not _is_from_this_test(kwargs):
            return
        self.errors.extend(f"{state}: {problem}" for problem in problems)
        self.events.append(_HookEvent(state, model, kwargs, response))
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._notify)

    def _notify(self) -> None:
        for predicate, event in self.waiters:
            if predicate(tuple(self.states)):
                event.set()

    async def until(self, predicate: Callable[[Sequence[_State]], bool]) -> tuple[_State, ...]:
        self.loop = asyncio.get_running_loop()
        if not predicate(tuple(self.states)):
            event: Final = asyncio.Event()
            self.waiters.append((predicate, event))
            await asyncio.wait_for(event.wait(), timeout=10)
        return tuple(self.states)

    def log_pre_api_call(self, model: object, messages: object, kwargs: Mapping[str, object]) -> None:
        problems: Final = (
            *(("model",) if not isinstance(model, str) else ()),
            *(("messages",) if not isinstance(messages, list) else ()),
            *_request_problems(kwargs),
        )
        self._record("sync_pre_api_call", model, kwargs, messages, problems)

    def log_post_api_call(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        problems: Final = (
            *(("start_time",) if not isinstance(start_time, datetime) else ()),
            *(("end_time",) if end_time is not None else ()),
            *(("response_obj",) if response_obj is not None else ()),
            *_request_problems(kwargs),
            *_call_detail_problems(kwargs),
        )
        self._record("post_api_call", kwargs.get("model"), kwargs, response_obj, problems)

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("sync_success", kwargs.get("model"), kwargs, response_obj, ())

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("sync_failure", kwargs.get("model"), kwargs, response_obj, ())

    async def async_log_stream_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("async_stream", kwargs.get("model"), kwargs, response_obj, ())

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        problems: Final = (
            *(("times",) if not (isinstance(start_time, datetime) and isinstance(end_time, datetime)) else ()),
            *(
                ("response_obj",)
                if not isinstance(response_obj, (litellm.ModelResponse, litellm.EmbeddingResponse))
                else ()
            ),
            *(("cache_hit",) if not isinstance(kwargs.get("cache_hit"), (bool, type(None))) else ()),
            *_request_problems(kwargs),
            *_call_detail_problems(kwargs),
        )
        self._record("async_success", kwargs.get("model"), kwargs, response_obj, problems)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        problems: Final = (
            *(("times",) if not (isinstance(start_time, datetime) and isinstance(end_time, datetime)) else ()),
            *(("response_obj",) if response_obj is not None else ()),
            *(("exception",) if not isinstance(kwargs.get("exception"), Exception) else ()),
            *_request_problems(kwargs),
            *_call_detail_problems(kwargs),
        )
        self._record("async_failure", kwargs.get("model"), kwargs, response_obj, problems)


def _settled(terminal: int, posts: int) -> Callable[[Sequence[_State]], bool]:
    def satisfied(states: Sequence[_State]) -> bool:
        terminals: Final = sum(state in ("async_success", "async_failure") for state in states)
        return terminals >= terminal and states.count("post_api_call") >= posts

    return satisfied


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _HookRecorder:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    hook_recorder: Final = _HookRecorder()
    monkeypatch.setattr(litellm, "callbacks", [hook_recorder])
    return hook_recorder


def _router(model: str, api_base: str = _PRIMARY) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "hooks-group",
                "litellm_params": {
                    "model": model,
                    "api_key": "sk-unit-test",
                    "api_base": api_base,
                    "api_version": _API_VERSION,
                },
                "model_info": {"base_model": model},
            }
        ],
        num_retries=0,
    )


class _RouterMetadata(TypedDict):
    model_group: ReadOnly[str]
    deployment: ReadOnly[str]


class _RouterModelInfo(TypedDict):
    id: ReadOnly[str]


class _RouterParams(TypedDict):
    metadata: ReadOnly[_RouterMetadata]
    model_info: ReadOnly[_RouterModelInfo]


def _router_params(event: _HookEvent) -> _RouterParams:
    return TypeAdapter(_RouterParams).validate_python(event.kwargs["litellm_params"])


def _model_group(event: _HookEvent) -> str:
    return _router_params(event)["metadata"]["model_group"]


def _model_id(event: _HookEvent) -> str:
    return _router_params(event)["model_info"]["id"]


def _of_state(recorder: _HookRecorder, state: _State) -> list[_HookEvent]:
    return [event for event in recorder.events if event.state == state]


def _completion(event: _HookEvent) -> litellm.ModelResponse:
    assert isinstance(event.response, litellm.ModelResponse)
    return event.response


@pytest.mark.asyncio
async def test_router_chat_success_streaming_and_failure_fire_the_hooks_in_order(
    recorder: _HookRecorder, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(url__startswith=f"{_PRIMARY}/openai/deployments/gpt-4.1-mini/chat/completions")
    router: Final = _router("azure/gpt-4.1-mini")

    route.mock(return_value=httpx.Response(200, json=_COMPLETION))
    await router.acompletion(model="hooks-group", messages=_MESSAGES)
    await recorder.until(_settled(1, posts=1))
    assert recorder.states == ["sync_pre_api_call", "post_api_call", "async_success"]
    pre, post, success = recorder.events
    assert pre.model == "gpt-4.1-mini"
    assert pre.response == _MESSAGES
    assert {_model_group(event) for event in recorder.events} == {"hooks-group"}
    assert len({_model_id(event) for event in recorder.events}) == 1
    assert post.kwargs["messages"] == _MESSAGES
    assert success.kwargs["stream"] is False
    assert _completion(success).choices[0].message.content == "hello"
    assert _completion(success).usage.total_tokens == _USAGE["total_tokens"]

    route.mock(return_value=httpx.Response(200, text=_STREAM, headers={"content-type": "text/event-stream"}))
    stream: Final = await router.acompletion(model="hooks-group", messages=_MESSAGES, stream=True)
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in stream]) == "hello"
    await recorder.until(_settled(2, posts=2))
    streamed: Final = recorder.events[3:]
    assert sorted(event.state for event in streamed[:2]) == ["post_api_call", "sync_pre_api_call"]
    assert [event.state for event in streamed[2:]] == ["async_success"]
    assert all(event.kwargs["stream"] is True for event in streamed)
    assert _completion(streamed[2]).choices[0].message.content == "hello"

    route.mock(return_value=_AUTH_ERROR)
    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(model="hooks-group", messages=_MESSAGES)
    await recorder.until(_settled(3, posts=3))
    failed: Final = recorder.events[6:]
    assert [event.state for event in failed] == ["sync_pre_api_call", "post_api_call", "async_failure"]
    assert failed[2].response is None
    assert isinstance(failed[2].kwargs["exception"], litellm.AuthenticationError)
    assert recorder.errors == []


@pytest.mark.asyncio
async def test_router_embedding_success_and_failure_fire_the_hooks_in_order(
    recorder: _HookRecorder, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(url__startswith=f"{_PRIMARY}/openai/deployments/text-embedding-3-small/embeddings")
    router: Final = _router("azure/text-embedding-3-small")

    route.mock(return_value=httpx.Response(200, json=_EMBEDDING))
    await router.aembedding(model="hooks-group", input=["hello"])
    await recorder.until(_settled(1, posts=1))
    assert recorder.states == ["sync_pre_api_call", "post_api_call", "async_success"]
    assert {event.model for event in recorder.events} == {"text-embedding-3-small"}
    assert {_model_group(event) for event in recorder.events} == {"hooks-group"}
    embedding: Final = recorder.events[2].response
    assert isinstance(embedding, litellm.EmbeddingResponse)
    assert embedding.model_dump()["data"][0]["embedding"] == _EMBEDDING_VECTOR
    assert embedding.usage.prompt_tokens == 3

    route.mock(return_value=_AUTH_ERROR)
    with pytest.raises(litellm.AuthenticationError):
        await router.aembedding(model="hooks-group", input=["hello"])
    await recorder.until(_settled(2, posts=2))
    assert recorder.states[3:] == ["sync_pre_api_call", "post_api_call", "async_failure"]
    assert recorder.events[5].response is None
    assert isinstance(recorder.events[5].kwargs["exception"], litellm.AuthenticationError)
    assert recorder.errors == []


@pytest.mark.asyncio
async def test_router_fallback_fires_failure_then_success_hooks(
    recorder: _HookRecorder, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post(url__startswith=f"{_PRIMARY}/openai/deployments/gpt-4.1-mini/chat/completions").mock(
        return_value=_AUTH_ERROR
    )
    fallback: Final = respx_mock.post(
        url__startswith=f"{_FALLBACK}/openai/deployments/gpt-4.1-mini/chat/completions"
    ).mock(return_value=httpx.Response(200, json=_COMPLETION))
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary-group",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "my-bad-key",
                    "api_base": _PRIMARY,
                    "api_version": _API_VERSION,
                },
            },
            {
                "model_name": "fallback-group",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "sk-unit-test",
                    "api_base": _FALLBACK,
                    "api_version": _API_VERSION,
                },
            },
        ],
        fallbacks=[{"primary-group": ["fallback-group"]}],
        num_retries=0,
    )

    await router.acompletion(model="primary-group", messages=_MESSAGES)
    await recorder.until(_settled(2, posts=2))

    assert fallback.call_count == 1
    assert recorder.states == [
        "sync_pre_api_call",
        "post_api_call",
        "async_failure",
        "sync_pre_api_call",
        "post_api_call",
        "async_success",
    ]
    assert [_model_group(event) for event in recorder.events] == ["primary-group"] * 3 + ["fallback-group"] * 3
    assert _model_id(recorder.events[0]) != _model_id(recorder.events[3])
    assert isinstance(recorder.events[2].kwargs["exception"], litellm.AuthenticationError)
    assert _completion(recorder.events[5]).choices[0].message.content == "hello"
    assert recorder.errors == []


@pytest.mark.asyncio
async def test_router_completion_cache_hit_fires_a_second_success_hook(
    recorder: _HookRecorder, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache())
    route: Final = respx_mock.post(url__startswith=f"{_PRIMARY}/openai/deployments/gpt-4.1-mini/chat/completions").mock(
        return_value=httpx.Response(200, json=_COMPLETION)
    )
    router: Final = _router("azure/gpt-4.1-mini")

    await router.acompletion(model="hooks-group", messages=_MESSAGES, caching=True)
    await recorder.until(_settled(1, posts=1))
    await router.acompletion(model="hooks-group", messages=_MESSAGES, caching=True)
    await recorder.until(_settled(2, posts=1))

    assert route.call_count == 1
    assert recorder.states == ["sync_pre_api_call", "post_api_call", "async_success", "async_success"]
    first, second = _of_state(recorder, "async_success")
    assert first.kwargs.get("cache_hit") is not True
    assert second.kwargs.get("cache_hit") is True
    assert _completion(second).choices[0].message.content == "hello"
    assert recorder.errors == []


@pytest.mark.asyncio
async def test_router_streaming_cache_hit_still_fires_the_success_hook(
    recorder: _HookRecorder, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache())
    route: Final = respx_mock.post(url__startswith=f"{_PRIMARY}/openai/deployments/gpt-4.1-mini/chat/completions").mock(
        return_value=httpx.Response(200, text=_STREAM, headers={"content-type": "text/event-stream"})
    )
    router: Final = _router("azure/gpt-4.1-mini")

    first: Final = await router.acompletion(model="hooks-group", messages=_MESSAGES, stream=True, caching=True)
    first_text: Final = "".join([chunk.choices[0].delta.content or "" async for chunk in first])
    await recorder.until(_settled(1, posts=1))
    states_after_first: Final = len(recorder.states)
    second: Final = await router.acompletion(model="hooks-group", messages=_MESSAGES, stream=True, caching=True)
    second_text: Final = "".join([chunk.choices[0].delta.content or "" async for chunk in second])
    await recorder.until(_settled(2, posts=1))

    assert route.call_count == 1
    assert first_text == second_text == "hello"
    assert sorted(recorder.states[:2]) == ["post_api_call", "sync_pre_api_call"]
    assert recorder.states[2:states_after_first] == ["async_success"]
    assert recorder.states[states_after_first:] == ["async_success"]
    first_success, second_success = _of_state(recorder, "async_success")
    assert first_success.kwargs.get("cache_hit") is not True
    assert second_success.kwargs.get("cache_hit") is True
    assert _completion(second_success).choices[0].message.content == "hello"
    assert recorder.errors == []


@pytest.mark.asyncio
async def test_router_embedding_cache_hit_fires_a_second_success_hook(
    recorder: _HookRecorder, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache())
    route: Final = respx_mock.post(
        url__startswith=f"{_PRIMARY}/openai/deployments/text-embedding-3-small/embeddings"
    ).mock(return_value=httpx.Response(200, json=_EMBEDDING))
    router: Final = _router("azure/text-embedding-3-small")

    await router.aembedding(model="hooks-group", input=["hello"], caching=True)
    await recorder.until(_settled(1, posts=1))
    await router.aembedding(model="hooks-group", input=["hello"], caching=True)
    await recorder.until(_settled(2, posts=1))

    assert route.call_count == 1
    assert recorder.states == ["sync_pre_api_call", "post_api_call", "async_success", "async_success"]
    first, second = _of_state(recorder, "async_success")
    assert first.kwargs.get("cache_hit") is not True
    assert second.kwargs.get("cache_hit") is True
    assert isinstance(second.response, litellm.EmbeddingResponse)
    assert second.response.model_dump()["data"][0]["embedding"] == _EMBEDDING_VECTOR
    assert recorder.errors == []
