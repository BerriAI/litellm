import asyncio
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Final, Literal

import httpx
import pytest
import respx

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
_EMBEDDING: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
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

_State = Literal[
    "sync_pre_api_call",
    "post_api_call",
    "async_stream",
    "sync_success",
    "async_success",
    "sync_failure",
    "async_failure",
]


class _HookRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.states: Final[list[_State]] = []
        self.errors: Final[list[str]] = []
        self.cache_hits: Final[list[object]] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.waiters: Final[list[tuple[Callable[[Sequence[_State]], bool], asyncio.Event]]] = []

    def _check(self, ok: bool, what: str) -> None:
        if not ok:
            self.errors.append(what)

    def _notify(self) -> None:
        for predicate, event in self.waiters:
            if predicate(tuple(self.states)):
                event.set()

    def _record(self, state: _State) -> None:
        self.states.append(state)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._notify)

    async def until(self, predicate: Callable[[Sequence[_State]], bool]) -> tuple[_State, ...]:
        self.loop = asyncio.get_running_loop()
        if not predicate(tuple(self.states)):
            event: Final = asyncio.Event()
            self.waiters.append((predicate, event))
            await asyncio.wait_for(event.wait(), timeout=10)
        return tuple(self.states)

    def log_pre_api_call(self, model: object, messages: object, kwargs: Mapping[str, object]) -> None:
        self._record("sync_pre_api_call")
        litellm_params: Final = kwargs.get("litellm_params")
        self._check(isinstance(model, str), "pre model")
        self._check(isinstance(messages, list), "pre messages")
        self._check(isinstance(kwargs.get("optional_params"), dict), "pre optional_params")
        self._check(isinstance(kwargs.get("stream"), bool), "pre stream")
        self._check(isinstance(litellm_params, dict), "pre litellm_params")
        if isinstance(litellm_params, dict):
            metadata: Final = litellm_params.get("metadata")
            model_info: Final = litellm_params.get("model_info")
            self._check(isinstance(metadata, dict) and isinstance(metadata.get("model_group"), str), "pre model_group")
            self._check(isinstance(metadata, dict) and isinstance(metadata.get("deployment"), str), "pre deployment")
            self._check(isinstance(model_info, dict) and isinstance(model_info.get("id"), str), "pre model_info id")

    def log_post_api_call(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("post_api_call")
        self._check(isinstance(kwargs.get("model"), str), "post model")
        self._check(isinstance(kwargs.get("litellm_params"), dict), "post litellm_params")

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("sync_success")

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("sync_failure")

    async def async_log_stream_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("async_stream")

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._check(isinstance(start_time, datetime) and isinstance(end_time, datetime), "success times")
        self._check(
            isinstance(response_obj, (litellm.ModelResponse, litellm.EmbeddingResponse)), "success response type"
        )
        self._check(isinstance(kwargs.get("model"), str), "success model")
        self.cache_hits.append(kwargs.get("cache_hit"))
        self._record("async_success")

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._check(isinstance(start_time, datetime) and isinstance(end_time, datetime), "failure times")
        self._check(isinstance(kwargs.get("exception"), Exception), "failure exception")
        self._record("async_failure")


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

    route.mock(return_value=httpx.Response(200, text=_STREAM, headers={"content-type": "text/event-stream"}))
    stream: Final = await router.acompletion(model="hooks-group", messages=_MESSAGES, stream=True)
    assert [chunk async for chunk in stream]
    await recorder.until(_settled(2, posts=2))
    streamed: Final = recorder.states[3:]
    assert len(streamed) >= 3
    assert Counter(streamed) >= Counter(("sync_pre_api_call", "post_api_call", "async_success"))
    assert streamed[-1] == "async_success"

    route.mock(return_value=_AUTH_ERROR)
    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(model="hooks-group", messages=_MESSAGES)
    await recorder.until(_settled(3, posts=3))
    assert recorder.states[3 + len(streamed) :] == ["sync_pre_api_call", "post_api_call", "async_failure"]
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

    route.mock(return_value=_AUTH_ERROR)
    with pytest.raises(litellm.AuthenticationError):
        await router.aembedding(model="hooks-group", input=["hello"])
    await recorder.until(_settled(2, posts=2))
    assert recorder.states[3:] == ["sync_pre_api_call", "post_api_call", "async_failure"]
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
    assert Counter(recorder.states) == Counter(
        ("sync_pre_api_call", "post_api_call", "async_failure", "sync_pre_api_call", "post_api_call", "async_success")
    )
    assert [state for state in recorder.states if state.startswith("async_")] == ["async_failure", "async_success"]
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
    assert recorder.cache_hits[0] is not True
    assert recorder.cache_hits[-1] is True
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
    assert recorder.states[states_after_first:][-1] == "async_success"
    assert recorder.cache_hits[0] is not True
    assert recorder.cache_hits[-1] is True
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
    assert recorder.cache_hits[0] is not True
    assert recorder.cache_hits[-1] is True
    assert recorder.errors == []
