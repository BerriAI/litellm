from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import CustomStreamWrapper, Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream

_MODEL: Final = "gpt-5.4"
_API_KEY: Final = "synthetic-fallback-errors-key"
_ROUTER_ONLY_KEYS: Final = ("include_fallback_errors", "silent_model")
_ANSWER: Final = "answered by"
_UNAUTHORIZED_MESSAGE: Final = "scripted 401: the primary key was revoked"
_NONCE: Final = re.compile(r"nonce=([0-9a-f]{32})")
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_OBJECT: Final = TypeAdapter(dict[str, object])
_ERRORS: Final = TypeAdapter(list[dict[str, JsonValue]])
_BURST: Final = 12
_CALLBACK_WINDOW: Final = 15.0
_STREAMING: Final = (pytest.param(False, id="non-stream"), pytest.param(True, id="stream"))
_NON_BOOLEAN_FLAGS: Final = (
    pytest.param(1, True, id="int"),
    pytest.param("", False, id="empty-string"),
    pytest.param([], False, id="list"),
    pytest.param("x" * 5120, True, id="five-kilobytes"),
    pytest.param(False, False, id="false"),
)
_UNAUTHORIZED: Final = Reply(
    status=401,
    body=json.dumps(
        {
            "error": {
                "message": _UNAUTHORIZED_MESSAGE,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
    ).encode(),
)


def _prompt(nonce: str) -> str:
    return f"which deployment answers this? nonce={nonce}"


def _nonce_of(body: Mapping[str, JsonValue]) -> str:
    found: Final = _NONCE.search(json.dumps(body))
    assert found is not None, body
    return found.group(1)


def _peer(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    deployment, _, route = request.target.lstrip("/").partition("/")
    assert route == "chat/completions", request.target
    if deployment == "primary":
        return _UNAUTHORIZED
    body: Final = _JSON.validate_json(request.body)
    identity: Final = f"chatcmpl-{deployment}-{_nonce_of(body)}"
    return chat_reply(identity, _MODEL, f"{_ANSWER} {deployment}", stream=body.get("stream") is True)


def _deployment(name: str, wire: Wire, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"openai/{_MODEL}",
            "api_base": f"{wire.url}/{name}",
            "api_key": _API_KEY,
            **extra,
        },
    }


def _router(wire: Wire, *, cache_responses: bool = False) -> Router:
    return Router(
        model_list=[
            _deployment("primary", wire),
            _deployment("backup", wire),
            _deployment("serving", wire),
            _deployment("mirrored", wire, silent_model="shadow"),
            _deployment("shadow", wire),
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=0,
        disable_cooldowns=True,
        cache_responses=cache_responses,
    )


@dataclass(frozen=True, slots=True)
class _Posted:
    deployment: str
    raw: str

    def leaked(self) -> tuple[str, ...]:
        return tuple(key for key in _ROUTER_ONLY_KEYS if key in self.raw)

    def nonce(self) -> str:
        found: Final = _NONCE.search(self.raw)
        assert found is not None, self.raw
        return found.group(1)


def _posted(wire: Wire) -> tuple[_Posted, ...]:
    return tuple(
        _Posted(request.target.lstrip("/").partition("/")[0], request.body.decode())
        for request in wire.drain()
        if request.method == "POST"
    )


def _leaks(posted: tuple[_Posted, ...]) -> tuple[str, ...]:
    return tuple(f"{item.deployment}:{','.join(item.leaked())}" for item in posted if item.leaked())


def _hit(posted: tuple[_Posted, ...]) -> tuple[str, ...]:
    return tuple(item.deployment for item in posted)


def _nonces_at(posted: tuple[_Posted, ...], deployment: str) -> tuple[str, ...]:
    return tuple(sorted(item.nonce() for item in posted if item.deployment == deployment))


def _hidden(response: object) -> Mapping[str, object]:
    return _OBJECT.validate_python(getattr(response, "_hidden_params", None) or {})


def _headers(response: object) -> Mapping[str, object]:
    return _OBJECT.validate_python(_hidden(response).get("additional_headers") or {})


def _cache_hit(response: object) -> bool:
    return _hidden(response).get("cache_hit") is True


def _error_messages(headers: Mapping[str, object]) -> tuple[str, ...]:
    raw: Final = headers.get("x-litellm-fallback-errors")
    if raw is None:
        return ()
    return tuple(str(error["message"]) for error in _ERRORS.validate_json(str(raw)))


def _delta(chunk: object) -> str:
    assert isinstance(chunk, ModelResponseStream), chunk
    return "".join(str(choice.delta.content or "") for choice in chunk.choices)


def _messages(nonce: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": _prompt(nonce)}]


def _typed_messages(nonce: str) -> list[AllMessageValues]:
    return [ChatCompletionUserMessage(role="user", content=_prompt(nonce))]


def _content(response: object) -> str:
    assert isinstance(response, ModelResponse), response
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices), choice
    return str(choice.message.content)


@dataclass(frozen=True, slots=True)
class _Served:
    text: str
    headers: Mapping[str, object]
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class _Outcome:
    text: str
    attempted: object
    errors: tuple[str, ...]
    hit: tuple[str, ...]
    leaks: tuple[str, ...]


def _outcome(wire: Wire, served: _Served) -> _Outcome:
    posted: Final = _posted(wire)
    return _Outcome(
        text=served.text,
        attempted=served.headers.get("x-litellm-attempted-fallbacks"),
        errors=_error_messages(served.headers),
        hit=_hit(posted),
        leaks=_leaks(posted),
    )


def _complete(router: Router, model: str, *, stream: bool, nonce: str | None = None, **request: object) -> _Served:
    response: Final = router.completion(
        model=model, messages=_messages(nonce or uuid.uuid4().hex), stream=stream, **request
    )
    if isinstance(response, CustomStreamWrapper):
        return _Served("".join(_delta(chunk) for chunk in response), _headers(response), _cache_hit(response))
    return _Served(_content(response), _headers(response), _cache_hit(response))


async def _acomplete(router: Router, model: str, *, stream: bool, **request: object) -> _Served:
    response: Final = await router.acompletion(
        model=model, messages=_typed_messages(uuid.uuid4().hex), stream=stream, **request
    )
    if isinstance(response, CustomStreamWrapper):
        parts: Final = [_delta(chunk) async for chunk in response]
        return _Served("".join(parts), _headers(response), _cache_hit(response))
    return _Served(_content(response), _headers(response), _cache_hit(response))


@pytest.mark.parametrize("stream", _STREAMING)
def test_sync_fallback_keeps_the_flag_off_the_wire_and_reports_the_errors(stream: bool) -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        outcome: Final = _outcome(wire, _complete(router, "primary", stream=stream, include_fallback_errors=True))
    assert outcome.leaks == (), outcome
    assert outcome.hit == ("primary", "backup"), outcome
    assert outcome.text == f"{_ANSWER} backup", outcome
    assert outcome.attempted == 1, outcome
    if not stream:
        assert len(outcome.errors) == 1 and _UNAUTHORIZED_MESSAGE in outcome.errors[0], outcome


@pytest.mark.parametrize("stream", _STREAMING)
def test_sync_matches_the_async_twin(stream: bool) -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        twin: Final = _outcome(
            wire, asyncio.run(_acomplete(router, "primary", stream=stream, include_fallback_errors=True))
        )
        observed: Final = _outcome(wire, _complete(router, "primary", stream=stream, include_fallback_errors=True))
    assert observed == twin, (observed, twin)
    assert twin.leaks == (), twin
    assert twin.hit == ("primary", "backup"), twin


def test_without_a_fallback_the_flag_still_stays_off_the_wire() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        outcome: Final = _outcome(wire, _complete(router, "serving", stream=False, include_fallback_errors=True))
    assert outcome.leaks == (), outcome
    assert outcome.hit == ("serving",), outcome
    assert outcome.text == f"{_ANSWER} serving", outcome
    assert outcome.attempted == 0, outcome
    assert outcome.errors == (), outcome


@pytest.mark.parametrize(("value", "errors_reported"), _NON_BOOLEAN_FLAGS)
def test_a_non_boolean_flag_stays_off_the_wire_and_the_errors_header_follows_its_truthiness(
    value: object, errors_reported: bool
) -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        outcome: Final = _outcome(wire, _complete(router, "primary", stream=False, include_fallback_errors=value))
    assert outcome.leaks == (), outcome
    assert outcome.hit == ("primary", "backup"), outcome
    assert outcome.text == f"{_ANSWER} backup", outcome
    assert outcome.attempted == 1, outcome
    assert (len(outcome.errors) == 1) is errors_reported, outcome


class _Recorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.mentions_the_flag: bool | None = None
        self.fired = threading.Event()

    def _record(self, kwargs: Mapping[str, object]) -> None:
        self.mentions_the_flag = "include_fallback_errors" in json.dumps(kwargs, default=str)
        self.fired.set()

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs)


async def _acomplete_and_await_the_logger(router: Router, recorder: _Recorder) -> _Served:
    served: Final = await _acomplete(router, "serving", stream=False, include_fallback_errors=True)
    assert await asyncio.get_running_loop().run_in_executor(None, recorder.fired.wait, _CALLBACK_WINDOW)
    return served


def test_sync_logger_kwargs_carry_the_flag_exactly_as_the_async_ones_do(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_recorder: Final = _Recorder()
    async_recorder: Final = _Recorder()
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        monkeypatch.setattr(litellm, "callbacks", [sync_recorder])
        _complete(router, "serving", stream=False, include_fallback_errors=True)
        assert sync_recorder.fired.wait(_CALLBACK_WINDOW)
        monkeypatch.setattr(litellm, "callbacks", [async_recorder])
        asyncio.run(_acomplete_and_await_the_logger(router, async_recorder))
        posted: Final = _posted(wire)
    assert _leaks(posted) == (), posted
    assert (sync_recorder.mentions_the_flag, async_recorder.mentions_the_flag) == (False, False)


def test_silent_model_shadow_traffic_carries_neither_router_only_key() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        served: Final = _complete(router, "mirrored", stream=False, include_fallback_errors=True)
        eventually(wire.received.qsize, lambda count: count >= 2, seconds=20)
        posted: Final = _posted(wire)
    assert served.text == f"{_ANSWER} mirrored", served
    assert tuple(sorted(_hit(posted))) == ("mirrored", "shadow"), posted
    assert _leaks(posted) == (), posted


def test_a_cache_hit_repeats_the_answer_without_a_wire_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", None)
    with wire_server(_peer) as wire:
        router: Final = _router(wire, cache_responses=True)
        nonce: Final = uuid.uuid4().hex
        first: Final = _complete(router, "serving", stream=False, nonce=nonce, include_fallback_errors=True)
        posted: Final = _posted(wire)
        second: Final = _complete(router, "serving", stream=False, nonce=nonce, include_fallback_errors=True)
        again: Final = _posted(wire)
    assert _hit(posted) == ("serving",), posted
    assert _leaks(posted) == (), posted
    assert again == (), again
    assert second.text == first.text == f"{_ANSWER} serving"
    assert (first.cache_hit, second.cache_hit) == (False, True), (first, second)


def _flagged(router: Router, nonce: str) -> _Served:
    return _complete(router, "primary", stream=False, nonce=nonce, include_fallback_errors=True)


def test_a_sync_burst_lands_every_prompt_once_on_each_side_of_the_fallback() -> None:
    nonces: Final = tuple(uuid.uuid4().hex for _ in range(_BURST))
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        with ThreadPoolExecutor(max_workers=_BURST) as pool:
            served: Final = tuple(pool.map(partial(_flagged, router), nonces))
        posted: Final = _posted(wire)
    assert tuple(item.text for item in served) == (f"{_ANSWER} backup",) * _BURST, served
    assert tuple(item.headers.get("x-litellm-attempted-fallbacks") for item in served) == (1,) * _BURST, served
    assert _leaks(posted) == (), posted
    assert _nonces_at(posted, "primary") == tuple(sorted(nonces)), posted
    assert _nonces_at(posted, "backup") == tuple(sorted(nonces)), posted
