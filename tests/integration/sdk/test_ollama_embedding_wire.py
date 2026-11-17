import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import JSON_OBJECT
from integration._support.timing import assert_cut_short
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm.types.utils import EmbeddingResponse

_MODEL: Final = "nomic-embed-text"
_EXPLICIT_KEY: Final = "synthetic-ollama-explicit-key"
_ENV_KEY: Final = "synthetic-ollama-env-key"
_GLOBAL_KEY: Final = "synthetic-global-key-never-sent"
_CALLER_TOKEN: Final = "Bearer synthetic-caller-token"
_DELAY_SECONDS: Final = 5.0
_SHORT_DELAY_SECONDS: Final = 1.0

Mode = Literal["sync", "async"]
_MODES: Final[tuple[Mode, ...]] = ("sync", "async")


def _vector(index: int) -> list[float]:
    return [0.25, float(index)]


def _embeddings(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    count: Final = len(body["input"]) if isinstance(body["input"], list) else 1
    return Reply(
        body=json.dumps({"embeddings": [_vector(index) for index in range(count)], "prompt_eval_count": 3}).encode()
    )


def _held(release: threading.Event, seconds: float) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        release.wait(timeout=seconds)
        return _embeddings(request)

    return respond


@pytest.fixture(autouse=True)
def _no_ambient_ollama_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)


@contextmanager
def _peer(respond: Callable[[Request], Reply] = _embeddings) -> Iterator[Wire]:
    with wire_server(respond) as wire:
        yield wire


async def _call(mode: Mode, **kwargs: object) -> EmbeddingResponse:
    if mode == "sync":
        return litellm.embedding(model=f"ollama/{_MODEL}", **kwargs)
    return await litellm.aembedding(model=f"ollama/{_MODEL}", **kwargs)


async def _seconds_until_timeout(mode: Mode, **kwargs: object) -> float:
    started: Final = time.monotonic()
    with pytest.raises(litellm.Timeout):
        await _call(mode, **kwargs)
    return time.monotonic() - started


def _only(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/api/embed")], [
        (request.method, request.target) for request in received
    ]
    return received[0]


def _values_of(request: Request, name: str) -> list[str]:
    return [value for header, value in request.header_lines if header.lower() == name.lower()]


def _text() -> str:
    return f"ollama embedding {uuid.uuid4().hex}"


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    (
        (_EXPLICIT_KEY, None, _EXPLICIT_KEY),
        (None, _ENV_KEY, _ENV_KEY),
        (_EXPLICIT_KEY, _ENV_KEY, _EXPLICIT_KEY),
    ),
    ids=("explicit-key", "env-key", "explicit-beats-env"),
)
async def test_regression_embedding_sends_the_resolved_ollama_key_as_one_bearer_header(
    monkeypatch: pytest.MonkeyPatch, mode: Mode, explicit: str | None, env: str | None, expected: str
) -> None:
    if env is not None:
        monkeypatch.setenv("OLLAMA_API_KEY", env)
    texts: Final = [_text(), _text()]
    with _peer() as wire:
        response: Final = await _call(mode, input=texts, api_base=wire.url, api_key=explicit)
        request: Final = _only(wire)
    assert [row["embedding"] for row in response.data] == [_vector(0), _vector(1)], response
    assert _values_of(request, "authorization") == [f"Bearer {expected}"], request.header_lines
    assert JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": texts}, request.body


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("auth_name", ("Authorization", "authorization"), ids=("canonical", "lowercase"))
@pytest.mark.parametrize("header_kwarg", ("extra_headers", "headers"))
async def test_regression_caller_headers_reach_ollama_and_override_the_key_once(
    mode: Mode, auth_name: str, header_kwarg: str
) -> None:
    caller: Final[dict[str, str]] = {auth_name: _CALLER_TOKEN, "X-Embed-Tenant": "tenant-a"}
    snapshot: Final = dict(caller)
    text: Final = _text()
    with _peer() as wire:
        response: Final = await _call(
            mode, input=text, api_base=wire.url, api_key=_EXPLICIT_KEY, **{header_kwarg: caller}
        )
        request: Final = _only(wire)
    assert [row["embedding"] for row in response.data] == [_vector(0)], response
    assert _values_of(request, "authorization") == [_CALLER_TOKEN], request.header_lines
    assert _values_of(request, "x-embed-tenant") == ["tenant-a"], request.header_lines
    assert caller == snapshot, "the caller's header map was changed by the call"
    assert JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": [text]}, request.body


@pytest.mark.parametrize("mode", _MODES)
async def test_regression_extra_headers_win_over_headers_and_both_reach_ollama(mode: Mode) -> None:
    extra: Final[dict[str, str]] = {"X-Embed-Tenant": "from-extra-headers"}
    extra_snapshot: Final = dict(extra)
    with _peer() as wire:
        await _call(
            mode,
            input=_text(),
            api_base=wire.url,
            headers={"X-Embed-Tenant": "from-headers", "X-Embed-Trace": "trace-1"},
            extra_headers=extra,
        )
        request: Final = _only(wire)
    assert _values_of(request, "x-embed-tenant") == ["from-extra-headers"], request.header_lines
    assert _values_of(request, "x-embed-trace") == ["trace-1"], request.header_lines
    assert extra == extra_snapshot, "the caller's extra_headers map was changed by the call"


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("timeout", (0.5, httpx.Timeout(0.5)), ids=("float", "httpx-timeout"))
async def test_regression_short_timeout_cuts_a_held_ollama_read(mode: Mode, timeout: float | httpx.Timeout) -> None:
    release: Final = threading.Event()
    with _peer(_held(release, _DELAY_SECONDS)) as wire:
        try:
            elapsed: Final = await _seconds_until_timeout(mode, input=_text(), api_base=wire.url, timeout=timeout)
        finally:
            release.set()
        _only(wire)
    assert_cut_short(elapsed, injected_delay=_DELAY_SECONDS)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("timeout_kwargs", ({}, {"timeout": None}), ids=("omitted", "none"))
async def test_guard_default_timeout_does_not_cut_a_slow_ollama_reply(
    mode: Mode, timeout_kwargs: Mapping[str, None]
) -> None:
    release: Final = threading.Event()
    with _peer(_held(release, _SHORT_DELAY_SECONDS)) as wire:
        response: Final = await _call(mode, input=_text(), api_base=wire.url, **timeout_kwargs)
        _only(wire)
    assert [row["embedding"] for row in response.data] == [_vector(0)], response


@pytest.mark.parametrize("mode", _MODES)
async def test_guard_no_ollama_key_sends_no_authorization_even_with_global_keys(
    monkeypatch: pytest.MonkeyPatch, mode: Mode
) -> None:
    monkeypatch.setattr(litellm, "api_key", _GLOBAL_KEY)
    monkeypatch.setattr(litellm, "ollama_key", _GLOBAL_KEY, raising=False)
    monkeypatch.setattr(litellm, "openai_key", _GLOBAL_KEY)
    texts: Final = [_text(), _text()]
    with _peer() as wire:
        response: Final = await _call(mode, input=texts, api_base=f"{wire.url}/")
        request: Final = _only(wire)
    assert [row["embedding"] for row in response.data] == [_vector(0), _vector(1)], response
    assert _values_of(request, "authorization") == [], request.header_lines
    assert not any(_GLOBAL_KEY in value for _, value in request.header_lines), request.header_lines
    assert JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": texts}, request.body


@pytest.mark.parametrize("mode", _MODES)
async def test_guard_non_text_input_is_rejected_before_ollama_is_called(mode: Mode) -> None:
    valid: Final = _text()
    with _peer() as wire:
        with pytest.raises(litellm.BadRequestError):
            await _call(mode, input=[[1, 2, 3]], api_base=wire.url, api_key=_EXPLICIT_KEY)
        await _call(mode, input=valid, api_base=wire.url)
        received: Final = wire.drain()
    assert [JSON_OBJECT.validate_json(request.body)["input"] for request in received] == [[valid]]


@pytest.mark.parametrize("mode", _MODES)
async def test_guard_ollama_error_reaches_the_caller_as_its_status(mode: Mode) -> None:
    marker: Final = f"scripted-unauthorized-{uuid.uuid4().hex}"

    def respond(_: Request) -> Reply:
        return Reply(status=401, body=json.dumps({"error": marker}).encode())

    with _peer(respond) as wire:
        with pytest.raises(litellm.AuthenticationError) as caught:
            await _call(mode, input=_text(), api_base=wire.url)
        _only(wire)
    assert marker in str(caught.value), caught.value
