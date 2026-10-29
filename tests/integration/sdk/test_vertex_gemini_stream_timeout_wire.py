from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Final

import pytest
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm.exceptions import MidStreamFallbackError

_MODEL: Final = "gemini-3.8-flash"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_MESSAGES: Final = [{"role": "user", "content": "does the per-request timeout reach a stalled stream?"}]
_REQUEST_TIMEOUT_SECONDS: Final = 1.0
_HANG_DEADLINE_SECONDS: Final = 15.0
_TOKEN_REPLY: Final = Reply(
    body=json.dumps({"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}).encode()
)


def _sse(text: str, finish_reason: str | None) -> bytes:
    candidate: Final = {
        "content": {"role": "model", "parts": [{"text": text}]},
        **({"finishReason": finish_reason} if finish_reason else {}),
    }
    usage: Final = {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11}
    return b"data: " + json.dumps({"candidates": [candidate], "usageMetadata": usage}).encode() + b"\n\n"


def _stream_that_stalls_after_the_first_chunk() -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(_sse("partial", None), _sse("", "STOP")),
        gate_after_first=threading.Event(),
    )


def _stream_that_completes() -> Reply:
    return Reply(content_type="text/event-stream", chunks=(_sse("scripted ", None), _sse("answer", "STOP")))


def _body_that_stalls_half_way() -> Reply:
    return Reply(chunks=(b'{"candidates": [', b"]}"), gate_after_first=threading.Event())


def _peer(reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            return _TOKEN_REPLY
        assert request.method == "POST", request.method
        return reply

    return respond


def _deployment(route: str, wire: Wire) -> dict[str, str]:
    if route == "gemini":
        return {"model": f"gemini/{_MODEL}", "api_base": wire.url, "api_key": "scripted-key"}
    return {
        "model": f"vertex_ai/{_MODEL}",
        "api_base": wire.url,
        "vertex_project": _PROJECT,
        "vertex_location": _LOCATION,
        "vertex_credentials": service_account_json(_PROJECT, wire.url),
    }


def _text_of(chunk: object) -> str:
    choices: Final = getattr(chunk, "choices", None) or ()
    return "".join(str(choice.delta.content or "") for choice in choices)


def _complete_sync(route: str, wire: Wire, stream: bool) -> str:
    response: Final = litellm.completion(
        **_deployment(route, wire), messages=_MESSAGES, stream=stream, timeout=_REQUEST_TIMEOUT_SECONDS
    )
    if stream:
        return "".join(_text_of(chunk) for chunk in response)
    return str(response.choices[0].message.content)


async def _complete_async(route: str, wire: Wire, stream: bool) -> str:
    response: Final = await litellm.acompletion(
        **_deployment(route, wire), messages=_MESSAGES, stream=stream, timeout=_REQUEST_TIMEOUT_SECONDS
    )
    if stream:
        parts: Final = [_text_of(chunk) async for chunk in response]
        return "".join(parts)
    return str(response.choices[0].message.content)


def _complete(client: str, route: str, wire: Wire, stream: bool) -> str:
    if client == "async":
        return asyncio.run(_complete_async(route, wire, stream))
    return _complete_sync(route, wire, stream)


def _before_the_hang_deadline(call: Callable[[], str]) -> str:
    pool: Final = ThreadPoolExecutor(max_workers=1)
    future: Final[Future[str]] = pool.submit(call)
    pool.shutdown(wait=False)
    try:
        return future.result(timeout=_HANG_DEADLINE_SECONDS)
    except FutureTimeout:
        pytest.fail(
            f"still waiting on the peer {_HANG_DEADLINE_SECONDS}s after a {_REQUEST_TIMEOUT_SECONDS}s request timeout"
        )


_CLIENTS: Final = ("sync", "async")
_ROUTES: Final = ("vertex_ai", "gemini")


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("client", _CLIENTS)
def test_stalled_stream_raises_at_the_request_timeout(client: str, route: str) -> None:
    with wire_server(_peer(_stream_that_stalls_after_the_first_chunk())) as wire:
        started: Final = time.monotonic()
        with pytest.raises(MidStreamFallbackError):
            _before_the_hang_deadline(lambda: _complete(client, route, wire, stream=True))
        elapsed: Final = time.monotonic() - started
        assert elapsed >= _REQUEST_TIMEOUT_SECONDS, elapsed
        assert len(wire.drain()) >= 1


@pytest.mark.parametrize("client", _CLIENTS)
def test_stalled_body_raises_at_the_request_timeout(client: str) -> None:
    with wire_server(_peer(_body_that_stalls_half_way())) as wire:
        started: Final = time.monotonic()
        with pytest.raises(litellm.Timeout):
            _before_the_hang_deadline(lambda: _complete(client, "vertex_ai", wire, stream=False))
        assert time.monotonic() - started >= _REQUEST_TIMEOUT_SECONDS


@pytest.mark.parametrize("client", _CLIENTS)
def test_stream_with_a_request_timeout_still_completes(client: str) -> None:
    with wire_server(_peer(_stream_that_completes())) as wire:
        assert _complete(client, "vertex_ai", wire, stream=True) == "scripted answer"
