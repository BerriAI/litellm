from __future__ import annotations

import itertools
import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Final

import pytest
from _pytest.mark.structures import ParameterSet
from integration._support.client import JSON_OBJECT
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._gemini_long_decode_support import (
    API_KEY,
    BACKEND,
    CAP,
    CHECKPOINT,
    FIRST_HALF,
    HOP_TOKEN,
    LOCATION,
    MODEL_PATH,
    PROJECT,
    PROMPT,
    PROVIDERS,
    SECOND_HALF,
    Provider,
    Seen,
    answers,
    seen,
    summed,
    target,
    upstream,
)
from pydantic import JsonValue

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

Respond = Callable[[Request], Reply]

_TOKEN_PATH: Final = "/_oauth/token"
_EXPIRED: Final = "The scripted Google refused an access token that had expired"
_REVOKED: Final = "The scripted token endpoint revoked this service account"
_JOINED: Final = FIRST_HALF.text + SECOND_HALF.text
_CHAIN_USAGE: Final = summed(FIRST_HALF, SECOND_HALF)


@dataclass(frozen=True, slots=True)
class _Route:
    api_base: str
    api_key: str | None
    vertex_project: str | None
    vertex_location: str | None
    vertex_credentials: str | None


def _mode(stream: bool) -> str:
    return "stream" if stream else "unary"


_CELLS: Final[tuple[ParameterSet, ...]] = tuple(
    pytest.param(provider, stream, id=f"{provider}-{_mode(stream)}")
    for provider, stream in itertools.product(PROVIDERS, (False, True))
)
_VERTEX_CELLS: Final[tuple[ParameterSet, ...]] = tuple(
    pytest.param(stream, id=f"vertex_ai-{_mode(stream)}") for stream in (False, True)
)


def _route(provider: Provider, wire: Wire) -> _Route:
    if provider == "gemini":
        return _Route(wire.url, API_KEY, None, None, None)
    return _Route(f"{wire.url}{MODEL_PATH}", None, PROJECT, LOCATION, service_account_json(PROJECT, wire.url))


def _complete(provider: Provider, wire: Wire, stream: bool) -> Seen:
    route: Final = _route(provider, wire)
    response: Final = litellm.completion(
        model=f"{provider}/{BACKEND}",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=CAP,
        stream=stream,
        stream_options={"include_usage": True} if stream else None,
        api_base=route.api_base,
        api_key=route.api_key,
        vertex_project=route.vertex_project,
        vertex_location=route.vertex_location,
        vertex_credentials=route.vertex_credentials,
        num_retries=0,
    )
    if isinstance(response, CustomStreamWrapper):
        chunks: Final = tuple(JSON_OBJECT.validate_json(chunk.model_dump_json()) for chunk in response)
        return seen("chat", True, chunks, "\n".join(json.dumps(chunk) for chunk in chunks))
    assert isinstance(response, ModelResponse), type(response)
    payload: Final = JSON_OBJECT.validate_json(response.model_dump_json())
    return seen("chat", False, (payload,), json.dumps(payload))


def _token_reply(token: str, lifetime: int) -> Reply:
    return Reply(body=json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": lifetime}).encode())


def _tokens_lasting_an_hour(google: Respond) -> Respond:
    issued: Final = itertools.count()

    def respond(request: Request) -> Reply:
        if request.target == _TOKEN_PATH:
            return _token_reply(f"issued-{next(issued)}", 3600)
        return google(request)

    return respond


def _tokens_that_expire_by_the_next_request(google: Respond) -> Respond:
    issued: Final = itertools.count()
    unspent: Final[SimpleQueue[str]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        if request.target == _TOKEN_PATH:
            token: Final = f"issued-{next(issued)}"
            unspent.put(token)
            return _token_reply(token, 0)
        valid: Final = frozenset(f"Bearer {unspent.get()}" for _ in range(unspent.qsize()))
        if request.headers.get("authorization") in valid:
            return google(request)
        error: Final = {"error": {"code": 401, "message": _EXPIRED, "status": "UNAUTHENTICATED"}}
        return Reply(status=401, body=json.dumps(error).encode())

    return respond


def _tokens_revoked_once_google_is_asked(google: Respond) -> Respond:
    issued: Final = itertools.count()
    asked: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if request.target != _TOKEN_PATH:
            asked.set()
            return google(request)
        if not asked.is_set():
            return _token_reply(f"issued-{next(issued)}", 0)
        refusal: Final = {"error": "invalid_grant", "error_description": _REVOKED}
        return Reply(status=400, body=json.dumps(refusal).encode())

    return respond


def _chain(first_id: str) -> Respond:
    return upstream(answers(first_id, FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF))


def _google_calls(requests: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in requests if request.target != _TOKEN_PATH)


def _sent_tokens(calls: tuple[Request, ...]) -> list[JsonValue]:
    return [JSON_OBJECT.validate_json(call.body).get("continuationToken") for call in calls]


@pytest.mark.parametrize(("provider", "stream"), _CELLS)
def test_sync_completion_joins_an_answer_google_cut_with_a_continuation_token(provider: Provider, stream: bool) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with wire_server(_tokens_lasting_an_hour(_chain(first_id))) as wire:
        observed: Final = _complete(provider, wire, stream)
        calls: Final = _google_calls(wire.drain())
    assert (observed.answer, observed.finishes, observed.usage) == (_JOINED, ("stop",), _CHAIN_USAGE), observed
    assert observed.tokens == () and observed.ids == frozenset({first_id}), observed
    assert CHECKPOINT not in observed.raw and HOP_TOKEN not in observed.raw, observed.raw
    assert [call.target for call in calls] == [target(provider, stream), target(provider, stream, True)], calls
    assert _sent_tokens(calls) == [None, HOP_TOKEN], calls


@pytest.mark.parametrize("stream", _VERTEX_CELLS)
def test_sync_vertex_continuation_sent_after_the_access_token_expired_carries_a_new_one_and_completes(
    stream: bool,
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with wire_server(_tokens_that_expire_by_the_next_request(_chain(first_id))) as wire:
        observed: Final = _complete("vertex_ai", wire, stream)
        calls: Final = _google_calls(wire.drain())
    assert (observed.answer, observed.finishes, observed.usage) == (_JOINED, ("stop",), _CHAIN_USAGE), observed
    assert observed.tokens == () and _EXPIRED not in observed.raw, observed.raw
    assert [call.target for call in calls] == [target("vertex_ai", stream), target("vertex_ai", stream, True)]
    assert len({call.headers.get("authorization") for call in calls}) == 2, calls
    assert _sent_tokens(calls) == [None, HOP_TOKEN], calls


@pytest.mark.parametrize("stream", _VERTEX_CELLS)
def test_sync_vertex_access_token_that_cannot_be_renewed_returns_the_answer_so_far_with_its_token(
    stream: bool,
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with wire_server(_tokens_revoked_once_google_is_asked(_chain(first_id))) as wire:
        observed: Final = _complete("vertex_ai", wire, stream)
        calls: Final = _google_calls(wire.drain())
    assert (observed.answer, observed.finishes, observed.usage) == (FIRST_HALF.text, ("length",), FIRST_HALF.usage)
    assert observed.tokens == (HOP_TOKEN,) and observed.ids == frozenset({first_id}), observed
    assert _REVOKED not in observed.raw and CHECKPOINT not in observed.raw, observed.raw
    assert [call.target for call in calls] == [target("vertex_ai", stream)], calls
