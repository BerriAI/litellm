from __future__ import annotations

import contextlib
import itertools
import json
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import anthropic
import httpx
import openai
import pytest
import yaml
from _pytest.mark.structures import ParameterSet
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._gemini_long_decode_support import (
    API_KEY,
    BACKEND,
    CAP,
    CHECKPOINT,
    CUT,
    CUT_TOKEN,
    ENDPOINTS,
    FIRST_HALF,
    HOP_TOKEN,
    NO_CACHE,
    NOTHING_EXTRA,
    PLAIN,
    PROMPT,
    PROVIDERS,
    REFUSAL,
    RESUMED,
    SECOND_HALF,
    SPEND_COLUMNS,
    Answer,
    Endpoint,
    Hop,
    Provider,
    Seen,
    answered,
    answers,
    assert_billed,
    assert_one_id,
    at,
    carries_token,
    complete,
    cost_of,
    cut_off,
    drops_before_answering,
    drops_mid_body,
    ends_cleanly,
    frame,
    hop_reply,
    items_of,
    number,
    output_text,
    payloads,
    post,
    received,
    refuses,
    register,
    register_vertex,
    request_body,
    seen,
    single,
    spend_rows,
    sse,
    stream_frames,
    streams,
    summed,
    target,
    text_of,
    upstream,
)
from pydantic import BaseModel, JsonValue

Client = Literal["sdk_sync", "sdk_async", "httpx"]
Failure = Literal["400", "500", "drop", "notjson", "errorframe"]
Dropped = Literal["first", "continued"]
Unread = Literal["dropped", "empty", "unfinished", "silent"]
DeploymentTimeout = Literal["timeout", "stream_timeout"]
Absent = Literal["missing", "null", "empty"]
Respond: TypeAlias = Callable[[Request], Reply]

_CLIENTS: Final[tuple[Client, ...]] = ("sdk_sync", "sdk_async", "httpx")
_FAILURES: Final[tuple[Failure, ...]] = ("400", "500", "drop", "notjson", "errorframe")
_UNREAD: Final[tuple[Unread, ...]] = ("dropped", "empty", "unfinished", "silent")
_DEPLOYMENT_TIMEOUTS: Final[tuple[DeploymentTimeout, ...]] = ("timeout", "stream_timeout")
_JOINED: Final = FIRST_HALF.text + SECOND_HALF.text
_CHAIN_USAGE: Final = summed(FIRST_HALF, SECOND_HALF)
_CHAT_MESSAGES: Final = ({"role": "user", "content": PROMPT},)
_USAGE_IN_STREAM: Final = {"include_usage": True}
_NATIVE_BODY: Final[Mapping[str, JsonValue]] = {
    "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
    "generationConfig": {"maxOutputTokens": CAP},
}


def _mode(stream: bool) -> str:
    return "stream" if stream else "sync"


def _cells(providers: Sequence[Provider], endpoints: Sequence[Endpoint]) -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(provider, endpoint, stream, id=f"{provider}-{endpoint}-{_mode(stream)}")
        for provider, endpoint, stream in itertools.product(providers, endpoints, (False, True))
    )


def _chain_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(provider, endpoint, stream, client, id=f"{provider}-{endpoint}-{_mode(stream)}-{client}")
        for provider, endpoint, stream, client in itertools.product(PROVIDERS, ENDPOINTS, (False, True), _CLIENTS)
    )


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_openai(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _anthropic(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_anthropic(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _dumped(endpoint: Endpoint, stream: bool, models: Sequence[BaseModel]) -> Seen:
    texts: Final = tuple(model.model_dump_json(by_alias=True, exclude_unset=True) for model in models)
    return seen(endpoint, stream, tuple(JSON_OBJECT.validate_json(text) for text in texts), "\n".join(texts))


def _chat_sync(gateway: Gateway, model: str, stream: bool) -> Seen:
    client: Final = _openai(gateway)
    if not stream:
        reply: Final = client.chat.completions.create(
            model=model, messages=_CHAT_MESSAGES, max_tokens=CAP, extra_body=dict(NO_CACHE)
        )
        return _dumped("chat", False, (reply,))
    chunks: Final = tuple(
        client.chat.completions.create(
            model=model,
            messages=_CHAT_MESSAGES,
            max_tokens=CAP,
            stream=True,
            stream_options=_USAGE_IN_STREAM,
            extra_body=dict(NO_CACHE),
        )
    )
    return _dumped("chat", True, chunks)


async def _chat_async(gateway: Gateway, model: str, stream: bool) -> Seen:
    client: Final = _async_openai(gateway)
    if not stream:
        reply: Final = await client.chat.completions.create(
            model=model, messages=_CHAT_MESSAGES, max_tokens=CAP, extra_body=dict(NO_CACHE)
        )
        return _dumped("chat", False, (reply,))
    chunks: Final = [
        chunk
        async for chunk in await client.chat.completions.create(
            model=model,
            messages=_CHAT_MESSAGES,
            max_tokens=CAP,
            stream=True,
            stream_options=_USAGE_IN_STREAM,
            extra_body=dict(NO_CACHE),
        )
    ]
    return _dumped("chat", True, chunks)


def _messages_sync(gateway: Gateway, model: str, stream: bool) -> Seen:
    client: Final = _anthropic(gateway)
    if not stream:
        reply: Final = client.messages.create(
            model=model, max_tokens=CAP, messages=_CHAT_MESSAGES, extra_body=dict(NO_CACHE)
        )
        return _dumped("messages", False, (reply,))
    events: Final = tuple(
        client.messages.create(
            model=model, max_tokens=CAP, messages=_CHAT_MESSAGES, stream=True, extra_body=dict(NO_CACHE)
        )
    )
    return _dumped("messages", True, events)


async def _messages_async(gateway: Gateway, model: str, stream: bool) -> Seen:
    client: Final = _async_anthropic(gateway)
    if not stream:
        reply: Final = await client.messages.create(
            model=model, max_tokens=CAP, messages=_CHAT_MESSAGES, extra_body=dict(NO_CACHE)
        )
        return _dumped("messages", False, (reply,))
    events: Final = [
        event
        async for event in await client.messages.create(
            model=model, max_tokens=CAP, messages=_CHAT_MESSAGES, stream=True, extra_body=dict(NO_CACHE)
        )
    ]
    return _dumped("messages", True, events)


def _responses_sync(gateway: Gateway, model: str, stream: bool) -> Seen:
    client: Final = _openai(gateway)
    if not stream:
        reply: Final = client.responses.create(
            model=model, input=PROMPT, max_output_tokens=CAP, extra_body=dict(NO_CACHE)
        )
        return _dumped("responses", False, (reply,))
    events: Final = tuple(
        client.responses.create(
            model=model, input=PROMPT, max_output_tokens=CAP, stream=True, extra_body=dict(NO_CACHE)
        )
    )
    return _dumped("responses", True, events)


async def _responses_async(gateway: Gateway, model: str, stream: bool) -> Seen:
    client: Final = _async_openai(gateway)
    if not stream:
        reply: Final = await client.responses.create(
            model=model, input=PROMPT, max_output_tokens=CAP, extra_body=dict(NO_CACHE)
        )
        return _dumped("responses", False, (reply,))
    events: Final = [
        event
        async for event in await client.responses.create(
            model=model, input=PROMPT, max_output_tokens=CAP, stream=True, extra_body=dict(NO_CACHE)
        )
    ]
    return _dumped("responses", True, events)


def _over_httpx(gateway: Gateway, endpoint: Endpoint, model: str, stream: bool) -> Seen:
    response: Final = post(gateway, endpoint, request_body(endpoint, model, stream))
    observed: Final = answered(endpoint, stream, response)
    assert not stream or ends_cleanly(endpoint, response.text), response.text
    return observed


async def _observe(gateway: Gateway, endpoint: Endpoint, client: Client, model: str, stream: bool) -> Seen:
    match (endpoint, client):
        case ("chat", "sdk_sync"):
            return _chat_sync(gateway, model, stream)
        case ("chat", "sdk_async"):
            return await _chat_async(gateway, model, stream)
        case ("messages", "sdk_sync"):
            return _messages_sync(gateway, model, stream)
        case ("messages", "sdk_async"):
            return await _messages_async(gateway, model, stream)
        case ("responses", "sdk_sync"):
            return _responses_sync(gateway, model, stream)
        case ("responses", "sdk_async"):
            return await _responses_async(gateway, model, stream)
        case _:
            return _over_httpx(gateway, endpoint, model, stream)


def _identities(observed: Seen, response_id: str) -> frozenset[str]:
    return frozenset({single(observed.ids), response_id})


@pytest.mark.parametrize(("provider", "endpoint", "stream", "client"), _chain_cells())
async def test_answer_google_ends_early_is_continued_into_one_joined_answer(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, client: Client
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    respond: Final = upstream(answers(first_id, FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        observed: Final = await _observe(gateway, endpoint, client, model, stream)
        assert (observed.answer, observed.streamed) == (_JOINED, _JOINED), observed
        assert observed.finishes == (complete(endpoint),), observed
        assert observed.usage == _CHAIN_USAGE, observed
        assert observed.initial_prompt_tokens == (FIRST_HALF.prompt_tokens if endpoint == "chat" else None), observed
        assert observed.tokens == () and observed.errors == (), observed
        assert CHECKPOINT not in observed.raw and HOP_TOKEN not in observed.raw, observed.raw
        assert_one_id(endpoint, stream, observed, first_id)
        opening, continuing = received(wire, provider, stream, (False, True))
        assert at(opening, "generationConfig", "max_output_tokens") == CAP, opening
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, _CHAIN_USAGE, _identities(observed, first_id))


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_answer_cut_at_the_callers_cap_returns_from_one_request_and_chat_carries_its_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    response_id: Final = f"cut-{uuid.uuid4().hex}"
    with wire_server(upstream(answers(response_id, CUT))) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = post(gateway, endpoint, request_body(endpoint, model, stream))
        observed: Final = answered(endpoint, stream, response)
        assert not stream or ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.streamed) == (CUT.text, CUT.text), observed
        assert observed.finishes == (cut_off(endpoint),), observed
        assert observed.incomplete_reason == ("max_output_tokens" if endpoint == "responses" else None), observed
        assert (observed.usage, observed.initial_prompt_tokens) == (CUT.usage, None), observed
        assert CHECKPOINT not in observed.raw, observed.raw
        assert observed.tokens == ((CUT_TOKEN,) if carries_token(endpoint, stream) else ()), observed
        assert (CUT_TOKEN in observed.raw) == carries_token(endpoint, stream), observed.raw
        assert observed.token_carriers == (observed.content_carriers[-1:] if endpoint == "chat" else ()), observed
        assert_one_id(endpoint, stream, observed, response_id)
        (only,) = received(wire, provider, stream, (False,))
        assert at(only, "generationConfig", "max_output_tokens") == CAP, only
        assert_billed(model, endpoint, CUT.usage, _identities(observed, response_id))


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_resend_with_a_continuation_token_reaches_google_and_returns_the_rest(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    resumed_id: Final = f"resumed-{uuid.uuid4().hex}"
    respond: Final = upstream(answers(f"restarted-{uuid.uuid4().hex}", PLAIN), answers(resumed_id, RESUMED))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body(endpoint, model, stream, {**NO_CACHE, "continuation_token": CUT_TOKEN})
        observed: Final = answered(endpoint, stream, post(gateway, endpoint, body))
        assert (observed.answer, observed.streamed) == (RESUMED.text, RESUMED.text), observed
        assert observed.finishes == (complete(endpoint),), observed
        assert (observed.usage, observed.tokens) == (RESUMED.usage, ()), observed
        assert CHECKPOINT not in observed.raw, observed.raw
        assert_one_id(endpoint, stream, observed, resumed_id)
        (only,) = received(wire, provider, stream, (True,))
        assert only["continuationToken"] == CUT_TOKEN and "continuation_token" not in json.dumps(only), only
        assert_billed(model, endpoint, RESUMED.usage, _identities(observed, resumed_id))


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_answer_google_finishes_itself_stays_one_request_with_no_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    response_id: Final = f"plain-{uuid.uuid4().hex}"
    with wire_server(upstream(answers(response_id, PLAIN))) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = post(gateway, endpoint, request_body(endpoint, model, stream))
        observed: Final = answered(endpoint, stream, response)
        assert not stream or ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.streamed) == (PLAIN.text, PLAIN.text), observed
        assert observed.finishes == (complete(endpoint),), observed
        assert (observed.usage, observed.initial_prompt_tokens) == (PLAIN.usage, None), observed
        assert (observed.tokens, observed.token_carriers) == ((), ()), observed
        assert CHECKPOINT not in observed.raw, observed.raw
        assert_one_id(endpoint, stream, observed, response_id)
        (only,) = received(wire, provider, stream, (False,))
        assert "continuationToken" not in only, only
        assert_billed(model, endpoint, PLAIN.usage, _identities(observed, response_id))


def _unavailable_mid_stream(stream: bool) -> Reply:
    if not stream:
        return refuses(503)(stream)
    error: Final = {"error": {"code": 503, "message": REFUSAL, "status": "UNAVAILABLE"}}
    return Reply(content_type="text/event-stream", body=b"data: " + json.dumps(error).encode() + b"\r\n\r\n")


def _failing(failure: Failure) -> Answer:
    match failure:
        case "400":
            return refuses(400)
        case "500":
            return refuses(500)
        case "drop":
            return drops_before_answering
        case "notjson":
            return lambda _stream: Reply(body=b"<html>not json</html>", content_type="text/html")
        case "errorframe":
            return _unavailable_mid_stream


def _failed_cells() -> tuple[ParameterSet, ...]:
    reached: Final = (
        *itertools.product(PROVIDERS, ("chat",), (False, True)),
        *itertools.product(("gemini",), ("messages", "responses"), (False, True)),
    )
    return tuple(
        pytest.param(provider, endpoint, stream, failure, id=f"{provider}-{endpoint}-{_mode(stream)}-{failure}")
        for (provider, endpoint, stream), failure in itertools.product(reached, _FAILURES)
    )


@pytest.mark.parametrize(("provider", "endpoint", "stream", "failure"), _failed_cells())
def test_continuation_request_that_fails_to_open_returns_the_answer_so_far_with_its_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, failure: Failure
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with (
        wire_server(upstream(answers(first_id, FIRST_HALF), _failing(failure))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = post(gateway, endpoint, request_body(endpoint, model, stream))
        observed: Final = answered(endpoint, stream, response)
        assert not stream or ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.streamed) == (FIRST_HALF.text, FIRST_HALF.text), observed
        assert observed.finishes == (cut_off(endpoint),), observed
        assert observed.incomplete_reason == ("max_output_tokens" if endpoint == "responses" else None), observed
        assert (observed.usage, observed.initial_prompt_tokens) == (FIRST_HALF.usage, None), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, stream) else ()), observed
        assert (HOP_TOKEN in observed.raw) == carries_token(endpoint, stream) and observed.errors == (), observed.raw
        assert CHECKPOINT not in observed.raw and REFUSAL not in observed.raw, observed.raw
        assert len(observed.token_carriers) == (1 if endpoint == "chat" and stream else 0), observed
        assert observed.token_carriers[:1] >= observed.content_carriers[-1:], observed
        assert_one_id(endpoint, stream, observed, first_id)
        opening, continuing = received(wire, provider, stream, (False, True))
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, FIRST_HALF.usage, _identities(observed, first_id))


_SECOND_TOKEN: Final = "token-after-the-second-request"
_MIDDLE: Final = Hop(("then the ", "middle, "), "CONTINUATION", _SECOND_TOKEN, 27, 5)
_LAST: Final = Hop(("and then ", "the end."), "STOP", None, 32, 4)


def _in_three_requests(first_id: str, last: Answer) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        token: Final = JSON_OBJECT.validate_json(request.body).get("continuationToken")
        if token is None:
            return hop_reply(first_id, FIRST_HALF, streams(request))
        if token == HOP_TOKEN:
            return hop_reply(f"middle-{uuid.uuid4().hex}", _MIDDLE, streams(request))
        assert token == _SECOND_TOKEN, request
        return last(streams(request))

    return respond


def _assert_each_request_sent_the_token_before_it(wire: Wire, provider: Provider, stream: bool) -> None:
    opening, second, third = received(wire, provider, stream, (False, True, True))
    assert (second, third) == (
        {**opening, "continuationToken": HOP_TOKEN},
        {**opening, "continuationToken": _SECOND_TOKEN},
    ), (opening, second, third)


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_answer_google_ends_early_twice_is_joined_from_three_requests_each_sent_the_token_before_it(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    joined: Final = FIRST_HALF.text + _MIDDLE.text + _LAST.text
    usage: Final = summed(FIRST_HALF, _MIDDLE, _LAST)
    respond: Final = _in_three_requests(first_id, answers(f"last-{uuid.uuid4().hex}", _LAST))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        observed: Final = _over_httpx(gateway, endpoint, model, stream)
        assert (observed.answer, observed.streamed) == (joined, joined), observed
        assert (observed.finishes, observed.usage) == ((complete(endpoint),), usage), observed
        assert observed.initial_prompt_tokens == (FIRST_HALF.prompt_tokens if endpoint == "chat" else None), observed
        assert observed.tokens == () and observed.errors == (), observed
        assert not any(token in observed.raw for token in (CHECKPOINT, HOP_TOKEN, _SECOND_TOKEN)), observed.raw
        assert_one_id(endpoint, stream, observed, first_id)
        _assert_each_request_sent_the_token_before_it(wire, provider, stream)
        assert_billed(model, endpoint, usage, _identities(observed, first_id))


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_third_request_that_fails_to_open_returns_both_earlier_parts_with_the_second_requests_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    so_far: Final = FIRST_HALF.text + _MIDDLE.text
    usage: Final = summed(FIRST_HALF, _MIDDLE)
    with wire_server(_in_three_requests(first_id, refuses(500))) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        observed: Final = _over_httpx(gateway, endpoint, model, stream)
        assert (observed.answer, observed.streamed) == (so_far, so_far), observed
        assert (observed.finishes, observed.usage) == ((cut_off(endpoint),), usage), observed
        assert observed.tokens == ((_SECOND_TOKEN,) if carries_token(endpoint, stream) else ()), observed
        assert (_SECOND_TOKEN in observed.raw) == carries_token(endpoint, stream), observed.raw
        assert not any(leaked in observed.raw for leaked in (CHECKPOINT, HOP_TOKEN, REFUSAL)), observed.raw
        assert observed.errors == (), observed
        assert_one_id(endpoint, stream, observed, first_id)
        _assert_each_request_sent_the_token_before_it(wire, provider, stream)
        assert_billed(model, endpoint, usage, _identities(observed, first_id))


_TOKEN_PATH: Final = "/_oauth/token"
_EXPIRED: Final = "The scripted Google refused an access token that had expired"
_REVOKED: Final = "The scripted token endpoint revoked this service account"
_CALLERS_AUTHORIZATION: Final = "Bearer the-callers-own-credential"


def _vertex_cells(endpoints: Sequence[Endpoint]) -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(endpoint, stream, id=f"vertex_ai-{endpoint}-{_mode(stream)}")
        for endpoint, stream in itertools.product(endpoints, (False, True))
    )


def _token_reply(token: str, lifetime: int) -> Reply:
    return Reply(body=json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": lifetime}).encode())


def _issuing_tokens(lifetime: int, google: Respond) -> Respond:
    issued: Final = itertools.count()

    def respond(request: Request) -> Reply:
        if request.target == _TOKEN_PATH:
            return _token_reply(f"issued-{next(issued)}", lifetime)
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


def _google_calls(requests: Sequence[Request], stream: bool, continued: tuple[bool, ...]) -> tuple[Request, ...]:
    calls: Final = tuple(request for request in requests if request.target != _TOKEN_PATH)
    assert [(call.method, call.target) for call in calls] == [
        ("POST", target("vertex_ai", stream, resumed)) for resumed in continued
    ], requests
    return calls


def _chain_through(token_endpoint: Callable[[Respond], Respond], first_id: str) -> Respond:
    return token_endpoint(upstream(answers(first_id, FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF)))


@pytest.mark.parametrize(("endpoint", "stream"), _vertex_cells(ENDPOINTS))
def test_vertex_continuation_request_sent_after_the_access_token_expired_carries_a_new_one_and_completes(
    gateway: Gateway, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with (
        wire_server(_chain_through(_tokens_that_expire_by_the_next_request, first_id)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = register_vertex(scenario, wire, wire.url)
        observed: Final = _over_httpx(gateway, endpoint, model, stream)
        assert (observed.answer, observed.streamed) == (_JOINED, _JOINED), observed
        assert (observed.finishes, observed.usage) == ((complete(endpoint),), _CHAIN_USAGE), observed
        assert observed.tokens == () and observed.errors == (), observed
        assert _EXPIRED not in observed.raw and HOP_TOKEN not in observed.raw, observed.raw
        assert_one_id(endpoint, stream, observed, first_id)
        _google_calls(wire.drain(), stream, (False, True))
        assert_billed(model, endpoint, _CHAIN_USAGE, _identities(observed, first_id))


@pytest.mark.parametrize(("endpoint", "stream"), _vertex_cells(ENDPOINTS))
def test_vertex_access_token_that_cannot_be_renewed_before_the_continuation_returns_the_answer_so_far_with_its_token(
    gateway: Gateway, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with (
        wire_server(_chain_through(_tokens_revoked_once_google_is_asked, first_id)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = register_vertex(scenario, wire, wire.url)
        observed: Final = _over_httpx(gateway, endpoint, model, stream)
        assert (observed.answer, observed.streamed) == (FIRST_HALF.text, FIRST_HALF.text), observed
        assert (observed.finishes, observed.usage) == ((cut_off(endpoint),), FIRST_HALF.usage), observed
        assert observed.incomplete_reason == ("max_output_tokens" if endpoint == "responses" else None), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, stream) else ()), observed
        assert (HOP_TOKEN in observed.raw) == carries_token(endpoint, stream) and observed.errors == (), observed.raw
        assert _REVOKED not in observed.raw and CHECKPOINT not in observed.raw, observed.raw
        assert_one_id(endpoint, stream, observed, first_id)
        _google_calls(wire.drain(), stream, (False,))
        assert_billed(model, endpoint, FIRST_HALF.usage, _identities(observed, first_id))


@pytest.mark.parametrize(("endpoint", "stream"), _vertex_cells(("chat",)))
def test_vertex_access_token_still_fresh_is_fetched_once_and_sent_on_every_request_of_a_continued_answer(
    gateway: Gateway, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with (
        wire_server(_chain_through(partial(_issuing_tokens, 3600), first_id)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = register_vertex(scenario, wire, wire.url)
        observed: Final = _over_httpx(gateway, endpoint, model, stream)
        assert (observed.answer, observed.finishes) == (_JOINED, (complete(endpoint),)), observed
        requests: Final = wire.drain()
        calls: Final = _google_calls(requests, stream, (False, True))
        assert [request.target for request in requests].count(_TOKEN_PATH) == 1, requests
        assert len({call.headers["authorization"] for call in calls}) == 1, requests


@pytest.mark.parametrize(("endpoint", "stream"), _vertex_cells(("chat",)))
def test_vertex_deployments_own_authorization_header_is_sent_on_every_request_of_a_continued_answer(
    gateway: Gateway, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    with (
        wire_server(_chain_through(partial(_issuing_tokens, 0), first_id)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = register_vertex(
            scenario, wire, wire.url, extra_headers={"Authorization": _CALLERS_AUTHORIZATION}
        )
        observed: Final = _over_httpx(gateway, endpoint, model, stream)
        assert (observed.answer, observed.finishes) == (_JOINED, (complete(endpoint),)), observed
        calls: Final = _google_calls(wire.drain(), stream, (False, True))
        assert [call.headers["authorization"] for call in calls] == [_CALLERS_AUTHORIZATION] * 2, calls


_SILENCE_SECONDS: Final = 5


def _silent(release: threading.Event) -> Answer:
    def hold(_stream: bool) -> Reply:
        assert release.wait(timeout=90), "The silent continuation request was never released"
        return Reply(drop_connection=True)

    return hold


@pytest.mark.parametrize(("provider", "endpoint"), tuple(itertools.product(PROVIDERS, ENDPOINTS)))
def test_non_streamed_continuation_request_left_unanswered_returns_the_answer_so_far_at_the_callers_timeout(
    gateway: Gateway, provider: Provider, endpoint: Endpoint
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    release: Final = threading.Event()
    with (
        wire_server(upstream(answers(first_id, FIRST_HALF), _silent(release))) as wire,
        contextlib.ExitStack() as cleanup,
        gateway.scenario() as scenario,
    ):
        cleanup.callback(release.set)
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body(endpoint, model, False, {**NO_CACHE, "timeout": _SILENCE_SECONDS})
        observed: Final = answered(endpoint, False, post(gateway, endpoint, body))
        assert (observed.answer, observed.finishes) == (FIRST_HALF.text, (cut_off(endpoint),)), observed
        assert (observed.usage, observed.errors) == (FIRST_HALF.usage, ()), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, False) else ()), observed
        assert_one_id(endpoint, False, observed, first_id)
        opening, continuing = received(wire, provider, False, (False, True))
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, FIRST_HALF.usage, _identities(observed, first_id))


@pytest.mark.parametrize(("provider", "endpoint"), tuple(itertools.product(PROVIDERS, ENDPOINTS)))
def test_streamed_continuation_request_left_unanswered_ends_the_stream_with_the_answer_so_far_at_the_callers_timeout(
    gateway: Gateway, provider: Provider, endpoint: Endpoint
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    release: Final = threading.Event()
    with (
        wire_server(upstream(answers(first_id, FIRST_HALF), _silent(release))) as wire,
        contextlib.ExitStack() as cleanup,
        gateway.scenario() as scenario,
    ):
        cleanup.callback(release.set)
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body(endpoint, model, True, {**NO_CACHE, "timeout": _SILENCE_SECONDS})
        response: Final = post(gateway, endpoint, body)
        observed: Final = answered(endpoint, True, response)
        assert ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.finishes) == (FIRST_HALF.text, (cut_off(endpoint),)), observed
        assert (observed.usage, observed.errors) == (FIRST_HALF.usage, ()), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, True) else ()), observed
        assert_one_id(endpoint, True, observed, first_id)
        opening, continuing = received(wire, provider, True, (False, True))
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, FIRST_HALF.usage, _identities(observed, first_id))


@pytest.mark.parametrize(
    ("provider", "endpoint", "setting"),
    tuple(itertools.product(PROVIDERS, ENDPOINTS, _DEPLOYMENT_TIMEOUTS)),
)
def test_streamed_continuation_request_left_unanswered_ends_at_the_deployments_timeout_with_the_answer_so_far(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, setting: DeploymentTimeout
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    release: Final = threading.Event()
    with (
        wire_server(upstream(answers(first_id, FIRST_HALF), _silent(release))) as wire,
        contextlib.ExitStack() as cleanup,
        gateway.scenario() as scenario,
    ):
        cleanup.callback(release.set)
        model: Final = register(gateway, scenario, provider, wire, settings={setting: _SILENCE_SECONDS})
        response: Final = post(gateway, endpoint, request_body(endpoint, model, True, NO_CACHE))
        observed: Final = answered(endpoint, True, response)
        assert ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.finishes) == (FIRST_HALF.text, (cut_off(endpoint),)), observed
        assert (observed.usage, observed.errors) == (FIRST_HALF.usage, ()), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, True) else ()), observed
        assert_one_id(endpoint, True, observed, first_id)
        opening, continuing = received(wire, provider, True, (False, True))
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, FIRST_HALF.usage, _identities(observed, first_id))


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_third_request_left_unanswered_returns_both_earlier_parts_and_the_second_token_at_the_callers_timeout(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    so_far: Final = FIRST_HALF.text + _MIDDLE.text
    usage: Final = summed(FIRST_HALF, _MIDDLE)
    release: Final = threading.Event()
    with (
        wire_server(_in_three_requests(first_id, _silent(release))) as wire,
        contextlib.ExitStack() as cleanup,
        gateway.scenario() as scenario,
    ):
        cleanup.callback(release.set)
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body(endpoint, model, stream, {**NO_CACHE, "timeout": _SILENCE_SECONDS})
        response: Final = post(gateway, endpoint, body)
        observed: Final = answered(endpoint, stream, response)
        assert not stream or ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.streamed) == (so_far, so_far), observed
        assert (observed.finishes, observed.usage, observed.errors) == ((cut_off(endpoint),), usage, ()), observed
        assert observed.tokens == ((_SECOND_TOKEN,) if carries_token(endpoint, stream) else ()), observed
        assert_one_id(endpoint, stream, observed, first_id)
        _assert_each_request_sent_the_token_before_it(wire, provider, stream)
        assert_billed(model, endpoint, usage, _identities(observed, first_id))


def _dropping(dropped: Dropped) -> Callable[[Request], Reply]:
    if dropped == "first":
        return upstream(drops_mid_body(f"dropped-{uuid.uuid4().hex}", PLAIN))
    return upstream(
        answers(f"first-{uuid.uuid4().hex}", FIRST_HALF), drops_mid_body(f"dropped-{uuid.uuid4().hex}", SECOND_HALF)
    )


def _delivered_before_the_drop(dropped: Dropped) -> str:
    if dropped == "first":
        return PLAIN.pieces[0]
    return FIRST_HALF.text + SECOND_HALF.pieces[0]


def _dropped_cells() -> tuple[ParameterSet, ...]:
    reached: Final = (("gemini", "chat"), ("vertex_ai", "chat"), ("gemini", "messages"), ("gemini", "responses"))
    return tuple(
        pytest.param(provider, endpoint, dropped, id=f"{provider}-{endpoint}-{dropped}")
        for (provider, endpoint), dropped in itertools.product(reached, ("first", "continued"))
    )


@pytest.mark.parametrize(("provider", "endpoint", "dropped"), _dropped_cells())
def test_stream_that_drops_mid_body_reaches_the_caller_as_an_error_with_no_finish_and_no_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, dropped: Dropped
) -> None:
    with wire_server(_dropping(dropped)) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = post(gateway, endpoint, request_body(endpoint, model, True))
        assert response.status_code == 200, response.text
        observed: Final = seen(endpoint, True, payloads(response.text, True), response.text)
        assert observed.errors != (), response.text
        assert observed.streamed == _delivered_before_the_drop(dropped), observed
        assert (observed.finishes, observed.tokens, observed.token_carriers) == ((), (), ()), observed
        assert CHECKPOINT not in response.text and HOP_TOKEN not in response.text, response.text
        bodies: Final = received(wire, provider, True, (False,) if dropped == "first" else (False, True))
        assert [body.get("continuationToken") for body in bodies] == [None, HOP_TOKEN][: len(bodies)], bodies


_UNFINISHED_FRAME: Final = b'data: {"candidates": [{"content": {"parts": [{"text": "never finished\n\n'


def _ends_on_an_unfinished_frame(response_id: str) -> Answer:
    finished: Final = tuple(f"data: {json.dumps(item)}\n\n".encode() for item in stream_frames(response_id, FIRST_HALF))
    return lambda _stream: Reply(content_type="text/event-stream", chunks=(*finished, _UNFINISHED_FRAME))


@pytest.mark.parametrize(("provider", "endpoint"), tuple(itertools.product(PROVIDERS, ENDPOINTS)))
def test_stream_that_ends_on_an_unfinished_frame_is_still_continued_into_one_joined_answer(
    gateway: Gateway, provider: Provider, endpoint: Endpoint
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    respond: Final = upstream(
        _ends_on_an_unfinished_frame(first_id), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF)
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        observed: Final = _over_httpx(gateway, endpoint, model, True)
        assert (observed.answer, observed.streamed) == (_JOINED, _JOINED), observed
        assert (observed.finishes, observed.usage) == ((complete(endpoint),), _CHAIN_USAGE), observed
        assert observed.tokens == () and observed.errors == (), observed
        assert_one_id(endpoint, True, observed, first_id)
        opening, continuing = received(wire, provider, True, (False, True))
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, _CHAIN_USAGE, _identities(observed, first_id))


def _opens_with_no_readable_frame(unread: Unread) -> Answer:
    match unread:
        case "dropped":
            return lambda stream: hop_reply(f"dropped-{uuid.uuid4().hex}", SECOND_HALF, stream, abort_after=0)
        case "empty":
            return lambda _stream: Reply(content_type="text/event-stream", chunks=())
        case "unfinished":
            return lambda _stream: Reply(content_type="text/event-stream", chunks=(_UNFINISHED_FRAME,))
        case "silent":
            return lambda _stream: Reply(
                content_type="text/event-stream", chunks=(b"\n",), gate_after_first=threading.Event()
            )


def _unread_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(provider, endpoint, unread, id=f"{provider}-{endpoint}-{unread}")
        for provider, endpoint, unread in itertools.product(PROVIDERS, ENDPOINTS, _UNREAD)
    )


@pytest.mark.parametrize(("provider", "endpoint", "unread"), _unread_cells())
def test_continuation_stream_that_opens_but_never_sends_a_readable_frame_ends_with_the_answer_so_far_and_its_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, unread: Unread
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    respond: Final = upstream(answers(first_id, FIRST_HALF), _opens_with_no_readable_frame(unread))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body(endpoint, model, True, {**NO_CACHE, "timeout": _SILENCE_SECONDS})
        response: Final = post(gateway, endpoint, body)
        observed: Final = answered(endpoint, True, response)
        assert ends_cleanly(endpoint, response.text), response.text
        assert (observed.answer, observed.streamed) == (FIRST_HALF.text, FIRST_HALF.text), observed
        assert (observed.finishes, observed.usage) == ((cut_off(endpoint),), FIRST_HALF.usage), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, True) else ()), observed
        assert observed.errors == () and CHECKPOINT not in response.text, response.text
        assert_one_id(endpoint, True, observed, first_id)
        opening, continuing = received(wire, provider, True, (False, True))
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        assert_billed(model, endpoint, FIRST_HALF.usage, _identities(observed, first_id))


_MALFORMED: Final[tuple[ParameterSet, ...]] = (
    pytest.param(12345, False, id="int-sync"),
    pytest.param(12345, True, id="int-stream"),
    pytest.param([CUT_TOKEN], False, id="list-sync"),
    pytest.param({"token": CUT_TOKEN}, False, id="object-sync"),
    pytest.param(True, False, id="bool-sync"),
)
_WRONG_TYPE: Final = "Invalid value at 'continuation_token' (TYPE_STRING)"


def _strict_about_the_token(resumed_id: str) -> Callable[[Request], Reply]:
    refused: Final = {"error": {"code": 400, "message": _WRONG_TYPE, "status": "INVALID_ARGUMENT"}}

    def respond(request: Request) -> Reply:
        body: Final = JSON_OBJECT.validate_json(request.body)
        if "continuationToken" not in body:
            return hop_reply(f"restarted-{uuid.uuid4().hex}", PLAIN, streams(request))
        if not isinstance(body["continuationToken"], str):
            return Reply(status=400, body=json.dumps(refused).encode())
        return hop_reply(resumed_id, RESUMED, streams(request))

    return respond


@pytest.mark.parametrize(("token", "stream"), _MALFORMED)
def test_continuation_token_of_the_wrong_type_is_refused_and_never_restarts_the_answer(
    gateway: Gateway, token: JsonValue, stream: bool
) -> None:
    with wire_server(_strict_about_the_token(f"resumed-{uuid.uuid4().hex}")) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, "gemini", wire)
        refused: Final = post(
            gateway, "chat", request_body("chat", model, stream, {**NO_CACHE, "continuation_token": token})
        )
        assert refused.status_code == 400, refused.text
        assert "continuation_token" in refused.text and PLAIN.text not in refused.text, refused.text
        sent: Final = tuple(JSON_OBJECT.validate_json(request.body) for request in wire.drain())
        assert len(sent) <= 1 and all(body.get("continuationToken") == token for body in sent), sent
        healthy: Final = answered("chat", stream, post(gateway, "chat", request_body("chat", model, stream)))
        assert (healthy.answer, healthy.finishes) == (PLAIN.text, ("stop",)), healthy


@pytest.mark.parametrize("provider", PROVIDERS)
def test_five_kilobyte_continuation_token_reaches_google_byte_for_byte(gateway: Gateway, provider: Provider) -> None:
    token: Final = f"{uuid.uuid4().hex}-" + "t" * 5000
    resumed_id: Final = f"resumed-{uuid.uuid4().hex}"
    with wire_server(_strict_about_the_token(resumed_id)) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body("chat", model, False, {**NO_CACHE, "continuation_token": token})
        observed: Final = answered("chat", False, post(gateway, "chat", body))
        assert (observed.answer, observed.finishes, observed.tokens) == (RESUMED.text, ("stop",), ()), observed
        (only,) = received(wire, provider, False, (True,))
        assert only["continuationToken"] == token, only


def test_continuation_token_sent_twice_in_one_body_reaches_google_once(gateway: Gateway) -> None:
    tokens: Final = (f"first-{uuid.uuid4().hex}", f"second-{uuid.uuid4().hex}")
    with wire_server(_strict_about_the_token(f"resumed-{uuid.uuid4().hex}")) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, "gemini", wire)
        whole: Final = json.dumps(request_body("chat", model, False))
        raw: Final = f'{whole[:-1]}, "continuation_token": "{tokens[0]}", "continuation_token": "{tokens[1]}"}}'
        response: Final = gateway.client.request(
            "POST",
            "/v1/chat/completions",
            content=raw.encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        observed: Final = answered("chat", False, response)
        assert observed.answer == RESUMED.text, observed
        (only,) = received(wire, "gemini", False, (True,))
        assert only["continuationToken"] in tokens, only


def _absent(kind: Absent) -> Mapping[str, JsonValue]:
    match kind:
        case "missing":
            return NO_CACHE
        case "null":
            return {**NO_CACHE, "continuation_token": None}
        case "empty":
            return {**NO_CACHE, "continuation_token": ""}


def _absent_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(kind, provider, stream, id=f"{kind}-{provider}-{_mode(stream)}")
        for kind, provider, stream in itertools.product(("missing", "null", "empty"), PROVIDERS, (False, True))
    )


@pytest.mark.parametrize(("kind", "provider", "stream"), _absent_cells())
def test_missing_null_and_empty_continuation_token_all_send_a_request_with_no_token(
    gateway: Gateway, kind: Absent, provider: Provider, stream: bool
) -> None:
    response_id: Final = f"plain-{uuid.uuid4().hex}"
    with wire_server(upstream(answers(response_id, PLAIN))) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        observed: Final = answered(
            "chat", stream, post(gateway, "chat", request_body("chat", model, stream, _absent(kind)))
        )
        assert (observed.answer, observed.finishes, observed.tokens) == (PLAIN.text, ("stop",), ()), observed
        assert_one_id("chat", stream, observed, response_id)
        (only,) = received(wire, provider, stream, (False,))
        assert "continuationToken" not in only and "continuation_token" not in json.dumps(only), only


_DRAFTS: Final = (("Draft A", "token-of-draft-a"), ("Draft B", "token-of-draft-b"))


def _two_drafts(request: Request) -> Reply:
    assert not streams(request), request.target
    candidates: Final[list[JsonValue]] = [
        {
            "content": {"role": "model", "parts": [{"text": text}]},
            "index": index,
            "finishReason": "CONTINUATION",
            "continuationToken": token,
        }
        for index, (text, token) in enumerate(_DRAFTS)
    ]
    body: Final = {
        "responseId": f"drafts-{uuid.uuid4().hex}",
        "candidates": candidates,
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 14, "totalTokenCount": 34},
        "modelVersion": BACKEND,
    }
    return Reply(body=json.dumps(body).encode())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_request_for_two_candidates_is_never_continued_and_each_choice_keeps_its_own_token(
    gateway: Gateway, provider: Provider
) -> None:
    with wire_server(_two_drafts) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = post(gateway, "chat", request_body("chat", model, False, {**NO_CACHE, "n": 2}))
        assert response.status_code == 200, response.text
        choices: Final = items_of(JSON_OBJECT.validate_json(response.content).get("choices"))
        assert [
            (
                at(choice, "message", "content"),
                at(choice, "finish_reason"),
                at(choice, "message", "provider_specific_fields", "continuation_token"),
            )
            for choice in choices
        ] == [(text, "length", token) for text, token in _DRAFTS], response.text
        (only,) = received(wire, provider, False, (False,))
        assert at(only, "generationConfig", "candidate_count") == 2, only


def _uncapped_cells() -> tuple[ParameterSet, ...]:
    reached: Final = (("gemini", "chat"), ("vertex_ai", "chat"), ("gemini", "responses"))
    return tuple(
        pytest.param(provider, endpoint, stream, id=f"{provider}-{endpoint}-{_mode(stream)}")
        for (provider, endpoint), stream in itertools.product(reached, (False, True))
    )


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _uncapped_cells())
def test_request_with_no_output_cap_is_never_continued_and_returns_googles_cut_with_its_token(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    respond: Final = upstream(answers(first_id, FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        observed: Final = answered(
            endpoint, stream, post(gateway, endpoint, request_body(endpoint, model, stream, cap=None))
        )
        assert (observed.answer, observed.streamed) == (FIRST_HALF.text, FIRST_HALF.text), observed
        assert observed.finishes == (cut_off(endpoint),), observed
        assert observed.tokens == ((HOP_TOKEN,) if carries_token(endpoint, stream) else ()), observed
        assert observed.token_carriers == (observed.content_carriers[-1:] if endpoint == "chat" else ()), observed
        assert CHECKPOINT not in observed.raw, observed.raw
        (only,) = received(wire, provider, stream, (False,))
        assert at(only, "generationConfig", "max_output_tokens") is None, only


_STRINGLY: Final[tuple[ParameterSet, ...]] = tuple(
    pytest.param(extra, cap, stream, id=f"{name}-{_mode(stream)}")
    for (name, extra, cap), stream in itertools.product(
        (("n", {"n": "1"}, CAP), ("max_tokens", {"max_tokens": "300"}, None)), (False, True)
    )
)


@pytest.mark.parametrize(("extra", "cap", "stream"), _STRINGLY)
def test_count_or_cap_sent_as_a_string_makes_one_request_and_is_never_continued(
    gateway: Gateway, extra: Mapping[str, JsonValue], cap: int | None, stream: bool
) -> None:
    respond: Final = upstream(
        answers(f"first-{uuid.uuid4().hex}", FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF)
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, "gemini", wire)
        body: Final = request_body("chat", model, stream, {**NO_CACHE, **extra}, cap=cap)
        observed: Final = answered("chat", stream, post(gateway, "chat", body))
        assert (observed.answer, observed.errors) == (FIRST_HALF.text, ()), observed
        (only,) = received(wire, "gemini", stream, (False,))
        assert "continuationToken" not in only, only


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _cells(PROVIDERS, ENDPOINTS))
def test_cache_hit_twin_of_a_continued_answer_replays_it_and_bills_the_chain_once(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    replayed: Final = (_JOINED, (complete(endpoint),), _CHAIN_USAGE)
    respond: Final = upstream(answers(first_id, FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        body: Final = request_body(endpoint, model, stream, NOTHING_EXTRA, prompt=f"{PROMPT} {uuid.uuid4().hex}")
        first: Final = answered(endpoint, stream, post(gateway, endpoint, body))
        assert (first.answer, first.finishes, first.usage) == replayed, first
        assert len(wire.drain()) == 2
        assert_billed(model, endpoint, _CHAIN_USAGE, _identities(first, first_id))
        second: Final = answered(endpoint, stream, post(gateway, endpoint, body))
        assert (second.answer, second.finishes, second.usage) == replayed and second.tokens == (), second
        assert CHECKPOINT not in second.raw and HOP_TOKEN not in second.raw, second.raw
        assert wire.drain() == ()
        rows: Final = spend_rows(model, expected=2)
        (served_from_cache,) = [row for row in rows if str(row["cache_hit"]) == "True"]
        (sent_to_google,) = [row for row in rows if str(row["cache_hit"]) != "True"]
        assert number(sent_to_google["spend"]) == pytest.approx(cost_of(_CHAIN_USAGE)), rows
        assert (served_from_cache["status"], number(served_from_cache["spend"])) == ("success", 0), rows


def _rest_for(token: str) -> Hop:
    return Hop((f"the rest for {token} ", "is here."), "STOP", None, 27, 5)


def _resumes_by_token(request: Request) -> Reply:
    token: Final = JSON_OBJECT.validate_json(request.body).get("continuationToken")
    hop: Final = _rest_for(token) if isinstance(token, str) else PLAIN
    return hop_reply(f"answer-{uuid.uuid4().hex}", hop, streams(request))


def test_request_with_a_token_is_never_served_the_cached_answer_of_one_without_it_or_with_another(
    gateway: Gateway,
) -> None:
    mine: Final = f"mine-{uuid.uuid4().hex}"
    other: Final = f"other-{uuid.uuid4().hex}"
    with wire_server(_resumes_by_token) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, "gemini", wire)
        body: Final = request_body("chat", model, False, NOTHING_EXTRA, prompt=f"{PROMPT} {uuid.uuid4().hex}")
        fresh: Final = answered("chat", False, post(gateway, "chat", body))
        spend_rows(model)
        cached: Final = answered("chat", False, post(gateway, "chat", body))
        assert (fresh.answer, cached.answer, len(wire.drain())) == (PLAIN.text, PLAIN.text, 1)
        resumed: Final = answered("chat", False, post(gateway, "chat", {**body, "continuation_token": mine}))
        assert resumed.answer == _rest_for(mine).text, resumed
        (sent,) = received(wire, "gemini", False, (True,))
        assert sent["continuationToken"] == mine, sent
        spend_rows(model, expected=3)
        replayed: Final = answered("chat", False, post(gateway, "chat", {**body, "continuation_token": mine}))
        assert (replayed.answer, wire.drain()) == (_rest_for(mine).text, ()), replayed
        another: Final = answered("chat", False, post(gateway, "chat", {**body, "continuation_token": other}))
        assert another.answer == _rest_for(other).text, another
        (crossed,) = received(wire, "gemini", False, (True,))
        assert crossed["continuationToken"] == other, crossed


@pytest.mark.parametrize("absent", ("", None), ids=("empty", "null"))
def test_empty_or_null_continuation_token_is_served_the_cached_answer_of_the_request_without_one(
    gateway: Gateway, absent: str | None
) -> None:
    with wire_server(_resumes_by_token) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, "gemini", wire)
        body: Final = request_body("chat", model, False, NOTHING_EXTRA, prompt=f"{PROMPT} {uuid.uuid4().hex}")
        fresh: Final = answered("chat", False, post(gateway, "chat", body))
        spend_rows(model)
        assert (fresh.answer, len(wire.drain())) == (PLAIN.text, 1), fresh
        twin: Final = answered("chat", False, post(gateway, "chat", {**body, "continuation_token": absent}))
        assert (twin.answer, wire.drain()) == (PLAIN.text, ()), twin
        rows: Final = spend_rows(model, expected=2)
        assert sorted(str(row["cache_hit"]) == "True" for row in rows) == [False, True], rows


def _native_path(model: str, stream: bool) -> str:
    return f"/v1beta/models/{model}:{'streamGenerateContent' if stream else 'generateContent'}"


def _native(gateway: Gateway, model: str, stream: bool, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", _native_path(model, stream), dict(body), params={"alt": "sse"} if stream else None)


def _native_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(provider, stream, id=f"{provider}-{_mode(stream)}")
        for provider, stream in itertools.product(PROVIDERS, (False, True))
    )


@pytest.mark.parametrize(("provider", "stream"), _native_cells())
def test_native_route_hands_googles_continuation_fields_to_the_caller_from_one_request(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    respond: Final = upstream(
        answers(f"first-{uuid.uuid4().hex}", FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF)
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = _native(gateway, model, stream, _NATIVE_BODY)
        assert response.status_code == 200, response.text
        candidates: Final = tuple(at(item, "candidates", 0) for item in payloads(response.text, stream))
        assert "".join(text_of(at(item, "content", "parts", 0, "text")) for item in candidates) == FIRST_HALF.text
        assert tuple(at(item, "continuationToken") for item in candidates) == (
            (CHECKPOINT, HOP_TOKEN) if stream else (HOP_TOKEN,)
        ), response.text
        assert at(candidates[-1], "finishReason") == "CONTINUATION", response.text
        (only,) = received(wire, provider, stream, (False,))
        assert "continuationToken" not in only, only
        (row,) = spend_rows(model)
        assert (row["status"], row["prompt_tokens"], row["completion_tokens"]) == ("success", *FIRST_HALF.usage), row
        assert number(row["spend"]) == pytest.approx(cost_of(FIRST_HALF.usage)), row


@pytest.mark.parametrize(("provider", "stream"), _native_cells())
def test_native_route_passes_the_callers_continuation_token_to_google(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    respond: Final = upstream(
        answers(f"restarted-{uuid.uuid4().hex}", PLAIN), answers(f"resumed-{uuid.uuid4().hex}", RESUMED)
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        response: Final = _native(gateway, model, stream, {**_NATIVE_BODY, "continuationToken": CUT_TOKEN})
        assert response.status_code == 200, response.text
        candidates: Final = tuple(at(item, "candidates", 0) for item in payloads(response.text, stream))
        assert "".join(text_of(at(item, "content", "parts", 0, "text")) for item in candidates) == RESUMED.text
        (sent,) = wire.drain()
        assert sent.target in (target(provider, stream), target(provider, stream, continued=True)), sent.target
        assert JSON_OBJECT.validate_json(sent.body)["continuationToken"] == CUT_TOKEN, sent.body


Searched = Literal["first", "each"]

_SEARCHES: Final = ("scripted search one", "scripted search two")


def _grounded(reply: dict[str, JsonValue], searches: Sequence[str]) -> dict[str, JsonValue]:
    candidate: Final = at(reply, "candidates", 0)
    assert isinstance(candidate, dict), reply
    grounding: Final[dict[str, JsonValue]] = {
        "webSearchQueries": [*searches],
        "groundingChunks": [{"web": {"uri": "https://example.com/source", "title": "example.com"}}],
    }
    return {**reply, "candidates": [{**candidate, "groundingMetadata": grounding}]}


def _searching(response_id: str, hop: Hop, searches: Sequence[str]) -> Answer:
    def reply(stream: bool) -> Reply:
        if not searches:
            return hop_reply(response_id, hop, stream)
        if stream:
            opening, closing = stream_frames(response_id, hop)
            return sse((opening, _grounded(closing, searches)))
        whole: Final = frame(response_id, hop.text, hop.usage, hop.finish, hop.token)
        return Reply(body=json.dumps(_grounded(whole, searches)).encode())

    return reply


def _searches_counted(response: httpx.Response, stream: bool) -> JsonValue:
    metered: Final = tuple(item for item in payloads(response.text, stream) if isinstance(item.get("usage"), dict))
    assert metered, response.text
    return at(metered[-1], "usage", "prompt_tokens_details", "web_search_requests")


def _search_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(provider, stream, searched, id=f"{provider}-{_mode(stream)}-{searched}")
        for provider, stream, searched in itertools.product(PROVIDERS, (False, True), ("first", "each"))
    )


@pytest.mark.parametrize(("provider", "stream", "searched"), _search_cells())
def test_google_searches_across_a_continued_answer_are_all_counted_and_billed_as_one_request_would_be(
    gateway: Gateway, provider: Provider, stream: bool, searched: Searched
) -> None:
    first_id: Final = f"first-{uuid.uuid4().hex}"
    opening_searches: Final = _SEARCHES if searched == "first" else _SEARCHES[:1]
    chained: Final = upstream(
        _searching(first_id, FIRST_HALF, opening_searches),
        _searching(f"second-{uuid.uuid4().hex}", SECOND_HALF, _SEARCHES[len(opening_searches) :]),
    )
    in_one_request: Final = upstream(_searching(f"plain-{uuid.uuid4().hex}", PLAIN, _SEARCHES))
    with wire_server(chained) as wire, wire_server(in_one_request) as twin_wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        twin: Final = register(gateway, scenario, provider, twin_wire)
        searching: Final[Mapping[str, JsonValue]] = {**NO_CACHE, "tools": [{"googleSearch": {}}]}
        response: Final = post(gateway, "chat", request_body("chat", model, stream, searching))
        observed: Final = answered("chat", stream, response)
        assert (observed.answer, observed.finishes, observed.usage) == (_JOINED, ("stop",), _CHAIN_USAGE), observed
        assert _searches_counted(response, stream) == len(_SEARCHES), response.text
        opening, continuing = received(wire, provider, stream, (False, True))
        assert "tools" in opening and continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)
        alone: Final = post(gateway, "chat", request_body("chat", twin, stream, searching))
        assert answered("chat", stream, alone).answer == PLAIN.text, alone.text
        assert _searches_counted(alone, stream) == len(_SEARCHES), alone.text
        (chain_row,) = spend_rows(model)
        (twin_row,) = spend_rows(twin)
        search_fee: Final = number(twin_row["spend"]) - cost_of(PLAIN.usage)
        assert search_fee > 0, twin_row
        assert number(chain_row["spend"]) == pytest.approx(cost_of(_CHAIN_USAGE) + search_fee), (chain_row, twin_row)


Spelling = Literal["citations", "citationSources"]
Earlier = Literal["multibyte", "thought"]

_SPELLINGS: Final[tuple[Spelling, ...]] = ("citations", "citationSources")
_EARLIER: Final[tuple[Earlier, ...]] = ("multibyte", "thought")
_CITED_OPENING: Final = Hop(("Grüße aus 東京 🌸, ", "wo der Fuji steht, "), "CONTINUATION", HOP_TOKEN, 20, 7)
_CITED_CLOSING: Final = Hop(("dann der ", "zweite Teil über Zürich."), "STOP", None, 27, 5)
_CITED_WHOLE: Final = Hop((_CITED_OPENING.text, _CITED_CLOSING.text), "STOP", None, 20, 12)
_THOUGHT: Final = "Ich plane die Antwort über 東京 zuerst."


@dataclass(frozen=True, slots=True)
class _Source:
    phrase: str
    uri: str


@dataclass(frozen=True, slots=True)
class _Carried:
    answer: str
    annotations: tuple[str, ...]
    citations: tuple[str, ...]
    retrieved: tuple[str, ...]


_OPENING_SOURCE: Final = _Source("der Fuji", "https://example.com/fuji")
_CLOSING_SOURCE: Final = _Source("über Zürich", "https://example.com/zurich")


def _byte_span(text: str, phrase: str) -> dict[str, JsonValue]:
    # Google measures citation and grounding segment offsets in UTF-8 bytes (ai.google.dev/api/generate-content,
    # Segment and CitationSource, read 2026-10-14)
    start: Final = text.encode().index(phrase.encode())
    return {"startIndex": start, "endIndex": start + len(phrase.encode())}


def _cited_metadata(text: str, sources: Sequence[_Source], spelling: Spelling) -> dict[str, JsonValue]:
    return {
        "citationMetadata": {spelling: [{**_byte_span(text, source.phrase), "uri": source.uri} for source in sources]},
        "urlContextMetadata": {
            "urlMetadata": [
                {"retrievedUrl": source.uri, "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"} for source in sources
            ]
        },
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": source.uri, "title": source.phrase}} for source in sources],
            "groundingSupports": [
                {
                    "segment": {**_byte_span(text, source.phrase), "text": source.phrase},
                    "groundingChunkIndices": [index],
                }
                for index, source in enumerate(sources)
            ],
        },
    }


def _opening_metadata(spelling: Spelling) -> dict[str, JsonValue]:
    return _cited_metadata(_CITED_OPENING.text, (_OPENING_SOURCE,), spelling)


def _decorated(
    reply: dict[str, JsonValue], thought: str | None, metadata: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    candidate: Final = at(reply, "candidates", 0)
    assert isinstance(candidate, dict), reply
    thoughts: Final[tuple[JsonValue, ...]] = ({"text": thought, "thought": True},) if thought else ()
    content: Final[dict[str, JsonValue]] = {
        "role": "model",
        "parts": [*thoughts, *items_of(at(candidate, "content", "parts"))],
    }
    return {**reply, "candidates": [{**candidate, "content": content, **metadata}]}


def _citing(response_id: str, hop: Hop, metadata: Mapping[str, JsonValue], thought: str | None) -> Answer:
    def reply(stream: bool) -> Reply:
        if stream:
            opening, closing = stream_frames(response_id, hop)
            return sse((_decorated(opening, thought, NOTHING_EXTRA), _decorated(closing, None, metadata)))
        whole: Final = frame(response_id, hop.text, hop.usage, hop.finish, hop.token)
        return Reply(body=json.dumps(_decorated(whole, thought, metadata)).encode())

    return reply


def _canonical(values: Iterable[JsonValue]) -> tuple[str, ...]:
    return tuple(sorted(json.dumps(value, sort_keys=True) for value in values))


def _entries(listed: JsonValue, *keys: str) -> Iterator[JsonValue]:
    return itertools.chain.from_iterable(
        items_of(at(entry, key)) for entry, key in itertools.product(items_of(listed), keys)
    )


def _chat_carried(payloads_seen: Sequence[dict[str, JsonValue]], message: str) -> _Carried:
    return _Carried(
        answer="".join(text_of(at(payload, "choices", 0, message, "content")) for payload in payloads_seen),
        annotations=_canonical(
            itertools.chain.from_iterable(
                items_of(at(payload, "choices", 0, message, "annotations")) for payload in payloads_seen
            )
        ),
        citations=_canonical(
            itertools.chain.from_iterable(
                _entries(payload.get("vertex_ai_citation_metadata"), "citations", "citationSources")
                for payload in payloads_seen
            )
        ),
        retrieved=_canonical(
            itertools.chain.from_iterable(
                _entries(payload.get("vertex_ai_url_context_metadata"), "urlMetadata") for payload in payloads_seen
            )
        ),
    )


def _output_parts(response: JsonValue) -> Iterator[JsonValue]:
    messages: Final = (item for item in items_of(at(response, "output")) if at(item, "type") == "message")
    return itertools.chain.from_iterable(items_of(at(item, "content")) for item in messages)


def _responses_carried(response: JsonValue) -> _Carried:
    return _Carried(
        answer=output_text(response),
        annotations=_canonical(
            itertools.chain.from_iterable(items_of(at(part, "annotations")) for part in _output_parts(response))
        ),
        citations=(),
        retrieved=(),
    )


def _completed(events: Sequence[dict[str, JsonValue]]) -> JsonValue:
    (completed,) = (event.get("response") for event in events if event.get("type") == "response.completed")
    return completed


def _carried(endpoint: Endpoint, stream: bool, response: httpx.Response) -> _Carried:
    assert response.status_code == 200, response.text
    seen_payloads: Final = payloads(response.text, stream)
    match (endpoint, stream):
        case ("chat", False):
            return _chat_carried(seen_payloads, "message")
        case ("chat", True):
            return _chat_carried(seen_payloads, "delta")
        case ("responses", False):
            return _responses_carried(seen_payloads[0])
        case ("responses", True):
            return _responses_carried(_completed(seen_payloads))
        case _:
            raise AssertionError(f"No citations are read from {endpoint}")


def _citation_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(
            provider,
            endpoint,
            stream,
            spelling,
            earlier,
            id=f"{provider}-{endpoint}-{_mode(stream)}-{spelling}-{earlier}",
        )
        for provider, endpoint, stream, spelling, earlier in itertools.product(
            PROVIDERS, ("chat", "responses"), (False, True), _SPELLINGS, _EARLIER
        )
    )


@pytest.mark.parametrize(("provider", "endpoint", "stream", "spelling", "earlier"), _citation_cells())
def test_citations_and_url_context_of_a_continued_answer_match_what_one_request_for_the_whole_answer_returns(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, spelling: Spelling, earlier: Earlier
) -> None:
    thought: Final = _THOUGHT if earlier == "thought" else None
    chained: Final = upstream(
        _citing(f"first-{uuid.uuid4().hex}", _CITED_OPENING, _opening_metadata(spelling), thought),
        _citing(
            f"second-{uuid.uuid4().hex}",
            _CITED_CLOSING,
            _cited_metadata(_CITED_CLOSING.text, (_CLOSING_SOURCE,), spelling),
            None,
        ),
    )
    in_one_request: Final = upstream(
        _citing(
            f"plain-{uuid.uuid4().hex}",
            _CITED_WHOLE,
            _cited_metadata(_CITED_WHOLE.text, (_OPENING_SOURCE, _CLOSING_SOURCE), spelling),
            thought,
        )
    )
    with wire_server(chained) as wire, wire_server(in_one_request) as twin_wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        twin: Final = register(gateway, scenario, provider, twin_wire)
        continued: Final = _carried(endpoint, stream, post(gateway, endpoint, request_body(endpoint, model, stream)))
        alone: Final = _carried(endpoint, stream, post(gateway, endpoint, request_body(endpoint, twin, stream)))
        assert alone.answer == _CITED_WHOLE.text and len(alone.annotations) == 2, alone
        assert endpoint != "chat" or (len(alone.citations), len(alone.retrieved)) == (2, 2), alone
        assert continued == alone, (continued, alone)
        assert len(wire.drain()) == 2


def _unplaced_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(provider, endpoint, stream, spelling, id=f"{provider}-{endpoint}-{_mode(stream)}-{spelling}")
        for provider, endpoint, stream, spelling in itertools.product(
            PROVIDERS, ("chat", "responses"), (False, True), _SPELLINGS
        )
    )


def _cited_uris(carried: _Carried) -> frozenset[JsonValue]:
    return frozenset(JSON_OBJECT.validate_json(cited).get("uri") for cited in carried.citations)


@pytest.mark.parametrize(("provider", "endpoint", "stream", "spelling"), _unplaced_cells())
def test_later_request_citing_a_source_with_no_offsets_returns_what_one_request_for_the_whole_answer_returns(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, spelling: Spelling
) -> None:
    # Google marks every CitationSource field optional, endIndex included (ai.google.dev/api/generate-content,
    # CitationSource, read 2026-10-14)
    unplaced: Final[JsonValue] = {"uri": _CLOSING_SOURCE.uri}
    placed: Final[JsonValue] = {**_byte_span(_CITED_OPENING.text, _OPENING_SOURCE.phrase), "uri": _OPENING_SOURCE.uri}
    chained: Final = upstream(
        _citing(f"first-{uuid.uuid4().hex}", _CITED_OPENING, {"citationMetadata": {spelling: [placed]}}, None),
        _citing(f"second-{uuid.uuid4().hex}", _CITED_CLOSING, {"citationMetadata": {spelling: [unplaced]}}, None),
    )
    in_one_request: Final = upstream(
        _citing(f"plain-{uuid.uuid4().hex}", _CITED_WHOLE, {"citationMetadata": {spelling: [placed, unplaced]}}, None)
    )
    with wire_server(chained) as wire, wire_server(in_one_request) as twin_wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire)
        twin: Final = register(gateway, scenario, provider, twin_wire)
        continued: Final = _carried(endpoint, stream, post(gateway, endpoint, request_body(endpoint, model, stream)))
        alone: Final = _carried(endpoint, stream, post(gateway, endpoint, request_body(endpoint, twin, stream)))
        assert alone.answer == continued.answer == _CITED_WHOLE.text, (continued, alone)
        assert stream or endpoint != "chat" or _OPENING_SOURCE.uri in _cited_uris(alone), alone
        assert _cited_uris(continued) == _cited_uris(alone), (continued, alone)
        assert len(wire.drain()) == 2


OddMetadata = Literal["citations_null", "citations_empty", "url_context_null", "url_context_empty", "both_missing"]

_ODD_METADATA: Final[Mapping[OddMetadata, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "citations_null": {"citationMetadata": None},
        "citations_empty": {"citationMetadata": {}},
        "url_context_null": {"urlContextMetadata": None},
        "url_context_empty": {"urlContextMetadata": {}},
        "both_missing": NOTHING_EXTRA,
    }
)
_OLDER_BACKEND: Final = "gemini-3.8-flash"
_ODD_ANSWERS: Final[tuple[tuple[str, tuple[Hop, ...]], ...]] = (
    (BACKEND, (PLAIN,)),
    (BACKEND, (FIRST_HALF, SECOND_HALF)),
    (_OLDER_BACKEND, (PLAIN,)),
    (_OLDER_BACKEND, (FIRST_HALF, SECOND_HALF)),
)


def _odd(response_id: str, hop: Hop, backend: str, metadata: Mapping[str, JsonValue]) -> Answer:
    def reply(stream: bool) -> Reply:
        frames: Final = (
            stream_frames(response_id, hop)
            if stream
            else (frame(response_id, hop.text, hop.usage, hop.finish, hop.token),)
        )
        odd: Final = tuple({**_decorated(item, None, metadata), "modelVersion": backend} for item in frames)
        return sse(odd) if stream else Reply(body=json.dumps(odd[0]).encode())

    return reply


def _odd_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(
            provider,
            endpoint,
            stream,
            backend,
            hops,
            odd,
            id=f"{provider}-{endpoint}-{_mode(stream)}-{backend}-{len(hops)}-requests-{odd}",
        )
        for provider, endpoint, stream, (backend, hops), odd in itertools.product(
            PROVIDERS, ENDPOINTS, (False, True), _ODD_ANSWERS, _ODD_METADATA
        )
    )


@pytest.mark.parametrize(("provider", "endpoint", "stream", "backend", "hops", "odd"), _odd_cells())
def test_null_empty_or_missing_citation_and_url_context_metadata_still_returns_the_whole_answer(
    gateway: Gateway,
    provider: Provider,
    endpoint: Endpoint,
    stream: bool,
    backend: str,
    hops: tuple[Hop, ...],
    odd: OddMetadata,
) -> None:
    replies: Final = tuple(_odd(f"odd-{uuid.uuid4().hex}", hop, backend, _ODD_METADATA[odd]) for hop in hops)
    with wire_server(upstream(*replies)) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, provider, wire, backend=backend)
        response: Final = post(gateway, endpoint, request_body(endpoint, model, stream))
        observed: Final = answered(endpoint, stream, response)
        assert not stream or ends_cleanly(endpoint, response.text), response.text
        assert observed.answer == "".join(hop.text for hop in hops) and observed.errors == (), observed
        assert observed.finishes == (complete(endpoint),), observed
        assert len(wire.drain()) == len(hops)


_OTHER_BODY: Final[Mapping[str, JsonValue]] = {
    **_NATIVE_BODY,
    "contents": [{"role": "user", "parts": [{"text": "Write the short answer."}]}],
}


def _pass_through_config(directory: Path, wire_url: str) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "gemini-pass-through.yaml"
    path.write_text(
        yaml.safe_dump({**base, "environment_variables": {"GEMINI_API_BASE": wire_url, "GEMINI_API_KEY": API_KEY}})
    )
    return path


def _scripted(script: Sequence[tuple[Mapping[str, JsonValue], Reply]]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        sent: Final = JSON_OBJECT.validate_json(request.body)
        matching: Final = tuple(reply for body, reply in script if body == sent)
        assert len(matching) == 1, request
        return matching[0]

    return respond


def _sent_by_google(reply: Reply, stream: bool) -> tuple[dict[str, JsonValue], ...]:
    return payloads(b"".join(reply.chunks or (reply.body,)).decode(), stream)


def _pass_through(gateway: Gateway, stream: bool, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        f"/gemini{_native_path(BACKEND, stream)}",
        dict(body),
        params={"alt": "sse"} if stream else None,
        headers={"x-goog-api-key": gateway.key},
    )


def _pass_through_row(response: httpx.Response) -> dict[str, JsonValue]:
    (row,) = eventually(
        lambda: read_rows(f"{SPEND_COLUMNS} WHERE request_id=%s", (response.headers["x-litellm-call-id"],)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return row


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize("stream", (False, True), ids=_mode)
def test_gemini_pass_through_route_relays_googles_continuation_fields_as_sent_and_bills_each_request_on_its_own(
    gateway: Gateway, tmp_path: Path, stream: bool
) -> None:
    script: Final = (
        (_NATIVE_BODY, hop_reply("pass-through-opening", FIRST_HALF, stream)),
        ({**_NATIVE_BODY, "continuationToken": HOP_TOKEN}, hop_reply("pass-through-rest", SECOND_HALF, stream)),
        (_OTHER_BODY, hop_reply("pass-through-other", PLAIN, stream)),
    )
    with wire_server(_scripted(script)) as wire:
        with owned_proxy_process(gateway, tmp_path, {}, config=_pass_through_config(tmp_path, wire.url)) as owned:
            responses: Final = tuple(_pass_through(owned.gateway, stream, body) for body, _ in script)
            assert [response.status_code for response in responses] == [200] * len(script), [
                response.text for response in responses
            ]
            assert [payloads(response.text, stream) for response in responses] == [
                _sent_by_google(reply, stream) for _, reply in script
            ]
            requests: Final = wire.drain()
            assert [(request.method, request.target.split("?", 1)[0]) for request in requests] == [
                ("POST", _native_path(BACKEND, stream))
            ] * len(script), requests
            assert [JSON_OBJECT.validate_json(request.body) for request in requests] == [body for body, _ in script]
            rows: Final = tuple(_pass_through_row(response) for response in responses)
            assert [
                (row["call_type"], row["status"], row["prompt_tokens"], row["completion_tokens"]) for row in rows
            ] == [("pass_through_endpoint", "success", *hop.usage) for hop in (FIRST_HALF, SECOND_HALF, PLAIN)], rows
            opening_spend, rest_spend, other_spend = (number(row["spend"]) for row in rows)
            assert opening_spend == pytest.approx(other_spend), rows
            assert min(opening_spend, rest_spend) > 0, rows
