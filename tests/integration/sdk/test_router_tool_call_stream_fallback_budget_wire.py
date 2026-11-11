from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm import Router

_MODEL: Final = "gpt-5.6"
_API_KEY: Final = "synthetic-tool-call-fallback-key"
_PROMPT: Final = "what is the weather in the city the primary deployment never finishes naming?"
_TOOLS: Final = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
)
_ERROR_FRAME: Final = (
    b"data: " + json.dumps({"error": {"message": "overloaded", "type": "server_error", "code": 500}}).encode() + b"\n\n"
)
_DONE: Final = b"data: [DONE]\n\n"
_PRIMARY_TO_BACKUP: Final = [{"primary": ["backup"]}]
_BOTH_WAYS: Final = [{"primary": ["backup"]}, {"backup": ["primary"]}]

pytestmark = pytest.mark.timeout(120)


def _chunk(delta: Mapping[str, object], finish_reason: str | None) -> bytes:
    body: Final = {
        "id": "chatcmpl-tool-call-fallback-wire",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": _MODEL,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _tool_call_open(call_id: str) -> bytes:
    return _chunk(
        {
            "role": "assistant",
            "tool_calls": [
                {"index": 0, "id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": ""}}
            ],
        },
        None,
    )


def _tool_call_arguments(fragment: str) -> bytes:
    return _chunk({"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}, None)


_FINISH: Final = _chunk({}, "tool_calls")


def _serves_tool_call(call_id: str, city: str) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _tool_call_open(call_id),
            _tool_call_arguments('{"city": "'),
            _tool_call_arguments(f'{city}"}}'),
            _FINISH,
            _DONE,
        ),
    )


def _dies_after_tool_call_opens(call_id: str) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(_tool_call_open(call_id), _tool_call_arguments('{"city": "'), _ERROR_FRAME, _DONE),
    )


_DIES_BEFORE_CONTENT: Final = Reply(content_type="text/event-stream", chunks=(_ERROR_FRAME, _DONE))
_DROPS_BEFORE_CONTENT: Final = Reply(content_type="text/event-stream", chunks=(_DONE,), abort_after=0)
_REJECTS_BEFORE_STREAMING: Final = Reply(
    status=500, body=json.dumps({"error": {"message": "overloaded", "type": "server_error", "code": 500}}).encode()
)


def _deployment_of(request: Request) -> str:
    deployment, _, route = request.target.lstrip("/").partition("/")
    assert request.method == "POST", request.method
    assert route == "chat/completions", request.target
    return deployment


def _peer(replies: Mapping[str, Reply]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body.get("stream") is True, body
        assert [tool["function"]["name"] for tool in body["tools"]] == ["get_weather"], body
        return replies[_deployment_of(request)]

    return respond


def _always(reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        _deployment_of(request)
        return reply

    return respond


def _fails_then_serves(failures: int, served: Reply) -> Callable[[Request], Reply]:
    attempts: Final = itertools.count(1)

    def respond(request: Request) -> Reply:
        _deployment_of(request)
        return _DIES_BEFORE_CONTENT if next(attempts) <= failures else served

    return respond


def _deployments_hit(wire: Wire) -> tuple[str, ...]:
    return tuple(_deployment_of(request) for request in wire.drain())


def _router(wire: Wire, deployments: tuple[str, ...], **settings: object) -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": f"openai/{_MODEL}", "api_base": f"{wire.url}/{name}", "api_key": _API_KEY},
            }
            for name in deployments
        ],
        disable_cooldowns=True,
        **settings,
    )


@dataclass(frozen=True, slots=True)
class _ToolCallStream:
    call_ids: tuple[str, ...]
    arguments: str
    finish_reasons: tuple[str, ...]


def _tool_call_deltas(chunk: object) -> tuple[object, ...]:
    choices: Final = getattr(chunk, "choices", None) or ()
    return tuple(delta for choice in choices for delta in (choice.delta.tool_calls or ()))


def _collect(chunks: tuple[object, ...]) -> _ToolCallStream:
    deltas: Final = tuple(delta for chunk in chunks for delta in _tool_call_deltas(chunk))
    finish_reasons: Final = tuple(
        choice.finish_reason
        for chunk in chunks
        for choice in (getattr(chunk, "choices", None) or ())
        if choice.finish_reason
    )
    return _ToolCallStream(
        call_ids=tuple(delta.id for delta in deltas if delta.id),
        arguments="".join(delta.function.arguments or "" for delta in deltas),
        finish_reasons=finish_reasons,
    )


def _request(**extra: object) -> dict[str, object]:
    return {
        "model": "primary",
        "messages": [{"role": "user", "content": _PROMPT}],
        "tools": list(_TOOLS),
        "stream": True,
        **extra,
    }


def _stream_sync(router: Router, **extra: object) -> _ToolCallStream:
    return _collect(tuple(router.completion(**_request(**extra))))


async def _stream_async(router: Router, **extra: object) -> _ToolCallStream:
    stream: Final = await router.acompletion(**_request(**extra))
    return _collect(tuple([chunk async for chunk in stream]))


def _stream(client: str, router: Router, **extra: object) -> _ToolCallStream:
    if client == "async":
        return asyncio.run(_stream_async(router, **extra))
    return _stream_sync(router, **extra)


_CLIENTS: Final = ("sync", "async")
_BACKUP_ANSWERS: Final = _ToolCallStream(
    call_ids=("call_backup",), arguments='{"city": "Lisbon"}', finish_reasons=("tool_calls",)
)


@pytest.mark.parametrize("client", _CLIENTS)
def test_primary_dies_before_the_tool_call_opens(client: str) -> None:
    replies: Final = {"primary": _DIES_BEFORE_CONTENT, "backup": _serves_tool_call("call_backup", "Lisbon")}
    with wire_server(_peer(replies)) as wire:
        router: Final = _router(wire, ("primary", "backup"), num_retries=0, fallbacks=_PRIMARY_TO_BACKUP)
        assert _stream(client, router) == _BACKUP_ANSWERS
        assert _deployments_hit(wire) == ("primary", "backup")


@pytest.mark.parametrize("client", _CLIENTS)
def test_primary_drops_the_connection_before_the_tool_call_opens(client: str) -> None:
    replies: Final = {"primary": _DROPS_BEFORE_CONTENT, "backup": _serves_tool_call("call_backup", "Lisbon")}
    with wire_server(_peer(replies)) as wire:
        router: Final = _router(wire, ("primary", "backup"), num_retries=0, fallbacks=_PRIMARY_TO_BACKUP)
        assert _stream(client, router) == _BACKUP_ANSWERS
        assert _deployments_hit(wire) == ("primary", "backup")


@pytest.mark.parametrize("client", _CLIENTS)
def test_primary_dies_after_tool_call_deltas_streamed(client: str) -> None:
    replies: Final = {
        "primary": _dies_after_tool_call_opens("call_primary"),
        "backup": _serves_tool_call("call_backup", "Lisbon"),
    }
    with wire_server(_peer(replies)) as wire:
        router: Final = _router(wire, ("primary", "backup"), num_retries=2, fallbacks=_PRIMARY_TO_BACKUP)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router)
        assert _deployments_hit(wire) == ("primary",)


@pytest.mark.parametrize("client", _CLIENTS)
def test_disable_fallbacks_holds_before_the_tool_call_opens(client: str) -> None:
    replies: Final = {"primary": _DIES_BEFORE_CONTENT, "backup": _serves_tool_call("call_backup", "Lisbon")}
    with wire_server(_peer(replies)) as wire:
        router: Final = _router(wire, ("primary", "backup"), num_retries=0, fallbacks=_PRIMARY_TO_BACKUP)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router, disable_fallbacks=True)
        assert _deployments_hit(wire) == ("primary",)


@pytest.mark.parametrize("client", _CLIENTS)
def test_primary_dies_before_the_tool_call_opens_is_retried_within_the_group(client: str) -> None:
    with wire_server(_fails_then_serves(1, _serves_tool_call("call_primary", "Lisbon"))) as wire:
        router: Final = _router(wire, ("primary",), num_retries=1, retry_after=0)
        assert _stream(client, router) == _ToolCallStream(
            call_ids=("call_primary",), arguments='{"city": "Lisbon"}', finish_reasons=("tool_calls",)
        )
        assert _deployments_hit(wire) == ("primary", "primary")


@pytest.mark.parametrize("client", _CLIENTS)
def test_request_num_retries_overrides_the_router_budget(client: str) -> None:
    with wire_server(_always(_DIES_BEFORE_CONTENT)) as wire:
        router: Final = _router(wire, ("primary",), num_retries=3, retry_after=0)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router, num_retries=1)
        assert _deployments_hit(wire) == ("primary", "primary")


@pytest.mark.parametrize("client", _CLIENTS)
@pytest.mark.parametrize("failure", (_DIES_BEFORE_CONTENT, _DROPS_BEFORE_CONTENT), ids=("error-frame", "dropped"))
def test_every_stream_dies_before_content_spends_exactly_the_budget(client: str, failure: Reply) -> None:
    num_retries: Final = 2
    with wire_server(_always(failure)) as wire:
        router: Final = _router(
            wire, ("primary", "backup"), num_retries=num_retries, fallbacks=_BOTH_WAYS, retry_after=0
        )
        with pytest.raises(litellm.APIConnectionError):
            _stream(client, router)
        assert _deployments_hit(wire) == ("primary",) * (num_retries + 1) + ("backup",) * (num_retries + 1)


@pytest.mark.parametrize("client", _CLIENTS)
def test_every_request_rejected_before_streaming_spends_a_bounded_budget(client: str) -> None:
    num_retries: Final = 1
    with wire_server(_always(_REJECTS_BEFORE_STREAMING)) as wire:
        router: Final = _router(
            wire, ("primary", "backup"), num_retries=num_retries, fallbacks=_BOTH_WAYS, retry_after=0
        )
        with pytest.raises(litellm.InternalServerError, match="overloaded"):
            _stream(client, router)
        hit: Final = _deployments_hit(wire)
    assert hit == ("primary",) * (num_retries + 1) + ("backup",) * (num_retries + 1), hit
