from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import anthropic
import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import group_members, owned_proxy, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Request, Wire
from integration._support.wire import wire_server as _wire_server
from openai import AsyncOpenAI, OpenAI

from litellm.responses.utils import ResponsesAPIRequestUtils

PRICE: Final = 0.019
PROMPT_TOKENS: Final = 11
COMPLETION_TOKENS: Final = 4
MODEL: Final = "openai/gpt-5.4-mini"
CALLBACK_IMPORT: Final = "integration.streaming.fallback_cost_recorder.proxy_handler_instance"


@contextmanager
def wire_server(respond: Callable[[Request], Reply], *, port: int = 0) -> Iterator[Wire]:
    def handle(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object":"list","data":[{"id":"gpt-5.4-mini","object":"model"}]}')
        return respond(request)

    with _wire_server(handle, port=port) as peer:
        yield peer


def _post_requests(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    name: str
    url: str
    model_id: str
    input_cost: float = 0.001
    output_cost: float = 0.002


@dataclass(frozen=True, slots=True)
class AuditProxy:
    gateway: Gateway
    callback_log: Path


def _frame(identity: str, delta: Mapping[str, object], finish: str | None = None) -> bytes:
    value: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4-mini",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def _chat_chunks(identity: str, *, text: str = "Hello 雪 café") -> tuple[bytes, ...]:
    usage: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4-mini",
        "choices": [],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS, "total_tokens": 15},
    }
    first, second = text[:6], text[6:]
    return (
        _frame(identity, {"role": "assistant", "content": first}),
        _frame(identity, {"content": second}),
        _frame(identity, {}, "stop"),
        b"data: " + json.dumps(usage).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    )


def _chat_response(identity: str, *, text: str = "Hello 雪 café", status: int = 200) -> Reply:
    body: Final = {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.4-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS, "total_tokens": 15},
    }
    return Reply(status=status, body=json.dumps(body, ensure_ascii=False).encode())


def _responses_events(identity: str, *, text: str = "Hello 雪 café") -> tuple[bytes, ...]:
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": [
            {
                "id": "msg-" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS, "total_tokens": 15},
    }
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": [], "usage": None},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg-" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return tuple(
        (f"event: {event['type']}\n".encode() + b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n")
        for event in events
    )


def _responses_response(identity: str, *, text: str = "Hello 雪 café") -> Reply:
    body: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": [
            {
                "id": "msg-" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS, "total_tokens": 15},
    }
    return Reply(body=json.dumps(body, ensure_ascii=False).encode())


def _sse_error() -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(
            b'data: {"error": {"message": "overloaded", "type": "server_error", "code": 500}}\n\n',
            b"data: [DONE]\n\n",
        ),
    )


def _responses_error() -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(
            b'data: {"type":"error","sequence_number":0,"error":{"type":"server_error","code":500,"message":"overloaded"}}\n\n',
        ),
    )


def _model(endpoint: ModelEndpoint) -> dict[str, object]:
    return {
        "model_name": endpoint.name,
        "litellm_params": {
            "model": MODEL,
            "api_key": "synthetic-fallback-key",
            "api_base": endpoint.url + "/v1",
            "input_cost_per_token": endpoint.input_cost,
            "output_cost_per_token": endpoint.output_cost,
        },
        "model_info": {"id": endpoint.model_id},
    }


@contextmanager
def _proxy(
    gateway: Gateway,
    tmp_path: Path,
    endpoints: Sequence[ModelEndpoint],
    fallbacks: Sequence[Mapping[str, Sequence[str]]],
    *,
    name: str,
    redis: tuple[str, int] | None = None,
    cache: bool = False,
    workers: int = 2,
    extra_overrides: Mapping[str, str] | None = None,
) -> Iterator[AuditProxy]:
    callback_log: Final = (
        Path(os.environ["INTEGRATION_RESULTS_DIR"]) / f"{name}-callback-{os.getpid()}-{id(endpoints)}.jsonl"
    )
    config: Final = {
        "model_list": [_model(endpoint) for endpoint in endpoints],
        "router_settings": {
            "num_retries": 0,
            "disable_cooldowns": True,
            "fallbacks": [dict(fallback) for fallback in fallbacks],
        },
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
        "litellm_settings": {
            "cache": cache,
            "callbacks": [CALLBACK_IMPORT],
            **(
                {
                    "cache_params": {
                        "type": "redis",
                        "host": "os.environ/REDIS_HOST",
                        "port": "os.environ/REDIS_PORT",
                    }
                }
                if cache
                else {}
            ),
        },
    }
    config_path: Final = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    environment_overrides: Final = {
        "LITELLM_FALLBACK_COST_LOG": str(callback_log),
        **({"REDIS_HOST": redis[0], "REDIS_PORT": str(redis[1])} if redis is not None else {}),
        **(extra_overrides or {}),
    }
    with owned_proxy(gateway, tmp_path, environment_overrides, config=config_path, workers=workers) as candidate:
        yield AuditProxy(candidate, callback_log)


def _callback_records(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def _costs(path: Path) -> tuple[object, ...]:
    return tuple(record["response_cost"] for record in _callback_records(path))


def _assert_usage_cost(path: Path, expected: float = PRICE) -> None:
    records: Final = _callback_records(path)
    assert records[-1]["usage_cost"] == pytest.approx(expected), records[-1]


def _assert_fallback_headers(response: httpx.Response, model_id: str = "backup-id", attempts: str = "1") -> None:
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-attempted-fallbacks"] == attempts, response.headers
    assert response.headers["x-litellm-model-id"] == model_id, response.headers


def _assert_spend(request_id: str, expected: float = PRICE, status: str = "success") -> tuple[dict[str, object], ...]:
    encoded_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        "openai", "backup-id", request_id
    )
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, spend, model, model_id, api_base, prompt_tokens, completion_tokens, status "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR request_id=%s',
            (request_id, encoded_response_id),
            database_url=os.environ.get("INTEGRATION_PROXY_DATABASE_URL"),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert float(rows[0]["spend"]) == pytest.approx(expected), rows
    assert rows[0]["status"] == status, rows
    return tuple(rows)


def _chat_sync(
    proxy: Gateway, model: str, text: str, *, stream_options: bool = True
) -> tuple[dict[str, str], tuple[object, ...]]:
    client: Final = OpenAI(base_url=str(proxy.client.base_url) + "/v1", api_key=proxy.key, max_retries=0)
    with client.chat.completions.with_streaming_response.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        stream=True,
        stream_options={"include_usage": True} if stream_options else None,
    ) as streamed:
        headers: Final = dict(streamed.headers)
        chunks: Final = tuple(streamed.parse())
    client.close()
    return headers, chunks


async def _chat_async_inner(
    proxy: Gateway, model: str, text: str, stream_options: bool = True
) -> tuple[dict[str, str], tuple[object, ...]]:
    client: Final = AsyncOpenAI(base_url=str(proxy.client.base_url) + "/v1", api_key=proxy.key, max_retries=0)
    async with client.chat.completions.with_streaming_response.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        stream=True,
        stream_options={"include_usage": True} if stream_options else None,
    ) as streamed:
        headers: Final = dict(streamed.headers)
        chunks: Final = tuple([chunk async for chunk in await streamed.parse()])
    await client.close()
    return headers, chunks


def _chat_async(
    proxy: Gateway, model: str, text: str, *, stream_options: bool = True
) -> tuple[dict[str, str], tuple[object, ...]]:
    return asyncio.run(_chat_async_inner(proxy, model, text, stream_options))


def _responses_sync(proxy: Gateway, model: str, text: str) -> tuple[dict[str, str], tuple[object, ...]]:
    client: Final = OpenAI(base_url=str(proxy.client.base_url) + "/v1", api_key=proxy.key, max_retries=0)
    with client.responses.with_streaming_response.create(model=model, input=text, stream=True) as streamed:
        headers: Final = dict(streamed.headers)
        events: Final = tuple(streamed.parse())
    client.close()
    return headers, events


async def _responses_async_inner(proxy: Gateway, model: str, text: str) -> tuple[dict[str, str], tuple[object, ...]]:
    client: Final = AsyncOpenAI(base_url=str(proxy.client.base_url) + "/v1", api_key=proxy.key, max_retries=0)
    async with client.responses.with_streaming_response.create(model=model, input=text, stream=True) as streamed:
        headers: Final = dict(streamed.headers)
        events: Final = tuple([event async for event in await streamed.parse()])
    await client.close()
    return headers, events


def _responses_async(proxy: Gateway, model: str, text: str) -> tuple[dict[str, str], tuple[object, ...]]:
    return asyncio.run(_responses_async_inner(proxy, model, text))


def _messages_sync(proxy: Gateway, model: str, text: str) -> tuple[dict[str, str], str, tuple[object, ...]]:
    client: Final = anthropic.Anthropic(base_url=str(proxy.client.base_url), api_key=proxy.key, max_retries=0)
    with client.messages.with_streaming_response.create(
        model=model, max_tokens=64, messages=[{"role": "user", "content": text}], stream=True
    ) as streamed:
        headers: Final = dict(streamed.headers)
        events: Final = tuple(streamed.parse())
    content: Final = "".join(
        event.delta.text for event in events if event.type == "content_block_delta" and event.delta.type == "text_delta"
    )
    client.close()
    return headers, content, events


async def _messages_async_inner(
    proxy: Gateway, model: str, text: str
) -> tuple[dict[str, str], str, tuple[object, ...]]:
    client: Final = anthropic.AsyncAnthropic(base_url=str(proxy.client.base_url), api_key=proxy.key, max_retries=0)
    async with client.messages.with_streaming_response.create(
        model=model, max_tokens=64, messages=[{"role": "user", "content": text}], stream=True
    ) as streamed:
        headers: Final = dict(streamed.headers)
        events: Final = tuple([event async for event in await streamed.parse()])
    content: Final = "".join(
        event.delta.text for event in events if event.type == "content_block_delta" and event.delta.type == "text_delta"
    )
    await client.close()
    return headers, content, events


def _messages_async(proxy: Gateway, model: str, text: str) -> tuple[dict[str, str], str, tuple[object, ...]]:
    return asyncio.run(_messages_async_inner(proxy, model, text))


def _assert_chat_chunks(chunks: Sequence[object], text: str = "Hello 雪 café") -> None:
    assert "".join(getattr(choice.delta, "content", "") or "" for chunk in chunks for choice in chunk.choices) == text
    assert chunks[-1].usage.prompt_tokens == PROMPT_TOKENS
    assert chunks[-1].usage.completion_tokens == COMPLETION_TOKENS


def _assert_response_events(events: Sequence[object], text: str = "Hello 雪 café") -> None:
    assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == text
    assert events[-1].type == "response.completed"
    assert events[-1].response.usage.input_tokens == PROMPT_TOKENS
    assert events[-1].response.usage.output_tokens == COMPLETION_TOKENS


def _response_event_id(events: Sequence[object]) -> str:
    return str(events[-1].response.id)


def _read_sse(response: httpx.Response) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line.removeprefix("data: "))
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _response_id(path: str, body: bytes) -> str:
    if path.endswith("/responses"):
        events: Final = _read_sse(httpx.Response(200, content=body))
        completed: Final = next(event for event in events if event["type"] == "response.completed")
        return str(completed["response"]["id"])
    if path.endswith("/messages"):
        events: Final = _read_sse(httpx.Response(200, content=body))
        started: Final = next(event for event in events if event["type"] == "message_start")
        return str(started["message"]["id"])
    if b"data: " in body:
        chunks: Final = _read_sse(httpx.Response(200, content=body))
        return str(chunks[0]["id"])
    return str(json.loads(body)["id"])


def _request_identity(body: Mapping[str, object]) -> str:
    input_text: Final = body.get("input")
    if isinstance(input_text, str):
        return input_text
    if isinstance(input_text, list):
        input_item: Final = next((item for item in input_text if isinstance(item, Mapping)), None)
        if input_item is not None:
            content: Final = input_item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                content_item: Final = next((item for item in content if isinstance(item, Mapping)), None)
                if content_item is not None:
                    text: Final = content_item.get("text")
                    if isinstance(text, str):
                        return text
    messages: Final = body.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[-1], Mapping):
        content: Final = messages[-1].get("content")
        if isinstance(content, str):
            return content
    return str(body.get("model", "fallback-cost"))


def _assert_wire_request(request: Request, expected_stream: bool) -> None:
    body: Final = json.loads(request.body)
    assert body["stream"] is expected_stream, body
    assert body["model"] in ("primary", "backup"), body


@pytest.fixture(scope="module")
def audit_redis(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, int]]:
    directory: Final = tmp_path_factory.mktemp("fallback-cost-redis")
    with owned_redis(directory) as redis:
        yield redis.host, redis.port


def test_h1_chat_sync_sdk_fallback_preserves_chunk_cost_and_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h1-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h1",
        ) as rig,
    ):
        headers, chunks = _chat_sync(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    _assert_chat_chunks(chunks)
    assert _costs(rig.callback_log) == (None, None, None, None), "H1 fallback chunk costs"
    _assert_usage_cost(rig.callback_log)
    rows: Final = _assert_spend(identity)
    assert rows[0]["model"] == MODEL, rows
    assert rows[0]["model_id"] == "backup-id", rows
    assert rows[0]["api_base"] == backup.url + "/v1", rows
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_h2_chat_async_sdk_fallback_preserves_chunk_cost_and_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h2-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h2",
        ) as rig,
    ):
        headers, chunks = _chat_async(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    _assert_chat_chunks(chunks)
    assert _costs(rig.callback_log) == (None, None, None, None), "H2 fallback chunk costs"
    _assert_usage_cost(rig.callback_log)
    rows: Final = _assert_spend(identity)
    assert rows[0]["model_id"] == "backup-id", rows
    assert rows[0]["api_base"] == backup.url + "/v1", rows
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_h3_chat_raw_httpx_fallback_preserves_bytes_and_chunk_cost(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h3-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h3",
        ) as rig,
    ):
        with rig.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {rig.gateway.key}"},
        ) as response:
            body: Final = response.read()
            headers: Final = dict(response.headers)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    assert b"Hello " in body and "雪 café".encode() in body
    assert _costs(rig.callback_log) == (None, None, None, None), "H3 fallback chunk costs"
    _assert_usage_cost(rig.callback_log)
    rows: Final = _assert_spend(identity)
    assert rows[0]["model_id"] == "backup-id", rows
    assert rows[0]["api_base"] == backup.url + "/v1", rows
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_h4_direct_backup_chat_stream_is_an_unchanged_control(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h4-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h4",
        ) as rig,
    ):
        headers, chunks = _chat_sync(rig.gateway, "backup", identity)
    assert headers["x-litellm-model-id"] == "backup-id", headers
    _assert_chat_chunks(chunks)
    assert _costs(rig.callback_log) == (None, None, None, None), "H4 direct-backup chunk costs"
    _assert_spend(identity)
    _assert_usage_cost(rig.callback_log)
    assert len(_post_requests(primary)) == 0
    assert len(_post_requests(backup)) == 1


def test_h5_chat_nonstreaming_fallback_preserves_response_cost_and_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h5-" + str(os.getpid())
    with (
        wire_server(lambda request: _chat_response(identity, status=500)) as primary,
        wire_server(lambda request: _chat_response(identity)) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h5",
        ) as rig,
    ):
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": "primary", "messages": [{"role": "user", "content": identity}]},
        )
    _assert_fallback_headers(response, "backup-id")
    assert response.headers["x-litellm-response-cost"] == "0.019", response.headers
    _assert_spend(identity)
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_h6_responses_sync_sdk_fallback_prices_only_completed_event(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h6-" + str(os.getpid())
    with (
        wire_server(lambda request: _responses_error()) as primary,
        wire_server(
            lambda request: Reply(content_type="text/event-stream", chunks=_responses_events(identity))
        ) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h6",
        ) as rig,
    ):
        headers, events = _responses_sync(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    _assert_response_events(events)
    assert _costs(rig.callback_log) == (None, None, pytest.approx(PRICE)), "H6 Responses event costs"
    rows: Final = _assert_spend(identity)
    assert rows[0]["model_id"] == "backup-id", rows
    assert rows[0]["api_base"] == backup.url + "/v1/responses", rows


def test_h7_responses_async_sdk_fallback_prices_only_completed_event(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h7-" + str(os.getpid())
    with (
        wire_server(lambda request: _responses_error()) as primary,
        wire_server(
            lambda request: Reply(content_type="text/event-stream", chunks=_responses_events(identity))
        ) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h7",
        ) as rig,
    ):
        headers, events = _responses_async(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    _assert_response_events(events)
    assert _costs(rig.callback_log) == (None, None, pytest.approx(PRICE)), "H7 Responses event costs"
    rows: Final = _assert_spend(identity)
    assert rows[0]["model_id"] == "backup-id", rows
    assert rows[0]["api_base"] == backup.url + "/v1/responses", rows


def test_h8_direct_backup_responses_stream_is_an_unchanged_control(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h8-" + str(os.getpid())
    with (
        wire_server(lambda request: _responses_error()) as primary,
        wire_server(
            lambda request: Reply(content_type="text/event-stream", chunks=_responses_events(identity))
        ) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h8",
        ) as rig,
    ):
        headers, events = _responses_sync(rig.gateway, "backup", identity)
    assert headers["x-litellm-model-id"] == "backup-id", headers
    _assert_response_events(events)
    assert _costs(rig.callback_log) == (None, None, pytest.approx(PRICE)), "H8 direct-backup Responses event costs"
    _assert_spend(identity)
    assert len(_post_requests(primary)) == 0
    assert len(_post_requests(backup)) == 1


def test_h9_responses_nonstreaming_fallback_preserves_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h9-" + str(os.getpid())
    with (
        wire_server(lambda request: Reply(status=500, body=b'{"error":{"message":"overloaded"}}')) as primary,
        wire_server(lambda request: _responses_response(identity)) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h9",
        ) as rig,
    ):
        response: Final = rig.gateway.request("POST", "/v1/responses", {"model": "primary", "input": identity})
        body: Final = response.json()
    _assert_fallback_headers(response, "backup-id")
    assert body["usage"]["input_tokens"] == PROMPT_TOKENS
    assert body["usage"]["output_tokens"] == COMPLETION_TOKENS
    _assert_spend(str(body["id"]))


def test_h10_messages_sync_and_async_stream_fallback_is_a_raw_bytes_control(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h10-" + str(os.getpid())

    def primary_reply(request: Request) -> Reply:
        return _responses_error() if request.target.endswith("/responses") else _sse_error()

    def backup_reply(request: Request) -> Reply:
        request_body: Final = json.loads(request.body)
        request_identity: Final = _request_identity(request_body)
        if request.target.endswith("/responses"):
            return Reply(content_type="text/event-stream", chunks=_responses_events(request_identity))
        return Reply(content_type="text/event-stream", chunks=_chat_chunks(request_identity))

    with (
        wire_server(primary_reply) as primary,
        wire_server(backup_reply) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h10",
        ) as rig,
    ):
        sync_headers, sync_text, sync_events = _messages_sync(rig.gateway, "primary", identity + "-sync")
        async_headers, async_text, async_events = _messages_async(rig.gateway, "primary", identity + "-async")
    _assert_fallback_headers(httpx.Response(200, headers=sync_headers), "backup-id")
    _assert_fallback_headers(httpx.Response(200, headers=async_headers), "backup-id")
    assert sync_text == "Hello 雪 café", sync_text
    assert async_text == "Hello 雪 café", async_text
    assert next(event for event in sync_events if event.type == "message_delta").usage.input_tokens == PROMPT_TOKENS
    assert next(event for event in async_events if event.type == "message_delta").usage.input_tokens == PROMPT_TOKENS
    assert (
        next(event for event in sync_events if event.type == "message_delta").usage.output_tokens == COMPLETION_TOKENS
    )
    assert (
        next(event for event in async_events if event.type == "message_delta").usage.output_tokens == COMPLETION_TOKENS
    )
    assert _costs(rig.callback_log) == (None, None, None, None, None, None, None, None, None, None), (
        "H10 raw-bytes callback costs"
    )
    sync_response_id: Final = str(next(event.message.id for event in sync_events if event.type == "message_start"))
    async_response_id: Final = str(next(event.message.id for event in async_events if event.type == "message_start"))
    _assert_spend(sync_response_id)
    _assert_spend(async_response_id)
    assert len(_post_requests(primary)) == 2
    assert len(_post_requests(backup)) == 2


def test_h11_messages_nonstreaming_fallback_preserves_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h11-" + str(os.getpid())
    with (
        wire_server(
            lambda request: (
                Reply(status=500, body=b'{"error":{"message":"overloaded"}}')
                if request.target.endswith("/responses")
                else _chat_response(identity, status=500)
            )
        ) as primary,
        wire_server(
            lambda request: (
                _responses_response(_request_identity(json.loads(request.body)))
                if request.target.endswith("/responses")
                else _chat_response(_request_identity(json.loads(request.body)))
            )
        ) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h11",
        ) as rig,
    ):
        client: Final = anthropic.Anthropic(
            base_url=str(rig.gateway.client.base_url), api_key=rig.gateway.key, max_retries=0
        )
        response: Final = client.messages.with_raw_response.create(
            model="primary", max_tokens=64, messages=[{"role": "user", "content": identity}]
        )
        message: Final = response.parse()
        client.close()
    assert response.headers["x-litellm-attempted-fallbacks"] == "1", response.headers
    assert message.content[0].text == "Hello 雪 café"
    _assert_spend(str(message.id))
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_h12_two_hop_chat_fallback_preserves_chunk_cost_and_attribution(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h12-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: _sse_error()) as mid,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (
                ModelEndpoint("primary", primary.url, "primary-id"),
                ModelEndpoint("mid", mid.url, "mid-id"),
                ModelEndpoint("backup", backup.url, "backup-id"),
            ),
            ({"primary": ("mid",)}, {"mid": ("backup",)}),
            name="h12",
        ) as rig,
    ):
        headers, chunks = _chat_sync(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers), attempts="2")
    _assert_chat_chunks(chunks)
    assert _costs(rig.callback_log) == (None, None, None, None), "H12 two-hop chunk costs"
    _assert_spend(identity)
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(mid)) == 1
    assert len(_post_requests(backup)) == 1


def test_h13_two_hop_responses_fallback_preserves_completed_cost(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h13-" + str(os.getpid())
    with (
        wire_server(lambda request: _responses_error()) as primary,
        wire_server(lambda request: _responses_error()) as mid,
        wire_server(
            lambda request: Reply(content_type="text/event-stream", chunks=_responses_events(identity))
        ) as backup,
        _proxy(
            gateway,
            tmp_path,
            (
                ModelEndpoint("primary", primary.url, "primary-id"),
                ModelEndpoint("mid", mid.url, "mid-id"),
                ModelEndpoint("backup", backup.url, "backup-id"),
            ),
            ({"primary": ("mid",)}, {"mid": ("backup",)}),
            name="h13",
        ) as rig,
    ):
        _headers, events = _responses_sync(rig.gateway, "primary", identity)
    _assert_response_events(events)
    assert _costs(rig.callback_log) == (None, None, pytest.approx(PRICE)), "H13 two-hop Responses event costs"
    _assert_spend(identity)
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(mid)) == 1
    assert len(_post_requests(backup)) == 1


def test_h14_pre_stream_http_failure_fallback_keeps_item_cost_none(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h14-" + str(os.getpid())
    with (
        wire_server(lambda request: Reply(status=500, body=b'{"error":{"message":"overloaded"}}')) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h14",
        ) as rig,
    ):
        headers, chunks = _chat_sync(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers))
    _assert_chat_chunks(chunks)
    assert _costs(rig.callback_log) == (None, None, None, None), "H14 pre-stream fallback costs"
    _assert_spend(identity)


def test_h15_openai_pass_through_does_not_stamp_router_item_metadata(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "h15-" + str(os.getpid())
    primary_chunks: Final = _chat_chunks(identity)
    with (
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=primary_chunks)) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=primary_chunks)) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="h15",
            extra_overrides={"OPENAI_API_BASE": primary.url, "OPENAI_API_KEY": "synthetic-openai-key"},
        ) as rig,
    ):
        response: Final = rig.gateway.request(
            "POST",
            "/openai/v1/chat/completions",
            {
                "model": "primary",
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert response.status_code == 200, response.text
    assert response.content == b"".join(primary_chunks)
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 0
    _assert_spend(response.headers["x-litellm-call-id"], expected=0.0)


def test_s1_chat_backup_sse_error_is_returned_and_spend_is_zero(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "s1-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: _sse_error()) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="s1",
        ) as rig,
    ):
        with rig.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "primary", "messages": [{"role": "user", "content": identity}], "stream": True},
            headers={"Authorization": f"Bearer {rig.gateway.key}"},
        ) as response:
            body: Final = response.read()
            headers: Final = dict(response.headers)
            status: Final = response.status_code
    assert status in (200, 500), body
    assert b"overloaded" in body
    assert headers["x-litellm-attempted-fallbacks"] == "1", headers
    _assert_spend(headers["x-litellm-call-id"], expected=0, status="failure")
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_s2_chat_backup_http_401_reaches_caller_and_counts_both_legs(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "s2-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(status=401, body=b'{"error":{"message":"unauthorized"}}')) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="s2",
        ) as rig,
    ):
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": "primary", "messages": [{"role": "user", "content": identity}], "stream": True},
        )
    assert response.status_code in (401, 500), response.text
    assert "error" in response.json(), response.text
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 1


def test_s3_chat_disable_fallbacks_returns_primary_error_without_backup(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "s3-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="s3",
        ) as rig,
    ):
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "primary",
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "disable_fallbacks": True,
            },
        )
    assert response.status_code in (200, 500), response.text
    assert "overloaded" in response.text, response.text
    assert len(_post_requests(primary)) == 1
    assert len(_post_requests(backup)) == 0


def test_s4_chat_without_stream_options_preserves_item_cost_and_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "s4-" + str(os.getpid())

    def backup_reply(request: Request) -> Reply:
        request_body: Final = json.loads(request.body)
        chunks: Final = _chat_chunks(_request_identity(request_body))
        return Reply(
            content_type="text/event-stream", chunks=tuple(chunk for chunk in chunks if b'"usage"' not in chunk)
        )

    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(backup_reply) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="s4",
        ) as rig,
    ):
        headers, chunks = _chat_sync(rig.gateway, "primary", identity, stream_options=False)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    assert (
        "".join(getattr(choice.delta, "content", "") or "" for chunk in chunks for choice in chunk.choices)
        == "Hello 雪 café"
    )
    assert all(chunk.usage is None for chunk in chunks), chunks
    assert _costs(rig.callback_log) == (None, None, None, None), "S4 no-usage-chunk fallback costs"
    spend_rows: Final = _assert_spend(identity, expected=0.022)
    assert (spend_rows[0]["prompt_tokens"], spend_rows[0]["completion_tokens"]) == (12, 5), spend_rows


def test_e1_zero_priced_backup_keeps_item_cost_contract_and_zero_spend(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "e1-" + str(os.getpid())
    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))) as backup,
        _proxy(
            gateway,
            tmp_path,
            (
                ModelEndpoint("primary", primary.url, "primary-id"),
                ModelEndpoint("backup", backup.url, "backup-id", input_cost=0.0, output_cost=0.0),
            ),
            ({"primary": ("backup",)},),
            name="e1",
        ) as rig,
    ):
        headers, chunks = _chat_sync(rig.gateway, "primary", identity)
    _assert_fallback_headers(httpx.Response(200, headers=headers), "backup-id")
    _assert_chat_chunks(chunks)
    assert _costs(rig.callback_log) == (None, None, None, None), "E1 zero-price fallback costs"
    _assert_spend(identity, expected=0)


def test_e2_three_sequential_fallbacks_create_three_priced_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    identities: Final = ("e2-a-" + str(os.getpid()), "e2-b-" + str(os.getpid()), "e2-c-" + str(os.getpid()))

    def backup_reply(request: Request) -> Reply:
        request_body: Final = json.loads(request.body)
        return Reply(content_type="text/event-stream", chunks=_chat_chunks(_request_identity(request_body)))

    with (
        wire_server(lambda request: _sse_error()) as primary,
        wire_server(backup_reply) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="e2",
        ) as rig,
    ):
        headers_list: Final = tuple(_chat_sync(rig.gateway, "primary", identity)[0] for identity in identities)
    assert all(cost is None for cost in _costs(rig.callback_log)), "E2 sequential fallback costs"
    assert tuple(headers["x-litellm-attempted-fallbacks"] for headers in headers_list) == ("1", "1", "1")
    assert tuple(headers["x-litellm-model-id"] for headers in headers_list) == ("backup-id", "backup-id", "backup-id")
    spend_rows: Final = tuple(_assert_spend(identity)[0] for identity in identities)
    assert tuple(row["request_id"] for row in spend_rows) == identities
    assert len(_post_requests(primary)) == 3
    assert len(_post_requests(backup)) == 3


def test_e3_redis_cache_hit_does_not_increase_backup_count(
    gateway: Gateway, tmp_path: Path, audit_redis: tuple[str, int]
) -> None:
    chat_identity: Final = "e3-chat-" + str(os.getpid())
    responses_identity: Final = "e3-responses-" + str(os.getpid())

    def backup_reply(request: Request) -> Reply:
        request_body: Final = json.loads(request.body)
        identity: Final = _request_identity(request_body)
        if request.target.endswith("/responses"):
            return Reply(content_type="text/event-stream", chunks=_responses_events(identity))
        return Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))

    with (
        wire_server(
            lambda request: _responses_error() if request.target.endswith("/responses") else _sse_error()
        ) as primary,
        wire_server(backup_reply) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="e3",
            redis=audit_redis,
            cache=True,
        ) as rig,
    ):
        first_chat: Final = _chat_sync(rig.gateway, "primary", chat_identity)
        first_chat_costs: Final = _costs(rig.callback_log)
        second_chat: Final = _chat_sync(rig.gateway, "primary", chat_identity)
        second_chat_costs: Final = _costs(rig.callback_log)
        first_responses: Final = _responses_sync(rig.gateway, "primary", responses_identity)
        first_responses_all_costs: Final = _costs(rig.callback_log)
        second_responses: Final = _responses_sync(rig.gateway, "primary", responses_identity)
        all_costs: Final = _costs(rig.callback_log)
        second_responses_costs: Final = all_costs[len(first_responses_all_costs) :]
    _assert_chat_chunks(first_chat[1])
    _assert_chat_chunks(second_chat[1])
    _assert_response_events(first_responses[1])
    _assert_response_events(second_responses[1])
    second_chat_item_costs: Final = second_chat_costs[len(first_chat_costs) :]
    assert first_chat_costs and all(cost == first_chat_costs[0] for cost in first_chat_costs), (
        "E3 first chat item costs"
    )
    assert all(cost == first_chat_costs[0] for cost in second_chat_item_costs), "E3 cache-hit chat item costs"
    assert second_responses_costs == (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        pytest.approx(0.019),
    ), "E3 cache-hit Responses event costs"
    assert len(_post_requests(backup)) == 2
    _assert_spend(chat_identity)
    _assert_spend(responses_identity)


def _concurrent_request(proxy: Gateway, path: str, body: Mapping[str, object]) -> tuple[int, str, bytes]:
    response: Final = proxy.request("POST", path, body)
    response_id: Final = (
        _response_id(path, response.content)
        if response.status_code == 200 and b'"error"' not in response.content
        else response.headers.get("x-litellm-call-id", "")
    )
    return response.status_code, response_id, response.content


def _concurrent_fresh_request(proxy: Gateway, path: str, body: Mapping[str, object]) -> tuple[int, str, bytes]:
    limits: Final = httpx.Limits(max_keepalive_connections=0)
    with httpx.Client(
        base_url=str(proxy.client.base_url),
        timeout=15,
        trust_env=False,
        limits=limits,
    ) as client:
        response: Final = client.post(
            path,
            json=body,
            headers={"Authorization": f"Bearer {proxy.key}", "Connection": "close"},
        )
    response_id: Final = (
        _response_id(path, response.content)
        if response.status_code == 200 and b'"error"' not in response.content
        else response.headers.get("x-litellm-call-id", "")
    )
    return response.status_code, response_id, response.content


def test_c1_thirty_concurrent_mixed_fallbacks_price_every_success_once(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "c1-" + str(os.getpid())
    counter: Final = iter(range(30))

    def primary_reply(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        identity: Final = _request_identity(body)
        if request.target.endswith("/responses"):
            return _responses_error()
        if body.get("stream") is True:
            return _sse_error()
        return _chat_response(identity, status=500)

    def backup_reply(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        identity: Final = _request_identity(body)
        if request.target.endswith("/responses"):
            if body.get("stream") is True:
                return Reply(content_type="text/event-stream", chunks=_responses_events(identity))
            return _responses_response(identity)
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_chat_chunks(identity))
        return _chat_response(identity)

    with (
        wire_server(primary_reply) as primary,
        wire_server(backup_reply) as backup,
        _proxy(
            gateway,
            tmp_path,
            (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
            ({"primary": ("backup",)},),
            name="c1",
        ) as rig,
    ):
        requests: Final = (
            tuple(
                (
                    "/v1/chat/completions",
                    {
                        "model": "primary",
                        "messages": [{"role": "user", "content": f"{identity}-chat-{next(counter)}"}],
                        "stream": True,
                    },
                )
                for _ in range(8)
            )
            + tuple(
                (
                    "/v1/responses",
                    {"model": "primary", "input": f"{identity}-response-{next(counter)}", "stream": True},
                )
                for _ in range(8)
            )
            + tuple(
                (
                    "/v1/messages",
                    {
                        "model": "primary",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": f"{identity}-message-{next(counter)}"}],
                        "stream": True,
                    },
                )
                for _ in range(7)
            )
            + tuple(
                (
                    "/v1/chat/completions",
                    {
                        "model": "primary",
                        "messages": [{"role": "user", "content": f"{identity}-nonstream-{next(counter)}"}],
                    },
                )
                for _ in range(7)
            )
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            initial_results: Final = tuple(
                executor.map(lambda item: _concurrent_fresh_request(rig.gateway, *item), requests)
            )

        def send_until_both_workers(
            sent_requests: tuple[tuple[str, Mapping[str, object]], ...],
            sent_results: tuple[tuple[int, str, bytes], ...],
        ) -> tuple[tuple[tuple[str, Mapping[str, object]], ...], tuple[tuple[int, str, bytes], ...]]:
            worker_pids: Final = frozenset(record["worker_pid"] for record in _callback_records(rig.callback_log))
            if len(worker_pids) == 2 or len(sent_requests) >= 120:
                return sent_requests, sent_results
            extra_count: Final = min(8, 120 - len(sent_requests))
            extra_requests: Final = tuple(
                (
                    "/v1/chat/completions",
                    {
                        "model": "primary",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{identity}-extra-chat-{len(sent_requests) + index}",
                            }
                        ],
                        "stream": True,
                    },
                )
                for index in range(extra_count)
            )
            with ThreadPoolExecutor(max_workers=8) as executor:
                extra_results: Final = tuple(
                    executor.map(lambda item: _concurrent_fresh_request(rig.gateway, *item), extra_requests)
                )
            return send_until_both_workers(
                (*sent_requests, *extra_requests),
                (*sent_results, *extra_results),
            )

        final_requests, final_results = send_until_both_workers(requests, initial_results)

    assert len(final_requests) <= 120
    assert all(result[0] == 200 for result in final_results), tuple(result[0] for result in final_results)
    response_ids: Final = tuple(result[1] for result in final_results)
    assert all(response_ids), response_ids
    assert len(set(response_ids)) == len(response_ids), response_ids
    for (path, request_body), (_, response_id, _) in zip(final_requests, final_results, strict=True):
        _assert_spend(_request_identity(request_body) if path.endswith("/responses") else response_id)
    chat_costs: Final = tuple(
        record["response_cost"]
        for record in _callback_records(rig.callback_log)
        if record["item_type"] == "ModelResponseStream" and record["event_type"] is None
    )
    assert chat_costs and all(cost is None for cost in chat_costs), "C1 chat stream item costs"
    assert len({record["worker_pid"] for record in _callback_records(rig.callback_log)}) == 2, "C1 worker coverage"
    assert len(_post_requests(primary)) == len(final_requests)
    assert len(_post_requests(backup)) == len(final_requests)


def test_c2_backup_restart_on_fixed_port_recovers_after_concurrent_outage(gateway: Gateway, tmp_path: Path) -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        backup_port: Final = reservation.getsockname()[1]
    identity: Final = "c2-" + str(os.getpid())

    def primary_reply(request: Request) -> Reply:
        return _sse_error()

    def backup_reply(request: Request) -> Reply:
        request_body: Final = json.loads(request.body)
        return Reply(content_type="text/event-stream", chunks=_chat_chunks(_request_identity(request_body)))

    def send_burst(stage: str) -> tuple[tuple[int, str, bytes], ...]:
        with ThreadPoolExecutor(max_workers=8) as executor:
            return tuple(
                executor.map(
                    lambda index: _concurrent_request(
                        rig.gateway,
                        "/v1/chat/completions",
                        {
                            "model": "primary",
                            "messages": [{"role": "user", "content": f"{identity}-{stage}-{index}"}],
                            "stream": True,
                        },
                    ),
                    range(8),
                )
            )

    with wire_server(primary_reply) as primary:
        backup_stack: Final = ExitStack()
        backup: Final = backup_stack.enter_context(wire_server(backup_reply, port=backup_port))
        try:
            with _proxy(
                gateway,
                tmp_path,
                (ModelEndpoint("primary", primary.url, "primary-id"), ModelEndpoint("backup", backup.url, "backup-id")),
                ({"primary": ("backup",)},),
                name="c2",
            ) as rig:
                primary.drain()
                backup.drain()
                available: Final = send_burst("available")
                assert tuple(result[0] for result in available) == (200, 200, 200, 200, 200, 200, 200, 200)
                assert len(_post_requests(primary)) == 8
                assert len(_post_requests(backup)) == 8
                for _, call_id, _ in available:
                    _assert_spend(call_id)

                backup_stack.close()
                outage: Final = send_burst("outage")
                assert tuple(result[0] for result in outage) == (500, 500, 500, 500, 500, 500, 500, 500)
                assert tuple(b'"error"' in result[2] for result in outage) == (
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ), outage
                assert len(_post_requests(primary)) == 8
                assert len(_post_requests(backup)) == 0
                for _, call_id, _ in outage:
                    _assert_spend(call_id, expected=0.0, status="failure")

                with wire_server(
                    backup_reply,
                    port=backup_port,
                ) as restarted:
                    recovered: Final = send_burst("recovered")
                    assert tuple(result[0] for result in recovered) == (200, 200, 200, 200, 200, 200, 200, 200)
                    assert tuple(b'"error"' not in result[2] for result in recovered) == (
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ), recovered
                    assert len(_post_requests(restarted)) == 8
                    assert len(_post_requests(primary)) == 8
                    for _, call_id, _ in recovered:
                        _assert_spend(call_id)
        finally:
            backup_stack.close()


def test_c3_surviving_proxy_worker_serves_after_one_worker_is_killed(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "c3-" + str(os.getpid())
    backup_gate: Final = threading.Event()
    backup_posts: Final = SimpleQueue[Request]()

    def primary_reply(request: Request) -> Reply:
        return _sse_error()

    def backup_reply(request: Request) -> Reply:
        backup_posts.put(request)
        request_body: Final = json.loads(request.body)
        return Reply(
            content_type="text/event-stream",
            chunks=_chat_chunks(_request_identity(request_body)),
            gate_after_first=backup_gate,
        )

    with (
        wire_server(primary_reply) as primary,
        wire_server(backup_reply) as backup,
    ):
        config: Final = {
            "model_list": [
                _model(ModelEndpoint("primary", primary.url, "primary-id")),
                _model(ModelEndpoint("backup", backup.url, "backup-id")),
            ],
            "router_settings": {"num_retries": 0, "disable_cooldowns": True, "fallbacks": [{"primary": ["backup"]}]},
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            "litellm_settings": {"callbacks": [CALLBACK_IMPORT]},
        }
        config_path: Final = tmp_path / "c3.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        callback_log: Final = Path(os.environ["INTEGRATION_RESULTS_DIR"]) / f"c3-callback-{os.getpid()}.jsonl"
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"LITELLM_FALLBACK_COST_LOG": str(callback_log)},
            config=config_path,
            workers=2,
        ) as owned:
            members: Final = group_members(owned.process.pid)
            workers: Final = tuple(member for member in members if member.pid != owned.process.pid)
            assert len(workers) >= 2, members
            primary.drain()
            backup.drain()
            try:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    in_flight: Final = tuple(
                        executor.submit(
                            _concurrent_request,
                            owned.gateway,
                            "/v1/chat/completions",
                            {
                                "model": "primary",
                                "messages": [{"role": "user", "content": f"{identity}-in-flight-{index}"}],
                                "stream": True,
                            },
                        )
                        for index in range(8)
                    )
                    eventually(lambda: backup_posts.qsize(), lambda count: count == 8, seconds=20)
                    os.kill(workers[0].pid, signal.SIGTERM)
                    backup_gate.set()
                    in_flight_responses: Final = tuple(future.result(timeout=60) for future in in_flight)
                    surviving_responses: Final = tuple(
                        executor.map(
                            lambda index: _concurrent_request(
                                owned.gateway,
                                "/v1/chat/completions",
                                {
                                    "model": "primary",
                                    "messages": [{"role": "user", "content": f"{identity}-surviving-{index}"}],
                                    "stream": True,
                                },
                            ),
                            range(8),
                        )
                    )
            finally:
                backup_gate.set()
    successful: Final = tuple(response for response in in_flight_responses + surviving_responses if response[0] == 200)
    assert tuple(result[0] for result in surviving_responses) == (
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
    ), surviving_responses
    assert len(successful) >= 8, (in_flight_responses, surviving_responses)
    for _, response_id, _ in successful:
        _assert_spend(response_id)
    assert len(_post_requests(primary)) == 16
    assert len(_post_requests(backup)) == 16
