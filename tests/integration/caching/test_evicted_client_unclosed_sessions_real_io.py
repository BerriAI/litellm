import asyncio
import base64
import gc
import json
import struct
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Final, TypeAlias

import pytest
from aiohttp import web
from pydantic import BaseModel, ConfigDict

import litellm
from litellm import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream
from tests.integration._support.ports import allocate_port

_API_VERSION: Final = "2024-10-21"
_MESSAGES: Final = ({"role": "user", "content": "hi"},)
_VECTOR: Final = (0.25, 0.5, 0.75)
_STREAM_WORDS: Final = ("Hello", " from", " a", " stream", " that", " outlived", " its", " client")
_DRAIN_SECONDS: Final = 10.0

_Call: TypeAlias = Callable[[str, str], Awaitable[object]]


class _RequestBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    stream: bool = False
    encoding_format: str | None = None


@dataclass(frozen=True, slots=True)
class _Upstream:
    base: str
    release_stream: asyncio.Event


def _keys_that_overflow_the_client_cache() -> int:
    return litellm.in_memory_llm_clients_cache.max_size_in_memory + 30


def _sse_chunk(word: str, finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake",
        "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _embedding_reply(as_base64: bool) -> web.Response:
    embedding: Final = base64.b64encode(struct.pack("<3f", *_VECTOR)).decode() if as_base64 else _VECTOR
    return web.json_response(
        {
            "object": "list",
            "model": "fake",
            "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )


def _chat_reply() -> web.Response:
    return web.json_response(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1,
            "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


async def _held_stream(request: web.Request, release: asyncio.Event) -> web.StreamResponse:
    response: Final = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    await response.write(_sse_chunk(_STREAM_WORDS[0], None))
    await release.wait()
    for word in _STREAM_WORDS[1:]:
        await response.write(_sse_chunk(word, None))
    await response.write(_sse_chunk("", "stop"))
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


@asynccontextmanager
async def _fake_openai() -> AsyncIterator[_Upstream]:
    release: Final = asyncio.Event()

    async def handle(request: web.Request) -> web.StreamResponse:
        body: Final = _RequestBody.model_validate_json(await request.read())
        if request.path.endswith("/embeddings"):
            return _embedding_reply(body.encoding_format == "base64")
        if body.stream:
            return await _held_stream(request, release)
        return _chat_reply()

    app: Final = web.Application()
    app.router.add_post("/{path:.*}", handle)
    runner: Final = web.AppRunner(app)
    await runner.setup()
    port: Final = allocate_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield _Upstream(base=f"http://127.0.0.1:{port}", release_stream=release)
    finally:
        release.set()
        await runner.cleanup()


@contextmanager
def _loop_reports() -> Iterator[list[str]]:
    loop: Final = asyncio.get_running_loop()
    previous: Final = loop.get_exception_handler()
    reports: Final[list[str]] = []
    loop.set_exception_handler(lambda _loop, context: reports.append(repr(context)))
    try:
        yield reports
    finally:
        loop.set_exception_handler(previous)


def _azure_chat(base: str, api_key: str) -> Awaitable[object]:
    return litellm.acompletion(
        model="azure/chat-deployment", api_base=base, api_version=_API_VERSION, api_key=api_key, messages=_MESSAGES
    )


def _azure_embedding(base: str, api_key: str) -> Awaitable[object]:
    return litellm.aembedding(
        model="azure/embedding-deployment", api_base=base, api_version=_API_VERSION, api_key=api_key, input=["hi"]
    )


def _openai_float_embedding(base: str, api_key: str) -> Awaitable[object]:
    return litellm.aembedding(model="openai/fake-embedding", api_base=f"{base}/v1", api_key=api_key, input=["hi"])


def _openai_base64_embedding(base: str, api_key: str) -> Awaitable[object]:
    return litellm.aembedding(
        model="openai/fake-embedding", api_base=f"{base}/v1", api_key=api_key, input=["hi"], encoding_format="base64"
    )


_Tasks: TypeAlias = frozenset[asyncio.Task[object]]


def _other_pending_tasks() -> _Tasks:
    return frozenset(asyncio.all_tasks() - {asyncio.current_task()})


async def _churn(call: _Call, base: str) -> _Tasks:
    prefix: Final = f"sk-churn-{uuid.uuid4().hex}"
    await call(base, f"{prefix}-warmup")
    warmup_tasks: Final = _other_pending_tasks()
    if warmup_tasks:
        await asyncio.wait(warmup_tasks, timeout=_DRAIN_SECONDS)
    resident: Final = _other_pending_tasks()
    for index in range(_keys_that_overflow_the_client_cache()):
        await call(base, f"{prefix}-{index}")
    return resident


async def _tasks_left_after_draining(resident: _Tasks, deadline: float) -> _Tasks:
    while _other_pending_tasks() - resident and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)
    return _other_pending_tasks() - resident


async def _settled_reports(reports: list[str], resident: _Tasks) -> tuple[str, ...]:
    gc.collect()
    leftover: Final = await _tasks_left_after_draining(resident, asyncio.get_running_loop().time() + _DRAIN_SECONDS)
    assert not leftover, f"tasks still pending {_DRAIN_SECONDS}s after eviction: {sorted(map(repr, leftover))}"
    gc.collect()
    return tuple(reports)


async def _loop_reports_after_eviction(call: _Call) -> tuple[str, ...]:
    with _loop_reports() as reports:
        async with _fake_openai() as upstream:
            resident: Final = await _churn(call, upstream.base)
        return await _settled_reports(reports, resident)


@dataclass(frozen=True, slots=True)
class _StreamOutcome:
    text: str
    loop_reports: tuple[str, ...]


def _chunk_text(chunk: ModelResponseStream) -> str:
    return "".join(choice.delta.content or "" for choice in chunk.choices)


async def _read_a_stream_while_its_client_is_evicted(upstream: _Upstream) -> tuple[str, _Tasks]:
    stream: Final = await litellm.acompletion(
        model="azure/chat-deployment",
        api_base=upstream.base,
        api_version=_API_VERSION,
        api_key=f"sk-streamer-{uuid.uuid4().hex}",
        messages=_MESSAGES,
        stream=True,
    )
    assert isinstance(stream, CustomStreamWrapper)
    first: Final = _chunk_text(await anext(stream))
    resident: Final = await _churn(_azure_chat, upstream.base)
    gc.collect()
    upstream.release_stream.set()
    return first + "".join([_chunk_text(chunk) async for chunk in stream]), resident


async def _stream_through_eviction() -> _StreamOutcome:
    with _loop_reports() as reports:
        async with _fake_openai() as upstream:
            text, resident = await _read_a_stream_while_its_client_is_evicted(upstream)
        return _StreamOutcome(text=text, loop_reports=await _settled_reports(reports, resident))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    (
        pytest.param(_azure_chat, id="azure-chat"),
        pytest.param(_azure_embedding, id="azure-embedding"),
        pytest.param(_openai_base64_embedding, id="openai-base64-embedding"),
        pytest.param(_openai_float_embedding, id="openai-float-embedding"),
    ),
)
async def test_clients_evicted_from_the_cache_report_nothing_to_the_loop_exception_handler(call: _Call) -> None:
    reports: Final = await _loop_reports_after_eviction(call)

    assert reports == (), reports


@pytest.mark.asyncio
async def test_an_azure_stream_reads_to_the_end_while_its_client_is_evicted_and_leaves_nothing_unclosed() -> None:
    outcome: Final = await _stream_through_eviction()

    assert outcome.text == "".join(_STREAM_WORDS)
    assert outcome.loop_reports == (), outcome.loop_reports
