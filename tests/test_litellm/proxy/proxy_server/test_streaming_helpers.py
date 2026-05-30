"""Behavior pins for the proxy_server streaming helpers.

Pins covered:
- ``data_generator``
- ``async_assistants_data_generator``
- ``_get_client_requested_model_for_streaming``
- ``_restamp_streaming_chunk_model``
- ``_fast_serialize_simple_model_response_stream``
- ``_serialize_streaming_chunk``
- ``_apply_streaming_chunk_hooks``
- ``_format_streaming_sse_chunk``
- ``async_data_generator``
- ``select_data_generator``
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import (
    _apply_streaming_chunk_hooks,
    _fast_serialize_simple_model_response_stream,
    _format_streaming_sse_chunk,
    _get_client_requested_model_for_streaming,
    _restamp_streaming_chunk_model,
    _serialize_streaming_chunk,
    async_assistants_data_generator,
    async_data_generator,
    data_generator,
    select_data_generator,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

from .conftest import normalize


def _user_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test-key", user_id="u")


def _simple_chunk(model: str = "gpt-4", content: str = "hi") -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content=content, role="assistant"),
            )
        ],
        created=0,
        model=model,
        object="chat.completion.chunk",
    )


async def _async_iter(items):
    for it in items:
        yield it


async def _async_iter_raises(exc: Exception):
    # yield once then raise — exercises the mid-stream failure branch
    yield _simple_chunk(content="partial")
    raise exc


# ---------------------------------------------------------------------------
# data_generator
# ---------------------------------------------------------------------------


def test_data_generator_yields_sse_lines_for_dict_chunks():
    class DictChunk:
        def __init__(self, payload):
            self._payload = payload

        def dict(self):
            return self._payload

    chunks = [
        DictChunk({"id": "1", "object": "chat.completion.chunk", "model": "gpt-4"}),
        DictChunk({"id": "2", "object": "chat.completion.chunk", "model": "gpt-4"}),
    ]
    out = list(data_generator(chunks))

    assert len(out) == 2
    payloads = [json.loads(line.removeprefix("data: ").rstrip("\n\n")) for line in out]
    assert normalize(payloads[0]) == {
        "id": "<VOLATILE>",
        "object": "chat.completion.chunk",
        "model": "gpt-4",
    }
    assert payloads[1]["model"] == "gpt-4"


def test_data_generator_fallback_when_dict_raises_exception():
    class BadChunk:
        def dict(self):
            raise RuntimeError("cannot serialize")

    # When .dict() raises, the inner json.dumps(chunk) on a non-JSON-serializable
    # instance also raises — the generator does not catch the second failure.
    with pytest.raises((TypeError, RuntimeError)):
        list(data_generator([BadChunk()]))


# ---------------------------------------------------------------------------
# async_assistants_data_generator
# ---------------------------------------------------------------------------


class _FakeAssistantsStream:
    """Mimic the async-context-manager + async-iterable shape of the
    assistants streaming object (e.g. AssistantEventHandler)."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


@pytest.mark.asyncio
async def test_async_assistants_data_generator_yields_sse_and_done(monkeypatch):
    chunk = _simple_chunk(content="hello")

    async def _passthrough_hook(*, user_api_key_dict, response, data, **kwargs):
        return response

    monkeypatch.setattr(
        ps.proxy_logging_obj,
        "async_post_call_streaming_hook",
        _passthrough_hook,
    )

    stream = _FakeAssistantsStream([chunk])
    out = []
    async for line in async_assistants_data_generator(
        response=stream,
        user_api_key_dict=_user_auth(),
        request_data={},
    ):
        out.append(line)

    assert out[-1] == "data: [DONE]\n\n"
    body = json.loads(out[0].removeprefix("data: ").rstrip("\n\n"))
    assert normalize(body) == {
        "id": "<VOLATILE>",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello", "role": "assistant"},
            }
        ],
    }


@pytest.mark.asyncio
async def test_async_assistants_data_generator_hook_failure_yields_error_chunk(
    monkeypatch,
):
    async def _boom_hook(*args, **kwargs):
        raise RuntimeError("hook exploded")

    async def _noop_failure(*args, **kwargs):
        return None

    monkeypatch.setattr(
        ps.proxy_logging_obj, "async_post_call_streaming_hook", _boom_hook
    )
    monkeypatch.setattr(ps.proxy_logging_obj, "post_call_failure_hook", _noop_failure)

    stream = _FakeAssistantsStream([_simple_chunk()])
    out = []
    async for line in async_assistants_data_generator(
        response=stream,
        user_api_key_dict=_user_auth(),
        request_data={},
    ):
        out.append(line)

    assert any("error" in line for line in out)
    assert out[-1].startswith('data: {"error":')


# ---------------------------------------------------------------------------
# _get_client_requested_model_for_streaming
# ---------------------------------------------------------------------------


def test_get_client_requested_model_for_streaming_prefers_client_requested():
    request_data = {
        "_litellm_client_requested_model": "gpt-4",
        "model": "openai/internal-gpt-4",
        "litellm_call_id": "abc",
    }
    result = _get_client_requested_model_for_streaming(request_data)
    assert result == "gpt-4"

    snapshot = {
        "result": result,
        "client_field_preserved": request_data["_litellm_client_requested_model"],
        "model_field_preserved": request_data["model"],
    }
    assert normalize(snapshot) == {
        "result": "gpt-4",
        "client_field_preserved": "gpt-4",
        "model_field_preserved": "openai/internal-gpt-4",
    }


def test_get_client_requested_model_for_streaming_falls_back_to_model_field():
    result = _get_client_requested_model_for_streaming({"model": "claude-sonnet"})
    assert result == "claude-sonnet"


def test_get_client_requested_model_for_streaming_missing_returns_empty_invalid():
    """When neither key is set or values are non-strings, the helper returns ""
    rather than raising — callers depend on this to skip restamping."""
    assert _get_client_requested_model_for_streaming({}) == ""
    assert _get_client_requested_model_for_streaming({"model": 123}) == ""


# ---------------------------------------------------------------------------
# _restamp_streaming_chunk_model
# ---------------------------------------------------------------------------


def test_restamp_streaming_chunk_model_overrides_model_on_basemodel():
    chunk = _simple_chunk(model="openai/internal-x")
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="gpt-4",
        request_data={"litellm_call_id": "id-1"},
        model_mismatch_logged=False,
    )
    snapshot = {
        "model": new_chunk.model,
        "logged": logged,
        "same_object": new_chunk is chunk,
    }
    assert snapshot == {"model": "gpt-4", "logged": True, "same_object": True}


def test_restamp_streaming_chunk_model_overrides_model_on_dict():
    chunk = {"model": "internal", "choices": []}
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="gpt-4",
        request_data={},
        model_mismatch_logged=True,
    )
    assert new_chunk["model"] == "gpt-4"
    assert logged is True


def test_restamp_streaming_chunk_model_invalid_chunk_type_unchanged():
    """For a non-BaseModel, non-dict chunk the helper returns it as-is
    along with the original ``model_mismatch_logged`` flag."""
    chunk = "raw string chunk"
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="gpt-4",
        request_data={},
        model_mismatch_logged=False,
    )
    assert new_chunk == "raw string chunk"
    assert logged is False


# ---------------------------------------------------------------------------
# _fast_serialize_simple_model_response_stream
# ---------------------------------------------------------------------------


def test_fast_serialize_simple_model_response_stream_returns_bytes_payload():
    chunk = _simple_chunk()
    result = _fast_serialize_simple_model_response_stream(chunk)
    assert isinstance(result, bytes)
    payload = json.loads(result)
    assert normalize(payload) == {
        "id": "<VOLATILE>",
        "object": "chat.completion.chunk",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
            }
        ],
    }


def test_fast_serialize_simple_model_response_stream_with_usage_returns_none_invalid():
    """Fast path bails (returns None) when ``usage`` is populated — the slow
    path is required to preserve usage fields. Returning None here is the
    "I cannot handle this" sentinel, not a hard error."""
    chunk = _simple_chunk()
    chunk.usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    assert _fast_serialize_simple_model_response_stream(chunk) is None


# ---------------------------------------------------------------------------
# _serialize_streaming_chunk
# ---------------------------------------------------------------------------


def test_serialize_streaming_chunk_simple_uses_fast_path_bytes():
    result = _serialize_streaming_chunk(_simple_chunk())
    assert isinstance(result, bytes)
    payload = json.loads(result)
    assert normalize(payload) == {
        "id": "<VOLATILE>",
        "object": "chat.completion.chunk",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
            }
        ],
    }


def test_serialize_streaming_chunk_invalid_input_raises_attribute_error():
    """The helper is typed as ``BaseModel`` — handing it a plain dict trips
    the attribute-access path (no ``model_dump_json``)."""
    with pytest.raises(AttributeError):
        _serialize_streaming_chunk({"not": "a model"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _apply_streaming_chunk_hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_streaming_chunk_hooks_appends_to_str_so_far(monkeypatch):
    chunk = _simple_chunk(content="abc")

    async def _passthrough(*, user_api_key_dict, response, data, str_so_far=None):
        return response

    monkeypatch.setattr(
        ps.proxy_logging_obj, "async_post_call_streaming_hook", _passthrough
    )

    new_chunk, new_str = await _apply_streaming_chunk_hooks(
        chunk=chunk,
        user_api_key_dict=_user_auth(),
        request_data={},
        str_so_far="prior:",
    )

    observed = {
        "chunk_is_basemodel": isinstance(new_chunk, ModelResponseStream),
        "str_so_far": new_str,
        "grew": len(new_str) > len("prior:"),
    }
    assert observed == {
        "chunk_is_basemodel": True,
        "str_so_far": "prior:abc",
        "grew": True,
    }


@pytest.mark.asyncio
async def test_apply_streaming_chunk_hooks_hook_raises_exception(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("hook failed")

    monkeypatch.setattr(ps.proxy_logging_obj, "async_post_call_streaming_hook", _boom)

    with pytest.raises(RuntimeError):
        await _apply_streaming_chunk_hooks(
            chunk=_simple_chunk(),
            user_api_key_dict=_user_auth(),
            request_data={},
            str_so_far="",
        )


# ---------------------------------------------------------------------------
# _format_streaming_sse_chunk
# ---------------------------------------------------------------------------


def test_format_streaming_sse_chunk_handles_bytes_and_str_shapes():
    bytes_out = _format_streaming_sse_chunk(b'{"a":1}')
    str_out = _format_streaming_sse_chunk('{"a":1}')

    snapshot = {
        "bytes_out": bytes_out,
        "str_out": str_out,
        "bytes_starts_with_data": bytes_out.startswith(b"data: "),
    }
    assert snapshot == {
        "bytes_out": b'data: {"a":1}\n\n',
        "str_out": 'data: {"a":1}\n\n',
        "bytes_starts_with_data": True,
    }


def test_format_streaming_sse_chunk_invalid_empty_string_still_wraps():
    """Edge case: empty string still gets the ``data: \\n\\n`` wrapping
    — clients expect SSE shape even on empty payloads."""
    result = _format_streaming_sse_chunk("")
    assert result == "data: \n\n"


# ---------------------------------------------------------------------------
# async_data_generator
# ---------------------------------------------------------------------------


def _patch_logging_flags(monkeypatch, needs_wrap=False, needs_per_chunk=False):
    monkeypatch.setattr(
        ps.proxy_logging_obj,
        "needs_iterator_wrap",
        lambda: needs_wrap,
    )
    monkeypatch.setattr(
        ps.proxy_logging_obj,
        "needs_per_chunk_streaming_hook",
        lambda: needs_per_chunk,
    )
    # ``_fire_deferred_stream_logging`` is a classmethod — patch the
    # underlying function so the no-wrap branch is a no-op rather than
    # touching real logging globals.
    monkeypatch.setattr(
        ps.ProxyLogging,
        "_fire_deferred_stream_logging",
        staticmethod(lambda request_data: None),
    )


@pytest.mark.asyncio
async def test_async_data_generator_yields_sse_chunks_and_done(monkeypatch):
    _patch_logging_flags(monkeypatch)

    response = _async_iter([_simple_chunk(content="hello")])
    out = []
    async for line in async_data_generator(
        response=response,
        user_api_key_dict=_user_auth(),
        request_data={"model": "gpt-4"},
    ):
        out.append(line)

    assert out[-1] == "data: [DONE]\n\n"
    # First chunk is bytes (fast path) wrapped via _format_streaming_sse_chunk.
    first = out[0]
    assert isinstance(first, bytes)
    payload = json.loads(first.removeprefix(b"data: ").rstrip(b"\n\n"))
    assert normalize(payload) == {
        "id": "<VOLATILE>",
        "object": "chat.completion.chunk",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hello"},
            }
        ],
    }


@pytest.mark.asyncio
async def test_async_data_generator_mid_stream_exception_yields_error_payload(
    monkeypatch,
):
    _patch_logging_flags(monkeypatch)

    async def _noop_failure(*args, **kwargs):
        return None

    monkeypatch.setattr(ps.proxy_logging_obj, "post_call_failure_hook", _noop_failure)

    response = _async_iter_raises(RuntimeError("upstream blew up"))
    out = []
    async for line in async_data_generator(
        response=response,
        user_api_key_dict=_user_auth(),
        request_data={},
    ):
        out.append(line)

    # First entry is the successful "partial" chunk (bytes), last is the error.
    assert any(
        isinstance(item, str) and item.startswith('data: {"error":') for item in out
    )


# ---------------------------------------------------------------------------
# select_data_generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_data_generator_returns_async_generator(monkeypatch):
    _patch_logging_flags(monkeypatch)

    response = _async_iter([_simple_chunk()])
    gen = select_data_generator(
        response=response,
        user_api_key_dict=_user_auth(),
        request_data={"model": "gpt-4"},
    )

    # Drain to confirm it really is an async iterator emitting SSE shape.
    collected = []
    async for line in gen:
        collected.append(line)

    snapshot = {
        "is_async_iterable": hasattr(gen, "__aiter__"),
        "yielded_at_least_one": len(collected) >= 1,
        "ends_with_done": collected[-1] == "data: [DONE]\n\n",
    }
    assert snapshot == {
        "is_async_iterable": True,
        "yielded_at_least_one": True,
        "ends_with_done": True,
    }


def test_select_data_generator_missing_required_kwarg_raises_type_error():
    """``select_data_generator`` requires all three keyword args — calling
    without ``request_data`` raises TypeError at the wrapper, before any
    streaming starts."""
    with pytest.raises(TypeError):
        select_data_generator(response=_async_iter([]), user_api_key_dict=_user_auth())  # type: ignore[call-arg]
