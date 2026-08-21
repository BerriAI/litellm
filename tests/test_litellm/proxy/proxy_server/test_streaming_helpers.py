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

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse

import litellm
from litellm.constants import RETURN_RAW_MODEL_NAME_METADATA_KEY
import litellm.proxy.proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import (
    _apply_streaming_chunk_hooks,
    _fast_serialize_simple_model_response_stream,
    _format_fallback_metadata_sse_event,
    _format_streaming_sse_chunk,
    _get_client_requested_model_for_streaming,
    _get_streaming_fallback_metadata,
    _is_positive_int_like,
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


class _FakeStream:
    def __init__(self, chunks, hidden_params=None):
        self._chunks = chunks
        self._hidden_params = hidden_params or {}

    def __aiter__(self):
        return _async_iter(self._chunks)


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

    monkeypatch.setattr(ps.proxy_logging_obj, "async_post_call_streaming_hook", _boom_hook)
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


@pytest.mark.parametrize("return_raw_model_name", [False, True])
def test_restamp_streaming_chunk_model_respects_raw_model_name_toggle(return_raw_model_name):
    chunk = _simple_chunk(model="gpt-4o-mini")
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="auto_router/complexity_router",
        request_data={"metadata": {RETURN_RAW_MODEL_NAME_METADATA_KEY: return_raw_model_name}},
        model_mismatch_logged=False,
    )

    expected_model = "gpt-4o-mini" if return_raw_model_name else "auto_router/complexity_router"
    assert new_chunk.model == expected_model
    assert logged is (not return_raw_model_name)


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


def test_restamp_streaming_chunk_model_uses_fallback_model_from_metadata():
    chunk = _simple_chunk(model="openai/internal-fallback")
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="primary-model",
        request_data={"litellm_call_id": "id-1"},
        model_mismatch_logged=False,
        fallback_was_attempted=True,
        fallback_model_from_metadata="fallback-model",
    )
    assert new_chunk.model == "fallback-model"
    assert logged is True


def test_restamp_streaming_chunk_model_preserves_fallback_model_without_group():
    chunk = _simple_chunk(model="openai/internal-fallback")
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="primary-model",
        request_data={},
        model_mismatch_logged=False,
        fallback_was_attempted=True,
        fallback_model_from_metadata=None,
    )
    assert new_chunk.model == "openai/internal-fallback"
    assert logged is False


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


def test_is_positive_int_like_invalid_and_edge_values():
    assert _is_positive_int_like(None) is False
    assert _is_positive_int_like("not-a-number") is False
    assert _is_positive_int_like(0) is False
    assert _is_positive_int_like(-1) is False
    assert _is_positive_int_like("1") is True
    assert _is_positive_int_like(2) is True


def test_get_streaming_fallback_metadata_reads_headers():
    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]
    stream = _FakeStream(
        [],
        hidden_params={
            "additional_headers": {
                "x-litellm-attempted-fallbacks": "1",
                "x-litellm-model-group": "fallback-model",
                "x-litellm-fallback-errors": json.dumps(fallback_errors),
            }
        },
    )
    assert _get_streaming_fallback_metadata(stream) == (
        True,
        "fallback-model",
        fallback_errors,
    )


def test_get_streaming_fallback_metadata_no_additional_headers():
    stream = _FakeStream([], hidden_params={})
    assert _get_streaming_fallback_metadata(stream) == (False, None, [])


def test_get_streaming_fallback_metadata_zero_fallback_count():
    stream = _FakeStream(
        [],
        hidden_params={"additional_headers": {"x-litellm-attempted-fallbacks": 0}},
    )
    assert _get_streaming_fallback_metadata(stream) == (False, None, [])


def test_get_streaming_fallback_metadata_no_model_group_returns_none_model():
    stream = _FakeStream(
        [],
        hidden_params={
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1,
            }
        },
    )
    was_attempted, fallback_model, errors = _get_streaming_fallback_metadata(stream)
    assert was_attempted is True
    assert fallback_model is None
    assert errors == []


def test_restamp_streaming_chunk_model_azure_router_preserves_model():
    chunk = _simple_chunk(model="azure_ai/internal-deployment")
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="azure_ai/model-router",
        request_data={},
        model_mismatch_logged=False,
    )
    assert new_chunk.model == "azure_ai/internal-deployment"
    assert logged is False


def test_restamp_streaming_chunk_model_fastest_response_preserves_model():
    chunk = _simple_chunk(model="winning-model")
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="gpt-4,claude-3",
        request_data={"fastest_response": True},
        model_mismatch_logged=False,
    )
    assert new_chunk.model == "winning-model"
    assert logged is False


def test_restamp_streaming_chunk_model_setattr_exception_logs_and_returns():
    from pydantic import ConfigDict

    class FrozenChunk(_simple_chunk().__class__):
        model_config = ConfigDict(frozen=True)

    chunk = FrozenChunk(
        id="chatcmpl-test",
        choices=[],
        created=0,
        model="openai/internal-x",
        object="chat.completion.chunk",
    )
    new_chunk, logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="gpt-4",
        request_data={"litellm_call_id": "test-id"},
        model_mismatch_logged=False,
    )
    assert new_chunk.model == "openai/internal-x"
    assert logged is True


def test_format_fallback_metadata_sse_event():
    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]

    event = _format_fallback_metadata_sse_event(
        fallback_model="fallback-model",
        fallback_errors=fallback_errors,
    )

    assert isinstance(event, str)
    assert event.startswith("data: ")
    payload = json.loads(event.removeprefix("data: ").removesuffix("\n\n"))
    assert payload["choices"] == []
    assert payload["litellm_fallback"] == {
        "fallback_model": "fallback-model",
        "errors": fallback_errors,
    }
    assert payload["id"] == "litellm-fallback-metadata"
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "fallback-model"
    assert isinstance(payload["created"], int)


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

    monkeypatch.setattr(ps.proxy_logging_obj, "async_post_call_streaming_hook", _passthrough)

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
    payload = json.loads(first.removeprefix(b"data: ").removesuffix(b"\n\n"))
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
async def test_async_data_generator_uses_response_fallback_metadata(monkeypatch):
    _patch_logging_flags(monkeypatch)

    response = _FakeStream(
        [_simple_chunk(model="openai/internal-fallback", content="hello")],
        hidden_params={
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1,
                "x-litellm-model-group": "fallback-model",
            }
        },
    )
    out = []
    async for line in async_data_generator(
        response=response,
        user_api_key_dict=_user_auth(),
        request_data={"model": "primary-model", "include_fallback_errors": True},
    ):
        out.append(line)

    first = out[0]
    assert isinstance(first, bytes)
    payload = json.loads(first.removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert payload["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_async_data_generator_uses_chunk_fallback_metadata(monkeypatch):
    _patch_logging_flags(monkeypatch)

    chunk = _simple_chunk(model="openai/internal-fallback", content="hello")
    chunk._hidden_params = {
        "additional_headers": {
            "x-litellm-attempted-fallbacks": 1,
            "x-litellm-model-group": "fallback-model",
        }
    }
    out = []
    async for line in async_data_generator(
        response=_async_iter([chunk]),
        user_api_key_dict=_user_auth(),
        request_data={"model": "primary-model"},
    ):
        out.append(line)

    first = out[0]
    assert isinstance(first, bytes)
    payload = json.loads(first.removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert payload["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_async_data_generator_switches_model_mid_stream_on_fallback(monkeypatch):
    """Pre-fallback chunks keep the client-requested model; once a chunk carries
    fallback metadata the model latches to the fallback group for the rest of the
    stream. This pins the client-visible mid-stream model change."""
    _patch_logging_flags(monkeypatch)

    primary_chunk = _simple_chunk(model="openai/internal-primary", content="hi")
    fallback_chunk = _simple_chunk(model="openai/internal-fallback", content="there")
    fallback_chunk._hidden_params = {
        "additional_headers": {
            "x-litellm-attempted-fallbacks": 1,
            "x-litellm-model-group": "fallback-model",
        }
    }
    out = []
    async for line in async_data_generator(
        response=_async_iter([primary_chunk, fallback_chunk]),
        user_api_key_dict=_user_auth(),
        request_data={"model": "primary-model"},
    ):
        out.append(line)

    first_payload = json.loads(out[0].removeprefix(b"data: ").removesuffix(b"\n\n"))
    second_payload = json.loads(out[1].removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert first_payload["model"] == "primary-model"
    assert second_payload["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_async_data_generator_emits_fallback_error_metadata_event(monkeypatch):
    _patch_logging_flags(monkeypatch)
    monkeypatch.setitem(ps.general_settings, "expose_fallback_errors_to_caller", True)

    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]
    response = _FakeStream(
        [_simple_chunk(model="openai/internal-fallback", content="hello")],
        hidden_params={
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1,
                "x-litellm-model-group": "fallback-model",
                "x-litellm-fallback-errors": json.dumps(fallback_errors),
            }
        },
    )
    out = []
    async for line in async_data_generator(
        response=response,
        user_api_key_dict=_user_auth(),
        request_data={"model": "primary-model", "include_fallback_errors": True},
    ):
        out.append(line)

    assert isinstance(out[0], bytes)
    chunk_payload = json.loads(out[0].removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert chunk_payload["model"] == "fallback-model"
    assert isinstance(out[1], str)
    assert out[1].startswith("data: ")
    metadata_payload = json.loads(out[1].removeprefix("data: ").removesuffix("\n\n"))
    assert metadata_payload["choices"] == []
    assert metadata_payload["litellm_fallback"] == {
        "fallback_model": "fallback-model",
        "errors": fallback_errors,
    }
    assert metadata_payload["id"] == "litellm-fallback-metadata"
    assert metadata_payload["object"] == "chat.completion.chunk"
    assert metadata_payload["model"] == "fallback-model"
    assert isinstance(metadata_payload["created"], int)


@pytest.mark.asyncio
async def test_async_data_generator_skips_fallback_error_event_without_opt_in(
    monkeypatch,
):
    _patch_logging_flags(monkeypatch)

    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]
    response = _FakeStream(
        [_simple_chunk(model="openai/internal-fallback", content="hello")],
        hidden_params={
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1,
                "x-litellm-model-group": "fallback-model",
                "x-litellm-fallback-errors": json.dumps(fallback_errors),
            }
        },
    )
    out = []
    async for line in async_data_generator(
        response=response,
        user_api_key_dict=_user_auth(),
        request_data={"model": "primary-model"},
    ):
        out.append(line)

    assert isinstance(out[0], bytes)
    payload = json.loads(out[0].removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert payload["model"] == "fallback-model"


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
    assert any(isinstance(item, str) and item.startswith('data: {"error":') for item in out)


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


# ---------------------------------------------------------------------------
# SSE keepalive helpers
# ---------------------------------------------------------------------------


from litellm.proxy.proxy_server import (  # noqa: E402
    _iter_with_keepalive,
    _keepalive_from_deployment_config,
    _make_keepalive_resolver,
    _resolve_keepalive_seconds,
)
from litellm.proxy.proxy_server import _KEEPALIVE_MAX_SECONDS, _KEEPALIVE_MIN_SECONDS  # noqa: E402


@pytest.mark.asyncio
async def test_iter_with_keepalive_hot_path_no_task_wrapping():
    """When keepalive_seconds <= 0, the generator is a transparent pass-through."""
    chunks = [_simple_chunk(content="a"), _simple_chunk(content="b")]
    out = []
    async for item in _iter_with_keepalive(_async_iter(chunks), lambda _: 0, keepalive_seconds=0):
        out.append(item)

    assert out == chunks
    assert ps._STREAM_KEEPALIVE not in out


@pytest.mark.asyncio
async def test_iter_with_keepalive_emits_sentinel_when_stream_stalls():
    """With a short keepalive interval and a stalled upstream, _STREAM_KEEPALIVE
    sentinels appear before the delayed chunk arrives. The resolver returns a
    constant interval, since this test pins the timing mechanics, not
    re-resolution."""
    import asyncio

    async def _slow_stream():
        yield _simple_chunk(content="first")
        await asyncio.sleep(0.3)
        yield _simple_chunk(content="second")

    items = []
    async for item in _iter_with_keepalive(_slow_stream(), lambda _: 0.05, keepalive_seconds=0.05):
        items.append(item)

    sentinels = [i for i in items if i is ps._STREAM_KEEPALIVE]
    real_chunks = [i for i in items if i is not ps._STREAM_KEEPALIVE]

    assert len(sentinels) >= 2, f"expected >= 2 sentinels during 0.3s stall; got {len(sentinels)}"
    assert len(real_chunks) == 2
    assert real_chunks[0].choices[0].delta.content == "first"
    assert real_chunks[1].choices[0].delta.content == "second"


@pytest.mark.asyncio
async def test_iter_with_keepalive_cancel_on_early_close():
    """Closing the generator early cancels the in-flight task without raising."""
    import asyncio

    async def _infinite_stream():
        while True:
            await asyncio.sleep(10)
            yield _simple_chunk()

    gen = _iter_with_keepalive(_infinite_stream(), lambda _: 0.05, keepalive_seconds=0.05)
    # Advance once to get the sentinel; then close before the real chunk.
    first = await gen.__anext__()
    assert first is ps._STREAM_KEEPALIVE
    # aclose must not raise, and must drain the cancelled task cleanly.
    await gen.aclose()


@pytest.mark.asyncio
async def test_iter_with_keepalive_disables_after_fallback_lowers_interval():
    """Greptile P1: a mid-stream router fallback can hand off to a deployment
    with a different (or disabled) keepalive policy partway through the same
    stream. The interval must be re-resolved against each chunk's own identity,
    not the one picked before iteration started, or heartbeats keep using the
    pre-fallback deployment's policy for the rest of the stream."""
    import asyncio

    async def _slow_stream():
        yield _simple_chunk(content="first")
        await asyncio.sleep(0.3)
        yield _simple_chunk(content="second")

    def _resolver(item):
        # First chunk resolves under the enabled interval used to start the
        # wrapper; every chunk after that resolves as if a fallback disabled it.
        return 0.0 if item.choices[0].delta.content == "first" else 999.0

    items = []
    async for item in _iter_with_keepalive(_slow_stream(), _resolver, keepalive_seconds=0.05):
        items.append(item)

    sentinels = [i for i in items if i is ps._STREAM_KEEPALIVE]
    real_chunks = [i for i in items if i is not ps._STREAM_KEEPALIVE]

    assert sentinels == [], f"expected no sentinels once the resolver disables keepalive; got {len(sentinels)}"
    assert len(real_chunks) == 2
    assert real_chunks[0].choices[0].delta.content == "first"
    assert real_chunks[1].choices[0].delta.content == "second"


@pytest.mark.asyncio
async def test_iter_with_keepalive_enables_after_fallback_raises_interval():
    """Symmetric case: a mid-stream fallback to a deployment with a *shorter*
    keepalive interval must take effect immediately, not stay pinned to the
    longer interval the stream started with. The interval used to wait for a
    chunk is resolved from the *previous* chunk (the only one seen so far when
    that wait begins), so the stall has to follow the fallback chunk rather
    than precede it: waiting for "third" is where the shorter interval bites."""
    import asyncio

    async def _slow_stream():
        yield _simple_chunk(content="first")
        yield _simple_chunk(content="second")
        await asyncio.sleep(0.3)
        yield _simple_chunk(content="third")

    def _resolver(item):
        # "first" resolves under an interval too long to fire before "second"
        # arrives; "second" (the fallback chunk) resolves as if the fallback
        # deployment enabled a much shorter interval for everything after it.
        return 999.0 if item.choices[0].delta.content == "first" else 0.05

    items = []
    async for item in _iter_with_keepalive(_slow_stream(), _resolver, keepalive_seconds=999.0):
        items.append(item)

    sentinels = [i for i in items if i is ps._STREAM_KEEPALIVE]
    real_chunks = [i for i in items if i is not ps._STREAM_KEEPALIVE]

    assert len(sentinels) >= 2, (
        f"expected >= 2 sentinels once the resolver enables a short interval; got {len(sentinels)}"
    )
    assert len(real_chunks) == 3


@pytest.mark.asyncio
async def test_iter_with_keepalive_activates_from_a_fully_disabled_start():
    """Greptile P1: a stream can start on a deployment with keepalive off
    (keepalive_seconds passed in as 0, not merely a long interval) and fall back
    mid-stream to one that enables it. The 0-second start must not be treated as
    a one-time decision to skip heartbeats for the rest of the stream: no task
    is created while inactive, but every chunk still re-resolves so the fallback
    chunk can switch the stream into task-wrapped mode."""
    import asyncio

    async def _slow_stream():
        yield _simple_chunk(content="first")
        yield _simple_chunk(content="second")
        await asyncio.sleep(0.3)
        yield _simple_chunk(content="third")

    def _resolver(item):
        # "first" resolves to stay off; "second" (the fallback chunk) resolves
        # as if the fallback deployment newly enabled a short interval.
        return 0.0 if item.choices[0].delta.content == "first" else 0.05

    items = []
    async for item in _iter_with_keepalive(_slow_stream(), _resolver, keepalive_seconds=0):
        items.append(item)

    sentinels = [i for i in items if i is ps._STREAM_KEEPALIVE]
    real_chunks = [i for i in items if i is not ps._STREAM_KEEPALIVE]

    assert len(sentinels) >= 2, (
        f"expected >= 2 sentinels once the resolver activates from a disabled start; got {len(sentinels)}"
    )
    assert len(real_chunks) == 3


def test_resolve_keepalive_seconds_client_value_ignored_without_override_permission(monkeypatch):
    """keepalive_seconds is operator-only by default: a deployment that hasn't set
    allow_client_keepalive_override must not let a client's request-level value
    change its behavior at all, since that would let any authenticated client
    unilaterally enable heartbeats (and the LB-idle-timeout evasion that comes
    with them) for a deployment that never opted in."""
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 15.0
    deployment.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-locked"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": 1}, response=response)
    assert result == 15.0


def test_resolve_keepalive_seconds_request_value_wins_when_override_allowed(monkeypatch):
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = None
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-opt-in"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": 30}, response=response)
    assert result == 30.0


def test_resolve_keepalive_seconds_explicit_zero_disables_when_override_allowed(monkeypatch):
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 20.0
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-opt-in"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": 0}, response=response)
    assert result == 0.0


def test_resolve_keepalive_seconds_clamps_below_minimum(monkeypatch):
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = None
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-opt-in"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": 0.001}, response=response)
    assert result == _KEEPALIVE_MIN_SECONDS


def test_resolve_keepalive_seconds_clamps_above_maximum(monkeypatch):
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = None
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-opt-in"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": 9999}, response=response)
    assert result == _KEEPALIVE_MAX_SECONDS


def test_resolve_keepalive_seconds_non_numeric_returns_zero(monkeypatch):
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = None
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-opt-in"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": "not-a-number"}, response=response)
    assert result == 0.0


def test_resolve_keepalive_seconds_absent_returns_zero(monkeypatch):
    monkeypatch.setattr(ps, "llm_router", None)
    result = _resolve_keepalive_seconds({}, response=None)
    assert result == 0.0


def test_resolve_keepalive_seconds_deployment_disable_cannot_be_overridden_by_request(monkeypatch):
    """A deployment that explicitly sets keepalive_seconds: 0 is a hard operator
    disable: an authenticated client must not be able to re-enable heartbeats for
    that deployment by passing a positive value in the request body, since that
    would let a client evade the deployment's idle-timeout behavior at will. This
    holds even if the deployment also grants override permission, since an
    explicit disable is a stronger, unconditional signal than an override grant."""
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 0
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-disabled"}

    result = _resolve_keepalive_seconds({"model": "my-model", "keepalive_seconds": 250}, response=response)
    assert result == 0.0


def test_resolve_keepalive_seconds_global_default_applies_when_unconfigured(monkeypatch):
    """litellm_settings.sse_keepalive_ping_interval_seconds is the operator's
    global default: it applies when neither the serving deployment nor the
    request supplies keepalive_seconds, including proxies with no router at
    all."""
    import litellm

    monkeypatch.setattr(ps, "llm_router", None)
    monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", 15.0)

    result = _resolve_keepalive_seconds({}, response=None)
    assert result == 15.0


def test_resolve_keepalive_seconds_global_default_is_clamped(monkeypatch):
    import litellm

    monkeypatch.setattr(ps, "llm_router", None)
    monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", 900.0)

    result = _resolve_keepalive_seconds({}, response=None)
    assert result == _KEEPALIVE_MAX_SECONDS


def test_resolve_keepalive_seconds_deployment_zero_beats_global_default(monkeypatch):
    """A deployment's explicit keepalive_seconds: 0 is a hard operator disable
    that must also win over the global default interval, or the global setting
    would silently re-enable heartbeats (and the LB-idle-timeout evasion that
    comes with them) for a deployment the operator opted out of."""
    from unittest.mock import MagicMock

    import litellm

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 0
    deployment.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", 15.0)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-disabled"}

    result = _resolve_keepalive_seconds({"model": "my-model"}, response=response)
    assert result == 0.0


def test_resolve_keepalive_seconds_deployment_value_beats_global_default(monkeypatch):
    from unittest.mock import MagicMock

    import litellm

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 30.0
    deployment.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", 15.0)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-tuned"}

    result = _resolve_keepalive_seconds({"model": "my-model"}, response=response)
    assert result == 30.0


def test_keepalive_from_deployment_config_reads_by_model_id(monkeypatch):
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 45.0
    deployment.litellm_params.allow_client_keepalive_override = True

    router = MagicMock()
    router.get_deployment.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "deploy-abc"}

    result = _keepalive_from_deployment_config({"model": "my-model"}, response)
    assert result == ps._DeploymentKeepaliveConfig(keepalive_seconds=45.0, allow_client_override=True)
    router.get_deployment.assert_called_once_with(model_id="deploy-abc")


def test_keepalive_from_deployment_config_stale_model_id_does_not_fall_through(monkeypatch):
    """A populated model_id names the specific deployment that served the stream.
    If that ID no longer resolves (e.g. removed by a config reload mid-stream),
    that's a stale identity, not an absent one: it must not fall through to the
    model_name fallback, since a currently-live sibling deployment's config was
    never what actually served this stream, even if that sibling's config is
    unambiguous on its own."""
    from unittest.mock import MagicMock

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 20.0}},
    ]

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {"model_id": "stale-deploy-id"}

    result = _keepalive_from_deployment_config({"model": "slow-model"}, response)
    assert result is None
    router.get_model_list.assert_not_called()


def test_keepalive_from_deployment_config_fallback_by_name(monkeypatch):
    from unittest.mock import MagicMock

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 20.0}},
    ]

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {}

    result = _keepalive_from_deployment_config({"model": "slow-model"}, response)
    assert result == ps._DeploymentKeepaliveConfig(keepalive_seconds=20.0, allow_client_override=False)
    router.get_model_list.assert_called_once_with(model_name="slow-model")


def test_keepalive_from_deployment_config_fallback_by_name_agreeing_deployments(monkeypatch):
    """Multiple deployments under the same model_name with the same keepalive_seconds
    is unambiguous, so the shared value is used even without a model_id."""
    from unittest.mock import MagicMock

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 20.0, "allow_client_keepalive_override": True}},
        {"litellm_params": {"keepalive_seconds": 20.0, "allow_client_keepalive_override": True}},
    ]

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {}

    result = _keepalive_from_deployment_config({"model": "slow-model"}, response)
    assert result == ps._DeploymentKeepaliveConfig(keepalive_seconds=20.0, allow_client_override=True)


def test_keepalive_from_deployment_config_fallback_by_name_conflicting_deployments(monkeypatch):
    """Without a model_id, if deployments under the same model_name disagree on
    keepalive_seconds, we can't tell which one served the stream: don't guess and
    apply the wrong deployment's interval (or override an explicit disable)."""
    from unittest.mock import MagicMock

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 20.0}},
        {"litellm_params": {"keepalive_seconds": 0}},
    ]

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {}

    result = _keepalive_from_deployment_config({"model": "slow-model"}, response)
    assert result is None


def test_keepalive_from_deployment_config_fallback_by_name_configured_plus_unset(monkeypatch):
    """A deployment that leaves keepalive_seconds unset entirely (not explicitly 0)
    must not inherit a sibling deployment's configured interval: without a model_id
    we can't tell which deployment served the stream, so mixing a configured
    deployment with an unconfigured one is just as ambiguous as two conflicting
    configured values."""
    from unittest.mock import MagicMock

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 20.0}},
        {"litellm_params": {}},
    ]

    monkeypatch.setattr(ps, "llm_router", router)

    response = MagicMock()
    response._hidden_params = {}

    result = _keepalive_from_deployment_config({"model": "slow-model"}, response)
    assert result is None


def test_keepalive_from_deployment_config_no_router_returns_none(monkeypatch):
    monkeypatch.setattr(ps, "llm_router", None)
    result = _keepalive_from_deployment_config({"model": "gpt-4"}, None)
    assert result is None


def test_make_keepalive_resolver_caches_by_model_id(monkeypatch):
    """The steady-state case (no fallback): every chunk shares the same
    model_id, so the deployment lookup must happen once, not once per chunk."""
    from unittest.mock import MagicMock

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = 5.0
    deployment.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.return_value = deployment
    monkeypatch.setattr(ps, "llm_router", router)

    resolve = _make_keepalive_resolver({"model": "my-model"})

    first = _simple_chunk(content="a")
    first._hidden_params = {"model_id": "deploy-steady"}
    second = _simple_chunk(content="b")
    second._hidden_params = {"model_id": "deploy-steady"}

    assert resolve(first) == 5.0
    assert resolve(second) == 5.0
    router.get_deployment.assert_called_once_with(model_id="deploy-steady")


def test_make_keepalive_resolver_reresolves_on_model_id_change(monkeypatch):
    """A mid-stream fallback changes model_id: the cache must miss and
    re-resolve against the new deployment, not keep serving the stale value."""
    from unittest.mock import MagicMock

    before = MagicMock()
    before.litellm_params.keepalive_seconds = 5.0
    before.litellm_params.allow_client_keepalive_override = False

    after = MagicMock()
    after.litellm_params.keepalive_seconds = 30.0
    after.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.side_effect = lambda model_id: {"deploy-a": before, "deploy-b": after}[model_id]
    monkeypatch.setattr(ps, "llm_router", router)

    resolve = _make_keepalive_resolver({"model": "my-model"})

    chunk_a = _simple_chunk(content="a")
    chunk_a._hidden_params = {"model_id": "deploy-a"}
    chunk_b = _simple_chunk(content="b")
    chunk_b._hidden_params = {"model_id": "deploy-b"}

    assert resolve(chunk_a) == 5.0
    assert resolve(chunk_b) == 30.0
    assert router.get_deployment.call_count == 2


def test_make_keepalive_resolver_missing_model_id_never_cached(monkeypatch):
    """Without a model_id there's no reliable cache key (see the model_name
    fallback in _keepalive_from_deployment_config), so every chunk must
    re-resolve fresh rather than reuse a stale guess."""
    from unittest.mock import MagicMock

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [{"litellm_params": {"keepalive_seconds": 12.0}}]
    monkeypatch.setattr(ps, "llm_router", router)

    resolve = _make_keepalive_resolver({"model": "slow-model"})

    chunk_a = _simple_chunk(content="a")
    chunk_a._hidden_params = {}
    chunk_b = _simple_chunk(content="b")
    chunk_b._hidden_params = {}

    assert resolve(chunk_a) == 12.0
    assert resolve(chunk_b) == 12.0
    assert router.get_model_list.call_count == 2


def test_make_keepalive_resolver_expires_cache_after_ttl(monkeypatch):
    """An operator's live config change (revoking override, disabling
    keepalive, removing the deployment) must be observed within
    _KEEPALIVE_CACHE_TTL_SECONDS, not frozen for the rest of an
    already-in-flight stream just because the model_id hasn't changed."""
    from unittest.mock import MagicMock

    before = MagicMock()
    before.litellm_params.keepalive_seconds = 20.0
    before.litellm_params.allow_client_keepalive_override = False

    after = MagicMock()
    after.litellm_params.keepalive_seconds = 0
    after.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.return_value = before
    monkeypatch.setattr(ps, "llm_router", router)

    clock = {"t": 0.0}
    monkeypatch.setattr(ps.time, "monotonic", lambda: clock["t"])

    resolve = _make_keepalive_resolver({"model": "my-model"})

    chunk = _simple_chunk(content="a")
    chunk._hidden_params = {"model_id": "deploy-live"}

    assert resolve(chunk) == 20.0
    assert router.get_deployment.call_count == 1

    # Still within the TTL: same model_id, cached value reused even though
    # the router's live config has since changed underneath it.
    router.get_deployment.return_value = after
    clock["t"] = ps._KEEPALIVE_CACHE_TTL_SECONDS - 0.01
    assert resolve(chunk) == 20.0
    assert router.get_deployment.call_count == 1

    # Past the TTL: the config-reload disable is now observed.
    clock["t"] = ps._KEEPALIVE_CACHE_TTL_SECONDS + 0.01
    assert resolve(chunk) == 0.0
    assert router.get_deployment.call_count == 2


def test_keepalive_seconds_in_all_litellm_params():
    from litellm.types.utils import all_litellm_params

    assert "keepalive_seconds" in all_litellm_params


def test_allow_client_keepalive_override_in_all_litellm_params():
    """allow_client_keepalive_override is a deployment-only control flag: if it's
    missing from all_litellm_params, it leaks straight through into the actual
    provider API call as an unrecognized field and gets rejected (confirmed live
    against the real Anthropic API, which returns 'Extra inputs are not
    permitted')."""
    from litellm.types.utils import all_litellm_params

    assert "allow_client_keepalive_override" in all_litellm_params


@pytest.mark.asyncio
async def test_async_data_generator_emits_ping_heartbeat(monkeypatch):
    """When keepalive_seconds is set on a deployment that allows client override,
    ': ping' frames appear during upstream stalls."""
    import asyncio
    from unittest.mock import MagicMock

    _patch_logging_flags(monkeypatch)
    monkeypatch.setattr(ps, "_KEEPALIVE_MIN_SECONDS", 0.05)

    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [{"litellm_params": {"allow_client_keepalive_override": True}}]
    monkeypatch.setattr(ps, "llm_router", router)

    async def _slow_response():
        yield _simple_chunk(content="hello")
        await asyncio.sleep(0.4)
        yield _simple_chunk(content="world")

    out = []
    async for line in async_data_generator(
        response=_slow_response(),
        user_api_key_dict=_user_auth(),
        request_data={"model": "gpt-4", "keepalive_seconds": 0.05},
    ):
        out.append(line)

    pings = [item for item in out if item == ": ping\n\n"]
    assert len(pings) >= 2, f"expected >= 2 ping frames; got {len(pings)}"
    assert out[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_async_data_generator_emits_ping_heartbeat_from_global_default_without_router(monkeypatch):
    """The global sse_keepalive_ping_interval_seconds must produce ': ping'
    frames even on a proxy with no router, where the wrap was previously
    skipped entirely because no deployment could ever resolve a non-zero
    interval."""
    import asyncio

    import litellm

    _patch_logging_flags(monkeypatch)
    monkeypatch.setattr(ps, "_KEEPALIVE_MIN_SECONDS", 0.05)
    monkeypatch.setattr(ps, "llm_router", None)
    monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", 0.05)

    async def _slow_response():
        yield _simple_chunk(content="hello")
        await asyncio.sleep(0.4)
        yield _simple_chunk(content="world")

    out = []
    async for line in async_data_generator(
        response=_slow_response(),
        user_api_key_dict=_user_auth(),
        request_data={"model": "gpt-4"},
    ):
        out.append(line)

    pings = [item for item in out if item == ": ping\n\n"]
    assert len(pings) >= 2, f"expected >= 2 ping frames; got {len(pings)}"
    assert out[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_async_data_generator_no_keepalive_no_pings(monkeypatch):
    """Without keepalive_seconds, no ': ping' frames are emitted."""
    _patch_logging_flags(monkeypatch)

    out = []
    async for line in async_data_generator(
        response=_async_iter([_simple_chunk(content="hello")]),
        user_api_key_dict=_user_auth(),
        request_data={"model": "gpt-4"},
    ):
        out.append(line)

    assert ": ping\n\n" not in out
    assert out[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_async_data_generator_resolves_deployment_once_per_steady_stream(monkeypatch):
    """Regression test for the per-chunk resolver cost: a stream where every
    real chunk comes from the same deployment (the common, no-fallback case)
    must only pay for one `llm_router.get_deployment()` call, not one per
    chunk. Before caching, this asserted 1 but got len(chunks) since the
    resolver re-ran the full deployment lookup after every single chunk.

    The very first resolve happens on the raw `response` object before any
    chunk is yielded; a bare async generator (unlike the real
    CustomStreamWrapper this stands in for) can't carry `_hidden_params`, so
    that one call goes through the model_name fallback instead of
    `get_deployment` — hence it's asserted separately.
    """
    from unittest.mock import MagicMock

    _patch_logging_flags(monkeypatch)

    deployment = MagicMock()
    deployment.litellm_params.keepalive_seconds = None
    deployment.litellm_params.allow_client_keepalive_override = False

    router = MagicMock()
    router.get_deployment.return_value = deployment
    router.get_model_list.return_value = [{"litellm_params": {}}]
    monkeypatch.setattr(ps, "llm_router", router)

    async def _steady_response():
        for content in ("a", "b", "c", "d", "e"):
            chunk = _simple_chunk(content=content)
            chunk._hidden_params = {"model_id": "deploy-steady"}
            yield chunk

    out = []
    async for line in async_data_generator(
        response=_steady_response(),
        user_api_key_dict=_user_auth(),
        request_data={"model": "gpt-4"},
    ):
        out.append(line)

    assert router.get_deployment.call_count == 1
    assert router.get_model_list.call_count == 1
    assert out[-1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# run_thread: SSE keepalives during the time-to-first-token
# ---------------------------------------------------------------------------


class _SlowAssistantsStream(_FakeAssistantsStream):
    """The assistants run only contacts the upstream when the stream is entered,
    and `create_response` buffers that first chunk, so the whole
    time-to-first-token is spent before a byte can be written."""

    def __init__(self, chunks, delay):
        super().__init__(chunks)
        self._delay = delay

    async def __aenter__(self):
        await asyncio.sleep(self._delay)
        return self


async def _run_thread_streaming(monkeypatch, interval, delay=0.3, fails_with=None):
    monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", interval)

    router = MagicMock()
    router.get_model_list.return_value = []
    if fails_with is None:
        router.arun_thread = AsyncMock(return_value=_SlowAssistantsStream([_simple_chunk(content="hi")], delay))
    else:

        async def _fails_after_the_first_ping(**kwargs):
            await asyncio.sleep(delay)
            raise fails_with

        router.arun_thread = _fails_after_the_first_ping
    monkeypatch.setattr(ps, "llm_router", router)

    async def _passthrough_hook(*, user_api_key_dict, response, data, **kwargs):
        return response

    monkeypatch.setattr(ps.proxy_logging_obj, "async_post_call_streaming_hook", _passthrough_hook)

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(ps, "add_litellm_data_to_request", _add_data)

    request = MagicMock()
    request.body = AsyncMock(return_value=b'{"assistant_id": "asst_1", "stream": true}')
    request.is_disconnected = AsyncMock(return_value=False)

    return await ps.run_thread(
        request=request,
        thread_id="thr_1",
        fastapi_response=Response(),
        user_api_key_dict=_user_auth(),
    )


@pytest.mark.asyncio
async def test_run_thread_pings_while_the_assistants_run_is_still_silent(monkeypatch):
    """Regression for LIT-5737. A streaming assistants run wrote zero bytes for the
    whole time-to-first-token, so an idle-timeout hop drops a healthy connection."""
    response = await _run_thread_streaming(monkeypatch, interval=0.05)

    assert isinstance(response, StreamingResponse)
    assert response.headers["x-accel-buffering"] == "no"
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks[0] == b": ping\n\n"
    assert chunks.count(b": ping\n\n") >= 3
    assert chunks[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_run_thread_audits_a_failure_that_arrives_after_the_first_ping(monkeypatch):
    """Once a ping is on the wire the run can no longer raise, so the handler's own
    `except` never runs. The failure still has to reach post_call_failure_hook or it
    goes unaudited, and it has to reach the client as an SSE frame."""
    audited = []

    async def _record_failure(*, user_api_key_dict, original_exception, request_data, **kwargs):
        audited.append(original_exception)
        return None

    monkeypatch.setattr(ps.proxy_logging_obj, "post_call_failure_hook", _record_failure)

    boom = RuntimeError("upstream died after the wire was already open")
    response = await _run_thread_streaming(monkeypatch, interval=0.05, fails_with=boom)

    assert isinstance(response, StreamingResponse)
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks[0] == b": ping\n\n"
    # The hook is the only thing that still sees the real exception; the client
    # gets the sanitized frame, under the 200 the ping already committed.
    assert audited == [boom]
    assert b"upstream died after the wire was already open" not in chunks[-2]
    assert json.loads(chunks[-2].removeprefix(b"data: "))["error"]["code"] == "500"
    assert chunks[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_run_thread_stream_is_untouched_while_keepalives_are_unconfigured(monkeypatch):
    """Off until an operator sets an interval, so the default run is unchanged."""
    response = await _run_thread_streaming(monkeypatch, interval=None, delay=0.15)

    assert isinstance(response, StreamingResponse)
    chunks = [chunk async for chunk in response.body_iterator]

    assert not any(chunk.startswith(": ping") for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
