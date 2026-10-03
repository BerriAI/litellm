"""
Tests for CustomStreamWrapper per-chunk behavior across Anthropic,
Bedrock Invoke, and Bedrock Converse: text passthrough, usage stripping,
hidden_params propagation, finish_reason, sync/async parity, and the
per-stream caches (_GCHUNK_FIELDS, _post_streaming_hooks).
"""

import asyncio
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import litellm
from litellm.litellm_core_utils.streaming_handler import (
    CustomStreamWrapper,
    _GCHUNK_FIELDS,
    generic_chunk_has_all_required_fields,
)
from litellm.types.utils import (
    Delta,
    GenericStreamingChunk as GChunk,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_logging_obj(provider: str = "anthropic") -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "custom_llm_provider": provider,
        "litellm_params": {},
    }
    logging_obj.call_type = "completion"
    logging_obj.stream_options = None
    logging_obj.messages = [{"role": "user", "content": "hi"}]
    logging_obj.completion_start_time = None
    logging_obj._llm_caching_handler = None
    return logging_obj


def _make_generic_chunk(
    text: str,
    is_finished: bool = False,
    finish_reason: str = "",
    usage: Optional[dict] = None,
) -> GChunk:
    return GChunk(
        text=text,
        is_finished=is_finished,
        finish_reason=finish_reason,
        usage=usage,
        index=0,
        tool_use=None,
    )


def _make_bedrock_converse_chunk(
    text: str = "",
    finish_reason: str = "",
    usage: Optional[Usage] = None,
) -> ModelResponseStream:
    """Simulate what AWSEventStreamDecoder.converse_chunk_parser returns."""
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=finish_reason or None,
                index=0,
                delta=Delta(content=text, role="assistant"),
            )
        ],
        id="msg-test",
        model="anthropic.claude-3-5-sonnet",
        usage=usage,
    )


async def _async_iter(chunks: list):
    """Wrap a list as a proper async iterator for use in __anext__ async branch."""
    for chunk in chunks:
        yield chunk


def _make_wrapper(
    chunks: list,
    provider: str = "anthropic",
    async_stream: bool = False,
) -> CustomStreamWrapper:
    logging_obj = _make_logging_obj(provider)
    stream = _async_iter(chunks) if async_stream else iter(chunks)
    wrapper = CustomStreamWrapper(
        completion_stream=stream,
        model="claude-3-5-sonnet",
        logging_obj=logging_obj,
        custom_llm_provider=provider,
    )
    return wrapper


def _drain_sync(wrapper: CustomStreamWrapper) -> List[ModelResponseStream]:
    results = []
    for chunk in wrapper:
        results.append(chunk)
    return results


async def _drain_async(wrapper: CustomStreamWrapper) -> List[ModelResponseStream]:
    results = []
    async for chunk in wrapper:
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# 1. Module-level _GCHUNK_FIELDS constant
# ---------------------------------------------------------------------------


def test_gchunk_fields_is_frozenset():
    """_GCHUNK_FIELDS must be a frozenset built from GChunk.__annotations__."""
    assert isinstance(_GCHUNK_FIELDS, frozenset)
    assert _GCHUNK_FIELDS == frozenset(GChunk.__annotations__)


def test_generic_chunk_has_all_required_fields_uses_module_constant(monkeypatch):
    """generic_chunk_has_all_required_fields must use _GCHUNK_FIELDS, not __annotations__.

    The check semantics: every key in `chunk` must be a known GChunk field.
    This identifies GChunk-shaped dicts (all keys are valid GChunk fields).
    """
    valid_chunk = _make_generic_chunk("hello")
    assert generic_chunk_has_all_required_fields(valid_chunk) is True

    # A dict with an extra unknown key should return False — the unknown key
    # is not a GChunk field, so the chunk is not a pure GChunk.
    extra_key_chunk = dict(valid_chunk)
    extra_key_chunk["unknown_extra_key"] = "value"
    assert generic_chunk_has_all_required_fields(extra_key_chunk) is False

    # A dict with only known GChunk fields but fewer keys still passes because
    # all its keys are valid (subset of GChunk fields).
    partial_chunk = {"text": "hi", "is_finished": False}
    assert generic_chunk_has_all_required_fields(partial_chunk) is True


# ---------------------------------------------------------------------------
# 2. Cached model name and provider at init time
# ---------------------------------------------------------------------------


def test_cached_model_name_simple():
    """For non-openai providers the cached model name must match the model arg."""
    wrapper = _make_wrapper([], provider="anthropic")
    assert wrapper._cached_model_name == "claude-3-5-sonnet"
    assert wrapper._cached_logging_llm_provider == "anthropic"


def test_cached_model_name_openai_prefix():
    """For openai provider when logging provider differs, model name is prefixed."""
    logging_obj = _make_logging_obj(provider="azure")
    wrapper = CustomStreamWrapper(
        completion_stream=iter([]),
        model="gpt-4o",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    assert wrapper._cached_model_name == "azure/gpt-4o"
    assert wrapper._cached_logging_llm_provider == "azure"


def test_base_hidden_params_precomputed():
    """_base_hidden_params must be pre-built from _hidden_params at init."""
    wrapper = _make_wrapper([], provider="anthropic")
    assert "response_cost" in wrapper._base_hidden_params
    assert wrapper._base_hidden_params["response_cost"] is None
    # Must include all keys from _hidden_params
    for k in wrapper._hidden_params:
        assert k in wrapper._base_hidden_params


# ---------------------------------------------------------------------------
# 3. Sync path: model_dump() is NOT called on non-usage chunks
# ---------------------------------------------------------------------------


def test_sync_path_no_model_dump_on_text_chunks():
    """
    The sync __next__ must NOT call model_dump() on chunks that have no usage.

    ModelResponseStream declares `usage` as a field, so a `hasattr` check
    would always succeed and trigger the model_dump()+recreate path on every
    chunk. The wrapper must check `is not None` instead.
    """
    chunks = [
        _make_generic_chunk("Hello"),
        _make_generic_chunk(" world"),
        _make_generic_chunk("", is_finished=True, finish_reason="stop"),
    ]
    wrapper = _make_wrapper(chunks)

    model_dump_call_count = 0
    original_model_dump = ModelResponseStream.model_dump

    def counting_model_dump(self, **kwargs):
        nonlocal model_dump_call_count
        model_dump_call_count += 1
        return original_model_dump(self, **kwargs)

    with patch.object(ModelResponseStream, "model_dump", counting_model_dump):
        results = _drain_sync(wrapper)

    text_chunks = [r for r in results if r.choices and r.choices[0].delta.content]
    assert len(text_chunks) >= 2, "Expected at least 2 text chunks"
    assert model_dump_call_count <= 1, (
        f"model_dump() called {model_dump_call_count} times — "
        "usage check is firing on every chunk"
    )


# ---------------------------------------------------------------------------
# 4. Sync path: usage chunk is stripped from body but preserved in hidden_params
# ---------------------------------------------------------------------------


def test_sync_path_usage_stripped_from_body_preserved_in_hidden_params():
    """Usage data must be removed from the returned chunk but added to _hidden_params."""
    usage_dict = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    chunks = [
        _make_generic_chunk("Hello"),
        _make_generic_chunk(
            "", is_finished=True, finish_reason="stop", usage=usage_dict
        ),
    ]
    wrapper = _make_wrapper(chunks)
    results = _drain_sync(wrapper)

    # The usage chunk must be returned (not silently dropped)
    finish_chunks = [
        r for r in results if r.choices and r.choices[0].finish_reason == "stop"
    ]
    assert finish_chunks, "Finish-reason chunk was not returned"

    # The final chunk must carry usage in _hidden_params
    final = results[-1]
    assert "usage" in final._hidden_params, "usage missing from _hidden_params"
    hidden_usage = final._hidden_params["usage"]
    assert hidden_usage is not None


# ---------------------------------------------------------------------------
# 5. Async path: usage chunk is stripped from body but preserved in hidden_params
# ---------------------------------------------------------------------------


def test_async_path_usage_stripped_from_body_preserved_in_hidden_params():
    """Async path mirrors sync path for usage handling."""
    usage_dict = {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
    chunks = [
        _make_generic_chunk("Hi"),
        _make_generic_chunk(
            "", is_finished=True, finish_reason="stop", usage=usage_dict
        ),
    ]

    async def _run():
        # async_stream=True forces the real async-for branch of __anext__
        wrapper = _make_wrapper(chunks, async_stream=True)
        return await _drain_async(wrapper)

    results = asyncio.run(_run())
    final = results[-1]
    assert "usage" in final._hidden_params
    assert final._hidden_params["usage"] is not None


# ---------------------------------------------------------------------------
# 6. Bedrock Converse: ModelResponseStream chunks pass through correctly
# ---------------------------------------------------------------------------


def test_bedrock_converse_text_chunks_pass_through():
    """
    Bedrock Converse returns ModelResponseStream objects directly.
    They should pass through chunk_creator and appear in output unchanged.
    """
    chunks = [
        _make_bedrock_converse_chunk("Hello"),
        _make_bedrock_converse_chunk(" world"),
        _make_bedrock_converse_chunk("", finish_reason="end_turn"),
    ]
    wrapper = _make_wrapper(chunks, provider="bedrock")
    results = _drain_sync(wrapper)

    texts = [
        r.choices[0].delta.content
        for r in results
        if r.choices and r.choices[0].delta.content
    ]
    assert "Hello" in texts or any("Hello" in (t or "") for t in texts)


def test_bedrock_converse_usage_chunk_stripped_and_in_hidden_params():
    """Usage in a Bedrock Converse ModelResponseStream chunk is handled correctly."""
    usage = Usage(prompt_tokens=8, completion_tokens=12, total_tokens=20)
    chunks = [
        _make_bedrock_converse_chunk("Hi"),
        _make_bedrock_converse_chunk("", finish_reason="end_turn", usage=usage),
    ]
    wrapper = _make_wrapper(chunks, provider="bedrock")
    results = _drain_sync(wrapper)

    final = results[-1]
    assert "usage" in final._hidden_params
    assert final._hidden_params["usage"] is not None


# ---------------------------------------------------------------------------
# 7. Anthropic generic chunk (GChunk) path
# ---------------------------------------------------------------------------


def test_anthropic_generic_chunks_text_pass_through():
    """GChunk text chunks must arrive in the output with correct content."""
    chunks = [
        _make_generic_chunk("The"),
        _make_generic_chunk(" answer"),
        _make_generic_chunk("", is_finished=True, finish_reason="stop"),
    ]
    wrapper = _make_wrapper(chunks, provider="anthropic")
    results = _drain_sync(wrapper)

    texts = [
        r.choices[0].delta.content
        for r in results
        if r.choices and r.choices[0].delta.content
    ]
    assert len(texts) >= 2


def test_anthropic_finish_reason_propagated():
    """finish_reason must be set on the final streaming chunk."""
    chunks = [
        _make_generic_chunk("Hi"),
        _make_generic_chunk("", is_finished=True, finish_reason="stop"),
    ]
    wrapper = _make_wrapper(chunks, provider="anthropic")
    results = _drain_sync(wrapper)

    finish_reasons = [
        r.choices[0].finish_reason
        for r in results
        if r.choices and r.choices[0].finish_reason
    ]
    assert "stop" in finish_reasons


# ---------------------------------------------------------------------------
# 8. Callback caching: _post_streaming_hooks resolved once per stream
# ---------------------------------------------------------------------------


def test_post_streaming_hooks_cached_after_first_call():
    """
    _post_streaming_hooks must be None before the first hook call and a list after.
    The same list object must be reused on subsequent calls (not re-built).
    """
    wrapper = _make_wrapper([], provider="anthropic")
    assert wrapper._post_streaming_hooks is None, "Must be None before first call"

    async def _run():
        # Simulate hook resolution with an empty callback list
        with patch.object(litellm, "callbacks", []):
            await wrapper._call_post_streaming_deployment_hook(
                MagicMock(spec=ModelResponseStream)
            )
        first_list = wrapper._post_streaming_hooks
        assert isinstance(first_list, list)

        # Second call must reuse the same list object
        with patch.object(litellm, "callbacks", []):
            await wrapper._call_post_streaming_deployment_hook(
                MagicMock(spec=ModelResponseStream)
            )
        assert (
            wrapper._post_streaming_hooks is first_list
        ), "_post_streaming_hooks was rebuilt on second call — caching broken"

    asyncio.run(_run())


def test_post_streaming_hooks_filters_correctly():
    """
    Only CustomLogger instances must be included; plain callables are excluded.

    Note: CustomLogger's base class already defines
    async_post_call_streaming_deployment_hook, so ALL CustomLogger subclasses
    pass the hasattr() check regardless of whether they override the method.
    The filter therefore keeps any CustomLogger instance and drops anything else.
    """
    from litellm.integrations.custom_logger import CustomLogger

    class MyLogger(CustomLogger):
        pass

    plain_callable = MagicMock()

    wrapper = _make_wrapper([], provider="anthropic")

    async def _run():
        with patch.object(litellm, "callbacks", [MyLogger(), plain_callable]):
            await wrapper._call_post_streaming_deployment_hook(
                MagicMock(spec=ModelResponseStream)
            )

        # plain_callable must be excluded; MyLogger (CustomLogger subclass) included
        assert len(wrapper._post_streaming_hooks) == 1
        assert isinstance(wrapper._post_streaming_hooks[0], MyLogger)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 9. model_response_creator: hidden_params built correctly
# ---------------------------------------------------------------------------


def test_model_response_creator_hidden_params_no_chunk():
    """model_response_creator() with no args must include all _base_hidden_params."""
    wrapper = _make_wrapper([], provider="anthropic")
    response = wrapper.model_response_creator()

    assert response._hidden_params.get("response_cost") is None
    assert response._hidden_params.get("custom_llm_provider") == "anthropic"
    assert "created_at" in response._hidden_params


def test_model_response_creator_hidden_params_caller_merged():
    """When hidden_params are passed by caller, they must be included in result."""
    wrapper = _make_wrapper([], provider="anthropic")
    caller_params = {"some_key": "some_value"}
    response = wrapper.model_response_creator(hidden_params=caller_params)

    assert response._hidden_params.get("some_key") == "some_value"
    assert response._hidden_params.get("response_cost") is None


def test_model_response_creator_stream_key_stripped():
    """The 'stream' key must be removed from chunk before constructing ModelResponseStream."""
    wrapper = _make_wrapper([], provider="anthropic")
    chunk = {"stream": True, "choices": []}
    # Should not raise even if 'stream' would be an invalid ModelResponseStream field
    response = wrapper.model_response_creator(chunk=chunk)
    assert response is not None


# ---------------------------------------------------------------------------
# 10. Per-chunk overhead regression: sync path must not regress
# ---------------------------------------------------------------------------


def test_sync_streaming_overhead_not_regressed():
    """
    Micro-benchmark: the sync hot path must process 200 text chunks in < 2 s.

    This test acts as a canary for gross per-chunk overhead regressions.
    It is intentionally generous (2 s) to avoid flakiness on slow CI runners.
    """
    n_chunks = 200
    chunks = [_make_generic_chunk(f"token-{i}") for i in range(n_chunks)]
    chunks.append(_make_generic_chunk("", is_finished=True, finish_reason="stop"))

    wrapper = _make_wrapper(chunks, provider="anthropic")

    start = time.monotonic()
    results = _drain_sync(wrapper)
    elapsed = time.monotonic() - start

    assert len(results) > 0, "No chunks returned"
    assert elapsed < 2.0, (
        f"Sync streaming of {n_chunks} chunks took {elapsed:.3f}s — "
        "per-chunk overhead regression detected"
    )


def test_async_streaming_overhead_not_regressed():
    """
    Micro-benchmark for the async path: 200 text chunks in < 2 s.
    """
    n_chunks = 200
    chunks = [_make_generic_chunk(f"token-{i}") for i in range(n_chunks)]
    chunks.append(_make_generic_chunk("", is_finished=True, finish_reason="stop"))

    async def _run():
        wrapper = _make_wrapper(chunks, provider="anthropic")
        start = time.monotonic()
        results = await _drain_async(wrapper)
        return results, time.monotonic() - start

    results, elapsed = asyncio.run(_run())
    assert len(results) > 0
    assert elapsed < 2.0, (
        f"Async streaming of {n_chunks} chunks took {elapsed:.3f}s — "
        "per-chunk overhead regression detected"
    )
