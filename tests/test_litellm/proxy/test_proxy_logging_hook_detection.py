import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging


def test_has_post_call_response_headers_callbacks_ignores_empty_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [])

    assert ProxyLogging.has_post_call_response_headers_callbacks() is False


def test_has_post_call_response_headers_callbacks_requires_override(
    monkeypatch,
):
    """A vanilla ``CustomLogger`` inherits the no-op response-headers hook;
    the capability flag must stay False so the proxy can skip the headers
    loop entirely.  Only callbacks that *override* the hook should flip it."""
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    assert ProxyLogging.has_post_call_response_headers_callbacks() is False

    class _AddsHeaders(CustomLogger):
        async def async_post_call_response_headers_hook(self, **kwargs):
            return {"x-custom": "1"}

    monkeypatch.setattr(litellm, "callbacks", [_AddsHeaders()])
    assert ProxyLogging.has_post_call_response_headers_callbacks() is True


def test_has_streaming_callbacks_uses_custom_logger_detection(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    assert ProxyLogging.has_streaming_callbacks() is False

    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    assert ProxyLogging.has_streaming_callbacks() is False

    class StreamingLogger(CustomLogger):
        async def async_post_call_streaming_hook(self, **kwargs):
            return kwargs.get("response")

    monkeypatch.setattr(litellm, "callbacks", [StreamingLogger()])
    assert ProxyLogging.has_streaming_callbacks() is True


def test_has_streaming_callbacks_detects_guardrails(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [CustomGuardrail()])
    assert ProxyLogging.has_streaming_callbacks() is True


@pytest.mark.asyncio
async def test_post_call_response_headers_hook_returns_early_without_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy_logging_obj = ProxyLogging(user_api_key_cache={})  # type: ignore[arg-type]

    result = await proxy_logging_obj.post_call_response_headers_hook(
        data={},
        user_api_key_dict=None,  # type: ignore[arg-type]
        response=None,
        request_headers={},
    )

    assert result == {}


def test_callback_capabilities_skips_default_custom_logger(monkeypatch):
    """
    Internal proxy hooks (e.g. _PROXY_MaxBudgetLimiter, ManagedFiles) inherit
    the default ``async_post_call_streaming_iterator_hook`` body.  The
    capability scanner must NOT report them as iterator overrides — wrapping
    the chunk stream through every no-op layer was responsible for ~10x
    streaming overhead on default deployments.
    """

    class _InternalNoopHook(CustomLogger):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_InternalNoopHook()])

    caps = ProxyLogging._callback_capabilities()
    # Subclass inherits the base no-op for every hook — every capability flag
    # must stay False so the proxy short-circuits the corresponding loops.
    assert caps.has_post_call_response_headers is False
    assert caps.iterator_overrides == ()
    assert caps.has_iterator_override is False
    assert caps.has_streaming_chunk_override is False
    assert caps.has_guardrail is False


def test_callback_capabilities_captures_iterator_override(monkeypatch):
    class _OverridesIterator(CustomLogger):
        async def async_post_call_streaming_iterator_hook(  # type: ignore[override]
            self, user_api_key_dict, response, request_data
        ):
            async for item in response:
                yield item

    override = _OverridesIterator()
    monkeypatch.setattr(litellm, "callbacks", [override])

    caps = ProxyLogging._callback_capabilities()
    assert caps.has_iterator_override is True
    assert len(caps.iterator_overrides) == 1
    resolved, kind = caps.iterator_overrides[0]
    assert resolved is override
    assert kind == "override"


def test_callback_capabilities_detects_inherited_streaming_chunk_override(monkeypatch):
    """
    ``async_post_call_streaming_hook`` must be detected even when the override
    lives on an intermediate parent class — a vendor base class can carry the
    override and the registered class can add nothing else. Before this PR the
    hook was unconditionally invoked, so a leaf-class ``__dict__`` miss here
    would silently drop the inherited hook.
    """
    ProxyLogging._callback_capabilities_cache.clear()

    class _StreamingBase(CustomLogger):
        async def async_post_call_streaming_hook(self, *args, **kwargs):  # type: ignore[override]
            return kwargs.get("response")

    class _LeafWithoutOverride(_StreamingBase):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_LeafWithoutOverride()])
    caps = ProxyLogging._callback_capabilities()
    assert caps.has_streaming_chunk_override is True


def test_callback_capabilities_cache_invalidates_on_list_change(monkeypatch):
    """The cache key includes (length, id-of-each-callback).  Mutating the
    callback list must produce a fresh capability snapshot."""
    monkeypatch.setattr(litellm, "callbacks", [])
    assert ProxyLogging._callback_capabilities().resolved_callbacks == ()

    class _OverridesPreCall(CustomLogger):
        async def async_pre_call_hook(self, *args, **kwargs):
            return kwargs.get("data")

    pre = _OverridesPreCall()
    monkeypatch.setattr(litellm, "callbacks", [pre])
    caps = ProxyLogging._callback_capabilities()
    assert caps.has_pre_call_override is True
    assert pre in caps.resolved_callbacks


def _sse_bytes(event: str, payload: dict) -> bytes:
    import json

    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _anthropic_stream_chunks(text_parts):
    chunks = [
        _sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "model": "claude-sonnet-5",
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        _sse_bytes(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    for part in text_parts:
        chunks.append(
            _sse_bytes(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": part}},
            )
        )
    chunks.append(_sse_bytes("content_block_stop", {"type": "content_block_stop", "index": 0}))
    chunks.append(
        _sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 20, "output_tokens": 8},
            },
        )
    )
    chunks.append(_sse_bytes("message_stop", {"type": "message_stop"}))
    return chunks


def _content_filter_guardrail(action: str, guardrail_cls=None, **guardrail_kwargs):
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )
    from litellm.types.guardrails import BlockedWord, ContentFilterAction

    cls = guardrail_cls or ContentFilterGuardrail
    return cls(
        guardrail_name="output-filter",
        blocked_words=[BlockedWord(keyword="zebra", action=ContentFilterAction(action))],
        event_hook="post_call",
        default_on=True,
        **guardrail_kwargs,
    )


def _streaming_logging_obj():
    import datetime
    import uuid

    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Reply with exactly: the zebra runs"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        function_id="test",
    )


def test_stream_requires_guardrail_translation_route_detection():
    from litellm.proxy._types import UserAPIKeyAuth

    assert (
        ProxyLogging._stream_requires_guardrail_translation(
            UserAPIKeyAuth(api_key="sk-1234", request_route="/v1/messages")
        )
        is True
    )
    assert (
        ProxyLogging._stream_requires_guardrail_translation(
            UserAPIKeyAuth(api_key="sk-1234", request_route="/chat/completions")
        )
        is False
    )
    assert ProxyLogging._stream_requires_guardrail_translation(UserAPIKeyAuth(api_key="sk-1234")) is False
    assert (
        ProxyLogging._stream_requires_guardrail_translation(
            UserAPIKeyAuth(api_key="sk-1234", request_route="/route/without/call/types")
        )
        is False
    )


@pytest.mark.asyncio
async def test_post_call_stream_guardrail_blocks_anthropic_messages_stream(monkeypatch):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/35257.

    /v1/messages streams raw Anthropic SSE bytes. A guardrail whose custom
    iterator hook only understands OpenAI ModelResponseStream chunks used to
    receive those bytes directly and silently pass every chunk through
    unscanned. The dispatch must route apply_guardrail-capable guardrails
    through unified_guardrail's anthropic translation so blocked output
    raises instead of streaming to the client. Because the guardrail's own
    iterator hook withheld content until scanned, the rerouted invocation
    defaults to buffer_until_moderated, so nothing may reach the client
    before the block fires.
    """
    from fastapi import HTTPException

    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    guardrail = _content_filter_guardrail("BLOCK")
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    request_data = {
        "model": "claude-sonnet-5",
        "litellm_logging_obj": _streaming_logging_obj(),
        "metadata": {},
    }

    async def fake_stream():
        for chunk in _anthropic_stream_chunks(["the", " zebra runs"]):
            yield chunk

    delivered = []
    with pytest.raises(HTTPException) as exc_info:
        async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
            response=fake_stream(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-1234", request_route="/v1/messages"),
            request_data=request_data,
        ):
            delivered.append(chunk)

    detail = exc_info.value.detail
    assert detail["guardrail_name"] == "output-filter"
    assert detail["keyword"] == "zebra"
    assert delivered == []


@pytest.mark.asyncio
async def test_post_call_stream_guardrail_keeps_own_iterator_on_chat_completions(monkeypatch):
    """
    On /chat/completions the guardrail's own iterator hook must keep running:
    it masks incrementally inside ModelResponseStream chunks, which the
    unified block_only path never does. Masked output proves the own-hook
    path was used.
    """
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    guardrail = _content_filter_guardrail("MASK")
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

    async def fake_stream():
        yield ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content="the zebra runs"))]
        )
        yield ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=""), finish_reason="stop")]
        )

    delivered_text = ""
    async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
        response=fake_stream(),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-1234", request_route="/chat/completions"),
        request_data={"model": "gpt-4o-mini", "metadata": {}},
    ):
        for choice in chunk.choices:
            delivered_text += choice.delta.content or ""

    assert "zebra" not in delivered_text
    assert delivered_text != ""


@pytest.mark.asyncio
async def test_unified_guardrail_iterator_accepts_explicit_guardrail(monkeypatch):
    """
    The dispatch passes each guardrail explicitly instead of through a shared
    request_data key, so chaining two unified-routed guardrails cannot drop
    all but the last one.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import unified_guardrail

    guardrail = _content_filter_guardrail("BLOCK")
    request_data = {
        "model": "claude-sonnet-5",
        "litellm_logging_obj": _streaming_logging_obj(),
        "metadata": {},
    }

    async def fake_stream():
        for chunk in _anthropic_stream_chunks(["the", " zebra runs"]):
            yield chunk

    with pytest.raises(HTTPException):
        async for _ in unified_guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-1234", request_route="/v1/messages"),
            response=fake_stream(),
            request_data=request_data,
            guardrail_to_apply=guardrail,
        ):
            pass


@pytest.mark.asyncio
async def test_post_call_stream_guardrail_reroutes_inherited_apply_guardrail(monkeypatch):
    """
    The reroute predicate must recognize apply_guardrail implementations
    inherited from a parent class, not only ones defined on the registered
    leaf class. A vendor base class can carry apply_guardrail while the leaf
    only overrides the streaming iterator; a leaf-class ``__dict__`` check
    would leave that guardrail on the raw Anthropic SSE path unscanned.
    """
    from fastapi import HTTPException

    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )

    class _InheritsApplyGuardrail(ContentFilterGuardrail):
        async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
            async for item in response:
                yield item

    guardrail = _content_filter_guardrail("BLOCK", guardrail_cls=_InheritsApplyGuardrail)
    assert "apply_guardrail" not in type(guardrail).__dict__
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    request_data = {
        "model": "claude-sonnet-5",
        "litellm_logging_obj": _streaming_logging_obj(),
        "metadata": {},
    }

    async def fake_stream():
        for chunk in _anthropic_stream_chunks(["the", " zebra runs"]):
            yield chunk

    delivered = []
    with pytest.raises(HTTPException) as exc_info:
        async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
            response=fake_stream(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-1234", request_route="/v1/messages"),
            request_data=request_data,
        ):
            delivered.append(chunk)

    assert exc_info.value.detail["keyword"] == "zebra"
    assert delivered == []


@pytest.mark.asyncio
async def test_post_call_stream_masking_guardrail_keeps_own_iterator_on_anthropic(monkeypatch):
    """
    A guardrail with mask_response_content=True must stay on its own iterator
    hook on /v1/messages. The unified streaming path cannot re-emit rewritten
    text on raw Anthropic SSE (block_only drops rewrites and buffered replay
    releases the unredacted originals), so rerouting such a guardrail would
    deliver content it decided to mask. PANW Prisma AIRS is the concrete
    case: its own hook parses the raw bytes and blocks instead of masking.
    """
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )

    own_hook_streams = []

    class _MasksViaOwnRawStreamHook(ContentFilterGuardrail):
        apply_guardrail = ContentFilterGuardrail.apply_guardrail

        async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
            own_hook_streams.append(request_data.get("model"))
            async for item in response:
                yield item

    guardrail = _content_filter_guardrail(
        "BLOCK", guardrail_cls=_MasksViaOwnRawStreamHook, mask_response_content=True
    )
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    chunks = _anthropic_stream_chunks(["the", " zebra runs"])

    async def fake_stream():
        for chunk in chunks:
            yield chunk

    delivered = []
    async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
        response=fake_stream(),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-1234", request_route="/v1/messages"),
        request_data={
            "model": "claude-sonnet-5",
            "litellm_logging_obj": _streaming_logging_obj(),
            "metadata": {},
        },
    ):
        delivered.append(chunk)

    assert own_hook_streams == ["claude-sonnet-5"]
    assert delivered == chunks
