import gc
import weakref

import pytest

from litellm.router_rust.invoke.streaming import _fallback_responses_stream_wrapper_class


class _Source:
    model = "gpt-4"
    custom_llm_provider = "openai"
    completed_response = None


@pytest.mark.asyncio
async def test_responses_stream_wrapper_releases_source_without_gc():
    """The wrapper class is built once and holds the source only as an attribute, so a consumed
    stream frees its source (and the request it carries) without waiting for a full collection."""

    async def events():
        yield object()

    source = _Source()
    source_ref = weakref.ref(source)
    wrapper_cls = _fallback_responses_stream_wrapper_class()
    assert wrapper_cls is _fallback_responses_stream_wrapper_class()

    gc.disable()
    try:
        wrapper = wrapper_cls(events(), source)
        assert len([item async for item in wrapper]) == 1
        del source, wrapper
        assert source_ref() is None
    finally:
        gc.enable()
