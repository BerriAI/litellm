import asyncio
import gc
import weakref

import pytest

import litellm


@pytest.mark.asyncio
async def test_streamed_completion_releases_logging_object_without_full_gc() -> None:
    """A streamed Router completion frees the logging object (and the request messages it holds)
    as soon as the caller drops the stream, without waiting for a full gc.collect()."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": "hi"},
            }
        ]
    )
    gc.collect()
    gc.disable()
    try:
        resp = await router.acompletion(model="m", messages=[{"role": "user", "content": "x" * 1_000_000}], stream=True)
        async for _ in resp:
            pass
        stream_ref = weakref.ref(resp)
        logging_ref = weakref.ref(resp.logging_obj)
        del resp
        await asyncio.sleep(0)
        assert stream_ref() is None
        assert logging_ref() is None
    finally:
        gc.enable()
