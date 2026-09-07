import asyncio
from datetime import datetime
from unittest import TestCase

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage


def logger_for(callbacks=(), stream=False):
    return Logging(
        model="test",
        messages=[{"role": "user", "content": "test"}],
        stream=stream,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="retained-test",
        function_id="retained-test",
        dynamic_async_success_callbacks=list(callbacks),
    )


async def real_async_logging(owners):
    observations = []
    task = asyncio.current_task()
    gate = asyncio.Event()
    result = ModelResponse(model="test", choices=[{"message": {"role": "assistant", "content": "original"}}])
    replacement = ModelResponse(model="test", choices=[{"message": {"role": "assistant", "content": "replacement"}}])
    shared = {}

    class Retain(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            observations.append(("retained", kwargs, result, asyncio.current_task()))
            return kwargs, result

    class MutateThenFail(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            asyncio.get_running_loop().call_soon(gate.set)
            await gate.wait()
            kwargs["retained_shared"]["changed"] = True
            result.choices[0].message.content = "mutated"
            raise RuntimeError("expected async callback failure")

    class Replace(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            observations.append(("replace", kwargs, result))
            return kwargs, replacement

    class Observe(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            observations.append(("observe", kwargs, result, asyncio.current_task()))
            return kwargs, result

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("success", kwargs, response_obj))

    logger = logger_for([Retain(), MutateThenFail(), Replace(), Observe()])
    logger.model_call_details["retained_shared"] = shared
    owner = owners.prepare(logger.async_success_handler, (), {"result": result}, awaited=True)
    await owner.invoke()
    owner.close()
    assert [entry[0] for entry in observations] == ["retained", "replace", "observe", "success"]
    assert observations[0][3] is task and observations[2][3] is task
    assert observations[0][2] is result and observations[1][2] is result
    assert observations[2][2] is replacement and observations[3][2] is replacement
    assert observations[0][1]["retained_shared"] is shared and shared["changed"]
    assert result.choices[0].message.content == "mutated"


class ControlledStream:
    def __init__(self):
        self.originals = [
            ModelResponseStream(model="test", choices=[StreamingChoices(delta=Delta(content="hello"), index=0)]),
            ModelResponseStream(
                model="test", choices=[StreamingChoices(delta=Delta(content=""), index=0, finish_reason="stop")]
            ),
            ModelResponseStream(
                model="test", choices=[], usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
            ),
        ]
        self.chunks = iter(self.originals)
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        self.closed += 1


async def drain_component_tasks():
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    if pending:
        await asyncio.gather(*pending)


async def real_stream_completion(owners):
    logger = logger_for(stream=True)
    completions = []
    cached = []

    class CacheRecorder:
        async def _add_streaming_response_to_cache(self, response):
            cached.append(response)

    logger._llm_caching_handler = CacheRecorder()

    async def complete(response, cache_hit):
        completions.append(response)

    logger._on_deferred_stream_complete = complete
    stream = ControlledStream()
    wrapper = CustomStreamWrapper(
        completion_stream=stream, model="test", logging_obj=logger, custom_llm_provider="bedrock"
    )
    pull = owners.prepare(wrapper.__anext__, (), awaited=True)
    chunks = []
    retained_hidden = None
    hidden_owner = None
    while True:
        try:
            chunk = await pull.invoke()
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].finish_reason:
                retained_hidden = chunk._hidden_params
                hidden_owner = owners.prepare(lambda value: value, (retained_hidden,))
                assert not completions
        except StopAsyncIteration:
            break
    pull.close()
    assert retained_hidden is not None
    assert retained_hidden["usage"].total_tokens == 8
    assert completions == []
    response, cache_hit = logger._deferred_stream_complete_args
    assert response.usage.total_tokens == 8
    deferred = owners.prepare(logger._on_deferred_stream_complete, (response, cache_hit), awaited=True)
    logger._on_deferred_stream_complete = None
    logger._deferred_stream_complete_args = None
    close = owners.prepare(wrapper.aclose, (), awaited=True)
    await close.invoke()
    close.close()
    del wrapper, logger
    await deferred.invoke()
    deferred.close()
    assert completions == [response]
    assert retained_hidden["usage"].total_tokens == 8
    assert hidden_owner.invoke() is retained_hidden
    hidden_owner.close()
    await drain_component_tasks()
    assert len(cached) == 1 and cached[0] is not response
    assert cached[0].choices[0] is not response.choices[0]
    cached[0].choices[0].message.content = "cache only"
    assert response.choices[0].message.content == "hello"


async def real_stream_close(owners):
    source = ControlledStream()
    logger = logger_for(stream=True)
    wrapper = CustomStreamWrapper(
        completion_stream=source, model="test", logging_obj=logger, custom_llm_provider="bedrock"
    )
    pull = owners.prepare(wrapper.__anext__, (), awaited=True)
    chunk = await pull.invoke()
    pull.close()
    retained = owners.prepare(lambda value: value, (chunk,))
    close = owners.prepare(wrapper.aclose, (), awaited=True)
    await close.invoke()
    await close.invoke()
    close.close()
    assert source.closed == 1
    assert retained.invoke() is chunk
    assert chunk.choices[0].delta.content == "hello"
    retained.close()
    await drain_component_tasks()


async def real_stream_cancellation(owners):
    entered = asyncio.Event()

    class SuspendedStream(ControlledStream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    source = SuspendedStream()
    logger = logger_for(stream=True)
    wrapper = CustomStreamWrapper(
        completion_stream=source, model="test", logging_obj=logger, custom_llm_provider="bedrock"
    )
    pull = owners.prepare(wrapper.__anext__, (), awaited=True)
    task = asyncio.create_task(pull.invoke())
    pull.close()
    await entered.wait()
    task.cancel()
    with TestCase().assertRaises(asyncio.CancelledError):
        await task
    close = owners.prepare(wrapper.aclose, (), awaited=True)
    await close.invoke()
    close.close()
    assert source.closed == 1
    await drain_component_tasks()
