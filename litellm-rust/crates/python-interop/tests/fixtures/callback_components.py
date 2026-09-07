import asyncio
import atexit
import contextvars
import gc
import json
import threading
import weakref
from datetime import datetime
from unittest import TestCase

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import safe_deep_copy
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import LoggingWorker
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage


def logger_for(callbacks=(), stream=False, input_callbacks=(), sync_callbacks=()):
    return Logging(
        model="test",
        messages=[{"role": "user", "content": "test"}],
        stream=stream,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="retained-test",
        function_id="retained-test",
        dynamic_async_success_callbacks=list(callbacks),
        dynamic_input_callbacks=list(input_callbacks),
        dynamic_success_callbacks=list(sync_callbacks),
    )


async def real_pre_call_logging(owners):
    retained = []
    observed = []
    ignored = {"replacement": True}
    metadata = {"secret": "private", "keep": []}
    removed = object()
    lock = threading.Lock()

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            retained.append(kwargs)
            return ignored

    class Mutate(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["normalized"] = "normalized"
            assert kwargs.pop("remove") is removed
            kwargs["retained_metadata"]["secret"] = "masked"
            return ignored

    class Fail(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["lock"] = lock
            kwargs["retained_metadata"]["keep"].append("before failure")
            raise RuntimeError("expected pre-call callback failure")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append((kwargs, messages))

    logger = logger_for(input_callbacks=[Retain(), Mutate(), Fail(), Observe()])
    details = logger.model_call_details
    details.update(retained_metadata=metadata, normalized=None, remove=removed)
    messages = logger.messages
    additional = {"headers": {"test": "header"}}
    owner = owners.prepare(logger.pre_call, (messages, "test-key"), {"additional_args": additional})
    try:
        assert owner.invoke() is None
    finally:
        owner.close()
    assert retained == [details] and observed == [(details, messages)]
    assert retained[0] is details and observed[0][0] is details
    assert observed[0][1] is messages and details["input"] is messages
    assert details["additional_args"] is additional
    assert details["retained_metadata"] is metadata
    assert metadata == {"secret": "masked", "keep": ["before failure"]}
    assert details["normalized"] == "normalized" and "remove" not in details
    assert details["lock"] is lock and "replacement" not in details
    with TestCase().assertRaises(TypeError):
        json.dumps({"lock": details["lock"]})


async def real_async_logging(owners):
    observations = []
    task = asyncio.current_task()
    gate = asyncio.Event()
    result = ModelResponse(model="test", choices=[{"message": {"role": "assistant", "content": "original"}}])
    replacement = ModelResponse(model="test", choices=[{"message": {"role": "assistant", "content": "replacement"}}])
    shared = {}
    side_channel = {}
    replaced_kwargs = []

    class Retain(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            observations.append(("retained", kwargs, result, asyncio.current_task()))
            return kwargs, result

    class MutateThenFail(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            asyncio.get_running_loop().call_soon(gate.set)
            await gate.wait()
            kwargs["retained_shared"]["changed"] = True
            side_channel["failed_hook"] = kwargs
            result.choices[0].message.content = "mutated"
            raise RuntimeError("expected async callback failure")

    class Replace(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            observations.append(("replace", kwargs, result))
            updated = {**kwargs, "adopted": True}
            replaced_kwargs.append(updated)
            side_channel["replacement"] = replacement
            return updated, replacement

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
    assert observations[0][1] is observations[1][1] is side_channel["failed_hook"]
    assert observations[2][1] is observations[3][1] is logger.model_call_details is replaced_kwargs[0]
    assert logger.model_call_details is not observations[0][1]
    assert logger.model_call_details["adopted"] and "adopted" not in observations[0][1]
    assert logger.model_call_details["retained_shared"] is shared
    assert side_channel["replacement"] is replacement
    assert result.choices[0].message.content == "mutated"
    observations[0][1]["retained_shared"]["after_replacement"] = True
    assert observations[3][1]["retained_shared"]["after_replacement"]


async def real_copy_boundaries(owners):
    lock = threading.Lock()
    shared = {"values": []}
    standard = {
        "messages": [{"role": "user", "content": "private"}],
        "response": {"choices": [{"message": {"content": "private"}}]},
        "metadata": shared,
    }
    details = {"standard_logging_object": standard, "shared": shared, "lock": lock}
    passthrough = owners.prepare(CustomLogger().redact_standard_logging_payload_from_model_call_details, (details,))
    try:
        assert passthrough.invoke() is details
    finally:
        passthrough.close()
    logger = CustomLogger(turn_off_message_logging=True)
    redact = owners.prepare(logger.redact_standard_logging_payload_from_model_call_details, (details,))
    try:
        redacted = redact.invoke()
    finally:
        redact.close()
    assert redacted is not details
    assert redacted["standard_logging_object"] is not standard
    assert redacted["shared"] is shared and redacted["lock"] is lock
    assert redacted["standard_logging_object"]["metadata"] is shared
    redacted["standard_logging_object"]["metadata"]["values"].append("shared mutation")
    assert shared["values"] == ["shared mutation"]
    assert redacted["standard_logging_object"]["messages"][0]["content"] == "redacted-by-litellm"
    assert redacted["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == "redacted-by-litellm"
    assert standard["messages"][0]["content"] == "private"
    assert standard["response"]["choices"][0]["message"]["content"] == "private"

    original_mode = litellm.safe_memory_mode
    try:
        for safe_mode in (False, True):
            litellm.safe_memory_mode = safe_mode
            uncopyable = {"lock": lock, "values": []}
            values = []
            data = {"copyable": {"values": values, "alias": values}, "uncopyable": uncopyable}
            owner = owners.prepare(safe_deep_copy, (data,))
            try:
                copied = owner.invoke()
            finally:
                owner.close()
            assert (copied is data) is safe_mode
            assert (copied["copyable"] is data["copyable"]) is safe_mode
            assert copied["copyable"]["values"] is copied["copyable"]["alias"]
            assert (copied["copyable"]["values"] is values) is safe_mode
            assert copied["uncopyable"] is uncopyable and copied["uncopyable"]["lock"] is lock
            copied["copyable"]["values"].append("copy")
            assert copied["copyable"]["alias"] == ["copy"]
            copied["uncopyable"]["values"].append("fallback")
            assert data["copyable"]["values"] == (["copy"] if safe_mode else [])
            assert uncopyable["values"] == ["fallback"]
    finally:
        litellm.safe_memory_mode = original_mode


async def real_logging_worker(owners):
    context = contextvars.ContextVar("component_worker_context", default="outside")
    entered, release = asyncio.Event(), asyncio.Event()
    observations = []
    worker = LoggingWorker(timeout=5, concurrency=1)

    class Payload:
        pass

    async def upload(value, *, alias):
        assert value is alias
        assert context.get() == "submitted"
        entered.set()
        await release.wait()
        observations.append((value.changed, context.get()))
        context.set("worker only")

    payload = Payload()
    payload.changed = False
    reference = weakref.ref(payload)
    invocation = owners.prepare(upload, (payload,), {"alias": payload}, awaited=True)
    pending = invocation.invoke()
    invocation.close()
    enqueue = owners.prepare(worker.ensure_initialized_and_enqueue, (pending,))
    stop = owners.prepare(worker.stop, (), awaited=True)
    flush = owners.prepare(worker.flush, (), awaited=True)
    token = context.set("submitted")
    try:
        enqueue.invoke()
        enqueue.close()
        del pending, payload
        context.set("consumer")
        await entered.wait()
        assert reference() is not None
        reference().changed = True
        release.set()
        await flush.invoke()
        assert observations == [(True, "submitted")]
        assert context.get() == "consumer"
    finally:
        release.set()
        enqueue.close()
        flush.close()
        await stop.invoke()
        stop.close()
        context.reset(token)
        atexit.unregister(worker._flush_on_exit)
    assert worker._worker_task is None and not worker._running_tasks and not worker._dequeued_tasks
    assert worker._queue.empty()
    gc.collect()
    assert reference() is None


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


async def real_stream_completion(owners):
    logger = logger_for(stream=True)
    completions = []
    cached = []
    cache_done = asyncio.Event()

    class CacheRecorder:
        async def _add_streaming_response_to_cache(self, response):
            cached.append(response)
            cache_done.set()

    logger._llm_caching_handler = CacheRecorder()

    async def complete(response, cache_hit):
        completions.append(response)

    logger._on_deferred_stream_complete = complete
    stream = ControlledStream()
    stream.originals[1].usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
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
            if len(chunks) == 1:
                assert wrapper.chunks[-1] is chunk
                chunk.choices[0].delta.content = "retained hello"
            if chunk.choices and chunk.choices[0].finish_reason:
                stored = wrapper.chunks[-1]
                assert stored is not chunk and stored is not stream.originals[1]
                assert stored.usage is stream.originals[1].usage
                assert getattr(chunk, "usage", None) is None and stored.usage.total_tokens == 2
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
    assert response.choices[0].message.content == "retained hello"
    assert retained_hidden["usage"] is response.usage
    usage_chunk = wrapper.chunks[-1]
    assert usage_chunk is not stream.originals[-1]
    assert usage_chunk.usage is stream.originals[-1].usage
    stream.originals[-1].usage.total_tokens = 13
    assert usage_chunk.usage.total_tokens == 13
    assert response.usage.total_tokens == 8
    deferred = owners.prepare(logger._on_deferred_stream_complete, (response, cache_hit), awaited=True)
    logger._on_deferred_stream_complete = None
    logger._deferred_stream_complete_args = None
    close = owners.prepare(wrapper.aclose, (), awaited=True)
    await close.invoke()
    await close.invoke()
    close.close()
    assert stream.closed == 1
    del wrapper, logger
    await deferred.invoke()
    deferred.close()
    assert completions == [response]
    assert retained_hidden["usage"].total_tokens == 8
    assert hidden_owner.invoke() is retained_hidden
    hidden_owner.close()
    await cache_done.wait()
    assert len(cached) == 1 and cached[0] is not response
    assert cached[0].choices[0] is not response.choices[0]
    cached[0].choices[0].message.content = "cache only"
    assert response.choices[0].message.content == "retained hello"


async def real_sync_stream_copies(owners):
    original_disable = litellm.disable_streaming_logging
    copy_attempts = []

    class Uncopyable:
        def __deepcopy__(self, memo):
            copy_attempts.append(True)
            raise RuntimeError("expected streaming deepcopy failure")

    class CacheRecorder:
        def __init__(self, responses):
            self.responses = responses

        def _sync_add_streaming_response_to_cache(self, response):
            self.responses.append(response)

    class Observe(CustomLogger):
        def __init__(self, responses, finished):
            super().__init__()
            self.responses = responses
            self.finished = finished

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.responses.append(response_obj)
            self.finished.set()

    try:
        litellm.disable_streaming_logging = True
        for fallback in (False, True):
            cached, logged = [], []
            finished = threading.Event()

            logger = logger_for(stream=True, sync_callbacks=[Observe(logged, finished)])
            logger._llm_caching_handler = CacheRecorder(cached)
            source = ControlledStream()
            source.originals[1].usage = source.originals[2].usage
            wrapper = CustomStreamWrapper(
                completion_stream=iter(source.originals[:2]),
                model="test",
                logging_obj=logger,
                custom_llm_provider="bedrock",
            )
            pull = owners.prepare(wrapper.__next__, ())
            close = owners.prepare(wrapper.aclose, (), awaited=True)
            try:
                first = pull.invoke()
                assert wrapper.chunks[0] is first
                first.choices[0].delta.content = "consumer mutation"
                shared = {"values": []}
                last = pull.invoke()
                assert last.choices[0].finish_reason == "stop"
                for chunk in wrapper.chunks:
                    chunk._hidden_params["retained_shared"] = shared
                    if fallback:
                        chunk._hidden_params["uncopyable"] = Uncopyable()
                retained_hidden = last._hidden_params
                with TestCase().assertRaises(StopIteration):
                    pull.invoke()
                assert await asyncio.to_thread(finished.wait, 5)
            finally:
                pull.close()
                await close.invoke()
                close.close()
            assert len(cached) == len(logged) == 1
            cache_response, log_response = cached[0], logged[0]
            assert cache_response is not log_response
            assert cache_response.choices[0].message.content == "consumer mutation"
            assert log_response.choices[0].message.content == "consumer mutation"
            assert retained_hidden["usage"].total_tokens == 8
            assert (cache_response.choices is log_response.choices) is fallback
            assert (cache_response.usage is log_response.usage) is fallback
            assert (cache_response.usage is retained_hidden["usage"]) is fallback
            assert (cache_response._hidden_params is log_response._hidden_params) is fallback
            assert (cache_response._hidden_params["retained_shared"] is shared) is fallback
            cache_response.choices[0].message.content = "cache mutation"
            cache_response._hidden_params["retained_shared"]["values"].append("cache mutation")
            assert log_response.choices[0].message.content == ("cache mutation" if fallback else "consumer mutation")
            assert shared["values"] == (["cache mutation"] if fallback else [])
            assert log_response._hidden_params["retained_shared"]["values"] == (["cache mutation"] if fallback else [])
        assert copy_attempts == [True]
    finally:
        litellm.disable_streaming_logging = original_disable


async def real_stream_close(owners):
    source = ControlledStream()
    logger = logger_for(stream=True)
    wrapper = CustomStreamWrapper(
        completion_stream=source, model="test", logging_obj=logger, custom_llm_provider="bedrock"
    )
    pull = owners.prepare(wrapper.__anext__, (), awaited=True)
    chunk = await pull.invoke()
    assert wrapper.chunks[0] is chunk
    pull.close()
    retained = owners.prepare(lambda value: value, (chunk,))
    close = owners.prepare(wrapper.aclose, (), awaited=True)
    await close.invoke()
    await close.invoke()
    close.close()
    assert source.closed == 1
    assert wrapper.completion_stream is None
    assert not getattr(logger, "_deferred_stream_complete_args", None)
    assert retained.invoke() is chunk
    assert chunk.choices[0].delta.content == "hello"
    retained.close()


async def real_stream_cancellation(owners):
    entered = asyncio.Event()

    class SuspendedStream(ControlledStream):
        async def __anext__(self):
            if self.chunks is not None:
                chunk = next(self.chunks)
                self.chunks = None
                return chunk
            entered.set()
            await asyncio.Event().wait()

    source = SuspendedStream()
    source.originals[0].usage = Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    logger = logger_for(stream=True)
    wrapper = CustomStreamWrapper(
        completion_stream=source, model="test", logging_obj=logger, custom_llm_provider="bedrock"
    )
    pull = owners.prepare(wrapper.__anext__, (), awaited=True)
    chunk = await pull.invoke()
    retained = owners.prepare(lambda value: value, (chunk,))
    assert wrapper.chunks[0] is not chunk
    assert wrapper.chunks[0].usage is source.originals[0].usage
    task = asyncio.create_task(pull.invoke())
    pull.close()
    await entered.wait()
    task.cancel()
    with TestCase().assertRaises(asyncio.CancelledError):
        await task
    assert logger.model_call_details.get("combined_usage_object") is None
    recover = owners.prepare(wrapper._record_partial_usage_for_failure, ())
    try:
        assert recover.invoke() is None
    finally:
        recover.close()
    usage = logger.model_call_details["combined_usage_object"]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 2, 5)
    assert usage is not source.originals[0].usage
    source.originals[0].usage.total_tokens = 99
    assert usage.total_tokens == 5
    retained_usage = owners.prepare(lambda value: value, (usage,))
    close = owners.prepare(wrapper.aclose, (), awaited=True)
    await close.invoke()
    await close.invoke()
    close.close()
    assert source.closed == 1
    assert wrapper.completion_stream is None and len(wrapper.chunks) == 1
    assert not getattr(logger, "_deferred_stream_complete_args", None)
    assert retained.invoke() is chunk and chunk.choices[0].delta.content == "hello"
    retained.close()
    del wrapper, logger, usage
    assert retained_usage.invoke().total_tokens == 5
    retained_usage.close()
