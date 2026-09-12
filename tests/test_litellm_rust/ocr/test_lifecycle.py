import asyncio
import datetime
import gc
import threading
import weakref
from contextvars import ContextVar
from typing import Final

import pytest

import litellm
from litellm._logging import trace_id_var
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import OCR_RESPONSE, call_aocr, call_ocr

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


@pytest.mark.asyncio
async def test_proxy_metadata_remains_python_owned(ocr_server: RecordingServer) -> None:
    from litellm.proxy._types import UserAPIKeyAuth

    recorder: Final = RecordingLogger()
    auth: Final = UserAPIKeyAuth(user_id="ocr-user")
    response: Final = await call_aocr(
        ocr_server, callbacks=[recorder], metadata={"user_api_key_auth": auth}, shared_session=object()
    )
    events: Final = await recorder.wait_for_async("async_log_success_event")
    assert response.pages[0].markdown == "native OCR response"
    assert events[0].kwargs["litellm_params"]["metadata"]["user_api_key_auth"].user_id == "ocr-user"
    assert "metadata" not in ocr_server.requests[0].body


@pytest.mark.asyncio
async def test_response_replacement_finalized_before_dispatch_in_caller_task(ocr_server: RecordingServer) -> None:
    caller: Final = asyncio.current_task()
    context: Final = ContextVar("lifecycle-test", default="before")
    observations: Final = []
    recorder: Final = RecordingLogger()

    class Replace(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            context.set("pre")
            observations.append(("pre", asyncio.current_task(), context.get()))
            return {**kwargs, "pages": [2]}

        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            observations.append(("post", asyncio.current_task(), context.get()))
            return response.model_copy(update={"model": "replaced"})

    litellm.callbacks.append(Replace())
    response: Final = await call_aocr(ocr_server, callbacks=[recorder], litellm_call_id="native-final")
    events: Final = await recorder.wait_for_async("async_log_success_event")
    assert observations == [("pre", caller, "pre"), ("post", caller, "pre")]
    assert context.get() == "pre"
    assert ocr_server.requests[0].body["pages"] == [2]
    assert response.model == "replaced"
    assert events[0].response is response
    assert response._hidden_params["litellm_call_id"] == "native-final"
    assert "response_cost" in response._hidden_params


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_metadata_failure_dispatches_only_failure_and_releases_logger(
    ocr_server: RecordingServer, asynchronous: bool
) -> None:
    failure: Final = RuntimeError("metadata failed")
    seen: Final = []

    class FailingMetadata(Logging):
        def _response_cost_calculator(self, *args, **kwargs):
            raise failure

        def success_handler(self, *args, **kwargs):
            seen.append("success")

        def failure_handler(self, exception, *args, **kwargs):
            seen.append(("sync", exception))

        async def async_failure_handler(self, exception, *args, **kwargs):
            seen.append(("async", exception))

    async def invoke():
        logger: Final = FailingMetadata(
            model="mistral-ocr-latest",
            messages=[],
            stream=False,
            call_type="aocr" if asynchronous else "ocr",
            start_time=datetime.datetime.now(),
            litellm_call_id="metadata",
            function_id="metadata",
        )
        reference: Final = weakref.ref(logger)
        with pytest.raises(RuntimeError) as caught:
            await call_aocr(ocr_server, litellm_logging_obj=logger) if asynchronous else call_ocr(
                ocr_server, litellm_logging_obj=logger
            )
        assert caught.value is failure
        failure.__traceback__ = None
        return reference

    reference: Final = await invoke()
    await drain_logging()
    gc.collect()
    assert seen == ([("sync", failure), ("async", failure)] if asynchronous else [("sync", failure)])
    assert reference() is None
    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
async def test_mapped_failure_identity_and_deployment_snapshot(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "unavailable"}, status=500))
    recorder: Final = RecordingLogger()
    snapshots: Final = []

    class Observe(CustomLogger):
        async def async_post_call_failure_deployment_hook(self, request_data, exception, call_type, **kwargs):
            snapshots.append(exception)
            exception.status_code = 418

    litellm.callbacks.append(Observe())
    with pytest.raises(litellm.InternalServerError) as caught:
        await call_aocr(ocr_server, callbacks=[recorder])
    failures: Final = tuple(event for event in recorder.events if "failure" in event.name)
    assert [event.name for event in failures] == ["log_failure_event", "async_log_failure_event"]
    assert all(event.kwargs["exception"] is caught.value for event in failures)
    assert caught.value.status_code == 500
    assert snapshots[0] is not caught.value
    assert snapshots[0].status_code == 418
    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "http", "post"])
async def test_cancellation_cleans_up_in_caller_task_without_terminal_dispatch(
    ocr_server: RecordingServer, phase: str
) -> None:
    entered: Final = asyncio.Event()
    recorder: Final = RecordingLogger()

    class Pause(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            if phase == "pre":
                entered.set()
                await asyncio.Event().wait()

        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            if phase == "post":
                entered.set()
                await asyncio.Event().wait()

    litellm.callbacks.append(Pause())
    if phase == "http":
        ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.2))
    if phase == "pre":
        ocr_server.expected_requests = 0
    restored: Final = []

    async def invoke():
        trace_id_var.set("parent")
        try:
            await call_aocr(ocr_server, callbacks=[recorder], litellm_trace_id="native-call")
        finally:
            restored.append(trace_id_var.get())

    task: Final = asyncio.create_task(invoke())
    if phase == "http":
        await ocr_server.wait_for_requests(1)
    else:
        await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await drain_logging()
    assert restored == ["parent"]
    assert not any("success" in name or "failure" in name for name in recorder.names)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True])
async def test_deferred_logging_requires_release_and_runs_at_most_once(
    ocr_server: RecordingServer, blocked: bool
) -> None:
    recorder: Final = RecordingLogger()
    logger: Final = Logging(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="deferred",
        function_id="deferred",
        dynamic_async_success_callbacks=[recorder],
    )
    logger._defer_async_logging = True
    response: Final = await call_aocr(ocr_server, litellm_logging_obj=logger)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, blocked)
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, blocked)
    await drain_logging()
    events: Final = tuple(event for event in recorder.events if event.name == "async_log_success_event")
    assert len(events) == int(not blocked)
    if events:
        assert events[0].response is response


@pytest.mark.asyncio
async def test_abandoned_deferred_logging_is_collectable(ocr_server: RecordingServer) -> None:
    async def invoke():
        logger: Final = Logging(
            model="mistral-ocr-latest",
            messages=[],
            stream=False,
            call_type="aocr",
            start_time=datetime.datetime.now(),
            litellm_call_id="abandoned",
            function_id="abandoned",
        )
        logger._defer_async_logging = True
        await call_aocr(ocr_server, litellm_logging_obj=logger)
        return weakref.ref(logger)

    reference: Final = await invoke()
    await drain_logging()
    gc.collect()
    assert reference() is None


def test_sync_success_uses_executor_and_copied_caller_context(ocr_server: RecordingServer) -> None:
    context: Final = ContextVar("sync-lifecycle", default="missing")
    context.set("caller")
    thread: Final = threading.current_thread()
    finished: Final = threading.Event()
    observations: Final = []

    class Observe(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            observations.append((threading.current_thread(), context.get(), response_obj))
            finished.set()

    response: Final = call_ocr(ocr_server, callbacks=[Observe()])
    assert finished.wait(5)
    assert observations[0][0] is not thread
    assert observations[0][1] == "caller"
    assert observations[0][2] is response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invalid_response_runs_post_call_before_failure(ocr_server: RecordingServer, asynchronous: bool) -> None:
    ocr_server.enqueue(ResponseSpec(body={"pages": "invalid"}))
    events: Final = []

    class Observe(Logging):
        def pre_call(self, *args, **kwargs):
            events.append("pre")
            return super().pre_call(*args, **kwargs)

        def post_call(self, *args, **kwargs):
            events.append(("post", kwargs["original_response"]))
            return super().post_call(*args, **kwargs)

        def success_handler(self, *args, **kwargs):
            events.append("success")

        def failure_handler(self, exception, *args, **kwargs):
            events.append(("failure", exception))

        async def async_failure_handler(self, exception, *args, **kwargs):
            events.append(("async_failure", exception))

    logger: Final = Observe(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr" if asynchronous else "ocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="invalid",
        function_id="invalid",
    )
    with pytest.raises(litellm.APIConnectionError) as caught:
        await call_aocr(ocr_server, litellm_logging_obj=logger) if asynchronous else call_ocr(
            ocr_server, litellm_logging_obj=logger
        )
    assert events[0] == "pre"
    assert events[1] == ("post", '{"pages": "invalid"}')
    assert events[2] == ("failure", caught.value)
    if asynchronous:
        assert events[3] == ("async_failure", caught.value)
    assert "success" not in events


@pytest.mark.asyncio
async def test_failing_terminal_handler_preserves_public_failure_and_runs_async_handler(
    ocr_server: RecordingServer,
) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider failure"}, status=500))
    failures: Final = []

    class BrokenHandler(Logging):
        def failure_handler(self, exception, *args, **kwargs):
            failures.append(exception)
            raise RuntimeError("handler failed")

        async def async_failure_handler(self, exception, *args, **kwargs):
            failures.append(exception)

    logger: Final = BrokenHandler(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="broken",
        function_id="broken",
    )
    with pytest.raises(litellm.InternalServerError) as caught:
        await call_aocr(ocr_server, litellm_logging_obj=logger)
    assert failures == [caught.value, caught.value]
    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
async def test_nested_native_calls_preserve_context_and_dispatch_each_outcome(ocr_server: RecordingServer) -> None:
    ocr_server.expected_requests = 2
    recorder: Final = RecordingLogger()
    outcomes: Final = []

    class Nested(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            if kwargs.get("litellm_call_id") == "outer":
                outcomes.append(await call_aocr(ocr_server, callbacks=[recorder], litellm_call_id="inner"))

    litellm.callbacks.append(Nested())
    outcomes.append(await call_aocr(ocr_server, callbacks=[recorder], litellm_call_id="outer"))
    events: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    assert [event.kwargs["litellm_call_id"] for event in events] == ["inner", "outer"]
    assert events[0].response is outcomes[0]
    assert events[1].response is outcomes[1]
    assert len(ocr_server.requests) == 2


def test_sync_pre_call_can_make_nested_native_request(ocr_server: RecordingServer) -> None:
    ocr_server.expected_requests = 2
    observed: Final = []

    class Nested(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            if kwargs["litellm_call_id"] == "outer-sync":
                observed.append(call_ocr(ocr_server, litellm_call_id="inner-sync"))

    response: Final = call_ocr(ocr_server, callbacks=[Nested()], litellm_call_id="outer-sync")
    assert observed[0].pages[0].markdown == response.pages[0].markdown
    assert len(ocr_server.requests) == 2
