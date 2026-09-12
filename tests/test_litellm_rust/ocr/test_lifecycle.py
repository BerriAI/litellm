import asyncio
import datetime
import gc
import json
import sys
import threading
import weakref
from collections.abc import Coroutine
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["deployment", "failure"])
async def test_cancellation_during_failure_obeys_phase_policy(ocr_server: RecordingServer, phase: str) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider failure"}, status=500))
    entered: Final = asyncio.Event()
    observed: Final = []

    class Observer(CustomLogger):
        async def async_post_call_failure_deployment_hook(self, request_data, exception, call_type, **kwargs):
            if phase == "deployment":
                entered.set()
                await asyncio.Event().wait()

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(kwargs["exception"])
            if phase == "failure":
                entered.set()
                await asyncio.Event().wait()

    observer: Final = Observer()
    litellm.callbacks.append(observer)
    task: Final = asyncio.create_task(call_aocr(ocr_server, callbacks=[observer]))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    if phase == "deployment":
        with pytest.raises(litellm.InternalServerError) as caught:
            await task
        assert observed == [caught.value]
    else:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(observed) == 1
        assert isinstance(observed[0], litellm.InternalServerError)


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
async def test_deployment_hook_replaces_complete_routing_request(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.05))
    original: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
    replacement: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
    observed: Final = []

    class Replace(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            return {
                **kwargs,
                "model": "azure_ai/mistral-ocr-latest",
                "custom_llm_provider": "azure_ai",
                "document": replacement,
                "api_key": "replacement-key",
                "api_base": ocr_server.base_url,
                "extra_headers": {"x-deployment": "replacement"},
                "timeout": 2,
                "pages": [2],
            }

    class Observe(Logging):
        def pre_call(self, input, api_key, additional_args):
            observed.append((additional_args["complete_input_dict"]["document"], api_key))

    litellm.callbacks.append(Replace())
    logger: Final = Observe(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="deployment-routing",
        function_id="deployment-routing",
    )
    response: Final = await call_aocr(
        ocr_server,
        document=original,
        timeout=0.001,
        litellm_logging_obj=logger,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert observed == [(replacement, "replacement-key")]
    assert observed[0][0] is replacement
    assert replacement == original
    assert replacement is not original
    assert original == {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
    assert ocr_server.requests[0].path == "/providers/mistral/azure/ocr"
    assert ocr_server.requests[0].headers["authorization"] == "Bearer replacement-key"
    assert ocr_server.requests[0].headers["x-deployment"] == "replacement"
    assert ocr_server.requests[0].body["document"] == replacement
    assert ocr_server.requests[0].body["pages"] == [2]


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
@pytest.mark.parametrize("failure", [RuntimeError("native enqueue failed"), asyncio.CancelledError("cancelled")])
async def test_deferred_release_handles_enqueue_failure_once_without_replay(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    import inspect

    from litellm.litellm_core_utils import logging_worker

    attempts: Final[list[Coroutine[object, object, object]]] = []
    diagnostics: Final = []

    class FailingWorker:
        def ensure_initialized_and_enqueue(self, coroutine: Coroutine[object, object, object]) -> None:
            attempts.append(coroutine)
            raise failure

    recorder: Final = RecordingLogger()
    logger: Final = Logging(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="release-failure",
        function_id="release-failure",
        dynamic_async_success_callbacks=[recorder],
    )
    logger._defer_async_logging = True
    response: Final = await call_aocr(ocr_server, litellm_logging_obj=logger)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", FailingWorker())
    monkeypatch.setattr(sys, "unraisablehook", lambda event: diagnostics.append(event.exc_value))

    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError, match="cancelled") as caught:
            ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, False)
        assert caught.value is failure
        assert diagnostics == []
    else:
        ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, False)
        assert diagnostics == [failure]
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, False)

    assert len(attempts) == 1
    assert inspect.getcoroutinestate(attempts[0]) == inspect.CORO_CLOSED
    assert response.pages[0].markdown == "native OCR response"
    assert len(ocr_server.requests) == 1
    assert not any("success" in name or "failure" in name for name in recorder.names)


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


@pytest.mark.asyncio
async def test_retained_argument_aliases_and_body_roots_survive_envelope_replacement(
    ocr_server: RecordingServer,
) -> None:
    pages: Final = [0]
    document: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
    opaque: Final = object()
    observed: Final = []

    class Observe(Logging):
        def pre_call(self, input, api_key, additional_args):
            body: Final = additional_args["complete_input_dict"]
            headers: Final = additional_args["headers"]
            observed.append((body["document"] is document, body["pages"] is pages))
            pages.append(2)
            headers["x-retained"] = "yes"
            additional_args["complete_input_dict"] = {"discarded": True}
            additional_args["headers"] = {}
            observed.append((body, headers))

        def post_call(self, original_response, additional_args):
            observed.append(
                (additional_args["complete_input_dict"] is observed[2][0], additional_args["headers"] is observed[2][1])
            )

    class Deployment(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            observed.append(("model" in kwargs, "document" in kwargs, kwargs["opaque"] is opaque))

    litellm.callbacks.append(Deployment())
    logger: Final = Observe(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="roots",
        function_id="roots",
    )
    response: Final = await litellm.aocr(
        "mistral/mistral-ocr-latest",
        document,
        api_key="test-key",
        api_base=ocr_server.base_url,
        pages=pages,
        opaque=opaque,
        litellm_logging_obj=logger,
    )
    assert response.pages[0].markdown == "native OCR response"
    assert observed[0] == (False, False, True)
    assert observed[1] == (True, True)
    assert observed[3] == (True, True)
    assert ocr_server.requests[0].body["pages"] == [0, 2]
    assert ocr_server.requests[0].headers["x-retained"] == "yes"


def test_unstarted_native_coroutine_releases_input_without_reading_file(ocr_server: RecordingServer) -> None:
    from litellm.ocr.main import _public_request
    from litellm.rust_bridge import _native

    ocr_server.expected_requests = 0
    effects: Final = []

    class File:
        def read(self):
            effects.append("read")
            return b"abc"

    def create():
        file: Final = File()
        kwargs: Final = {"model": "mistral/mistral-ocr-latest", "document": {"type": "file", "file": file}}
        coroutine: Final = _native._ocr_lifecycle(_public_request("aocr", (), kwargs), (), kwargs, True)
        file.owner = coroutine
        coroutine.close()
        return weakref.ref(file)

    reference: Final = create()
    gc.collect()
    assert reference() is None
    assert effects == []


@pytest.mark.asyncio
async def test_file_read_happens_after_deployment_hook_in_caller_task(ocr_server: RecordingServer) -> None:
    effects: Final = []
    caller: Final = asyncio.current_task()

    class File:
        def read(self):
            effects.append(("read", asyncio.current_task()))
            return b"abc"

    class Deployment(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            await asyncio.sleep(0)
            effects.append(("hook", asyncio.current_task()))

    litellm.callbacks.append(Deployment())
    await call_aocr(ocr_server, document={"type": "file", "file": File()})
    assert effects == [("hook", caller), ("read", caller)]


@pytest.mark.asyncio
async def test_failure_callbacks_continue_within_both_families(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "failed"}, status=500))
    observed: Final = []

    class Broken(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("broken-sync", kwargs["exception"]))
            raise RuntimeError("sync observer")

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("broken-async", kwargs["exception"]))
            raise RuntimeError("async observer")

    class Following(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("following-sync", kwargs["exception"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("following-async", kwargs["exception"]))

    with pytest.raises(litellm.InternalServerError) as caught:
        await call_aocr(ocr_server, callbacks=[Broken(), Following()])
    assert [name for name, _ in observed] == ["broken-sync", "following-sync", "broken-async", "following-async"]
    assert all(error is caught.value for _, error in observed)


@pytest.mark.asyncio
async def test_cancelling_native_transport_closes_connection_before_return() -> None:
    received: Final = asyncio.Event()
    disconnected: Final = asyncio.Event()

    async def provider(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        headers: Final = await reader.readuntil(b"\r\n\r\n")
        length: Final = next(
            int(line.split(b":", 1)[1])
            for line in headers.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
        await reader.readexactly(length)
        received.set()
        assert await reader.read() == b""
        disconnected.set()
        writer.close()
        await writer.wait_closed()

    server: Final = await asyncio.start_server(provider, "127.0.0.1", 0)
    async with server:
        port: Final = server.sockets[0].getsockname()[1]
        task: Final = asyncio.create_task(
            litellm.aocr(
                model="mistral/mistral-ocr-latest",
                document={"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                api_key="test-key",
                api_base=f"http://127.0.0.1:{port}",
            )
        )
        await asyncio.wait_for(received.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(disconnected.wait(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["reducto/parse-v3", "reducto/parse-legacy"])
async def test_reducto_lifecycle_retains_upload_parse_and_post_call_boundaries(
    ocr_server: RecordingServer, model: str
) -> None:
    ocr_server.expected_requests = 2
    ocr_server.enqueue(ResponseSpec(body={"file_id": "reducto://uploaded.pdf"}))
    ocr_server.enqueue(ResponseSpec(body={"result": {"chunks": [{"content": "parsed"}]}}))
    boundaries: Final = []
    recorder: Final = RecordingLogger()

    class Observe(Logging):
        def post_call(self, *args, **kwargs):
            boundaries.append(tuple(request.path for request in ocr_server.requests))
            return super().post_call(*args, **kwargs)

    logger: Final = Observe(
        model=model,
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="upload",
        function_id="upload",
        dynamic_async_success_callbacks=[recorder],
    )
    response: Final = await call_aocr(ocr_server, model=model, litellm_logging_obj=logger)
    events: Final = await recorder.wait_for_async("async_log_success_event")
    assert boundaries == [("/upload", "/parse")]
    assert b"abc" in ocr_server.requests[0].raw_body
    assert "multipart/form-data" in ocr_server.requests[0].headers["content-type"]
    assert ocr_server.requests[1].body["input" if model.endswith("v3") else "document_url"] == "reducto://uploaded.pdf"
    assert response.pages[0].markdown == "parsed"
    assert events[0].response is response


@pytest.mark.asyncio
async def test_document_intelligence_post_call_observes_submission_and_final_result(
    ocr_server: RecordingServer,
) -> None:
    ocr_server.expected_requests = 2
    ocr_server.enqueue(
        ResponseSpec(
            body={"status": "running"},
            status=202,
            headers={"Operation-Location": f"{ocr_server.base_url}/operations/1", "Retry-After": "0"},
        )
    )
    ocr_server.enqueue(ResponseSpec(body={"status": "succeeded", "analyzeResult": {"pages": []}}))
    boundaries: Final = []

    class Observe(Logging):
        def post_call(self, *args, **kwargs):
            boundaries.append((tuple(request.method for request in ocr_server.requests), kwargs["original_response"]))
            return super().post_call(*args, **kwargs)

    logger: Final = Observe(
        model="azure_ai/doc-intelligence/prebuilt-read",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="poll",
        function_id="poll",
    )
    response: Final = await call_aocr(
        ocr_server, model="azure_ai/doc-intelligence/prebuilt-read", litellm_logging_obj=logger
    )
    assert [methods for methods, _ in boundaries] == [("POST",), ("POST", "GET")]
    assert json.loads(boundaries[0][1])["status"] == "running"
    assert json.loads(boundaries[1][1])["status"] == "succeeded"
    assert [request.method for request in ocr_server.requests] == ["POST", "GET"]
    assert ocr_server.requests[1].path == "/operations/1"
    assert response.pages == []


@pytest.mark.asyncio
async def test_vertex_deepseek_public_lifecycle_normalizes_before_success(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(
        ResponseSpec(body={"choices": [{"message": {"content": "recognized"}}], "usage": {"prompt_tokens": 1}})
    )
    recorder: Final = RecordingLogger()
    response: Final = await call_aocr(
        ocr_server,
        model="vertex_ai/deepseek-ocr-maas",
        document={"type": "document_url", "document_url": "gs://bucket/document.pdf"},
        vertex_project="project-1",
        vertex_location="europe-west4",
        callbacks=[recorder],
    )
    events: Final = await recorder.wait_for_async("async_log_success_event")
    assert response.pages[0].markdown == "recognized"
    assert events[0].response is response
    assert (
        ocr_server.requests[0].path
        == "/v1/projects/project-1/locations/europe-west4/endpoints/openapi/chat/completions"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit", ["budget", "retries"])
async def test_shared_call_limits_still_reject_before_reading_ocr_file(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, limit: str
) -> None:
    ocr_server.expected_requests = 0
    reads: Final = []

    class File:
        def read(self):
            reads.append("read")
            return b"abc"

    monkeypatch.setattr(litellm, "max_budget", 1 if limit == "budget" else None)
    monkeypatch.setattr(litellm, "_current_cost", 2)
    monkeypatch.setattr(litellm, "num_retries_per_request", 1 if limit == "retries" else None)
    expected: Final = litellm.BudgetExceededError if limit == "budget" else RuntimeError
    arguments: Final = {"document": {"type": "file", "file": File()}, "metadata": {"previous_models": ["earlier"]}}
    with pytest.raises(expected, match=r"Budget has been exceeded|Max retries per request hit"):
        await call_aocr(ocr_server, **arguments) if asynchronous else call_ocr(ocr_server, **arguments)
    assert reads == []
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("extra_bytes", [0, 1])
async def test_response_limit_is_enforced_at_the_public_boundary(
    ocr_server: RecordingServer, asynchronous: bool, extra_bytes: int
) -> None:
    limit: Final = len(json.dumps(OCR_RESPONSE).encode()) - extra_bytes
    if extra_bytes:
        with pytest.raises(litellm.APIConnectionError, match="OCR response exceeds the size limit"):
            await call_aocr(ocr_server, max_response_bytes=limit) if asynchronous else call_ocr(
                ocr_server, max_response_bytes=limit
            )
    else:
        response: Final = (
            await call_aocr(ocr_server, max_response_bytes=limit)
            if asynchronous
            else call_ocr(ocr_server, max_response_bytes=limit)
        )
        assert response.pages[0].markdown == "native OCR response"
    assert len(ocr_server.requests) == 1
    body: Final = ocr_server.requests[0].body
    assert isinstance(body, dict)
    assert "max_response_bytes" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_empty_callbacks_keep_bookkeeping_without_optional_dispatch(
    ocr_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    failure: bool,
    created_loggers: list[Logging],
) -> None:
    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging, logging_worker

    class DispatchProbe:
        deployments = 0
        submissions = 0
        enqueues = 0

        def deployment(self, *args: object, **kwargs: object) -> None:
            self.deployments += 1

        def submit(self, *args: object, **kwargs: object) -> None:
            self.submissions += 1

        def ensure_initialized_and_enqueue(self, coroutine: Coroutine[object, object, object]) -> None:
            self.enqueues += 1
            coroutine.close()

    probe: Final = DispatchProbe()
    for name in (
        "async_pre_call_deployment_hook",
        "async_post_call_success_deployment_hook",
        "async_post_call_failure_deployment_hook",
    ):
        monkeypatch.setattr(utils, name, probe.deployment)
    monkeypatch.setattr(litellm_logging, "executor", probe)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", probe)
    if failure:
        ocr_server.enqueue(ResponseSpec(body={"message": "provider failed"}, status=500))
    trace_id_var.set("callback-free-parent")
    arguments: Final = {"litellm_trace_id": "callback-free-call", "litellm_call_id": "callback-free-id"}
    if failure:
        with pytest.raises(litellm.InternalServerError):
            await call_aocr(ocr_server, **arguments) if asynchronous else call_ocr(ocr_server, **arguments)
    else:
        response: Final = (
            await call_aocr(ocr_server, **arguments) if asynchronous else call_ocr(ocr_server, **arguments)
        )
        assert response.pages[0].markdown == "native OCR response"
        assert response._hidden_params["litellm_call_id"] == "callback-free-id"
        assert response._hidden_params["response_cost"] is not None
        assert response._hidden_params["_response_ms"] > 0
    assert trace_id_var.get() == "callback-free-parent"
    assert probe.deployments == probe.submissions == probe.enqueues == 0
    assert len(created_loggers) == 1
    logger: Final = created_loggers[0]
    assert not hasattr(logger, "_native_pending_logging")
    assert logger.model_call_details["first_api_call_start_time"] <= logger.model_call_details["end_time"]
    assert "standard_logging_object" not in logger.model_call_details
    assert (
        "original_response" not in logger.model_call_details or logger.model_call_details["original_response"] is None
    )
    assert "complete_input_dict" not in logger.model_call_details.get("additional_args", {})
    assert logger.model_call_details["response_cost"] == (0 if failure else response._hidden_params["response_cost"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registration", ["success_callback", "_async_success_callback", "failure_callback", "_async_failure_callback"]
)
async def test_terminal_registration_added_during_http_is_observed(
    ocr_server: RecordingServer, registration: str
) -> None:
    failure: Final = "failure" in registration
    observer: Final = RecordingLogger()
    ocr_server.enqueue(
        ResponseSpec(
            body={"message": "provider failed"} if failure else OCR_RESPONSE, status=500 if failure else 200, delay=0.1
        )
    )
    task: Final = asyncio.create_task(
        asyncio.to_thread(call_ocr, ocr_server) if registration == "success_callback" else call_aocr(ocr_server)
    )
    await ocr_server.wait_for_requests(1)
    getattr(litellm, registration).append(observer)
    if failure:
        with pytest.raises(litellm.InternalServerError):
            await task
    else:
        await task
    event: Final = ("async_" if registration.startswith("_async") else "") + (
        "log_failure_event" if failure else "log_success_event"
    )
    await observer.wait_for_async(event)
    assert event in observer.names


@pytest.fixture
def created_loggers(monkeypatch: pytest.MonkeyPatch) -> list[Logging]:
    from litellm import utils

    original_setup: Final = utils.function_setup
    loggers: Final[list[Logging]] = []

    def setup(
        call_type: str,
        rules: utils.Rules,
        start: datetime.datetime,
        *args: object,
        is_async_call: bool = True,
        **kwargs: object,
    ) -> tuple[Logging, dict[str, object]]:
        logger, prepared = original_setup(call_type, rules, start, *args, is_async_call=is_async_call, **kwargs)
        assert isinstance(logger, Logging)
        setattr(logger, "_defer_async_logging", True)
        loggers.append(logger)
        return logger, prepared

    monkeypatch.setattr(utils, "function_setup", setup)
    return loggers


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["logger_fn", "raw_global", "request_debug"])
async def test_explicit_logging_consumers_keep_request_and_response_payloads(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, created_loggers: list[Logging], consumer: str
) -> None:
    snapshots: Final[list[dict[str, object]]] = []
    if consumer == "raw_global":
        monkeypatch.setattr(litellm, "log_raw_request_response", True)
    arguments: Final = {
        "logger_fn": {"logger_fn": lambda details: snapshots.append(dict(details))},
        "raw_global": {},
        "request_debug": {"litellm_request_debug": True},
    }[consumer]
    response: Final = await call_aocr(ocr_server, **arguments)
    details: Final = created_loggers[0].model_call_details
    assert details["additional_args"]["complete_input_dict"]["model"] == "mistral-ocr-latest"
    assert json.loads(details["original_response"])["pages"][0]["markdown"] == response.pages[0].markdown
    if consumer.startswith("raw_"):
        assert details["raw_request_typed_dict"]["raw_request_body"]["model"] == "mistral-ocr-latest"
    if consumer == "logger_fn":
        assert [item["log_event_type"] for item in snapshots] == ["pre_api_call", "post_api_call"]


@pytest.mark.asyncio
async def test_registration_removed_before_deferred_release_skips_queue(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, created_loggers: list[Logging]
) -> None:
    from litellm.litellm_core_utils import logging_worker

    class QueueProbe:
        enqueues = 0

        def ensure_initialized_and_enqueue(self, coroutine: Coroutine[object, object, object]) -> None:
            self.enqueues += 1
            coroutine.close()

    observer: Final = RecordingLogger()
    litellm._async_success_callback.append(observer)
    await call_aocr(ocr_server)
    logger: Final = created_loggers[0]
    assert hasattr(logger, "_native_pending_logging")
    litellm._async_success_callback.clear()
    probe: Final = QueueProbe()
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", probe)
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, False)
    assert probe.enqueues == 0
    assert not observer.names
    assert logger.model_call_details["response_cost"] is not None
