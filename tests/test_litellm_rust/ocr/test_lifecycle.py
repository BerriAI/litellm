import asyncio
import datetime
import gc
import json
import threading
import weakref
from collections.abc import Awaitable, Callable, Coroutine
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
    assert response._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
    assert events[0].kwargs["litellm_params"]["metadata"]["user_api_key_auth"].user_id == "ocr-user"
    assert "metadata" not in ocr_server.requests[0].body


@pytest.mark.asyncio
async def test_request_level_custom_pricing_reaches_logging_params_and_bills_the_call(
    ocr_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()
    response: Final = await call_aocr(ocr_server, callbacks=[recorder], ocr_cost_per_page=0.05)
    events: Final = await recorder.wait_for_async("async_log_success_event")

    assert response.usage_info is not None and response.usage_info.pages_processed == 1
    assert events[0].kwargs["litellm_params"]["ocr_cost_per_page"] == 0.05
    assert response._hidden_params["response_cost"] == pytest.approx(0.05)
    assert "ocr_cost_per_page" not in ocr_server.requests[0].body


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
@pytest.mark.parametrize("phase", ["pre", "http"])
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


def test_unstarted_native_coroutine_releases_input_without_reading_file(ocr_server: RecordingServer) -> None:
    from litellm.ocr.dispatch import _public_request
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
        coroutine: Final = _native.aocr(_public_request("aocr", (), kwargs), (), kwargs)
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
async def test_response_limit_is_enforced_at_the_public_boundary(ocr_server: RecordingServer) -> None:
    limit: Final = len(json.dumps(OCR_RESPONSE).encode()) - 1
    with pytest.raises(litellm.APIConnectionError, match="OCR response exceeds the size limit"):
        await call_aocr(ocr_server, max_response_bytes=limit)
    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_empty_callbacks_run_deployment_hooks_and_defer_like_the_python_client_wrapper(
    ocr_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
    created_loggers: list[Logging],
) -> None:
    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging, logging_worker

    class DispatchProbe:
        deployments = 0
        submissions = 0
        enqueues = 0

        def counting(self, hook: Callable[..., Awaitable[object]]) -> Callable[..., Awaitable[object]]:
            async def counted(*args: object, **kwargs: object) -> object:
                self.deployments += 1
                return await hook(*args, **kwargs)

            return counted

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
        monkeypatch.setattr(utils, name, probe.counting(getattr(utils, name)))
    monkeypatch.setattr(litellm_logging, "executor", probe)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", probe)
    if failure:
        ocr_server.enqueue(ResponseSpec(body={"message": "provider failed"}, status=500))
    trace_id_var.set("callback-free-parent")
    arguments: Final = {"litellm_trace_id": "callback-free-call", "litellm_call_id": "callback-free-id"}
    if failure:
        with pytest.raises(litellm.InternalServerError):
            await call_aocr(ocr_server, **arguments)
    else:
        response: Final = await call_aocr(ocr_server, **arguments)
        assert response.pages[0].markdown == "native OCR response"
        assert response._hidden_params["litellm_call_id"] == "callback-free-id"
        assert response._hidden_params["response_cost"] is not None
        assert response._hidden_params["_response_ms"] > 0
    assert trace_id_var.get() == "callback-free-parent"
    assert probe.deployments == 2
    assert probe.submissions == probe.enqueues == 0
    assert len(created_loggers) == 1
    logger: Final = created_loggers[0]
    if failure:
        assert logger.model_call_details["first_api_call_start_time"] <= logger.model_call_details["end_time"]
        assert logger.model_call_details["response_cost"] == 0
    else:
        assert getattr(logger, "_native_pending_logging", None) is not None
        assert "end_time" not in logger.model_call_details


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
