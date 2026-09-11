import asyncio
import copy
import queue
import threading
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.requests import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
    request_body,
    request_headers,
)
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_pre_call_retains_context_and_allows_nested_call(
    ocr_server: RecordingServer, asynchronous: bool
) -> None:
    from contextvars import ContextVar
    from litellm.rust_bridge import _native

    context: Final = ContextVar("ocr-callback-context", default="missing")
    context.set("caller")
    caller_thread: Final = threading.current_thread()
    observations: Final = []
    ocr_server.expected_requests = 2

    class NestedCall(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((context.get(), threading.current_thread()))
            nested: Final = _native.ocr(
                model="mistral/mistral-ocr-latest",
                document=OCR_DOCUMENT,
                api_key="test-key",
                api_base=ocr_server.base_url,
            )
            observations.append(nested["pages"][0]["markdown"])
            request_headers(kwargs)["x-outer"] = "retained"

    if asynchronous:
        await call_native_aocr(ocr_server, callbacks=[NestedCall()])
    else:
        call_native_ocr(ocr_server, callbacks=[NestedCall()])

    assert observations == [("caller", caller_thread), "native OCR response"]
    assert "x-outer" not in ocr_server.requests[0].headers
    assert ocr_server.requests[1].headers["x-outer"] == "retained"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_binding_drives_pre_call_and_preserves_host_exception(
    ocr_server: RecordingServer, asynchronous: bool
) -> None:
    from litellm.rust_bridge import _native

    class Abort(BaseException):
        pass

    error: Final = Abort("stop before transport")
    original: Final = dict(OCR_DOCUMENT)
    observed: Final = []
    ocr_server.expected_requests = 0

    class HostLogger:
        def pre_call(self, *, input, api_key, model, additional_args):
            observed.append(additional_args["complete_input_dict"]["document"])
            raise error

    arguments: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": original,
        "api_key": "test-key",
        "api_base": ocr_server.base_url,
        "logging_obj": HostLogger(),
    }
    async def invoke() -> None:
        if asynchronous:
            await _native.aocr(**arguments, callback_loop=asyncio.get_running_loop())
        else:
            _native.ocr(**arguments)

    with pytest.raises(Abort) as caught:
        await invoke()

    assert observed == [original]
    assert observed[0] is original
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("rust_enabled", [False, True], ids=["python", "rust"])
async def test_callback_reference_and_response_identity_match_python(
    ocr_server: RecordingServer, rust_enabled: bool
) -> None:
    from tests.test_litellm_rust.support.requests import call_aocr

    original: Final = dict(OCR_DOCUMENT)
    token: Final = object()
    observed: Final = []
    finished: Final = asyncio.Event()

    class Logger(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs)["document"] is original)
            kwargs["retained-token"] = token
            request_body(kwargs)["include_image_base64"] = True

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(kwargs["retained-token"] is token)
            observed.append(response_obj)
            finished.set()

    litellm.rust(rust_enabled)
    response: Final = await call_aocr(ocr_server, document=original, callbacks=[Logger()])
    await asyncio.wait_for(finished.wait(), timeout=10)
    assert observed[:2] == [True, True]
    assert observed[2] is response
    assert ocr_server.requests[0].body["include_image_base64"] is True


@pytest.mark.asyncio
async def test_token_provider_document_edit_keeps_live_reference(ocr_server: RecordingServer) -> None:
    original: Final = dict(OCR_DOCUMENT)
    observed: Final = []

    def provider() -> str:
        original["document_url"] = "data:application/pdf;base64,ZGVm"
        return "token"

    class Logger(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs)["document"] is original)

    await call_native_aocr(
        ocr_server,
        model="azure_ai/mistral-ocr-latest",
        document=original,
        api_key=None,
        azure_ad_token_provider=provider,
        callbacks=[Logger()],
    )

    assert observed == [True]
    assert ocr_server.requests[0].body["document"] == original


def call_native_ocr_with_callbacks(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return call_native_ocr(server, callbacks=callbacks, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_callback_task_and_context_boundaries(
    ocr_server: RecordingServer, asynchronous: bool
) -> None:
    from contextvars import ContextVar

    context: Final = ContextVar("ocr-task-boundaries", default="missing")
    context.set("caller")
    caller: Final = asyncio.current_task()
    observations: Final = []

    def provider() -> str:
        observations.append(("token", asyncio.current_task(), context.get()))
        context.set("token")
        return "token"

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append(("pre", asyncio.current_task(), context.get()))
            context.set("pre")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append(("next", asyncio.current_task(), context.get()))

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": provider,
        "callbacks": [Edit(), Observe()],
    }
    if asynchronous:
        await call_native_aocr(ocr_server, **arguments)
    else:
        call_native_ocr(ocr_server, **arguments)

    assert [phase for phase, _, _ in observations] == ["token", "pre", "next"]
    assert [value for _, _, value in observations] == ["caller", "caller" if asynchronous else "token", "pre"]
    token_task: Final = observations[0][1]
    pre_task: Final = observations[1][1]
    assert observations[2][1] is pre_task
    if asynchronous:
        assert token_task is not None and pre_task is not None
        assert token_task is not caller and pre_task is not caller
        assert token_task is not pre_task
        assert token_task.done() and pre_task.done()
        assert context.get() == "caller"
    else:
        assert token_task is caller and pre_task is caller
        assert context.get() == "pre"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["token", "pre_call"])
async def test_native_cancellation_cancels_pending_callback_and_releases_references(
    ocr_server: RecordingServer, phase: str
) -> None:
    import gc
    import weakref

    from litellm.rust_bridge import _native

    loop: Final = asyncio.get_running_loop()
    previous_factory: Final = loop.get_task_factory()
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()
    tasks: Final = []
    calls: Final = []
    ocr_server.expected_requests = 0

    async def gate(coroutine):
        try:
            entered.set()
            await release.wait()
            return await coroutine
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            coroutine.close()

    def factory(event_loop, coroutine, **kwargs):
        if getattr(getattr(coroutine, "cr_code", None), "co_name", None) == "invoke_callback":
            task: Final = asyncio.Task(gate(coroutine), loop=event_loop, **kwargs)
            tasks.append(task)
            return task
        if previous_factory is not None:
            return previous_factory(event_loop, coroutine, **kwargs)
        return asyncio.Task(coroutine, loop=event_loop, **kwargs)

    class Host:
        def __call__(self):
            calls.append("token")
            return "token"

        def pre_call(self, **kwargs):
            calls.append("pre_call")

    def start():
        host: Final = Host()
        reference: Final = weakref.ref(host)
        future: Final = _native.aocr(
            model="azure_ai/mistral-ocr-latest",
            document=OCR_DOCUMENT,
            api_key=None if phase == "token" else "test-key",
            api_base=ocr_server.base_url,
            logging_obj=host,
            token_provider=host if phase == "token" else None,
            callback_loop=loop,
        )
        return future, reference

    loop.set_task_factory(factory)
    try:
        future, reference = start()
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert reference() is not None
        assert len(tasks) == 1
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await future
        await asyncio.wait_for(cancelled.wait(), timeout=5)
        await asyncio.gather(*tasks, return_exceptions=True)
        assert tasks[0].cancelled()
        tasks.clear()
        async with asyncio.timeout(5):
            while reference() is not None:
                gc.collect()
                await asyncio.sleep(0.01)
        assert calls == []
        assert ocr_server.requests == []
    finally:
        loop.set_task_factory(previous_factory)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["token", "pre_call"])
async def test_native_callback_cancelled_error_aborts_before_provider(
    ocr_server: RecordingServer, phase: str
) -> None:
    from litellm.rust_bridge import _native

    calls: Final = []
    ocr_server.expected_requests = 0

    class Host:
        def __call__(self):
            calls.append("token")
            raise asyncio.CancelledError("callback cancelled")

        def pre_call(self, **kwargs):
            calls.append("pre_call")
            raise asyncio.CancelledError("callback cancelled")

    host: Final = Host()
    future: Final = _native.aocr(
        model="azure_ai/mistral-ocr-latest",
        document=OCR_DOCUMENT,
        api_key=None if phase == "token" else "test-key",
        api_base=ocr_server.base_url,
        logging_obj=host,
        token_provider=host if phase == "token" else None,
        callback_loop=asyncio.get_running_loop(),
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(future, timeout=5)
    assert calls == [phase]
    assert ocr_server.requests == []


async def call_native_aocr_with_callbacks(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return await call_native_aocr(server, callbacks=callbacks, **kwargs)


def test_native_ocr_pre_call_callback_receives_transformed_provider_request(ocr_server: RecordingServer) -> None:
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((model, copy.deepcopy(kwargs["additional_args"])))

    call_native_ocr_with_callbacks(ocr_server, [Observe()], pages=[0])

    assert len(observations) == 1
    model, additional_args = observations[0]
    assert model == "mistral-ocr-latest"
    assert additional_args["api_base"] == f"{ocr_server.base_url}/v1/ocr"
    assert additional_args["complete_input_dict"] == {
        "model": "mistral-ocr-latest",
        "document": OCR_DOCUMENT,
        "pages": [0],
    }


@pytest.mark.parametrize("raise_after_edit", [False, True], ids=["callback-returns", "callback-raises"])
def test_native_ocr_pre_call_body_edit_reaches_next_callback_and_provider(
    ocr_server: RecordingServer, raise_after_edit: bool
) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["include_image_base64"] = True
            if raise_after_edit:
                raise RuntimeError("pre-call callback failed")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(copy.deepcopy(request_body(kwargs)))

    call_native_ocr_with_callbacks(ocr_server, [Edit(), Observe()], include_image_base64=False)

    assert observed[0]["include_image_base64"] is True
    assert ocr_server.requests[0].body["include_image_base64"] is True


def test_native_ocr_pre_call_header_edit_reaches_next_callback_and_provider(ocr_server: RecordingServer) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_headers(kwargs)["x-audit-tag"] = "reviewed"

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(dict(request_headers(kwargs)))

    call_native_ocr_with_callbacks(ocr_server, [Edit(), Observe()])

    assert observed[0]["x-audit-tag"] == "reviewed"
    assert ocr_server.requests[0].headers["x-audit-tag"] == "reviewed"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ocr_pre_call_nested_document_edit_updates_caller_callback_and_provider_references(
    ocr_server: RecordingServer, asynchronous: bool
) -> None:
    original: Final = dict(OCR_DOCUMENT)
    replacement_url: Final = "data:application/pdf;base64,ZGVm"
    retained: Final = []
    aliases: Final = []

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            aliases.append(request_body(kwargs)["document"] is original)
            retained.append(request_body(kwargs)["document"])

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            original["document_url"] = replacement_url

    arguments: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": original,
        "api_key": "test-key",
        "api_base": ocr_server.base_url,
        "callbacks": [Retain(), Edit()],
    }
    response: Final = (
        await call_native_aocr(ocr_server, **arguments)
        if asynchronous
        else call_native_ocr(ocr_server, **arguments)
    )

    assert aliases == [True]
    assert retained[0]["document_url"] == replacement_url
    assert original["document_url"] == replacement_url
    assert ocr_server.requests[0].body["document"]["document_url"] == replacement_url
    assert response.pages[0].markdown == "native OCR response"


def test_native_ocr_pre_call_document_replacement_does_not_mutate_original_document(
    ocr_server: RecordingServer,
) -> None:
    original: Final = dict(OCR_DOCUMENT)
    replacement: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,ZGVm"}
    retained: Final = []

    class RetainAndReplace(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = request_body(kwargs)
            retained.append(body["document"])
            body["document"] = replacement

    call_native_ocr(
        ocr_server,
        document=original,
        callbacks=[RetainAndReplace()],
    )

    assert retained[0] is original
    assert original["document_url"] == OCR_DOCUMENT["document_url"]
    assert ocr_server.requests[0].body["document"] == replacement


def test_native_ocr_pre_call_body_rebinding_is_visible_to_callbacks_but_not_provider(
    ocr_server: RecordingServer,
) -> None:
    observed: Final = []

    class Rebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs))

    call_native_ocr_with_callbacks(ocr_server, [Rebind(), Observe()])

    assert observed == [{"replacement": True}]
    assert ocr_server.requests[0].body == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT}


def test_native_ocr_callback_retained_body_observes_later_callback_mutation(ocr_server: RecordingServer) -> None:
    queued: Final = []

    class QueuePayload(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            queued.append(request_body(kwargs))

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["queued-edit"] = True

    call_native_ocr_with_callbacks(ocr_server, [QueuePayload(), Edit()])

    assert queued[0]["queued-edit"] is True


def test_native_ocr_success_callback_receives_state_added_by_pre_call_callback(ocr_server: RecordingServer) -> None:
    token: Final = object()
    terminal_tokens: queue.SimpleQueue[object] = queue.SimpleQueue()
    finished: Final = threading.Event()

    class Stash(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["test-token"] = token

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            terminal_tokens.put(kwargs["test-token"])
            finished.set()

    call_native_ocr_with_callbacks(ocr_server, [Stash()])

    assert finished.wait(10)
    assert terminal_tokens.get_nowait() is token


@pytest.mark.asyncio
async def test_native_aocr_success_callback_receives_call_id_metadata_and_response(
    ocr_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    await call_native_aocr_with_callbacks(
        ocr_server,
        [recorder],
        litellm_call_id="ocr-success",
        metadata={"source": "callback-test"},
    )
    events: Final = await recorder.wait_for_async("async_log_success_event")

    assert len(events) == 1
    assert events[0].call_type == "aocr"
    assert events[0].kwargs["litellm_call_id"] == "ocr-success"
    assert events[0].kwargs["litellm_params"]["metadata"]["source"] == "callback-test"
    assert events[0].response.pages[0].markdown == "native OCR response"


@pytest.mark.asyncio
async def test_native_aocr_failure_callbacks_receive_call_type_error_and_no_response(
    ocr_server: RecordingServer,
) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider unavailable"}, status=500))
    observations: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("sync", kwargs["call_type"], kwargs["exception"], response_obj))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("async", kwargs["call_type"], kwargs["exception"], response_obj))

    with pytest.raises(litellm.InternalServerError):
        await call_native_aocr_with_callbacks(ocr_server, [Observe()])

    assert [observation[0] for observation in observations] == ["sync", "async"]
    assert all(observation[1] == "aocr" for observation in observations)
    assert all(isinstance(observation[2], litellm.InternalServerError) for observation in observations)
    assert all(observation[3] is None for observation in observations)


@pytest.mark.asyncio
async def test_native_aocr_pre_call_callback_runs_on_caller_loop_and_thread(ocr_server: RecordingServer) -> None:
    caller_loop: Final = asyncio.get_running_loop()
    caller_thread: Final = threading.current_thread()
    recorder: Final = RecordingLogger()

    await call_native_aocr_with_callbacks(ocr_server, [recorder])

    events: Final = await recorder.wait_for_async("log_pre_api_call")
    assert len(events) == 1
    assert events[0].loop is caller_loop
    assert events[0].thread is caller_thread


@pytest.mark.asyncio
async def test_native_aocr_failure_callbacks_receive_state_added_by_pre_call_callback(
    ocr_server: RecordingServer,
) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider unavailable"}, status=500))
    token: Final = object()
    observed: Final = []

    class TrackInFlightRequest(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["request-token"] = token

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("sync", kwargs["request-token"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("async", kwargs["request-token"]))

    with pytest.raises(litellm.InternalServerError):
        await call_native_aocr_with_callbacks(ocr_server, [TrackInFlightRequest()])

    assert [event for event, _ in observed] == ["sync", "async"]
    assert all(observed_token is token for _, observed_token in observed)


@pytest.mark.asyncio
async def test_native_aocr_callback_error_does_not_mask_provider_error_or_skip_later_failure_callbacks(
    ocr_server: RecordingServer,
) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider unavailable"}, status=500))
    recorder: Final = RecordingLogger()

    class FailingCallback(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            raise RuntimeError("failure callback failed")

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            raise RuntimeError("failure callback failed")

    with pytest.raises(litellm.InternalServerError) as caught:
        await call_native_aocr_with_callbacks(ocr_server, [FailingCallback(), recorder])

    sync_events: Final = tuple(event for event in recorder.events if event.name == "log_failure_event")
    async_events: Final = tuple(event for event in recorder.events if event.name == "async_log_failure_event")
    assert len(sync_events) == 1
    assert len(async_events) == 1
    assert sync_events[0].kwargs["exception"] is caught.value
    assert async_events[0].kwargs["exception"] is caught.value
    assert "async_log_success_event" not in recorder.names


def test_native_ocr_dispatches_each_callback_phase_once_when_logger_is_registered_multiple_times(
    ocr_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    call_native_ocr_with_callbacks(
        ocr_server,
        [recorder, recorder],
        success_callback=[recorder],
        failure_callback=[recorder],
    )
    recorder.wait_for("log_success_event")

    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count("logging_hook") == 1
    assert recorder.names.count("log_success_event") == 1
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_resolves_token_before_pre_call_on_caller_context(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
) -> None:
    from contextvars import ContextVar
    context: Final = ContextVar("azure-token-context", default="missing")
    context.set("caller")
    caller_thread: Final = threading.current_thread()
    caller_loop: Final = asyncio.get_running_loop()
    observations: Final = []

    class Provider:
        def __call__(self) -> str:
            assert context.get() == "caller"
            assert threading.current_thread() is caller_thread
            assert asyncio.get_running_loop() is caller_loop
            observations.append("token")
            return "caller-token"

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            assert request_headers(kwargs)["Authorization"] == "Bearer caller-token"
            observations.append("pre_call")
            request_headers(kwargs)["Authorization"] = "Bearer edited"

    provider: Final = Provider()
    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": provider,
        "callbacks": [Edit()],
    }
    response: Final = (
        await call_native_aocr(ocr_server, **arguments)
        if asynchronous
        else call_native_ocr(ocr_server, **arguments)
    )
    assert response.pages[0].markdown == "native OCR response"
    assert observations == ["token", "pre_call"]
    assert ocr_server.requests[0].headers["authorization"] == "Bearer edited"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_token_provider_can_make_nested_native_ocr_call(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
) -> None:
    ocr_server.expected_requests = 2
    calls: Final = []

    def provider() -> str:
        calls.append("token")
        nested: Final = call_native_ocr(ocr_server)
        assert nested.pages[0].markdown == "native OCR response"
        return "outer-token"

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": provider,
    }
    response: Final = (
        await call_native_aocr(ocr_server, **arguments)
        if asynchronous
        else call_native_ocr(ocr_server, **arguments)
    )
    assert response.pages[0].markdown == "native OCR response"
    assert calls == ["token"]
    assert [request.headers["authorization"] for request in ocr_server.requests] == [
        "Bearer test-key",
        "Bearer outer-token",
    ]


@pytest.mark.asyncio
async def test_concurrent_native_azure_ocr_calls_isolate_token_results_and_error(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
) -> None:
    ocr_server.expected_requests = 2

    async def request(token: str, fail: bool) -> object:
        def provider() -> str:
            if fail:
                raise ValueError(token)
            return token

        return await call_native_aocr(
            ocr_server,
            model="azure_ai/mistral-ocr-latest",
            api_key=None,
            azure_ad_token_provider=provider,
        )

    responses: Final = await asyncio.gather(
        request("first", False),
        request("failed", True),
        request("second", False),
        return_exceptions=True,
    )
    assert isinstance(responses[0], OCRResponse)
    assert isinstance(responses[1], litellm.APIConnectionError)
    assert "Failed to get Azure AD token: failed" in str(responses[1])
    assert isinstance(responses[2], OCRResponse)
    assert sorted(request.headers["authorization"] for request in ocr_server.requests) == [
        "Bearer first",
        "Bearer second",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancellation"])
async def test_native_azure_ocr_releases_token_provider_after_terminal_outcome(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    outcome: str,
) -> None:
    import gc
    import weakref
    from tests.test_litellm_rust.support.callback_recorder import drain_logging
    class Provider:
        def __call__(self) -> str:
            if outcome == "failure":
                raise ValueError("unavailable")
            return "caller-token"

    async def invoke() -> weakref.ReferenceType[Provider]:
        provider: Final = Provider()
        reference: Final = weakref.ref(provider)
        if outcome == "failure":
            ocr_server.expected_requests = 0
            with pytest.raises(litellm.APIConnectionError):
                await call_native_aocr(
                    ocr_server, model="azure_ai/mistral-ocr-latest", api_key=None, azure_ad_token_provider=provider
                )
        elif outcome == "cancellation":
            ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.1))
            task: Final = asyncio.create_task(
                call_native_aocr(
                    ocr_server,
                    model="azure_ai/mistral-ocr-latest",
                    api_key=None,
                    azure_ad_token_provider=provider,
                )
            )
            await ocr_server.wait_for_requests(1)
            assert reference() is provider
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            response: Final = await call_native_aocr(
                ocr_server,
                model="azure_ai/mistral-ocr-latest",
                api_key=None,
                azure_ad_token_provider=provider,
            )
            assert response.pages[0].markdown == "native OCR response"
        return reference

    reference: Final = await invoke()
    await drain_logging()
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None
