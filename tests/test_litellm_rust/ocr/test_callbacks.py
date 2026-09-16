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
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
    request_body,
    request_headers,
)

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def call_native_ocr_with_callbacks(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return call_native_ocr(server, callbacks=callbacks, **kwargs)


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


def test_native_ocr_pre_call_header_rebinding_does_not_replace_execution_root(ocr_server: RecordingServer) -> None:
    retained: Final = []
    observed: Final = []

    class RetainMutateAndRebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            headers = request_headers(kwargs)
            retained.append(headers)
            kwargs["additional_args"]["headers"] = {"x-rebound": "not-sent"}
            headers["x-retained"] = "sent"

    class ObserveRebinding(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(dict(request_headers(kwargs)))

    call_native_ocr_with_callbacks(ocr_server, [RetainMutateAndRebind(), ObserveRebinding()])

    assert observed == [{"x-rebound": "not-sent"}]
    assert retained[0]["x-retained"] == "sent"
    assert ocr_server.requests[0].headers["x-retained"] == "sent"
    assert "x-rebound" not in ocr_server.requests[0].headers


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
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
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
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
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


FORBIDDEN_ORCHESTRATION: Final = (
    "pre_call",
    "post_call",
    "success_handler",
    "async_success_handler",
    "failure_handler",
    "async_failure_handler",
    "_success_handler_body",
    "_async_success_handler_body",
    "_failure_handler_body",
    "_async_failure_handler_body",
    "dispatch_success_handlers",
    "dispatch_failure_handlers",
    "handle_sync_success_callbacks_for_async_calls",
)


@pytest.fixture
def legacy_orchestration_disabled(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from litellm import utils
    from litellm.litellm_core_utils.litellm_logging import Logging

    reached: Final[list[str]] = []

    def forbid(name: str):
        def method(self, *args, **kwargs):
            reached.append(name)
            raise AssertionError(f"legacy orchestration reached: {name}")

        return method

    for name in FORBIDDEN_ORCHESTRATION:
        monkeypatch.setattr(Logging, name, forbid(name))

    def forbidden_setup(*args, **kwargs):
        reached.append("function_setup")
        raise AssertionError("legacy orchestration reached: function_setup")

    monkeypatch.setattr(utils, "function_setup", forbidden_setup)
    return reached


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ocr_success_runs_integrations_without_legacy_orchestration(
    ocr_server: RecordingServer, legacy_orchestration_disabled: list[str], asynchronous: bool
) -> None:
    recorder: Final = RecordingLogger()
    arguments: Final = {"callbacks": [recorder]}
    response: Final = (
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    )
    assert response.pages[0].markdown == "native OCR response"
    success_event: Final = "async_log_success_event" if asynchronous else "log_success_event"
    events: Final = await recorder.wait_for_async(success_event)
    assert legacy_orchestration_disabled == []
    assert recorder.names.count("log_pre_api_call") == 1
    assert events[0].kwargs["standard_logging_object"]["status"] == "success"
    assert events[0].kwargs["response_cost"] is not None
    assert events[0].kwargs["litellm_params"]["api_base"] == f"{ocr_server.base_url}/v1/ocr"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ocr_failure_runs_integrations_without_legacy_orchestration(
    ocr_server: RecordingServer, legacy_orchestration_disabled: list[str], asynchronous: bool
) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider unavailable"}, status=500))
    recorder: Final = RecordingLogger()
    arguments: Final = {"callbacks": [recorder]}
    with pytest.raises(litellm.InternalServerError) as caught:
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    assert legacy_orchestration_disabled == []
    failures: Final = tuple(event for event in recorder.events if event.name.endswith("log_failure_event"))
    assert [event.name for event in failures] == (
        ["log_failure_event", "async_log_failure_event"] if asynchronous else ["log_failure_event"]
    )
    assert all(event.kwargs["exception"] is caught.value for event in failures)
    assert all(event.kwargs["standard_logging_object"]["status"] == "failure" for event in failures)
    assert "log_success_event" not in recorder.names


def test_native_ocr_success_hooks_all_run_before_any_success_dispatch(ocr_server: RecordingServer) -> None:
    order: Final = []
    finished: Final = threading.Event()

    class Hooked(CustomLogger):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def logging_hook(self, kwargs, result, call_type):
            order.append(("hook", self.name))
            kwargs[f"seen-by-{self.name}"] = True
            return kwargs, result

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            order.append(("log", self.name, kwargs.get("seen-by-a"), kwargs.get("seen-by-b")))
            if self.name == "b":
                finished.set()

    call_native_ocr_with_callbacks(ocr_server, [Hooked("a"), Hooked("b")])

    assert finished.wait(10)
    assert order == [("hook", "a"), ("hook", "b"), ("log", "a", True, True), ("log", "b", True, True)]


def test_native_ocr_hook_failure_is_contained_and_target_still_dispatches(ocr_server: RecordingServer) -> None:
    order: Final = []
    finished: Final = threading.Event()

    class Broken(CustomLogger):
        def logging_hook(self, kwargs, result, call_type):
            raise RuntimeError("hook failed")

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            order.append("broken-log")

    class Healthy(CustomLogger):
        def logging_hook(self, kwargs, result, call_type):
            order.append("healthy-hook")
            return kwargs, result

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            order.append("healthy-log")
            finished.set()

    call_native_ocr_with_callbacks(ocr_server, [Broken(), Healthy()])

    assert finished.wait(10)
    assert order == ["healthy-hook", "broken-log", "healthy-log"]


@pytest.mark.asyncio
async def test_native_aocr_hook_replacement_of_result_reaches_success_dispatch(ocr_server: RecordingServer) -> None:
    replacement: Final = object()
    observed: Final = []

    class Replace(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            return kwargs, replacement

    class Observe(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(response_obj)

    recorder: Final = RecordingLogger()
    await call_native_aocr_with_callbacks(ocr_server, [Replace(), Observe(), recorder])
    await recorder.wait_for_async("async_log_success_event")

    assert observed == [replacement]


@pytest.mark.asyncio
async def test_native_aocr_shared_logging_object_dispatches_success_once(ocr_server: RecordingServer) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    ocr_server.expected_requests = 2
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(recorder)
    logger: Final = Logging(
        model="mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=__import__("datetime").datetime.now(),
        litellm_call_id="shared",
        function_id="shared",
    )
    logger.dynamic_async_success_callbacks = [recorder]

    await call_native_aocr(ocr_server, litellm_logging_obj=logger)
    await call_native_aocr(ocr_server, litellm_logging_obj=logger)
    await recorder.wait_for_async("async_log_success_event")
    from tests.test_litellm_rust.support.callback_recorder import drain_logging

    await drain_logging()

    assert logger.model_call_details["has_logged_async_success"] is True
    assert recorder.names.count("async_log_success_event") == 1


def test_native_ocr_logging_preparation_failure_does_not_fail_request(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.litellm_core_utils import litellm_logging

    def broken_payload(*args, **kwargs):
        raise RuntimeError("payload unavailable")

    monkeypatch.setattr(litellm_logging, "get_standard_logging_object_payload", broken_payload)
    unraisable: Final = []
    monkeypatch.setattr(__import__("sys"), "unraisablehook", lambda event: unraisable.append(event.exc_value))
    recorder: Final = RecordingLogger()

    response: Final = call_native_ocr_with_callbacks(ocr_server, [recorder])
    events: Final = recorder.wait_for("log_success_event")

    assert response.pages[0].markdown == "native OCR response"
    assert len(events) == 1
    assert events[0].thread is not threading.current_thread()
    assert any(str(error) == "payload unavailable" for error in unraisable)


def test_native_ocr_writes_success_marker_and_honours_existing_marker(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.rust_bridge import setup as native_setup

    ocr_server.expected_requests = 2
    loggers: Final = []
    original_build: Final = native_setup.build_logging

    def build_logging(**kwargs):
        logger = original_build(**kwargs)
        if loggers:
            logger.model_call_details["has_logged_sync_success"] = True
        loggers.append(logger)
        return logger

    monkeypatch.setattr(native_setup, "build_logging", build_logging)
    recorder: Final = RecordingLogger()

    call_native_ocr_with_callbacks(ocr_server, [recorder])
    recorder.wait_for("log_success_event")
    assert loggers[0].model_call_details["has_logged_sync_success"] is True

    call_native_ocr_with_callbacks(ocr_server, [recorder])
    from litellm.litellm_core_utils.thread_pool_executor import executor

    executor.submit(lambda: None).result(10)
    assert recorder.names.count("log_success_event") == 1
    assert recorder.names.count("logging_hook") == 1
