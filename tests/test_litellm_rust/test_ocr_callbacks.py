import asyncio
import copy
import queue
import threading
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from tests.test_litellm_rust.callback_recorder import RecordingLogger
from tests.test_litellm_rust.contracts import (
    OCR_DOCUMENT,
    OCR_MODEL,
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
    request_body,
    request_headers,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def call_ocr(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return call_native_ocr(server, callbacks=callbacks, **kwargs)


async def call_aocr(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return await call_native_aocr(server, callbacks=callbacks, **kwargs)


def test_pre_call_receives_expected_provider_request(ocr_server: RecordingServer) -> None:
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((model, messages, copy.deepcopy(kwargs["additional_args"])))

    call_ocr(ocr_server, [Observe()], pages=[0])

    assert len(observations) == 1
    model, messages, additional_args = observations[0]
    assert model == "mistral-ocr-latest"
    assert messages == [{"role": "user", "content": "default-message-value"}]
    assert additional_args["api_base"] == f"{ocr_server.base_url}/v1/ocr"
    assert additional_args["complete_input_dict"] == {
        "model": "mistral-ocr-latest",
        "document": OCR_DOCUMENT,
        "pages": [0],
    }


@pytest.mark.parametrize("raise_after_edit", [False, True])
def test_pre_call_body_edits_reach_later_callbacks_and_provider(
    ocr_server: RecordingServer, raise_after_edit: bool
) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["include_image_base64"] = True
            if raise_after_edit:
                raise RuntimeError("audit exporter unavailable")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(copy.deepcopy(request_body(kwargs)))

    call_ocr(ocr_server, [Edit(), Observe()], include_image_base64=False)

    assert observed[0]["include_image_base64"] is True
    assert ocr_server.requests[0].body["include_image_base64"] is True


def test_pre_call_header_edits_reach_later_callbacks_and_provider(ocr_server: RecordingServer) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_headers(kwargs)["x-audit-tag"] = "reviewed"

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(dict(request_headers(kwargs)))

    call_ocr(ocr_server, [Edit(), Observe()])

    assert observed[0]["x-audit-tag"] == "reviewed"
    assert ocr_server.requests[0].headers["x-audit-tag"] == "reviewed"


def test_pre_call_nested_mutation_updates_retained_references(ocr_server: RecordingServer) -> None:
    original: Final = dict(OCR_DOCUMENT)
    replacement_url: Final = "data:application/pdf;base64,ZGVm"
    retained: Final = []

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            retained.append(request_body(kwargs)["document"])

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["document"]["document_url"] = replacement_url

    call_native_ocr(
        ocr_server,
        document=original,
        callbacks=[Retain(), Edit()],
    )

    assert retained[0]["document_url"] == replacement_url
    assert original["document_url"] == replacement_url
    assert ocr_server.requests[0].body["document"]["document_url"] == replacement_url


def test_pre_call_field_replacement_preserves_original_references(ocr_server: RecordingServer) -> None:
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


def test_pre_call_body_rebinding_does_not_replace_inflight_request(ocr_server: RecordingServer) -> None:
    observed: Final = []

    class Rebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs))

    call_ocr(ocr_server, [Rebind(), Observe()])

    assert observed == [{"replacement": True}]
    assert ocr_server.requests[0].body == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT}


def test_queued_payload_observes_later_callback_mutations(ocr_server: RecordingServer) -> None:
    queued: Final = []

    class QueuePayload(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            queued.append(request_body(kwargs))

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["queued-edit"] = True

    call_ocr(ocr_server, [QueuePayload(), Edit()])

    assert queued[0]["queued-edit"] is True


def test_pre_call_state_reaches_terminal_callbacks(ocr_server: RecordingServer) -> None:
    token: Final = object()
    terminal_tokens: queue.SimpleQueue[object] = queue.SimpleQueue()
    finished: Final = threading.Event()

    class Stash(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["test-token"] = token

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            terminal_tokens.put(kwargs["test-token"])
            finished.set()

    call_ocr(ocr_server, [Stash()])

    assert finished.wait(10)
    assert terminal_tokens.get_nowait() is token


@pytest.mark.asyncio
async def test_success_callbacks_receive_expected_context_and_response(ocr_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call_aocr(
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
async def test_failure_callbacks_receive_expected_context_and_error(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "provider unavailable"}, status=500))
    observations: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("sync", kwargs["call_type"], kwargs["exception"], response_obj))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("async", kwargs["call_type"], kwargs["exception"], response_obj))

    with pytest.raises(litellm.InternalServerError):
        await call_aocr(ocr_server, [Observe()])

    assert [observation[0] for observation in observations] == ["sync", "async"]
    assert all(observation[1] == "aocr" for observation in observations)
    assert all(isinstance(observation[2], litellm.InternalServerError) for observation in observations)
    assert all(observation[3] is None for observation in observations)


@pytest.mark.asyncio
async def test_pre_call_runs_in_callers_execution_context(ocr_server: RecordingServer) -> None:
    caller_loop: Final = asyncio.get_running_loop()
    caller_thread: Final = threading.current_thread()
    recorder: Final = RecordingLogger()

    await call_aocr(ocr_server, [recorder])

    events: Final = await recorder.wait_for_async("log_pre_api_call")
    assert len(events) == 1
    assert events[0].loop is caller_loop
    assert events[0].thread is caller_thread
