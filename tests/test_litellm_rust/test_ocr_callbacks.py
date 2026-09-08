import asyncio
import copy
import json
import queue
import threading
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from tests.test_litellm_rust.ocr_test_server import OCRTestServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

DOCUMENT: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
MODEL: Final = "mistral/mistral-ocr-latest"


def call_ocr(server: OCRTestServer, callbacks: list[CustomLogger], **kwargs: object):
    return litellm.ocr(
        model=MODEL,
        document=dict(DOCUMENT),
        api_key="test-key",
        api_base=server.base_url,
        callbacks=callbacks,
        **kwargs,
    )


async def call_aocr(server: OCRTestServer, callbacks: list[CustomLogger], **kwargs: object):
    return await litellm.aocr(
        model=MODEL,
        document=dict(DOCUMENT),
        api_key="test-key",
        api_base=server.base_url,
        callbacks=callbacks,
        **kwargs,
    )


def request_body(kwargs: dict) -> dict:
    return kwargs["additional_args"]["complete_input_dict"]


def request_headers(kwargs: dict) -> dict:
    return kwargs["additional_args"]["headers"]


def test_pre_call_receives_expected_provider_request(ocr_server: OCRTestServer) -> None:
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
        "document": DOCUMENT,
        "pages": [0],
    }


@pytest.mark.parametrize("raise_after_edit", [False, True])
def test_pre_call_body_edits_reach_later_callbacks_and_provider(
    ocr_server: OCRTestServer, raise_after_edit: bool
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


def test_pre_call_header_edits_reach_later_callbacks_and_provider(ocr_server: OCRTestServer) -> None:
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


def test_pre_call_nested_mutation_updates_retained_references(ocr_server: OCRTestServer) -> None:
    original: Final = dict(DOCUMENT)
    replacement_url: Final = "data:application/pdf;base64,ZGVm"
    retained: Final = []

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            retained.append(request_body(kwargs)["document"])

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["document"]["document_url"] = replacement_url

    litellm.ocr(
        model=MODEL,
        document=original,
        api_key="test-key",
        api_base=ocr_server.base_url,
        callbacks=[Retain(), Edit()],
    )

    assert retained[0]["document_url"] == replacement_url
    assert original["document_url"] == replacement_url
    assert ocr_server.requests[0].body["document"]["document_url"] == replacement_url


def test_pre_call_field_replacement_preserves_original_references(ocr_server: OCRTestServer) -> None:
    original: Final = dict(DOCUMENT)
    replacement: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,ZGVm"}
    retained: Final = []

    class RetainAndReplace(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = request_body(kwargs)
            retained.append(body["document"])
            body["document"] = replacement

    litellm.ocr(
        model=MODEL,
        document=original,
        api_key="test-key",
        api_base=ocr_server.base_url,
        callbacks=[RetainAndReplace()],
    )

    assert retained[0] is original
    assert original["document_url"] == DOCUMENT["document_url"]
    assert ocr_server.requests[0].body["document"] == replacement


def test_pre_call_body_rebinding_does_not_replace_inflight_request(ocr_server: OCRTestServer) -> None:
    observed: Final = []

    class Rebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs))

    call_ocr(ocr_server, [Rebind(), Observe()])

    assert observed == [{"replacement": True}]
    assert ocr_server.requests[0].body == {"model": "mistral-ocr-latest", "document": DOCUMENT}


def test_queued_payload_observes_later_callback_mutations(ocr_server: OCRTestServer) -> None:
    queued: Final = []

    class QueuePayload(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            queued.append(request_body(kwargs))

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["queued-edit"] = True

    call_ocr(ocr_server, [QueuePayload(), Edit()])

    assert queued[0]["queued-edit"] is True


def test_callback_copies_preserve_expected_sharing(ocr_server: OCRTestServer) -> None:
    copies: Final = {}
    replacement_url: Final = "data:application/pdf;base64,ZGVm"

    class CopyPayload(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = request_body(kwargs)
            copies["shallow"] = dict(body)
            copies["deep"] = copy.deepcopy(body)
            copies["serialized"] = json.dumps(body)

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["document"]["document_url"] = replacement_url

    call_ocr(ocr_server, [CopyPayload(), Edit()])

    assert copies["shallow"]["document"]["document_url"] == replacement_url
    assert copies["deep"]["document"]["document_url"] == "data:application/pdf;base64,YWJj"
    assert json.loads(copies["serialized"])["document"]["document_url"] == "data:application/pdf;base64,YWJj"


def test_pre_call_state_reaches_terminal_callbacks(ocr_server: OCRTestServer) -> None:
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
async def test_success_callbacks_receive_expected_context_and_response(ocr_server: OCRTestServer) -> None:
    observations: Final = []
    finished: Final = asyncio.Event()

    class Observe(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(
                (
                    kwargs["call_type"],
                    kwargs["litellm_call_id"],
                    kwargs["litellm_params"]["metadata"]["source"],
                    response_obj.pages[0].markdown,
                )
            )
            finished.set()

    await call_aocr(
        ocr_server,
        [Observe()],
        litellm_call_id="ocr-success",
        metadata={"source": "callback-test"},
    )
    await asyncio.wait_for(finished.wait(), timeout=10)

    assert observations == [("aocr", "ocr-success", "callback-test", "native OCR response")]


@pytest.mark.asyncio
async def test_failure_callbacks_receive_expected_context_and_error(ocr_server: OCRTestServer) -> None:
    ocr_server.enqueue(ResponseSpec(status=500, body={"message": "provider unavailable"}))
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


def test_background_callback_can_mutate_retained_state_after_return(ocr_server: OCRTestServer) -> None:
    release: Final = threading.Event()
    finished: Final = threading.Event()
    retained: Final = []

    class BackgroundEdit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = request_body(kwargs)
            retained.append(body)

            def edit() -> None:
                release.wait(10)
                body["background-edit"] = True
                finished.set()

            threading.Thread(target=edit, daemon=True).start()

    call_ocr(ocr_server, [BackgroundEdit()])

    assert "background-edit" not in retained[0]
    release.set()
    assert finished.wait(10)
    assert retained[0]["background-edit"] is True


@pytest.mark.asyncio
async def test_pre_call_runs_in_callers_execution_context(ocr_server: OCRTestServer) -> None:
    caller_loop: Final = asyncio.get_running_loop()
    caller_thread: Final = threading.current_thread()
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((asyncio.get_running_loop(), threading.current_thread()))

    await call_aocr(ocr_server, [Observe()])

    assert observations == [(caller_loop, caller_thread)]
