import asyncio
import copy
import gc
import json
import queue
import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.ocr.data import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call,
    call_aocr,
    call_ocr,
    ocr_reply,
)
from tests.test_litellm_rust.support.callback_recorder import (
    RecordingLogger,
    drain_logging,
    request_body,
    request_headers,
)
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound
from tests.test_litellm_rust.support.routing import assert_served_natively
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension


def test_native_ocr_pre_call_callback_receives_transformed_provider_request(server: SharedProvider) -> None:
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((model, copy.deepcopy(kwargs["additional_args"])))

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[Observe()], pages=[0])

    requests: Final = assert_served_natively(server)
    assert len(observations) == 1
    model, additional_args = observations[0]
    assert model == "mistral-ocr-latest"
    assert additional_args["api_base"] == f"{server.wire.url}/v1/ocr"
    assert additional_args["complete_input_dict"] == {
        "model": "mistral-ocr-latest",
        "document": OCR_DOCUMENT,
        "pages": [0],
    }


@pytest.mark.asyncio
async def test_native_ocr_does_not_restore_keywords_deleted_by_a_pre_call_hook(server: SharedProvider) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("pages", "extra_headers")}

    server.expect(ocr_reply())
    with rebound(litellm, "callbacks", [DropKeywords()]):
        await call_aocr(server, pages=[0], extra_headers={"x-deleted": "remove this header"})

    requests: Final = assert_served_natively(server)
    assert len(requests) == 1
    sent: Final = requests[0]
    assert json.loads(sent.body) == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT}
    assert sent.headers.get("x-deleted") is None


@pytest.mark.parametrize("raise_after_edit", [False, True], ids=["callback-returns", "callback-raises"])
def test_native_ocr_pre_call_body_edit_reaches_next_callback_and_provider(
    server: SharedProvider, raise_after_edit: bool
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

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[Edit(), Observe()], include_image_base64=False)

    requests: Final = assert_served_natively(server)
    assert observed[0]["include_image_base64"] is True
    assert json.loads(requests[0].body)["include_image_base64"] is True


def test_native_ocr_pre_call_header_edit_reaches_next_callback_and_provider(server: SharedProvider) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_headers(kwargs)["x-audit-tag"] = "reviewed"

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(dict(request_headers(kwargs)))

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[Edit(), Observe()])

    requests: Final = assert_served_natively(server)
    assert observed[0]["x-audit-tag"] == "reviewed"
    assert requests[0].headers["x-audit-tag"] == "reviewed"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ocr_pre_call_nested_document_edit_updates_caller_callback_and_provider_references(
    server: SharedProvider, asynchronous: bool
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
        "api_base": server.wire.url,
        "callbacks": [Retain(), Edit()],
    }
    server.expect(ocr_reply())
    response: Final = await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)

    requests: Final = assert_served_natively(server, response)
    assert aliases == [True]
    assert retained[0]["document_url"] == replacement_url
    assert original["document_url"] == replacement_url
    assert json.loads(requests[0].body)["document"]["document_url"] == replacement_url
    assert response.pages[0].markdown == "native OCR response"


def test_native_ocr_pre_call_body_rebinding_is_visible_to_callbacks_but_not_provider(
    server: SharedProvider,
) -> None:
    observed: Final = []

    class Rebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs))

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[Rebind(), Observe()])

    requests: Final = assert_served_natively(server)
    assert observed == [{"replacement": True}]
    assert json.loads(requests[0].body) == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT}


def test_native_ocr_callback_retained_body_observes_later_callback_mutation(server: SharedProvider) -> None:
    queued: Final = []

    class QueuePayload(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            queued.append(request_body(kwargs))

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["queued-edit"] = True

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[QueuePayload(), Edit()])

    assert queued[0]["queued-edit"] is True


def test_native_ocr_success_callback_receives_state_added_by_pre_call_callback(server: SharedProvider) -> None:
    token: Final = object()
    terminal_tokens: queue.SimpleQueue[object] = queue.SimpleQueue()
    finished: Final = threading.Event()

    class Stash(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["test-token"] = token

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            terminal_tokens.put(kwargs["test-token"])
            finished.set()

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[Stash()])

    assert finished.wait(10)
    assert terminal_tokens.get_nowait() is token


@pytest.mark.asyncio
async def test_native_aocr_success_callback_receives_call_id_metadata_and_response(
    server: SharedProvider,
) -> None:
    recorder: Final = RecordingLogger()

    server.expect(ocr_reply())
    await call_aocr(
        server,
        callbacks=[recorder],
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
    server: SharedProvider,
) -> None:
    server.expect(ocr_reply({"message": "provider unavailable"}, status=500))
    observations: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("sync", kwargs["call_type"], kwargs["exception"], response_obj))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("async", kwargs["call_type"], kwargs["exception"], response_obj))

    with pytest.raises(litellm.InternalServerError):
        await call_aocr(server, callbacks=[Observe()])

    assert [observation[0] for observation in observations] == ["sync", "async"]
    assert all(observation[1] == "aocr" for observation in observations)
    assert all(isinstance(observation[2], litellm.InternalServerError) for observation in observations)
    assert all(observation[3] is None for observation in observations)
    assert_served_natively(server)


@pytest.mark.asyncio
async def test_native_aocr_pre_call_callback_runs_on_caller_loop_and_thread(server: SharedProvider) -> None:
    caller_loop: Final = asyncio.get_running_loop()
    caller_thread: Final = threading.current_thread()
    recorder: Final = RecordingLogger()

    server.expect(ocr_reply())
    await call_aocr(server, callbacks=[recorder])
    assert_served_natively(server)

    events: Final = await recorder.wait_for_async("log_pre_api_call")
    assert len(events) == 1
    assert events[0].loop is caller_loop
    assert events[0].thread is caller_thread


@pytest.mark.asyncio
async def test_native_aocr_failure_callbacks_receive_state_added_by_pre_call_callback(
    server: SharedProvider,
) -> None:
    server.expect(ocr_reply({"message": "provider unavailable"}, status=500))
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
        await call_aocr(server, callbacks=[TrackInFlightRequest()])

    assert [event for event, _ in observed] == ["sync", "async"]
    assert all(observed_token is token for _, observed_token in observed)
    assert_served_natively(server)


def test_native_ocr_dispatches_each_callback_phase_once_when_logger_is_registered_multiple_times(
    server: SharedProvider,
) -> None:
    recorder: Final = RecordingLogger()

    server.expect(ocr_reply())
    call_ocr(
        server,
        callbacks=[recorder, recorder],
        success_callback=[recorder],
        failure_callback=[recorder],
    )
    recorder.wait_for("log_success_event")
    assert_served_natively(server)

    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count("logging_hook") == 1
    assert recorder.names.count("log_success_event") == 1
    assert "log_failure_event" not in recorder.names


JSON_SCALARS: Final = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8)
)
JSON_VALUES: Final = st.recursive(
    JSON_SCALARS,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3),
    max_leaves=8,
)


class ApplyEdits(CustomLogger):
    def __init__(self, edits: Mapping[str, object]) -> None:
        super().__init__()
        self.edits: Final = edits

    def log_pre_api_call(self, model, messages, kwargs):
        request_body(kwargs).update(copy.deepcopy(dict(self.edits)))


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(edits=st.dictionaries(st.from_regex(r"x_[a-z]{1,6}", fullmatch=True), JSON_VALUES, max_size=3))
def test_native_ocr_provider_receives_the_body_exactly_as_pre_call_callbacks_left_it(
    server: SharedProvider, edits: dict[str, object]
) -> None:
    server.expect(ocr_reply())

    with isolated_callback_registries():
        call_ocr(server, callbacks=[ApplyEdits(MappingProxyType(edits))])

    requests: Final = assert_served_natively(server)
    assert json.loads(requests[0].body) == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT, **edits}


@pytest.mark.parametrize("hook", ["log_pre_api_call", "logging_hook", "log_success_event"])
def test_native_ocr_sync_hooks_see_no_running_event_loop(server: SharedProvider, hook: str) -> None:
    recorder: Final = RecordingLogger()

    server.expect(ocr_reply())
    call_ocr(server, callbacks=[recorder])
    assert_served_natively(server)

    [event] = recorder.wait_for(hook)
    assert event.loop is None
    assert (event.thread is threading.current_thread()) == (hook == "log_pre_api_call")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ocr_payload_a_callback_retains_outlives_the_call_intact(
    server: SharedProvider, asynchronous: bool
) -> None:
    retained: Final = []

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            retained.append((kwargs, request_body(kwargs), request_headers(kwargs)))

    server.expect(ocr_reply())
    await call(server, asynchronous, callbacks=[Retain()])
    await drain_logging()
    gc.collect()

    requests: Final = assert_served_natively(server)
    [(details, body, headers)] = retained
    assert body == json.loads(requests[0].body)
    assert headers
    assert all(requests[0].headers[name.lower()] == value for name, value in headers.items())
    assert details["additional_args"]["complete_input_dict"] is body
    assert details["additional_args"]["headers"] is headers


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["sync", "async"])
async def test_native_ocr_success_callbacks_share_one_logging_payload(server: SharedProvider, family: str) -> None:
    queued: Final = []
    finished: Final = threading.Event()

    def queue_payload(kwargs: dict[str, object]) -> None:
        queued.append(kwargs["standard_logging_object"])

    def strip_payload(kwargs: dict[str, object]) -> None:
        payload: Final = kwargs["standard_logging_object"]
        assert isinstance(payload, dict)
        payload["stripped-by-a-later-callback"] = True
        finished.set()

    class QueuePayload(CustomLogger):
        if family == "sync":

            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                queue_payload(kwargs)

        else:

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                queue_payload(kwargs)

    class StripPayload(CustomLogger):
        if family == "sync":

            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                strip_payload(kwargs)

        else:

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                strip_payload(kwargs)

    server.expect(ocr_reply())
    await call(server, family == "async", callbacks=[QueuePayload(), StripPayload()])
    await drain_logging()
    assert_served_natively(server)

    assert await asyncio.to_thread(finished.wait, 10)
    assert [payload["stripped-by-a-later-callback"] for payload in queued] == [True]


@pytest.mark.asyncio
async def test_native_aocr_state_stashed_before_a_blocking_hook_raises_reaches_failure_callbacks(
    server: SharedProvider,
) -> None:
    token: Final = object()
    observed: Final = []

    class Blocked(Exception):
        pass

    class Block(CustomGuardrail):
        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            request_data["litellm_logging_obj"].model_call_details["blocked-by"] = token
            raise Blocked("blocked after the provider answered")

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("success", None, None))

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("sync", kwargs.get("blocked-by"), kwargs["exception"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("async", kwargs.get("blocked-by"), kwargs["exception"]))

    litellm.callbacks.append(Block())

    server.expect(ocr_reply())
    with pytest.raises(Blocked) as raised:
        await call_aocr(server)
    await drain_logging()
    assert_served_natively(server)

    assert observed == [("sync", token, raised.value), ("async", token, raised.value)]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_resolves_token_before_pre_call_on_caller_context(
    server: SharedProvider,
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
    server.expect(ocr_reply())
    response: Final = await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)
    requests: Final = assert_served_natively(server, response)
    assert response.pages[0].markdown == "native OCR response"
    assert observations == ["token", "pre_call"]
    assert requests[0].headers["authorization"] == "Bearer edited"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_token_provider_can_make_nested_native_ocr_call(
    server: SharedProvider,
    isolated_azure_auth: None,
    asynchronous: bool,
) -> None:
    server.expect(ocr_reply(), ocr_reply())
    calls: Final = []

    def provider() -> str:
        calls.append("token")
        nested: Final = call_ocr(server)
        assert nested.pages[0].markdown == "native OCR response"
        return "outer-token"

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": provider,
    }
    response: Final = await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)
    assert response.pages[0].markdown == "native OCR response"
    assert calls == ["token"]
    requests: Final = assert_served_natively(server, response, count=2, include_model_discovery=True)
    assert [request.headers["authorization"] for request in requests] == [
        "Bearer test-key",
        "Bearer outer-token",
    ]


@pytest.mark.asyncio
async def test_concurrent_native_azure_ocr_calls_isolate_token_results_and_error(
    server: SharedProvider,
    isolated_azure_auth: None,
) -> None:
    server.expect(ocr_reply(), ocr_reply())

    async def request(token: str, fail: bool) -> object:
        def provider() -> str:
            if fail:
                raise ValueError(token)
            return token

        return await call_aocr(
            server,
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
    requests: Final = assert_served_natively(server, count=2)
    assert sorted(request.headers["authorization"] for request in requests) == [
        "Bearer first",
        "Bearer second",
    ]


@pytest.mark.asyncio
async def test_native_azure_ocr_releases_token_provider_after_cancellation(
    server: SharedProvider,
    isolated_azure_auth: None,
) -> None:
    import gc
    import weakref

    from tests.test_litellm_rust.support.callback_recorder import drain_logging

    class Provider:
        def __call__(self) -> str:
            return "caller-token"

    async def invoke() -> weakref.ReferenceType[Provider]:
        provider: Final = Provider()
        reference: Final = weakref.ref(provider)
        gate: Final = threading.Event()
        server.expect(Reply(chunks=(json.dumps(OCR_RESPONSE).encode(),), gate_before_first=gate))
        task: Final = asyncio.create_task(
            call_aocr(
                server,
                model="azure_ai/mistral-ocr-latest",
                api_key=None,
                azure_ad_token_provider=provider,
            )
        )
        await server.wait_for_requests(1)
        requests: Final = server.received(include_model_discovery=True)
        assert len(requests) == 1
        assert reference() is provider
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            gate.set()
        return reference

    reference: Final = await invoke()
    await drain_logging()
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None
