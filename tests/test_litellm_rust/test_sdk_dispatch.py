import hashlib
import hmac
import json
from typing import Final
from urllib.parse import urlsplit

import pytest

import litellm
from tests.test_litellm_rust.contracts import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE
from tests.test_litellm_rust.recording_server import RecordedRequest, RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


def _verify_sigv4(request: RecordedRequest, secret_key: str) -> None:
    authorization: Final = request.headers["authorization"]
    algorithm, attributes_text = authorization.split(" ", 1)
    attributes: Final = dict(item.split("=", 1) for item in attributes_text.split(", "))
    credential_scope: Final = attributes["Credential"].split("/", 1)[1]
    signed_names: Final = attributes["SignedHeaders"].split(";")
    canonical_headers: Final = "".join(f"{name}:{' '.join(request.headers[name].split())}\n" for name in signed_names)
    parsed_path: Final = urlsplit(request.path)
    canonical_request: Final = "\n".join(
        (
            request.method,
            parsed_path.path,
            parsed_path.query,
            canonical_headers,
            attributes["SignedHeaders"],
            hashlib.sha256(request.raw_body).hexdigest(),
        )
    )
    amz_date: Final = request.headers["x-amz-date"]
    string_to_sign: Final = "\n".join(
        (algorithm, amz_date, credential_scope, hashlib.sha256(canonical_request.encode()).hexdigest())
    )
    date, region, service, terminator = credential_scope.split("/")
    date_key: Final = hmac.new(f"AWS4{secret_key}".encode(), date.encode(), hashlib.sha256).digest()
    region_key: Final = hmac.new(date_key, region.encode(), hashlib.sha256).digest()
    service_key: Final = hmac.new(region_key, service.encode(), hashlib.sha256).digest()
    signing_key: Final = hmac.new(service_key, terminator.encode(), hashlib.sha256).digest()
    expected: Final = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    assert hmac.compare_digest(attributes["Signature"], expected)


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def test_public_ocr_entrypoint_uses_native_transport_when_enabled(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_public_aocr_entrypoint_uses_native_transport_when_enabled(ocr_server: RecordingServer) -> None:
    response: Final = await litellm.aocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] == "identity"


def test_public_ocr_falls_back_when_native_transport_declines(ocr_server: RecordingServer) -> None:
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] != "identity"


def test_public_ocr_uses_python_transport_when_disabled(ocr_server: RecordingServer) -> None:
    litellm.rust(False)

    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["accept-encoding"] != "identity"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
@pytest.mark.parametrize("rebind_logging_view", [False, True])
@pytest.mark.parametrize("status", [200, 429])
@pytest.mark.parametrize("native", [False, True])
async def test_chat_retains_callback_edits_through_public_dispatch(
    recording_server: RecordingServer,
    asynchronous: bool,
    provider: str,
    rebind_logging_view: bool,
    status: int,
    native: bool,
) -> None:
    import threading

    litellm.rust(native)

    from tests.test_litellm_rust.callback_recorder import RecordingLogger
    from tests.test_litellm_rust.contracts import MESSAGES_RESPONSE

    recording_server.default_response = ResponseSpec(
        status=status,
        body=(
            MESSAGES_RESPONSE
            if provider == "anthropic"
            else {
                "output": {"message": {"role": "assistant", "content": [{"text": "native chat"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9},
                "metrics": {"latencyMs": 1},
            }
        ),
    )
    caller_thread: Final = threading.current_thread()
    observations: Final = []

    class EditingLogger(RecordingLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            super().log_pre_api_call(model, messages, kwargs)
            body = kwargs["additional_args"]["complete_input_dict"]
            headers = kwargs["additional_args"]["headers"]
            if provider == "anthropic":
                assert isinstance(body, dict)
                body["messages"][0]["content"][0]["text"] = "edited by callback"
                body["max_tokens"] = 32
            else:
                assert isinstance(body, str)
                assert json.loads(body)["messages"][0]["content"][0]["text"] == "original"
            headers["x-retained-callback"] = "original"
            observations.append((threading.current_thread(), body, headers))
            if rebind_logging_view:
                kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}
                kwargs["additional_args"]["headers"] = {"x-retained-callback": "replacement"}

    recorder: Final = EditingLogger()
    kwargs: Final = {
        "model": "anthropic/claude-opus-5" if provider == "anthropic" else "bedrock/anthropic.claude-opus-5",
        "messages": [{"role": "user", "content": "original"}],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
        "callbacks": [recorder],
        "num_retries": 0,
        **(
            {"aws_access_key_id": "test", "aws_secret_access_key": "test", "aws_region_name": "us-east-1"}
            if provider == "bedrock"
            else {}
        ),
    }
    if status != 200:
        with pytest.raises((litellm.APIError, litellm.RateLimitError)) as raised:
            await litellm.acompletion(**kwargs) if asynchronous else litellm.completion(**kwargs)
        failure_event: Final = "async_log_failure_event" if asynchronous else "log_failure_event"
        failures: Final = await recorder.wait_for_async(failure_event)
        assert raised.value.status_code == status
        assert failures[0].kwargs["exception"] is raised.value
        assert recorder.names.count(failure_event) == 1
        assert recorder.names.count("log_pre_api_call") == 1
        assert len(recording_server.requests) == 1
        return

    response: Final = await litellm.acompletion(**kwargs) if asynchronous else litellm.completion(**kwargs)
    event_name: Final = "async_log_success_event" if asynchronous else "log_success_event"
    events: Final = await recorder.wait_for_async(event_name)

    if native:
        assert response._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count(event_name) == 1
    if asynchronous and provider == "anthropic" and not native:
        assert observations[0][0] is not caller_thread
    else:
        assert observations[0][0] is caller_thread
    assert events[0].response is response
    assert recording_server.requests[0].body["messages"][0]["content"][0]["text"] == (
        "edited by callback" if provider == "anthropic" else "original"
    )
    assert recording_server.requests[0].headers["x-retained-callback"] == "original"
    body: Final = recording_server.requests[0].body
    assert (body["max_tokens"] if provider == "anthropic" else body["inferenceConfig"]["maxTokens"]) == (
        32 if provider == "anthropic" else 64
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [200, 429])
@pytest.mark.parametrize("native", [False, True])
async def test_bedrock_callbacks_share_state_without_replacing_signed_transport(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    status: int,
    native: bool,
) -> None:
    from tests.test_litellm_rust.callback_recorder import RecordingLogger

    litellm.rust(native)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    recording_server.default_response = ResponseSpec(
        status=status,
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "native chat"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9},
            "metrics": {"latencyMs": 1},
        },
    )
    observations: Final[list[tuple[int, int]]] = []
    replacement_body: Final = {"replacement": True}
    rebound_headers: Final = {"x-callback-header": "rebound"}

    class FirstCallback(RecordingLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            additional: Final = kwargs["additional_args"]
            callback_body: Final = additional["complete_input_dict"]
            original_headers: Final = additional["headers"]
            observations.append((id(kwargs), id(additional)))
            assert isinstance(callback_body, str)
            assert json.loads(callback_body)["messages"][0]["content"][0]["text"] == "original"
            kwargs["callback_marker"] = "visible"
            original_headers["x-callback-header"] = "in-place"
            additional["complete_input_dict"] = replacement_body
            additional["headers"] = rebound_headers
            raise RuntimeError("first callback failure is non-blocking")

    class SecondCallback(RecordingLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            additional: Final = kwargs["additional_args"]
            observations.append((id(kwargs), id(additional)))
            assert kwargs["callback_marker"] == "visible"
            assert additional["complete_input_dict"] is replacement_body
            assert additional["headers"] is rebound_headers

    request: Final = {
        "model": "bedrock/anthropic.claude-opus-5",
        "messages": [{"role": "user", "content": "original"}],
        "max_tokens": 64,
        "api_base": recording_server.base_url,
        "callbacks": [FirstCallback(), SecondCallback()],
        "num_retries": 0,
        "aws_access_key_id": "test-access",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
    if status == 200:
        await litellm.acompletion(**request) if asynchronous else litellm.completion(**request)
    else:
        with pytest.raises((litellm.APIError, litellm.RateLimitError)):
            await litellm.acompletion(**request) if asynchronous else litellm.completion(**request)

    assert observations[0] == observations[1]
    recorded: Final = recording_server.requests[0]
    assert recorded.body["messages"][0]["content"][0]["text"] == "original"
    assert "replacement" not in recorded.body
    assert recorded.headers["x-callback-header"] == "in-place"
    assert recorded.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    _verify_sigv4(recorded, "test-secret")


@pytest.mark.asyncio
async def test_native_chat_pre_call_preserves_the_caller_task_and_context(
    recording_server: RecordingServer,
) -> None:
    import asyncio
    from contextvars import ContextVar

    from litellm.integrations.custom_logger import CustomLogger
    from tests.test_litellm_rust.contracts import MESSAGES_RESPONSE

    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    marker: Final[ContextVar[str]] = ContextVar("native_chat_marker", default="missing")
    marker.set("caller")
    caller_task: Final = asyncio.current_task()
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((asyncio.current_task(), marker.get()))
            marker.set("callback")

    response: Final = await litellm.acompletion(
        model="anthropic/claude-opus-5",
        messages=[{"role": "user", "content": "task context"}],
        max_tokens=16,
        api_key="test-key",
        api_base=recording_server.base_url,
        callbacks=[Observe()],
        num_retries=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
    assert observations == [(caller_task, "caller")]
    assert marker.get() == "callback"


@pytest.mark.asyncio
async def test_native_chat_concurrent_calls_keep_callback_roots_isolated(
    recording_server: RecordingServer,
) -> None:
    import asyncio

    from litellm.integrations.custom_logger import CustomLogger
    from tests.test_litellm_rust.contracts import MESSAGES_RESPONSE

    call_ids: Final = tuple(f"chat-{index}" for index in range(4))
    recording_server.expected_requests = len(call_ids)
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    class Correlate(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body: Final = kwargs["additional_args"]["complete_input_dict"]
            body["messages"][0]["content"][0]["text"] = kwargs["litellm_call_id"]

    logger: Final = Correlate()

    async def invoke(call_id: str) -> None:
        await litellm.acompletion(
            model="anthropic/claude-opus-5",
            messages=[{"role": "user", "content": call_id}],
            max_tokens=16,
            api_key="test-key",
            api_base=recording_server.base_url,
            callbacks=[logger],
            litellm_call_id=call_id,
            num_retries=0,
        )

    await asyncio.gather(*(invoke(call_id) for call_id in call_ids))

    sent: Final = {request.body["messages"][0]["content"][0]["text"] for request in recording_server.requests}
    assert sent == set(call_ids)


@pytest.mark.asyncio
async def test_native_chat_callback_can_make_a_nested_native_call(
    recording_server: RecordingServer,
) -> None:
    import threading

    from litellm.integrations.custom_logger import CustomLogger
    from tests.test_litellm_rust.contracts import MESSAGES_RESPONSE

    recording_server.expected_requests = 2
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    nested_responses: Final = []
    nested_started: Final = threading.Event()

    class NestedCall(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            if nested_started.is_set():
                return
            nested_started.set()
            nested_responses.append(
                litellm.completion(
                    model="anthropic/claude-opus-5",
                    messages=[{"role": "user", "content": "nested"}],
                    max_tokens=16,
                    api_key="test-key",
                    api_base=recording_server.base_url,
                    num_retries=0,
                )
            )

    outer: Final = await litellm.acompletion(
        model="anthropic/claude-opus-5",
        messages=[{"role": "user", "content": "outer"}],
        max_tokens=16,
        api_key="test-key",
        api_base=recording_server.base_url,
        callbacks=[NestedCall()],
        num_retries=0,
    )

    assert outer._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
    assert nested_responses[0]._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
    sent: Final = {request.body["messages"][0]["content"][0]["text"] for request in recording_server.requests}
    assert sent == {"outer", "nested"}


@pytest.mark.asyncio
async def test_native_chat_cancellation_during_io_does_not_publish_a_terminal(
    recording_server: RecordingServer,
) -> None:
    import asyncio

    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from tests.test_litellm_rust.callback_recorder import RecordingLogger
    from tests.test_litellm_rust.contracts import MESSAGES_RESPONSE

    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE, delay=0.5)
    recorder: Final = RecordingLogger()
    task: Final = asyncio.create_task(
        litellm.acompletion(
            model="anthropic/claude-opus-5",
            messages=[{"role": "user", "content": "cancel"}],
            max_tokens=16,
            api_key="test-key",
            api_base=recording_server.base_url,
            callbacks=[recorder],
            num_retries=0,
        )
    )
    async with asyncio.timeout(10):
        while not recording_server.requests:
            await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)

    assert recorder.names.count("log_pre_api_call") == 1
    assert not {
        "log_success_event",
        "async_log_success_event",
        "log_failure_event",
        "async_log_failure_event",
    }.intersection(recorder.names)
