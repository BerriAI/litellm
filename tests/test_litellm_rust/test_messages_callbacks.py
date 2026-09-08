import asyncio
import copy
import json
import threading
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from tests.test_litellm_rust.callback_recorder import RecordingLogger
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
MESSAGES: Final = [{"role": "user", "content": "Hello"}]
MESSAGES_RESPONSE: Final = {
    "id": "msg_native",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "Hello from native Messages"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 4},
}


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


async def call_messages(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return await litellm.anthropic.messages.acreate(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_key="test-key",
        api_base=server.base_url,
        callbacks=callbacks,
        **kwargs,
    )


def request_body(kwargs: dict) -> dict:
    return kwargs["additional_args"]["complete_input_dict"]


def request_headers(kwargs: dict) -> dict:
    return kwargs["additional_args"]["headers"]


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="UC-MSG-PRECALL-VIEW: native pre-call arguments differ from legacy")
async def test_messages_pre_call_receives_expected_provider_request(messages_server: RecordingServer) -> None:
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((model, messages, copy.deepcopy(kwargs["additional_args"])))

    await call_messages(messages_server, [Observe()])

    assert len(observations) == 1
    model, messages, additional_args = observations[0]
    assert model == "claude-sonnet-4-5-20250929"
    assert messages == [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "model": "claude-sonnet-4-5-20250929",
                    "messages": MESSAGES,
                    "max_tokens": 64,
                }
            ),
        }
    ]
    assert additional_args["api_base"] == f"{messages_server.base_url}/v1/messages"
    assert additional_args["complete_input_dict"] == {
        "model": "claude-sonnet-4-5-20250929",
        "messages": MESSAGES,
        "max_tokens": 64,
    }
    assert additional_args["headers"]["x-api-key"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_after_edit", [False, True])
@pytest.mark.xfail(
    strict=True,
    reason="UC-MSG-PRECALL-MUTATION: native transport snapshots body and headers before pre-call",
)
async def test_messages_pre_call_edits_reach_later_callbacks_and_provider(
    messages_server: RecordingServer, raise_after_edit: bool
) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["temperature"] = 0.25
            request_headers(kwargs)["x-audit-tag"] = "reviewed"
            if raise_after_edit:
                raise RuntimeError("audit exporter unavailable")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append((copy.deepcopy(request_body(kwargs)), dict(request_headers(kwargs))))

    await call_messages(messages_server, [Edit(), Observe()])

    assert observed[0][0]["temperature"] == 0.25
    assert observed[0][1]["x-audit-tag"] == "reviewed"
    assert messages_server.requests[0].body["temperature"] == 0.25
    assert messages_server.requests[0].headers["x-audit-tag"] == "reviewed"


@pytest.mark.asyncio
async def test_messages_pre_call_rebinding_does_not_replace_inflight_request(
    messages_server: RecordingServer,
) -> None:
    observed: Final = []

    class Rebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs))

    await call_messages(messages_server, [Rebind(), Observe()])

    assert observed == [{"replacement": True}]
    assert messages_server.requests[0].body["messages"] == MESSAGES


@pytest.mark.asyncio
async def test_messages_pre_call_state_reaches_terminal_callbacks(messages_server: RecordingServer) -> None:
    token: Final = object()
    observed: Final = []
    finished: Final = asyncio.Event()

    class Stash(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["test-token"] = token

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(kwargs["test-token"])
            finished.set()

    await call_messages(messages_server, [Stash()])
    await asyncio.wait_for(finished.wait(), timeout=10)

    assert observed == [token]


@pytest.mark.asyncio
async def test_messages_callbacks_run_once(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call_messages(messages_server, [recorder])
    await recorder.wait_for_async("async_log_success_event")

    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count("async_logging_hook") == 1
    assert recorder.names.count("async_log_success_event") == 1


@pytest.mark.asyncio
async def test_messages_success_callbacks_receive_expected_context(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call_messages(
        messages_server,
        [recorder],
        litellm_call_id="messages-success",
        metadata={"source": "callback-test"},
    )
    async_events: Final = await recorder.wait_for_async("async_log_success_event")

    assert len(async_events) == 1
    event: Final = async_events[0]
    assert event.call_type == "anthropic_messages"
    assert event.kwargs["litellm_call_id"] == "messages-success"
    assert event.kwargs["litellm_params"]["metadata"]["source"] == "callback-test"
    assert event.response.choices[0].message.content == "Hello from native Messages"


@pytest.mark.asyncio
async def test_messages_pre_call_runs_in_callers_execution_context(messages_server: RecordingServer) -> None:
    caller_thread: Final = threading.current_thread()
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((asyncio.get_running_loop(), threading.current_thread()))

    caller_loop: Final = asyncio.get_running_loop()
    await call_messages(messages_server, [Observe()])

    assert observations == [(caller_loop, caller_thread)]


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="UC-MSG-STREAM-COMPLETION: native fake stream logs before assembled stream finalization",
)
async def test_messages_stream_logs_success_after_exhaustion(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()
    stream: Final = await call_messages(messages_server, [recorder], stream=True)

    assert "async_log_success_event" not in recorder.names
    chunks: Final = [chunk async for chunk in stream]
    events: Final = await recorder.wait_for_async("async_log_success_event")

    assert chunks
    assert len(events) == 1
    assert "async_log_stream_event" not in recorder.names
    assert events[0].kwargs["complete_streaming_response"] is not None
