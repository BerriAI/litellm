import json
from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

from tests.test_litellm_rust.support.stream_callback_contract import (
    TestStreamCallbackContract as TestStreamCallbackContract,
)

from tests.test_litellm_rust.support.callback_contract import (
    CallbackRoute,
    TestCallbackContract as TestCallbackContract,
)
from tests.test_litellm_rust.support.callback_routes import messages_contract

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    request_body,
)

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture(autouse=True)
def opt_messages_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def assert_served_natively(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
async def test_native_messages_callbacks_see_the_provider_request_and_the_public_response(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], litellm_call_id="messages-success")
    )

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    sent: Final = messages_server.requests[0]
    assert sent.path == "/v1/messages"
    assert sent.body == {"model": "claude-sonnet-5", "messages": list(MESSAGES), "max_tokens": 64, "stream": False}
    pre_call: Final = recorder.wait_for("log_pre_api_call")
    assert request_body(pre_call[0].kwargs) == sent.body
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].call_type == "anthropic_messages"
    assert success[0].kwargs["litellm_call_id"] == "messages-success"
    assert success[0].response.choices[0].message.content == "Hello from native Messages"


@pytest.mark.asyncio
async def test_native_messages_pre_call_body_edit_reaches_the_provider(messages_server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["temperature"] = 0.25

    await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Edit()]))

    assert messages_server.requests[0].body["temperature"] == 0.25


@pytest.mark.asyncio
async def test_native_messages_provider_error_reaches_caller_and_failure_callbacks_as_one_public_error(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, status=400)
    )
    observed: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("sync", kwargs["exception"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("async", kwargs["exception"]))

    with pytest.raises(litellm.BadRequestError) as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Observe()]))

    assert_served_natively(messages_server)
    assert [phase for phase, _ in observed] == ["sync", "async"]
    assert all(error is raised.value for _, error in observed)


def sse_payload() -> bytes:
    return b"".join(STREAM.payloads())


@pytest.mark.asyncio
async def test_native_messages_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    assert get_hidden_params_dict(stream)["additional_headers"]["x-litellm-rust"] == "true"
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert first + b"".join(rest) == sse_payload()
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].kwargs["stream"] is True
    assert success[0].kwargs["completion_start_time"] is not None
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_messages_stream_closed_early_logs_success_once_for_what_was_delivered(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()

    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


def test_native_sync_messages_stream_relays_provider_events_and_logs_success_once(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = litellm.anthropic.messages.create(**arguments(messages_server, stream=True, callbacks=[recorder]))
    assert isinstance(stream, Iterator)
    assert get_hidden_params_dict(stream)["additional_headers"]["x-litellm-rust"] == "true"

    assert b"".join(stream) == sse_payload()
    assert_served_natively(messages_server)
    assert len(recorder.wait_for("async_log_success_event")) == 1


def test_native_sync_messages_returns_the_provider_message(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = litellm.anthropic.messages.create(**arguments(messages_server, callbacks=[recorder]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert len(recorder.wait_for("log_success_event")) == 1


@pytest.mark.asyncio
async def test_native_messages_pre_call_sees_the_shaped_optional_params(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], temperature=0.2, top_k=3, drop_params=True)
    )

    sent: Final = messages_server.requests[0].body
    assert not {"temperature", "top_k"} & sent.keys()
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    optional_params: Final = pre_call["optional_params"]
    assert isinstance(optional_params, dict)
    assert not {"model", "messages", "temperature", "top_k"} & optional_params.keys()
    assert optional_params["max_tokens"] == sent["max_tokens"]


@pytest.mark.asyncio
async def test_native_messages_failing_pre_call_logger_does_not_fail_the_call(messages_server: RecordingServer) -> None:
    class Broken(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            raise RuntimeError("logger exploded")

    response: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Broken()]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]


@pytest.mark.asyncio
async def test_native_messages_stream_success_log_carries_usage_rebuilt_from_the_relayed_events(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    async for _ in stream:
        pass

    success: Final = await recorder.wait_for_async("async_log_success_event")
    usage: Final = success[0].response.usage
    assert usage.completion_tokens == MESSAGES_EVENTS[4][1]["usage"]["output_tokens"]
    assert usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert success[0].response.choices[0].message.content == "Hello from native Messages"


@pytest.fixture
def callback_route() -> CallbackRoute:
    return messages_contract()


@pytest.fixture
def callback_stream_response() -> ResponseSpec:
    return STREAM


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine",
    (
        "python",
        pytest.param(
            "rust",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason="Messages native dispatch omits async_pre_request_hook"
            ),
        ),
    ),
)
async def test_messages_request_hook_changes_provider_request(
    messages_server: RecordingServer, callback_route: CallbackRoute, engine: str
) -> None:
    from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages

    class Edit(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: object, kwargs: dict[str, object]
        ) -> dict[str, object]:
            return {**kwargs, "stop_sequences": ["callback-stop"]}

    litellm.callbacks.append(Edit())
    if engine == "python":
        await anthropic_messages(**arguments(messages_server))
    else:
        await callback_route.invoke(messages_server)
    assert messages_server.requests[0].body.get("stop_sequences") == ["callback-stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine",
    (
        "python",
        pytest.param(
            "rust",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason="Messages native dispatch omits agentic completion hooks"
            ),
        ),
    ),
)
async def test_messages_agentic_hook_replaces_public_response(
    messages_server: RecordingServer, callback_route: CallbackRoute, engine: str
) -> None:
    from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages

    class Complete(CustomLogger):
        async def async_should_run_agentic_loop(self, **kwargs: object) -> tuple[bool, dict[str, object]]:
            return True, {"tool_calls": [{"name": "callback_tool", "input": {}}]}

        async def async_run_agentic_loop(self, **kwargs: object) -> object:
            return callback_route.replace(kwargs["response"], "completed by agentic hook")

    litellm.callbacks.append(Complete())
    response: Final = (
        await anthropic_messages(**arguments(messages_server))
        if engine == "python"
        else await callback_route.invoke(messages_server)
    )
    assert callback_route.text(response) == "completed by agentic hook"
    assert len(messages_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine",
    (
        "python",
        pytest.param(
            "rust",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="Messages native pre_call uses a label and credential instead of the Python input contract",
            ),
        ),
    ),
)
async def test_messages_pre_api_input_describes_the_request_without_credentials(
    messages_server: RecordingServer, callback_route: CallbackRoute, engine: str
) -> None:
    from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages

    recorder: Final = RecordingLogger()
    if engine == "python":
        await anthropic_messages(**arguments(messages_server, callbacks=[recorder]))
    else:
        await callback_route.invoke(messages_server, callbacks=[recorder])
    details: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(details, dict)
    assert details.get("api_key") == ""
    request_input: Final = details["input"]
    assert isinstance(request_input, list) and len(request_input) == 1
    assert request_input[0]["role"] == "user"
    assert json.loads(request_input[0]["content"]) == messages_server.requests[0].body
