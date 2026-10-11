from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

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


@pytest.mark.asyncio
async def test_native_messages_does_not_restore_keywords_deleted_by_a_pre_call_hook(
    messages_server: RecordingServer,
) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("system", "extra_headers")}

    call_arguments: Final = arguments(
        messages_server,
        system="remove this instruction",
        extra_headers={"x-deleted": "remove this header"},
        headers={"x-kept": "keep this header"},
        stream=False,
    )
    with rebound(litellm, "callbacks", [DropKeywords()]):
        await litellm.anthropic.messages.acreate(**call_arguments)

    assert_served_natively(messages_server)
    sent: Final = messages_server.requests[0]
    assert sent.body == {
        "model": MESSAGES_MODEL.removeprefix("anthropic/"),
        "messages": call_arguments["messages"],
        "max_tokens": call_arguments["max_tokens"],
        "stream": call_arguments["stream"],
    }
    assert sent.headers.get("x-deleted") is None
    assert sent.headers.get("x-kept") == "keep this header"


TOOL: Final = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
TOOL_USE_RESPONSE: Final = {
    **MESSAGES_RESPONSE,
    "id": "msg_tool_use",
    "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
    "stop_reason": "tool_use",
}
FINAL_RESPONSE: Final = {
    **MESSAGES_RESPONSE,
    "id": "msg_final",
    "content": [{"type": "text", "text": "18C and raining in Paris"}],
}


def agentic_arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return arguments(server, tools=[TOOL], **kwargs)


class RewriteOneField(CustomLogger):
    def __init__(self, field: str, value: object) -> None:
        super().__init__()
        self.field = field
        self.value = value
        self.seen: list[dict[str, object]] = []

    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object]:
        self.seen.append(dict(kwargs))
        return {**kwargs, self.field: self.value}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("system", "rewritten by the body hook"),
        ("stop_sequences", ["END"]),
        ("max_tokens", 32),
        ("metadata", {"user_id": "hook"}),
        ("tools", [TOOL]),
    ),
    ids=("system", "stop_sequences", "max_tokens", "metadata", "tools"),
)
@pytest.mark.parametrize("stream", (False, True), ids=("complete", "stream"))
async def test_native_messages_pre_request_edit_reaches_the_provider_and_the_pre_call_log(
    messages_server: RecordingServer, field: str, value: object, stream: bool
) -> None:
    if stream:
        messages_server.enqueue(STREAM)
    hook: Final = RewriteOneField(field, value)
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [hook]):
        result: Final = await litellm.anthropic.messages.acreate(
            **arguments(messages_server, stream=stream, callbacks=[recorder])
        )
        if stream:
            assert isinstance(result, AsyncIterator)
            assert b"".join([chunk async for chunk in result]) == sse_payload()
        else:
            assert result["content"] == MESSAGES_RESPONSE["content"]

    assert_served_natively(messages_server)
    sent: Final = messages_server.requests[0].body
    assert sent[field] == value
    assert sent["stream"] is stream
    assert len(hook.seen) == 1
    assert hook.seen[0]["stream"] is stream
    assert hook.seen[0]["litellm_params"] == {"custom_llm_provider": "anthropic"}
    assert field not in hook.seen[0] or hook.seen[0][field] != value
    pre_call: Final = recorder.wait_for("log_pre_api_call")
    assert request_body(pre_call[0].kwargs)[field] == value


@pytest.mark.asyncio
async def test_native_messages_pre_request_hooks_run_in_the_caller_task_and_chain_their_edits(
    messages_server: RecordingServer,
) -> None:
    import asyncio

    caller: Final = asyncio.current_task()
    first: Final = RewriteOneField("system", "first")
    second: Final = RewriteOneField("stop_sequences", ["END"])

    class InCallerTask(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> None:
            await asyncio.sleep(0)
            assert asyncio.current_task() is caller

    with rebound(litellm, "callbacks", [first, InCallerTask(), second]):
        await litellm.anthropic.messages.acreate(**arguments(messages_server))

    sent: Final = messages_server.requests[0].body
    assert sent["system"] == "first"
    assert sent["stop_sequences"] == ["END"]
    assert second.seen[0]["system"] == "first"


def test_native_sync_messages_run_no_body_hooks(messages_server: RecordingServer) -> None:
    hook: Final = RewriteOneField("system", "never sent")

    with rebound(litellm, "callbacks", [hook]):
        response: Final = litellm.anthropic.messages.create(**arguments(messages_server))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert hook.seen == []
    assert "system" not in messages_server.requests[0].body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "logged"),
    ((RuntimeError("pre-request rejected"), True), (__import__("asyncio").CancelledError(), False)),
    ids=("exception", "cancellation"),
)
async def test_native_messages_pre_request_failure_never_sends_and_ends_the_call(
    messages_server: RecordingServer, failure: BaseException, logged: bool
) -> None:
    messages_server.expected_requests = 0
    observed: Final = []

    class Reject(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> None:
            raise failure

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(kwargs["exception"])

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observed.append("success")

    with rebound(litellm, "callbacks", [Reject()]), pytest.raises(type(failure)) as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert raised.value is failure
    assert messages_server.requests == []
    await drain_logging()
    assert observed == ([failure] if logged else [])


class Gate(CustomLogger):
    streaming: bool = False

    def __init__(self, decision: str) -> None:
        super().__init__()
        self.decision = decision
        self.seen: list[object] = []

    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ) -> tuple[bool, dict[str, object]]:
        self.seen.append(response)
        assert model == "claude-sonnet-5"
        assert custom_llm_provider == "anthropic"
        assert stream is self.streaming
        assert tools == [TOOL]
        if self.decision == "crash":
            raise RuntimeError("gate crashed")
        return self.decision != "decline", {}

    async def async_run_agentic_loop(
        self,
        tools,
        model,
        messages,
        response,
        anthropic_messages_provider_config,
        anthropic_messages_optional_request_params,
        logging_obj,
        stream,
        kwargs,
    ) -> object:
        assert anthropic_messages_optional_request_params["max_tokens"] == 64
        return FINAL_RESPONSE


POST_LOOP_RESPONSE: Final = {**FINAL_RESPONSE, "id": "msg_post_loop"}
TOOL_RESULT_TURNS: Final = (
    {"role": "assistant", "content": TOOL_USE_RESPONSE["content"]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
)


class Planned(Gate):
    """Builds a typed plan instead of running the loop itself; the handler executes the plan."""

    def __init__(self, plan: str) -> None:
        super().__init__("run")
        self.plan = plan
        self.post_loop_inputs: list[tuple[str, str]] = []

    async def async_build_agentic_loop_plan(
        self,
        tools,
        model,
        messages,
        response,
        anthropic_messages_provider_config,
        anthropic_messages_optional_request_params,
        logging_obj,
        stream,
        kwargs,
    ):
        from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch

        match self.plan:
            case "override":
                return AgenticLoopPlan(response_override=FINAL_RESPONSE)
            case "follow-up":
                return AgenticLoopPlan(
                    run_agentic_loop=True,
                    request_patch=AgenticLoopRequestPatch(messages=[*messages, *TOOL_RESULT_TURNS]),
                )
            case "terminate":
                return AgenticLoopPlan(terminate=True, stop_reason="refused by the plan")
            case _:
                return AgenticLoopPlan(run_agentic_loop=False)

    async def async_post_agentic_loop_response_hook(self, response, plan, kwargs):
        self.post_loop_inputs.append((response["id"], str(plan.stop_reason)))
        return POST_LOOP_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback", "expected", "provider_calls"),
    (
        (Gate("run"), FINAL_RESPONSE, 1),
        (Gate("decline"), TOOL_USE_RESPONSE, 1),
        (Gate("crash"), TOOL_USE_RESPONSE, 1),
        (Planned("override"), FINAL_RESPONSE, 1),
        (Planned("skip"), TOOL_USE_RESPONSE, 1),
        (Planned("terminate"), POST_LOOP_RESPONSE, 1),
        (Planned("follow-up"), POST_LOOP_RESPONSE, 2),
    ),
    ids=(
        "loop-runs",
        "gate-declines",
        "gate-crashes",
        "plan-overrides",
        "plan-skips",
        "plan-terminates",
        "plan-follows-up",
    ),
)
async def test_native_messages_agentic_hooks_select_what_the_caller_and_the_success_log_see(
    messages_server: RecordingServer, callback: Gate, expected: dict[str, object], provider_calls: int
) -> None:
    messages_server.expected_requests = provider_calls
    messages_server.enqueue(ResponseSpec(body=TOOL_USE_RESPONSE))
    if provider_calls == 2:
        messages_server.enqueue(ResponseSpec(body=FINAL_RESPONSE))
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [callback]):
        response: Final = await litellm.anthropic.messages.acreate(
            **agentic_arguments(messages_server, callbacks=[recorder])
        )

    assert len(messages_server.requests) == provider_calls
    assert callback.seen[0]["content"] == TOOL_USE_RESPONSE["content"]
    assert response["id"] == expected["id"]
    assert response["content"] == expected["content"]
    success: Final = await recorder.wait_for_async("async_log_success_event", count=provider_calls)
    logged: Final = [event.response.id for event in success]
    assert logged[-1] == expected["id"]
    assert len(logged) == provider_calls
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_messages_plan_follow_up_is_a_second_native_call_whose_answer_reaches_the_post_loop_hook(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 2
    messages_server.enqueue(ResponseSpec(body=TOOL_USE_RESPONSE))
    messages_server.enqueue(ResponseSpec(body=FINAL_RESPONSE))
    callback: Final = Planned("follow-up")

    with rebound(litellm, "callbacks", [callback]):
        response: Final = await litellm.anthropic.messages.acreate(**agentic_arguments(messages_server))

    assert response["id"] == POST_LOOP_RESPONSE["id"]
    first, second = messages_server.requests
    assert first.body["messages"] == list(MESSAGES)
    assert second.body["messages"] == [*MESSAGES, *TOOL_RESULT_TURNS]
    assert second.body["tools"] == [TOOL]
    assert second.body["max_tokens"] == first.body["max_tokens"]
    assert all(not request.headers.get("user-agent", "").startswith("python-httpx") for request in (first, second))
    assert callback.post_loop_inputs == [("msg_final", "None")]


@pytest.mark.asyncio
async def test_native_messages_terminated_plan_hands_the_post_loop_hook_the_refused_turn(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(ResponseSpec(body=TOOL_USE_RESPONSE))
    callback: Final = Planned("terminate")

    with rebound(litellm, "callbacks", [callback]):
        response: Final = await litellm.anthropic.messages.acreate(**agentic_arguments(messages_server))

    assert response["id"] == POST_LOOP_RESPONSE["id"]
    assert callback.post_loop_inputs == [("msg_tool_use", "refused by the plan")]


@pytest.mark.asyncio
async def test_native_messages_agentic_loop_follow_up_is_a_second_native_call_with_the_tool_result(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 2
    messages_server.enqueue(ResponseSpec(body=TOOL_USE_RESPONSE))
    messages_server.enqueue(ResponseSpec(body=FINAL_RESPONSE))

    class ServerSideTool(Gate):
        async def async_run_agentic_loop(
            self,
            tools,
            model,
            messages,
            response,
            anthropic_messages_provider_config,
            anthropic_messages_optional_request_params,
            logging_obj,
            stream,
            kwargs,
        ) -> object:
            call: Final = response["content"][0]
            return await litellm.anthropic.messages.acreate(
                model=f"anthropic/{model}",
                max_tokens=anthropic_messages_optional_request_params["max_tokens"],
                messages=[
                    *messages,
                    {"role": "assistant", "content": response["content"]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "18C"}]},
                ],
                tools=anthropic_messages_optional_request_params["tools"],
                api_key=kwargs["api_key"],
                api_base=kwargs["api_base"],
            )

    recorder: Final = RecordingLogger()
    with rebound(litellm, "callbacks", [ServerSideTool("run")]):
        response: Final = await litellm.anthropic.messages.acreate(
            **agentic_arguments(messages_server, callbacks=[recorder])
        )

    assert response["content"] == FINAL_RESPONSE["content"]
    first, second = messages_server.requests
    assert first.body["messages"] == list(MESSAGES)
    assert second.body["messages"][1:] == [
        {"role": "assistant", "content": TOOL_USE_RESPONSE["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
    ]
    assert second.body["tools"] == [TOOL]
    assert all(not request.headers.get("user-agent", "").startswith("python-httpx") for request in (first, second))
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert {event.response.id for event in success} == {"msg_final"}


@pytest.mark.asyncio
async def test_native_messages_agentic_hooks_run_before_the_deployment_success_hook(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(ResponseSpec(body=TOOL_USE_RESPONSE))
    order: Final = []

    class Observe(Gate):
        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            order.append(("deployment", response["id"]))

    class Loop(Observe):
        async def async_should_run_agentic_loop(
            self, response, model, messages, tools, stream, custom_llm_provider, kwargs
        ):
            order.append(("agentic", response["id"]))
            return await super().async_should_run_agentic_loop(
                response, model, messages, tools, stream, custom_llm_provider, kwargs
            )

    with rebound(litellm, "callbacks", [Loop("run")]):
        await litellm.anthropic.messages.acreate(**agentic_arguments(messages_server))

    assert order == [("agentic", "msg_tool_use"), ("deployment", "msg_final")]


class ServerSideWeather(Gate):
    """Fulfils `get_weather` itself, and tells the gateway so a streamed tool call is never relayed."""

    server_fulfilled_tool_names: Final = frozenset({"get_weather"})
    streaming = True


def tool_use_events() -> tuple[tuple[str, dict[str, object]], ...]:
    return (
        (
            "message_start",
            {"type": "message_start", "message": {**TOOL_USE_RESPONSE, "content": [], "stop_reason": None}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def events_of(chunks: list[bytes]) -> list[tuple[str, dict[str, object]]]:
    import json

    events: list[tuple[str, dict[str, object]]] = []
    for block in b"".join(chunks).decode().split("\n\n"):
        lines = [line for line in block.split("\n") if line]
        if not lines:
            continue
        name = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


@pytest.mark.asyncio
async def test_native_messages_hold_a_stream_back_for_a_server_fulfilled_tool_and_stream_the_answer(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(ResponseSpec(body=None, events=tool_use_events()))
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [ServerSideWeather("run")]):
        stream: Final = await litellm.anthropic.messages.acreate(
            **agentic_arguments(messages_server, stream=True, callbacks=[recorder])
        )
        assert isinstance(stream, AsyncIterator)
        chunks: Final = [chunk async for chunk in stream]

    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    events: Final = events_of(chunks)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0][1]["message"]["id"] == "msg_final"
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "18C and raining in Paris"}
    assert all("tool_use" not in chunk.decode() for chunk in chunks)
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].kwargs["stream"] is True


@pytest.mark.asyncio
async def test_native_messages_relay_a_stream_live_and_skip_the_agentic_hooks_when_no_callback_declares_the_tool(
    messages_server: RecordingServer,
) -> None:
    """A callback that overrides the agentic gate without declaring the tool in
    `server_fulfilled_tool_names` gets a live relay and no end-of-stream hooks. Python would
    relay the tool call and then append the loop's answer as a second message on the same
    stream, which no Anthropic SSE client can consume, so the native path does not port that
    mode. Declaring the tool moves such a callback into hold-back."""
    messages_server.enqueue(ResponseSpec(body=None, events=tool_use_events()))
    gate: Final = Gate("run")

    with rebound(litellm, "callbacks", [gate]):
        stream: Final = await litellm.anthropic.messages.acreate(**agentic_arguments(messages_server, stream=True))
        assert isinstance(stream, AsyncIterator)
        chunks: Final = [chunk async for chunk in stream]

    assert_served_natively(messages_server)
    assert gate.seen == []
    assert [name for name, _ in events_of(chunks)][:2] == ["message_start", "content_block_start"]
    assert events_of(chunks)[1][1]["content_block"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_native_messages_held_back_stream_whose_tool_call_nobody_fulfilled_fails_instead_of_leaking(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(ResponseSpec(body=None, events=tool_use_events()))
    observed: Final = []

    class Declines(ServerSideWeather):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(kwargs["exception"])

    with rebound(litellm, "callbacks", [Declines("decline")]):
        stream: Final = await litellm.anthropic.messages.acreate(**agentic_arguments(messages_server, stream=True))
        assert isinstance(stream, AsyncIterator)
        with pytest.raises(litellm.InternalServerError) as raised:
            await anext(stream)

    assert "Server-side tool retrieval failed" in str(raised.value)
    assert_served_natively(messages_server)
    await drain_logging()
    assert observed == [raised.value]


@pytest.mark.asyncio
async def test_native_messages_synthesize_a_stream_when_a_pre_request_hook_turns_streaming_off(
    messages_server: RecordingServer,
) -> None:
    class TurnStreamOff(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> dict[str, object]:
            return {**kwargs, "stream": False}

    with rebound(litellm, "callbacks", [TurnStreamOff()]):
        stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
        assert isinstance(stream, AsyncIterator)
        chunks: Final = [chunk async for chunk in stream]

    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is False
    events: Final = events_of(chunks)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"]["text"] == MESSAGES_RESPONSE["content"][0]["text"]
    assert events[4][1]["delta"]["stop_reason"] == "end_turn"


SHORT_CIRCUIT_ANSWER: Final = {
    **MESSAGES_RESPONSE,
    "id": "msg_short_circuit",
    "content": [{"type": "text", "text": "answered without the model"}],
}


class OwnsAdvisorTools:
    """A `MessagesInterceptor` stand-in that owns any request carrying an `advisor` tool."""

    def __init__(self) -> None:
        self.handled: list[dict[str, object]] = []

    def can_handle(self, tools, custom_llm_provider) -> bool:
        return bool(tools) and any(tool.get("type") == "advisor" for tool in tools)

    async def handle(self, **kwargs):
        self.handled.append(kwargs)
        return dict(SHORT_CIRCUIT_ANSWER)


@pytest.fixture
def advisor_interceptor() -> Iterator[OwnsAdvisorTools]:
    from litellm.llms.anthropic.pass_through.messages import interceptors

    interceptor: Final = OwnsAdvisorTools()
    with rebound(interceptors, "_interceptors", [interceptor]):
        yield interceptor


@pytest.mark.asyncio
async def test_native_messages_let_an_interceptor_answer_without_a_provider_request(
    messages_server: RecordingServer, advisor_interceptor: OwnsAdvisorTools
) -> None:
    messages_server.expected_requests = 0
    recorder: Final = RecordingLogger()
    order: Final = []

    class Observe(Gate):
        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            order.append(("deployment", response["id"]))

    with rebound(litellm, "callbacks", [Observe("run")]):
        response: Final = await litellm.anthropic.messages.acreate(
            **arguments(messages_server, tools=[{"type": "advisor", "name": "advisor"}], callbacks=[recorder])
        )

    assert messages_server.requests == []
    assert response["id"] == "msg_short_circuit"
    [handled] = advisor_interceptor.handled
    assert handled["model"] == MESSAGES_MODEL
    assert handled["stream"] is False
    assert handled["max_tokens"] == 64
    assert handled["api_base"] == messages_server.base_url
    assert order == [("deployment", "msg_short_circuit")]
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert success[0].response.id == "msg_short_circuit"
    assert "log_pre_api_call" not in recorder.names


@pytest.mark.asyncio
async def test_native_messages_replay_an_interceptor_answer_to_a_streaming_caller(
    messages_server: RecordingServer, advisor_interceptor: OwnsAdvisorTools
) -> None:
    messages_server.expected_requests = 0

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, tools=[{"type": "advisor", "name": "advisor"}])
    )
    assert isinstance(stream, AsyncIterator)
    chunks: Final = [chunk async for chunk in stream]

    assert messages_server.requests == []
    events: Final = events_of(chunks)
    assert [name for name, _ in events][:2] == ["message_start", "content_block_start"]
    assert events[0][1]["message"]["id"] == "msg_short_circuit"
    assert events[2][1]["delta"]["text"] == "answered without the model"
    assert events[-1][0] == "message_stop"
    [handled] = advisor_interceptor.handled
    assert handled["stream"] is False


@pytest.mark.asyncio
async def test_native_messages_let_the_websearch_short_circuit_answer_before_any_interceptor(
    messages_server: RecordingServer, advisor_interceptor: OwnsAdvisorTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.integrations.websearch_interception import handler as websearch

    messages_server.expected_requests = 0

    class SearchAnswers(CustomLogger):
        async def try_short_circuit_search(self, model, messages, tools, custom_llm_provider, kwargs=None):
            return {**SHORT_CIRCUIT_ANSWER, "id": "msg_searched"}

    monkeypatch.setattr(websearch, "WebSearchInterceptionLogger", SearchAnswers)
    with rebound(litellm, "callbacks", [SearchAnswers()]):
        response: Final = await litellm.anthropic.messages.acreate(
            **arguments(
                messages_server, tools=[{"type": "web_search_20250305", "name": "web_search"}, {"type": "advisor"}]
            )
        )

    assert messages_server.requests == []
    assert response["id"] == "msg_searched"
    assert advisor_interceptor.handled == []


@pytest.mark.asyncio
async def test_native_messages_reach_the_provider_when_no_short_circuit_owns_the_request(
    messages_server: RecordingServer, advisor_interceptor: OwnsAdvisorTools
) -> None:
    response: Final = await litellm.anthropic.messages.acreate(**agentic_arguments(messages_server))

    assert_served_natively(messages_server)
    assert response["id"] == MESSAGES_RESPONSE["id"]
    assert advisor_interceptor.handled == []
