import asyncio
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.integrations.custom_logger import (
    RESPONSES_AGENTIC_SURFACE,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.test_inference import RESPONSES_MODEL, RESPONSES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

TOOL: Final = {
    "type": "function",
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
TOOL_CALL_RESPONSE: Final = {
    **RESPONSES_RESPONSE,
    "id": "resp_tool_call",
    "output": [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
        }
    ],
}
FINAL_RESPONSE: Final = {
    **RESPONSES_RESPONSE,
    "id": "resp_final",
    "output": [
        {
            "type": "message",
            "id": "msg_final",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "18C and raining in Paris", "annotations": []}],
        }
    ],
}
POST_LOOP_RESPONSE: Final = {**FINAL_RESPONSE, "id": "resp_post_loop"}
INPUT: Final = "What is the weather in Paris?"
TOOL_RESULT_ITEMS: Final = (
    TOOL_CALL_RESPONSE["output"][0],
    {"type": "function_call_output", "call_id": "call_1", "output": "18C"},
)


@pytest.fixture(autouse=True)
def opt_responses_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.RESPONSES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def responses_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=TOOL_CALL_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": RESPONSES_MODEL,
        "input": INPUT,
        "tools": [TOOL],
        "max_output_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def assert_served_natively(server: RecordingServer, count: int = 1) -> None:
    assert len(server.requests) == count
    assert all(request.path == "/responses" for request in server.requests)
    assert all(not request.headers.get("user-agent", "").startswith("python-httpx") for request in server.requests)


class Gate(CustomLogger):
    def __init__(self, decision: str) -> None:
        super().__init__()
        self.decision = decision
        self.seen: list[dict[str, object]] = []

    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ) -> tuple[bool, dict[str, object]]:
        self.seen.append(
            {
                "id": response.id,
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": stream,
                "provider": custom_llm_provider,
                "surface": kwargs.get("_agentic_loop_api_surface"),
                "api_key": kwargs.get("api_key"),
            }
        )
        if self.decision == "crash":
            raise RuntimeError("gate crashed")
        asks_for_a_tool: Final = any(item.type == "function_call" for item in response.output)
        return self.decision != "decline" and asks_for_a_tool, {}

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
        assert anthropic_messages_optional_request_params["max_output_tokens"] == 64
        return litellm.types.llms.openai.ResponsesAPIResponse.model_validate(FINAL_RESPONSE)


class Planned(Gate):
    def __init__(self, plan: str) -> None:
        super().__init__("run")
        self.plan = plan
        self.post_loop_inputs: list[tuple[str, str]] = []
        self.cleaned_up: list[str] = []

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
    ) -> AgenticLoopPlan:
        match self.plan:
            case "override":
                return AgenticLoopPlan(
                    response_override=litellm.types.llms.openai.ResponsesAPIResponse.model_validate(FINAL_RESPONSE)
                )
            case "follow-up":
                return AgenticLoopPlan(
                    run_agentic_loop=True,
                    request_patch=AgenticLoopRequestPatch(
                        messages=[{"role": "user", "content": INPUT}, *TOOL_RESULT_ITEMS], tools=[TOOL]
                    ),
                )
            case _:
                return AgenticLoopPlan(run_agentic_loop=False)

    async def async_post_agentic_loop_response_hook(self, response, plan, kwargs):
        from litellm.responses.utils import ResponsesAPIRequestUtils

        original_id: Final = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            response.id
        )
        self.post_loop_inputs.append((original_id, str(plan.stop_reason)))
        return litellm.types.llms.openai.ResponsesAPIResponse.model_validate(POST_LOOP_RESPONSE)

    async def async_agentic_loop_cleanup_hook(self, plan, kwargs) -> None:
        self.cleaned_up.append(str(plan.run_agentic_loop))


def expected_seen(**overrides: object) -> dict[str, object]:
    return {
        "id": "resp_tool_call",
        "model": RESPONSES_MODEL.removeprefix("openai/"),
        "messages": [{"role": "user", "content": INPUT}],
        "tools": [TOOL],
        "stream": False,
        "provider": "openai",
        "surface": RESPONSES_AGENTIC_SURFACE,
        "api_key": "test-key",
        **overrides,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback", "expected", "provider_calls"),
    (
        (Gate("run"), FINAL_RESPONSE, 1),
        (Gate("decline"), TOOL_CALL_RESPONSE, 1),
        (Gate("crash"), TOOL_CALL_RESPONSE, 1),
        (Planned("override"), FINAL_RESPONSE, 1),
        (Planned("skip"), TOOL_CALL_RESPONSE, 1),
        (Planned("follow-up"), POST_LOOP_RESPONSE, 2),
    ),
    ids=("loop-runs", "gate-declines", "gate-crashes", "plan-overrides", "plan-skips", "plan-follows-up"),
)
async def test_native_responses_agentic_hooks_select_what_the_caller_and_the_success_log_see(
    responses_server: RecordingServer, callback: Gate, expected: dict[str, object], provider_calls: int
) -> None:
    responses_server.expected_requests = provider_calls
    if provider_calls == 2:
        responses_server.enqueue(ResponseSpec(body=TOOL_CALL_RESPONSE))
        responses_server.enqueue(ResponseSpec(body=FINAL_RESPONSE))
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [callback]):
        response: Final = await litellm.aresponses(**arguments(responses_server, callbacks=[recorder]))

    assert_served_natively(responses_server, provider_calls)
    assert callback.seen[0] == expected_seen()
    assert len(callback.seen) == provider_calls
    assert response.id == expected["id"]
    assert response.output[0].type == expected["output"][0]["type"]
    success: Final = await recorder.wait_for_async("async_log_success_event", count=provider_calls)
    logged: Final = [event.response.id for event in success]
    assert expected["id"] in logged
    assert len(logged) == provider_calls
    assert "log_failure_event" not in recorder.names


def test_native_sync_responses_run_the_agentic_hooks_too(responses_server: RecordingServer) -> None:
    callback: Final = Gate("run")
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [callback]):
        response: Final = litellm.responses(**arguments(responses_server, callbacks=[recorder]))

    assert_served_natively(responses_server)
    assert callback.seen == [expected_seen()]
    assert response.id == "resp_final"
    success: Final = recorder.wait_for("log_success_event")
    assert success[0].response.id == "resp_final"


@pytest.mark.asyncio
async def test_native_responses_plan_follow_up_is_a_second_native_call_then_post_loop_and_cleanup_hooks(
    responses_server: RecordingServer,
) -> None:
    responses_server.expected_requests = 2
    responses_server.enqueue(ResponseSpec(body=TOOL_CALL_RESPONSE))
    responses_server.enqueue(ResponseSpec(body=FINAL_RESPONSE))
    callback: Final = Planned("follow-up")

    with rebound(litellm, "callbacks", [callback]):
        response: Final = await litellm.aresponses(**arguments(responses_server))

    assert response.id == "resp_post_loop"
    first, second = responses_server.requests
    assert first.body["input"] == INPUT
    assert second.body["input"] == [{"role": "user", "content": INPUT}, *TOOL_RESULT_ITEMS]
    assert second.body["tools"] == [TOOL]
    assert [entry["id"] for entry in callback.seen] == ["resp_tool_call", "resp_final"]
    assert callback.seen[1]["messages"] == [{"role": "user", "content": INPUT}, *TOOL_RESULT_ITEMS]
    assert callback.post_loop_inputs == [("resp_final", "None")]
    assert callback.cleaned_up == ["True"]


@pytest.mark.asyncio
async def test_native_responses_agentic_hooks_run_before_the_deployment_success_hook(
    responses_server: RecordingServer,
) -> None:
    order: Final = []

    class Loop(Gate):
        async def async_should_run_agentic_loop(
            self, response, model, messages, tools, stream, custom_llm_provider, kwargs
        ):
            order.append(("agentic", response.id))
            return await super().async_should_run_agentic_loop(
                response, model, messages, tools, stream, custom_llm_provider, kwargs
            )

        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            order.append(("deployment", response.id))

    with rebound(litellm, "callbacks", [Loop("run")]):
        await litellm.aresponses(**arguments(responses_server))

    assert order == [("agentic", "resp_tool_call"), ("deployment", "resp_final")]


@pytest.mark.asyncio
async def test_native_responses_without_agentic_callbacks_keep_the_provider_response(
    responses_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [CustomLogger()]):
        response: Final = await litellm.aresponses(**arguments(responses_server, callbacks=[recorder]))

    assert_served_natively(responses_server)
    assert response.id == "resp_tool_call"
    assert asyncio.iscoroutine(response) is False
