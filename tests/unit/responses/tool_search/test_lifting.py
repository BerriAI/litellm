import json
from datetime import datetime
from typing import Final

import httpx
import pytest
from openai.types.responses import ResponseToolSearchCall

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.hosted_vllm.responses.transformation import HostedVLLMResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.responses.tool_search.lifting import (
    ToolSearchEventLifter,
    lift_tool_search_calls,
    lift_tool_search_stream,
)
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamEvents

SEARCH_CALL: Final = {
    "type": "function_call",
    "id": "fc_search",
    "call_id": "call_search",
    "name": "tool_search",
    "arguments": '{"query": "calendar create", "limit": 1}',
    "status": "completed",
}
SHELL_CALL: Final = {
    "type": "function_call",
    "id": "fc_shell",
    "call_id": "call_shell",
    "name": "exec_command",
    "arguments": '{"cmd": "ls"}',
    "status": "completed",
}
NAMESPACED_SEARCH_CALL: Final = {**SHELL_CALL, "id": "fc_ns", "name": "tool_search", "namespace": "docs"}
MESSAGE: Final = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Searching the lab tools", "annotations": []}],
}


def _response(*output: dict[str, object]) -> dict[str, object]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "qwen",
        "status": "completed",
        "output": list(output),
        "tools": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
    }


def _event(event_type: str, **fields: object) -> dict[str, object]:
    return {"type": event_type, **fields}


STREAM_EVENTS: Final = [
    _event("response.created", response={**_response(), "status": "in_progress"}),
    _event(
        "response.output_item.added", output_index=0, item={**SEARCH_CALL, "arguments": "", "status": "in_progress"}
    ),
    _event("response.function_call_arguments.delta", item_id="fc_search", output_index=0, delta='{"query": '),
    _event("response.function_call_arguments.done", item_id="fc_search", output_index=0, arguments="{}"),
    _event("response.output_item.done", output_index=0, item=SEARCH_CALL),
    _event("response.output_item.added", output_index=1, item={**SHELL_CALL, "arguments": ""}),
    _event("response.function_call_arguments.delta", item_id="fc_shell", output_index=1, delta='{"cmd": "ls"}'),
    _event("response.output_item.done", output_index=1, item=SHELL_CALL),
    _event("response.completed", response=_response(SEARCH_CALL, SHELL_CALL)),
]


def _lifted_search_call() -> dict[str, object]:
    return {
        "type": "tool_search_call",
        "id": "tsc_search",
        "call_id": "call_search",
        "execution": "client",
        "status": "completed",
        "arguments": {"query": "calendar create", "limit": 1},
    }


def test_search_function_call_comes_back_as_a_tool_search_call():
    response: Final = ResponsesAPIResponse.model_validate(
        _response(SEARCH_CALL, SHELL_CALL, NAMESPACED_SEARCH_CALL, MESSAGE)
    )

    lifted: Final = lift_tool_search_calls(response)

    assert lifted.output[0] == ResponseToolSearchCall.model_validate(_lifted_search_call())
    assert all(after is before for after, before in zip(lifted.output[1:], response.output[1:], strict=True))


def test_response_without_a_search_call_is_returned_as_is():
    response: Final = ResponsesAPIResponse.model_validate(_response(SHELL_CALL))

    assert lift_tool_search_calls(response) is response


def test_search_arguments_that_are_not_a_json_object_become_the_query():
    response: Final = ResponsesAPIResponse.model_validate(_response({**SEARCH_CALL, "arguments": "calendar"}))

    lifted_item: Final = lift_tool_search_calls(response).output[0]

    assert isinstance(lifted_item, ResponseToolSearchCall)
    assert lifted_item.arguments == {"query": "calendar"}


def test_stream_events_of_a_search_call_are_lifted_and_its_argument_deltas_dropped():
    lifter: Final = ToolSearchEventLifter()

    lifted: Final = [lifter.lift(json.dumps(event)) for event in STREAM_EVENTS]

    kept: Final = [json.loads(data) for data in lifted if data is not None]
    assert [event["type"] for event in kept] == [
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert kept[1]["item"] == {**_lifted_search_call(), "status": "in_progress", "arguments": {}}
    assert kept[2]["item"] == _lifted_search_call()
    assert kept[3]["item"]["name"] == "exec_command"
    assert kept[6]["response"]["output"] == [_lifted_search_call(), SHELL_CALL]


def test_stream_data_that_is_not_json_passes_through():
    assert ToolSearchEventLifter().lift("[DONE]") == "[DONE]"


SSE_BODY: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in STREAM_EVENTS)


def _sse_upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=SSE_BODY.encode(), headers={"content-type": "text/event-stream"})


def _logging_obj(call_type: str) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type=call_type,
        start_time=datetime.now(),
        litellm_call_id=f"tool-search-{call_type}",
        function_id=f"tool-search-{call_type}",
    )
    logging_obj.model_call_details["litellm_params"] = {"aresponses": call_type == "aresponses"}
    return logging_obj


def _assert_search_lifted_and_logged(events: list[object], stream: BaseResponsesAPIStreamingIterator) -> None:
    assert [str(getattr(event, "type", "")) for event in events].count(
        str(ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA)
    ) == 1
    assert stream.completed_response is not None
    logged_output: Final = stream.completed_response.response.output
    assert [item["type"] if isinstance(item, dict) else item.type for item in logged_output] == [
        "tool_search_call",
        "function_call",
    ]


@pytest.mark.asyncio
async def test_a_live_stream_logs_the_lifted_response():
    async with httpx.AsyncClient(transport=httpx.MockTransport(_sse_upstream)) as client:
        upstream_response: Final = await client.send(
            client.build_request("POST", "http://vllm.test/v1/responses"), stream=True
        )
        stream: Final = lift_tool_search_stream(
            ResponsesAPIStreamingIterator(
                response=upstream_response,
                model="qwen",
                responses_api_provider_config=HostedVLLMResponsesAPIConfig(),
                logging_obj=_logging_obj("aresponses"),
                custom_llm_provider="hosted_vllm",
            )
        )
        events: Final = [event async for event in stream]

    _assert_search_lifted_and_logged(events, stream)


def test_a_live_sync_stream_logs_the_lifted_response():
    with httpx.Client(transport=httpx.MockTransport(_sse_upstream)) as client:
        upstream_response: Final = client.send(
            client.build_request("POST", "http://vllm.test/v1/responses"), stream=True
        )
        stream: Final = lift_tool_search_stream(
            SyncResponsesAPIStreamingIterator(
                response=upstream_response,
                model="qwen",
                responses_api_provider_config=HostedVLLMResponsesAPIConfig(),
                logging_obj=_logging_obj("responses"),
                custom_llm_provider="hosted_vllm",
            )
        )
        events: Final = list(stream)

    _assert_search_lifted_and_logged(events, stream)
