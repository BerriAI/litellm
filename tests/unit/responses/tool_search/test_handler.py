import asyncio
import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

CLIENT_TOOL_SEARCH: Final = {
    "type": "tool_search",
    "execution": "client",
    "description": "Search the deferred tools",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}
SEARCH_FUNCTION_CALL: Final = {
    "type": "function_call",
    "id": "fc_search",
    "call_id": "call_search",
    "name": "tool_search",
    "arguments": '{"query": "calendar"}',
    "status": "completed",
}
UPSTREAM_RESPONSE: Final = {
    "id": "resp_upstream",
    "object": "response",
    "created_at": 1,
    "model": "qwen",
    "status": "completed",
    "output": [SEARCH_FUNCTION_CALL],
    "tools": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
}


class _Upstream:
    def __init__(self) -> None:
        self.bodies: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=UPSTREAM_RESPONSE)


class _SuccessLog(CustomLogger):
    def __init__(self, call_id: str) -> None:
        super().__init__()
        self.call_id: Final = call_id
        self.outputs: list[list[object]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("litellm_call_id") == self.call_id:
            self.outputs.append(list(response_obj.output))


def _types(tools: object) -> list[object]:
    assert isinstance(tools, list)
    return [tool.get("type") for tool in tools]


def _output_types(output: object) -> list[object]:
    assert isinstance(output, list)
    return [item.get("type") if isinstance(item, dict) else getattr(item, "type", None) for item in output]


async def _hosted_vllm_call(upstream: _Upstream, **overrides: object):
    request: Final = {"input": "find a calendar tool", "tools": [CLIENT_TOOL_SEARCH], **overrides}
    return await litellm.aresponses(
        model="hosted_vllm/qwen",
        api_base="http://vllm.test/v1",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
        **request,
    )


@pytest.mark.asyncio
async def test_hosted_vllm_receives_a_function_and_the_client_a_tool_search_call():
    upstream: Final = _Upstream()

    response: Final = await _hosted_vllm_call(upstream)

    assert _types(upstream.bodies[0]["tools"]) == ["function"]
    assert _output_types(response.output) == ["tool_search_call"]
    assert response.output[0].arguments == {"query": "calendar"}


@pytest.mark.asyncio
async def test_a_streamed_hosted_vllm_search_reaches_the_client_as_tool_search_call_events():
    upstream: Final = _Upstream()

    stream: Final = await _hosted_vllm_call(upstream, stream=True)
    events: Final = [event async for event in stream]

    item_events: Final = [event for event in events if getattr(event, "item", None) is not None]
    assert [event.item.type for event in item_events] == ["tool_search_call", "tool_search_call"]
    assert not [event for event in events if "function_call_arguments" in str(event.type.value)]


@pytest.mark.asyncio
async def test_the_next_turn_replays_the_search_as_function_items_with_the_loaded_tools():
    upstream: Final = _Upstream()
    loaded_namespace: Final = {
        "type": "namespace",
        "name": "calendar",
        "description": "Calendar tools",
        "tools": [{"type": "function", "name": "create_event", "defer_loading": True, "parameters": {}}],
    }

    await _hosted_vllm_call(
        upstream,
        input=[
            {"role": "user", "content": "add a meeting"},
            {"type": "tool_search_call", "call_id": "call_search", "execution": "client", "arguments": {"query": "x"}},
            {
                "type": "tool_search_output",
                "call_id": "call_search",
                "execution": "client",
                "tools": [loaded_namespace],
            },
        ],
    )

    sent: Final = upstream.bodies[0]
    assert _output_types(sent["input"])[1:] == ["function_call", "function_call_output"]
    assert _types(sent["tools"]) == ["function", "namespace"]


@pytest.mark.asyncio
async def test_a_native_openai_deployment_keeps_tool_search():
    upstream: Final = _Upstream()

    await litellm.aresponses(
        model="openai/gpt-tool-search-test",
        api_key="test-key",
        api_base="http://openai.test/v1",
        input="find a calendar tool",
        tools=[CLIENT_TOOL_SEARCH],
        client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
    )

    assert _types(upstream.bodies[0]["tools"]) == ["tool_search"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "supports_tool_search", "expected_tool_type"),
    [
        ("openai/gpt-tool-search-test", False, "function"),
        ("hosted_vllm/qwen", True, "tool_search"),
    ],
)
async def test_the_deployment_model_info_decides_whether_tool_search_is_native(
    model: str, supports_tool_search: bool, expected_tool_type: str
):
    upstream: Final = _Upstream()

    await litellm.aresponses(
        model=model,
        api_key="test-key",
        api_base="http://upstream.test/v1",
        input="find a calendar tool",
        tools=[CLIENT_TOOL_SEARCH],
        model_info={"supports_tool_search": supports_tool_search},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
    )

    assert _types(upstream.bodies[0]["tools"]) == [expected_tool_type]


@pytest.mark.asyncio
async def test_a_client_function_named_tool_search_is_rejected_before_any_upstream_call():
    upstream: Final = _Upstream()

    with pytest.raises(litellm.BadRequestError, match="named 'tool_search' can't be declared alongside"):
        await _hosted_vllm_call(
            upstream, tools=[CLIENT_TOOL_SEARCH, {"type": "function", "name": "tool_search", "parameters": {}}]
        )

    assert upstream.bodies == []


@pytest.mark.asyncio
async def test_emulated_file_search_leaves_client_tool_search_as_declared():
    upstream: Final = _Upstream()

    await litellm.aresponses(
        model="fireworks_ai/qwen",
        api_key="test-key",
        api_base="http://fireworks.test/v1",
        input="find a calendar tool",
        tools=[CLIENT_TOOL_SEARCH, {"type": "file_search", "vector_store_ids": ["vs_lab"]}],
        client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
    )

    assert _types(upstream.bodies[0]["tools"]) == ["tool_search", "function"]


@pytest.mark.asyncio
async def test_a_client_function_named_like_the_file_search_emulation_keeps_tool_search_lowered():
    upstream: Final = _Upstream()
    own_function: Final = {
        "type": "function",
        "name": "litellm_file_search",
        "parameters": {"type": "object", "properties": {}},
    }

    await _hosted_vllm_call(upstream, tools=[CLIENT_TOOL_SEARCH, own_function])

    assert [tool["name"] for tool in upstream.bodies[0]["tools"]] == ["tool_search", "litellm_file_search"]


@pytest.mark.asyncio
async def test_the_call_is_logged_once_with_the_tool_search_call(monkeypatch: pytest.MonkeyPatch):
    success_log: Final = _SuccessLog("tool-search-logged-once")
    monkeypatch.setattr(litellm, "callbacks", [success_log])

    await _hosted_vllm_call(_Upstream(), litellm_call_id="tool-search-logged-once")
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)

    assert [_output_types(output) for output in success_log.outputs] == [["tool_search_call"]]


def test_the_sync_api_lowers_and_lifts_too():
    upstream: Final = _Upstream()

    response: Final = litellm.responses(
        model="hosted_vllm/qwen",
        api_base="http://vllm.test/v1",
        input="find a calendar tool",
        tools=[CLIENT_TOOL_SEARCH],
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream))),
    )

    assert _types(upstream.bodies[0]["tools"]) == ["function"]
    assert _output_types(response.output) == ["tool_search_call"]
