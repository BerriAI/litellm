import json
import sys
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.responses.mcp.tool_precheck import may_name_litellm_mcp_gateway

GATEWAY_REGISTRY_MODULE: Final = "litellm.proxy.mcp_server.mcp_server_manager"
GATEWAY_HANDLER_MODULES: Final = (
    "litellm.responses.mcp.litellm_proxy_mcp_handler",
    "litellm.responses.mcp.chat_completions_handler",
    "litellm.llms.anthropic.messages.native.mcp_handler",
)

FUNCTION_TOOL: Final = {
    "type": "function",
    "function": {"name": "noop", "description": "No operation", "parameters": {"type": "object", "properties": {}}},
}
RESPONSES_FUNCTION_TOOL: Final = {
    "type": "function",
    "name": "noop",
    "description": "No operation",
    "parameters": {"type": "object", "properties": {}},
}
ANTHROPIC_FUNCTION_TOOL: Final = {
    "name": "noop",
    "description": "No operation",
    "input_schema": {"type": "object", "properties": {}},
}
ANTHROPIC_MESSAGE_PAYLOAD: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
GATEWAY_MCP_TOOL: Final = {"type": "mcp", "server_label": "litellm", "server_url": "litellm_proxy"}
GATEWAY_SERVER_MCP_TOOL: Final = {"type": "mcp", "server_label": "x", "server_url": "litellm_proxy/mcp/github"}
GATEWAY_PATH_MCP_TOOL: Final = {"type": "mcp", "server_label": "x", "server_url": "https://gw.example.com/mcp/github"}
REMOTE_MCP_TOOL: Final = {
    "type": "mcp",
    "server_label": "deepwiki",
    "server_url": "https://mcp.deepwiki.com/sse",
    "require_approval": "never",
}

RESPONSES_PAYLOAD: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1734366691,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "tools": [],
}


@pytest.fixture
def gateway_handler_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in GATEWAY_HANDLER_MODULES:
        monkeypatch.setitem(sys.modules, module_name, None)


class _FailingImport(MetaPathFinder):
    def __init__(self, module_name: str, error: ImportError) -> None:
        self.module_name: Final = module_name
        self.error: Final = error

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == self.module_name:
            raise self.error
        return None


@pytest.fixture(
    params=[
        ModuleNotFoundError("No module named 'mcp'"),
        ImportError("cannot import name 'CacheableResult' from 'mcp.types'"),
        ModuleNotFoundError("No module named 'fastapi'"),
    ],
    ids=["mcp-missing", "mcp-1x", "proxy-extras-missing"],
)
def gateway_registry_unavailable(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, gateway_handler_unavailable: None
) -> None:
    monkeypatch.delitem(sys.modules, GATEWAY_REGISTRY_MODULE, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FailingImport(GATEWAY_REGISTRY_MODULE, request.param), *sys.meta_path])


class _ToolList(list[Mapping[str, object]]):
    pass


@pytest.mark.parametrize(
    "tools",
    [[FUNCTION_TOOL], [REMOTE_MCP_TOOL], _ToolList([FUNCTION_TOOL]), (FUNCTION_TOOL, REMOTE_MCP_TOOL)],
    ids=["function-tool", "remote-mcp-tool", "list-subclass", "tuple"],
)
@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_completion_without_a_gateway_tool_does_not_import_the_gateway_handler(tools: Iterable[Mapping[str, object]]):
    response: Final = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        mock_response="ok",
    )

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_handler_unavailable")
async def test_acompletion_with_function_tool_does_not_import_the_gateway_handler():
    response: Final = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[FUNCTION_TOOL],
        mock_response="ok",
    )

    assert response.choices[0].message.content == "ok"


@pytest.mark.parametrize(
    "tool",
    [GATEWAY_MCP_TOOL, GATEWAY_SERVER_MCP_TOOL, GATEWAY_PATH_MCP_TOOL],
    ids=["litellm-proxy", "litellm-proxy-server", "gateway-path-url"],
)
@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_completion_with_gateway_mcp_tool_still_reaches_the_handler(tool: Mapping[str, object]):
    with pytest.raises(ModuleNotFoundError, match=r"litellm_proxy_mcp_handler|chat_completions_handler"):
        litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[tool],
            mock_response="ok",
        )


@pytest.mark.parametrize(
    "tools",
    [None, [RESPONSES_FUNCTION_TOOL], [REMOTE_MCP_TOOL]],
    ids=["no-tools", "function-tool", "remote-mcp-tool"],
)
@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_responses_without_a_gateway_tool_does_not_import_the_gateway_handler(
    tools: list[Mapping[str, object]] | None, respx_mock: respx.MockRouter
):
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=RESPONSES_PAYLOAD)
    )

    response: Final = litellm.responses(model="openai/gpt-4o-mini", input="hi", tools=tools, api_key="sk-test")

    assert upstream.called
    assert response.output[0].content[0].text == "Done."


@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_responses_with_gateway_mcp_tool_still_reaches_the_handler():
    with pytest.raises(litellm.APIConnectionError, match=r"litellm_proxy_mcp_handler"):
        litellm.responses(model="openai/gpt-4o-mini", input="hi", tools=[GATEWAY_MCP_TOOL], api_key="sk-test")


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_handler_unavailable")
async def test_anthropic_messages_with_function_tool_does_not_import_the_gateway_handler(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    upstream: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGE_PAYLOAD)
    )

    response: Final = await litellm.anthropic_interface.messages.acreate(
        model="anthropic/claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        tools=[ANTHROPIC_FUNCTION_TOOL],
        api_key="sk-test",
    )

    assert upstream.called
    assert response["content"][0]["text"] == "ok"


@pytest.mark.parametrize(
    "tools, expected",
    [
        (None, False),
        ([], False),
        ([FUNCTION_TOOL], False),
        ([RESPONSES_FUNCTION_TOOL], False),
        ([{"type": "web_search"}], False),
        (["not-a-mapping"], False),
        ([REMOTE_MCP_TOOL], False),
        ([{"type": "mcp", "server_label": "x"}], False),
        ([{"type": "mcp", "server_label": "x", "server_url": None}], False),
        ([{"server_url": "litellm_proxy"}], False),
        ([FUNCTION_TOOL, GATEWAY_MCP_TOOL], True),
        ([GATEWAY_SERVER_MCP_TOOL], True),
        ([GATEWAY_PATH_MCP_TOOL], True),
        ([REMOTE_MCP_TOOL, GATEWAY_PATH_MCP_TOOL], True),
    ],
    ids=[
        "none",
        "empty",
        "chat-function",
        "responses-function",
        "web-search",
        "non-mapping",
        "remote-mcp",
        "mcp-without-url",
        "mcp-null-url",
        "gateway-url-without-mcp-type",
        "gateway-among-functions",
        "gateway-server",
        "gateway-path-url",
        "gateway-path-after-remote-mcp",
    ],
)
def test_may_name_litellm_mcp_gateway_matches_only_gateway_mcp_tools(tools: list[object] | None, expected: bool):
    assert may_name_litellm_mcp_gateway(tools) is expected


@pytest.mark.parametrize(
    "tools, expected",
    [
        ([GATEWAY_PATH_MCP_TOOL], False),
        ([FUNCTION_TOOL, GATEWAY_PATH_MCP_TOOL], False),
        ([GATEWAY_MCP_TOOL], True),
        ([GATEWAY_SERVER_MCP_TOOL, GATEWAY_PATH_MCP_TOOL], True),
    ],
    ids=["gateway-path-url", "gateway-path-among-functions", "litellm-proxy", "litellm-proxy-server-and-path-url"],
)
@pytest.mark.usefixtures("gateway_registry_unavailable")
def test_proxy_path_url_names_the_gateway_only_when_its_registry_loads(tools: list[object], expected: bool):
    assert may_name_litellm_mcp_gateway(tools) is expected


@pytest.mark.usefixtures("gateway_registry_unavailable")
def test_responses_sends_a_proxy_path_mcp_url_to_the_provider_without_the_gateway_registry(
    respx_mock: respx.MockRouter,
):
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=RESPONSES_PAYLOAD)
    )

    response: Final = litellm.responses(
        model="openai/gpt-4o-mini", input="hi", tools=[GATEWAY_PATH_MCP_TOOL], api_key="sk-test"
    )

    assert response.output[0].content[0].text == "Done."
    assert json.loads(upstream.calls.last.request.content)["tools"] == [GATEWAY_PATH_MCP_TOOL]


@pytest.mark.usefixtures("gateway_registry_unavailable")
def test_completion_sends_a_proxy_path_mcp_url_to_the_provider_without_the_gateway_registry():
    response: Final = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[GATEWAY_PATH_MCP_TOOL],
        mock_response="ok",
    )

    assert response.choices[0].message.content == "ok"


class _ToolTuple(tuple[Mapping[str, object], ...]):
    pass


@pytest.mark.parametrize(
    "make_tools",
    [
        lambda items: (item for item in items),
        lambda items: iter(items),
        lambda items: deque(items),
        lambda items: _ToolList(items),
        lambda items: _ToolTuple(items),
    ],
    ids=["generator", "iterator", "deque", "list-subclass", "tuple-subclass"],
)
@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_precheck_reads_every_iterable_shape_without_the_gateway_handler(
    make_tools: Callable[[list[Mapping[str, object]]], Iterable[Mapping[str, object]]],
):
    assert may_name_litellm_mcp_gateway(make_tools([FUNCTION_TOOL, REMOTE_MCP_TOOL])) is False
    assert may_name_litellm_mcp_gateway(make_tools([FUNCTION_TOOL, GATEWAY_MCP_TOOL])) is True
