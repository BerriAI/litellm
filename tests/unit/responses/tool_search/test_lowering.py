import json
from typing import Final

import pytest

from litellm.responses.tool_search.lowering import (
    LoweredToolSearchRequest,
    ToolSearchFunctionNameTaken,
    lower_tool_search_request,
    needs_tool_search_lowering,
)

CLIENT_TOOL_SEARCH: Final = {
    "type": "tool_search",
    "execution": "client",
    "description": "Search the deferred tools",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "number"}},
        "required": ["query"],
    },
}
SHELL_TOOL: Final = {"type": "function", "name": "exec_command", "parameters": {"type": "object", "properties": {}}}
SEARCH_CALL: Final = {
    "type": "tool_search_call",
    "call_id": "call_search",
    "execution": "client",
    "status": "completed",
    "arguments": {"query": "calendar create", "limit": 1},
}
CALENDAR_NAMESPACE: Final = {
    "type": "namespace",
    "name": "calendar",
    "description": "Calendar tools",
    "tools": [
        {
            "type": "function",
            "name": "create_event",
            "description": "Create an event",
            "defer_loading": True,
            "parameters": {"properties": {"title": {"type": "string"}}},
        }
    ],
}
SEARCH_OUTPUT: Final = {
    "type": "tool_search_output",
    "call_id": "call_search",
    "execution": "client",
    "status": "completed",
    "tools": [CALENDAR_NAMESPACE],
}


def _lowered(input: object, tools: list[object], tool_choice: object = None) -> LoweredToolSearchRequest:
    lowering: Final = lower_tool_search_request(input=input, tools=tools, tool_choice=tool_choice)
    assert isinstance(lowering, LoweredToolSearchRequest)
    return lowering


def _tool_named(tools: tuple[object, ...], name: str) -> dict[str, object]:
    matches: Final = [tool for tool in tools if isinstance(tool, dict) and tool.get("name") == name]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    ("input", "tools", "expected"),
    [
        ("hi", [SHELL_TOOL], False),
        ("hi", [SHELL_TOOL, CLIENT_TOOL_SEARCH], True),
        ("hi", [{"type": "tool_search"}], False),
        ([{"role": "user", "content": "hi"}, SEARCH_CALL], [SHELL_TOOL], True),
        ([{"role": "user", "content": "hi"}, SEARCH_OUTPUT], None, True),
    ],
)
def test_only_client_tool_search_state_needs_lowering(input: object, tools: list[object] | None, expected: bool):
    assert needs_tool_search_lowering(input, tools) is expected


def test_client_tool_search_becomes_a_function_with_its_declared_schema():
    lowered: Final = _lowered("find a calendar tool", [SHELL_TOOL, CLIENT_TOOL_SEARCH])

    assert lowered.tools[0] is SHELL_TOOL
    assert _tool_named(lowered.tools, "tool_search") == {
        "type": "function",
        "name": "tool_search",
        "description": "Search the deferred tools",
        "parameters": CLIENT_TOOL_SEARCH["parameters"],
        "strict": False,
    }


def test_client_tool_search_without_schema_gets_a_query_parameter():
    lowered: Final = _lowered("hi", [{"type": "tool_search", "execution": "client"}])

    search_function: Final = _tool_named(lowered.tools, "tool_search")
    assert search_function["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def test_hosted_tool_search_is_left_to_the_provider():
    hosted: Final = {"type": "tool_search"}
    lowered: Final = _lowered([SEARCH_CALL], [hosted])

    assert lowered.tools == (hosted,)


def test_replayed_search_call_becomes_a_function_call_with_the_same_call_id():
    lowered: Final = _lowered([{"role": "user", "content": "hi"}, SEARCH_CALL], [CLIENT_TOOL_SEARCH])

    assert isinstance(lowered.input, list)
    function_call: Final = lowered.input[1]
    assert isinstance(function_call, dict)
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "call_search"
    assert function_call["name"] == "tool_search"
    assert json.loads(function_call["arguments"]) == SEARCH_CALL["arguments"]


def test_replayed_search_output_loads_its_tools_for_the_model():
    lowered: Final = _lowered([SEARCH_CALL, SEARCH_OUTPUT], [SHELL_TOOL, CLIENT_TOOL_SEARCH])

    assert isinstance(lowered.input, list)
    function_output: Final = lowered.input[1]
    assert isinstance(function_output, dict)
    assert function_output["type"] == "function_call_output"
    assert function_output["call_id"] == "call_search"
    assert json.loads(function_output["output"]) == {
        "tools": [{"type": "namespace", "name": "calendar", "description": "Calendar tools"}]
    }
    namespace: Final = _tool_named(lowered.tools, "calendar")
    assert namespace["tools"] == [
        {
            "type": "function",
            "name": "create_event",
            "description": "Create an event",
            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
        }
    ]


def test_loaded_members_join_a_namespace_the_request_already_declares():
    declared_namespace: Final = {
        "type": "namespace",
        "name": "calendar",
        "description": "Calendar tools",
        "tools": [{"type": "function", "name": "list_events", "parameters": {"type": "object", "properties": {}}}],
    }
    lowered: Final = _lowered([SEARCH_CALL, SEARCH_OUTPUT], [declared_namespace, CLIENT_TOOL_SEARCH])

    namespace: Final = _tool_named(lowered.tools, "calendar")
    assert isinstance(namespace["tools"], list)
    assert [member["name"] for member in namespace["tools"]] == ["list_events", "create_event"]


def test_tool_choice_for_the_search_tool_targets_the_search_function():
    lowered: Final = _lowered("hi", [CLIENT_TOOL_SEARCH], {"type": "tool_search"})

    assert lowered.tool_choice == {"type": "function", "name": "tool_search"}


def test_a_client_function_already_named_tool_search_is_rejected():
    lowering: Final = lower_tool_search_request(
        input="hi",
        tools=[CLIENT_TOOL_SEARCH, {"type": "function", "name": "tool_search", "parameters": {}}],
        tool_choice=None,
    )

    assert isinstance(lowering, ToolSearchFunctionNameTaken)
