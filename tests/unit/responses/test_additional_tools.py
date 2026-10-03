from litellm.responses.additional_tools import hoist_additional_tools

_EXEC_TOOL = {"type": "custom", "name": "exec", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}
_WAIT_TOOL = {"type": "function", "name": "wait", "parameters": {"type": "object", "properties": {}}}
_TOP_LEVEL_TOOL = {"type": "function", "name": "top_level", "parameters": {"type": "object", "properties": {}}}
_USER_MESSAGE = {"type": "message", "role": "user", "content": "Run ls"}


def test_string_input_passes_through_with_existing_tools():
    hoisted = hoist_additional_tools("hello", [_TOP_LEVEL_TOOL])

    assert hoisted.input == "hello"
    assert hoisted.tools == (_TOP_LEVEL_TOOL,)
    assert hoisted.hoisted == ()


def test_input_without_additional_tools_items_is_returned_untouched():
    request_input = [_USER_MESSAGE]

    hoisted = hoist_additional_tools(request_input, None)

    assert hoisted.input is request_input
    assert hoisted.tools == ()
    assert hoisted.hoisted == ()


def test_additional_tools_items_are_stripped_and_appended_after_top_level_tools_in_item_order():
    request_input = [
        {"type": "additional_tools", "id": "at_1", "role": "developer", "tools": [_EXEC_TOOL]},
        _USER_MESSAGE,
        {"type": "additional_tools", "id": "at_2", "role": "developer", "tools": [_WAIT_TOOL]},
    ]

    hoisted = hoist_additional_tools(request_input, [_TOP_LEVEL_TOOL])

    assert hoisted.input == [_USER_MESSAGE]
    assert hoisted.tools == (_TOP_LEVEL_TOOL, _EXEC_TOOL, _WAIT_TOOL)
    assert hoisted.hoisted == (_EXEC_TOOL, _WAIT_TOOL)


def test_additional_tools_item_without_a_tools_list_is_stripped_and_contributes_nothing():
    request_input = [{"type": "additional_tools", "id": "at_1", "role": "developer", "tools": "exec"}, _USER_MESSAGE]

    hoisted = hoist_additional_tools(request_input, None)

    assert hoisted.input == [_USER_MESSAGE]
    assert hoisted.tools == ()
    assert hoisted.hoisted == ()


def test_tool_search_output_tools_are_hoisted_without_removing_the_history_item():
    request_input = [
        _USER_MESSAGE,
        {
            "type": "tool_search_output",
            "call_id": "call_tool_search",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
        },
    ]

    hoisted = hoist_additional_tools(request_input, [_TOP_LEVEL_TOOL])

    assert hoisted.input == request_input
    assert hoisted.tools == (
        _TOP_LEVEL_TOOL,
        {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
    )


def test_tool_search_output_tools_are_deduplicated_by_type_and_name():
    found_tool = {"type": "function", "name": "get_weather"}
    request_input = [
        {
            "type": "tool_search_output",
            "call_id": "call_tool_search",
            "tools": [found_tool, {"type": "function", "name": "get_weather"}, _EXEC_TOOL],
        },
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [_EXEC_TOOL, {"type": "function", "name": "get_weather"}],
        },
    ]

    hoisted = hoist_additional_tools(request_input, None)

    assert hoisted.hoisted == (found_tool, _EXEC_TOOL)


def test_tool_search_output_tools_do_not_override_top_level_tools():
    request_input = [
        {
            "type": "tool_search_output",
            "call_id": "call_tool_search",
            "tools": [{"type": "function", "name": "get_weather"}],
        }
    ]

    hoisted = hoist_additional_tools(request_input, [_TOP_LEVEL_TOOL])

    assert hoisted.tools == (_TOP_LEVEL_TOOL, {"type": "function", "name": "get_weather"})


def test_mcp_tools_without_names_are_deduplicated_by_server_label():
    first_mcp_tool = {"type": "mcp", "server_label": "first"}
    second_mcp_tool = {"type": "mcp", "server_label": "second"}
    request_input = [
        {
            "type": "tool_search_output",
            "call_id": "call_tool_search",
            "tools": [first_mcp_tool, second_mcp_tool],
        },
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [first_mcp_tool],
        },
    ]

    hoisted = hoist_additional_tools(request_input, None)

    assert hoisted.hoisted == (first_mcp_tool, second_mcp_tool)


def test_invalid_input_items_do_not_break_tool_hoisting():
    """Invalid input items are treated as ordinary history, not hoisting candidates."""
    request_input = [
        {"role": "user", "content": None},
        {
            "type": "tool_search_output",
            "call_id": "call_tool_search",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
    ]

    hoisted = hoist_additional_tools(request_input, None)

    assert hoisted.input == request_input
    assert hoisted.hoisted == ({"type": "function", "name": "get_weather"},)
