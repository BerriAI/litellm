from typing import Final

from pydantic import JsonValue

from litellm.responses.additional_tools import hoist_additional_tools

_EXEC_TOOL = {
    "type": "custom",
    "name": "exec",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
}
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


def _tool_list(*tools: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "additional_tools", "role": "developer", "tools": list(tools)}


def _namespace(name: str, *members: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "namespace", "name": name, "description": "", "tools": list(members)}


def _function(name: str, description: str = "run it") -> dict[str, JsonValue]:
    return {"type": "function", "name": name, "description": description, "parameters": {"type": "object"}}


def test_tool_list_repeated_on_every_turn_keeps_each_tool_once_with_the_newest_definition() -> None:
    collaboration: Final = _namespace("collaboration", _function("followup_task"))
    request_input: Final = [
        _tool_list(collaboration, _function("exec", "old")),
        _USER_MESSAGE,
        _tool_list(collaboration, _function("exec", "new")),
        _USER_MESSAGE,
    ]

    hoisted: Final = hoist_additional_tools(request_input, None)

    assert hoisted.tools == (collaboration, _function("exec", "new"))


def test_namespace_sent_in_several_turns_keeps_every_distinct_member_once() -> None:
    request_input: Final = [
        _tool_list(_namespace("mcp__docs__", _function("search"), _function("fetch", "old"))),
        _USER_MESSAGE,
        _tool_list(_namespace("mcp__docs__", _function("fetch", "new"), _function("list"))),
    ]

    hoisted: Final = hoist_additional_tools(request_input, None)

    assert hoisted.tools == (
        _namespace("mcp__docs__", _function("search"), _function("fetch", "new"), _function("list")),
    )


def test_unnamed_tools_collapse_only_when_identical() -> None:
    docs_server: Final[dict[str, JsonValue]] = {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://docs.example",
    }
    wiki_server: Final[dict[str, JsonValue]] = {
        "type": "mcp",
        "server_label": "wiki",
        "server_url": "https://wiki.example",
    }
    request_input: Final = [_tool_list(docs_server, wiki_server), _USER_MESSAGE, _tool_list(docs_server)]

    hoisted: Final = hoist_additional_tools(request_input, None)

    assert hoisted.tools == (docs_server, wiki_server)


def test_top_level_tool_repeated_in_a_tool_list_item_is_kept_once() -> None:
    request_input: Final = [_tool_list(_TOP_LEVEL_TOOL, _WAIT_TOOL), _USER_MESSAGE]

    hoisted: Final = hoist_additional_tools(request_input, [_TOP_LEVEL_TOOL])

    assert hoisted.tools == (_TOP_LEVEL_TOOL, _WAIT_TOOL)
