from typing import Final

from litellm.responses.litellm_completion_transformation.client_tool_search import bridge_client_tool_search
from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponseInputParam

_CODEX_TOOL_SEARCH: Final = {"type": "tool_search", "execution": "client"}
_SHELL: Final = {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}}
_SEARCH_CALL: Final = {
    "type": "tool_search_call",
    "call_id": "search-1",
    "execution": "client",
    "arguments": {"query": "shell"},
}


def _search_output(*tools: dict[str, object]) -> dict[str, object]:
    return {
        "type": "tool_search_output",
        "call_id": "search-1",
        "status": "completed",
        "execution": "client",
        "tools": list(tools),
    }


def test_requests_without_client_tool_search_pass_through_untouched() -> None:
    request_input: Final[ResponseInputParam] = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "tool_search_call", "call_id": "srv-1", "execution": "server", "arguments": {"query": "x"}},
    ]
    tools: Final[list[ALL_RESPONSES_API_TOOL_PARAMS]] = [_SHELL, {"type": "tool_search", "execution": "server"}]

    bridged: Final = bridge_client_tool_search(request_input, tools)

    assert bridged.bridged is False
    assert bridged.input is request_input
    assert bridged.tools == tuple(tools)


def test_a_search_that_found_nothing_answers_the_call_and_loads_no_tools() -> None:
    bridged: Final = bridge_client_tool_search([_SEARCH_CALL, _search_output()], [_CODEX_TOOL_SEARCH, _SHELL])

    assert bridged.tools == (_CODEX_TOOL_SEARCH, _SHELL)
    assert bridged.input[1] == {
        "type": "function_call_output",
        "call_id": "search-1",
        "output": "No tools matched the search.",
    }


def test_a_loaded_tool_already_offered_eagerly_is_not_sent_twice() -> None:
    loaded_shell: Final = {**_SHELL, "description": "loaded copy", "defer_loading": True}

    bridged: Final = bridge_client_tool_search(
        [_SEARCH_CALL, _search_output(loaded_shell)], [_CODEX_TOOL_SEARCH, _SHELL]
    )

    assert bridged.tools == (_CODEX_TOOL_SEARCH, _SHELL)


def test_deferred_request_tools_are_held_back_only_when_the_client_searches() -> None:
    deferred: Final = {**_SHELL, "name": "deploy", "defer_loading": True}

    with_search: Final = bridge_client_tool_search("hi", [_CODEX_TOOL_SEARCH, _SHELL, deferred])
    without_search: Final = bridge_client_tool_search("hi", [_SHELL, deferred])

    assert with_search.tools == (_CODEX_TOOL_SEARCH, _SHELL)
    assert without_search.tools == (_SHELL, deferred)
