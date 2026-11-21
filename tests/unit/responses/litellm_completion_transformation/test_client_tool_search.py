from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.responses.litellm_completion_transformation.client_tool_search import bridge_client_tool_search
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
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


def test_a_loaded_deferred_tool_keeps_its_deferral_so_its_calls_read_back_as_namespace_equals_name() -> None:
    deferred: Final = {**_SHELL, "name": "deploy", "defer_loading": True}
    loaded_plain: Final = {**_SHELL, "name": "deploy", "description": "loaded copy"}

    bridged: Final = bridge_client_tool_search(
        [_SEARCH_CALL, _search_output(loaded_plain)], [_CODEX_TOOL_SEARCH, _SHELL, deferred]
    )

    assert bridged.tools == (_CODEX_TOOL_SEARCH, _SHELL, {**loaded_plain, "defer_loading": True})
    assert LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
        LiteLLMCompletionResponsesConfig.request_tools({"tools": list(bridged.tools)})
    ) == {"deploy": ("deploy", "deploy")}


def test_a_loaded_tool_that_was_never_deferred_stays_undeferred() -> None:
    loaded_lint: Final = {**_SHELL, "name": "lint"}

    bridged: Final = bridge_client_tool_search([_SEARCH_CALL, _search_output(loaded_lint)], [_CODEX_TOOL_SEARCH])

    assert bridged.tools == (_CODEX_TOOL_SEARCH, loaded_lint)


class _ChatFunctionName(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class _ChatFunctionTool(BaseModel):
    model_config = ConfigDict(frozen=True)

    function: _ChatFunctionName


_CHAT_FUNCTION_TOOLS: Final = TypeAdapter(tuple[_ChatFunctionTool, ...])
_LONG_TOOL: Final = "read_the_current_secret_value_from_the_vault_that_this_codex_session_was_given"
_VAULT: Final[dict[str, object]] = {
    "type": "namespace",
    "name": "mcp__vault",
    "description": "Vault tools",
    "tools": [{**_SHELL, "name": _LONG_TOOL}],
}


def _chat_names(tools: tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]) -> tuple[str, ...]:
    chat_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=list(tools)
    )
    return tuple(tool.function.name for tool in _CHAT_FUNCTION_TOOLS.validate_python(chat_tools))


def test_a_loaded_namespace_tool_over_64_characters_is_announced_by_the_name_the_model_can_call() -> None:
    bridged: Final = bridge_client_tool_search([_SEARCH_CALL, _search_output(_VAULT)], [_CODEX_TOOL_SEARCH])
    vault_chat_names: Final = tuple(name for name in _chat_names(bridged.tools) if _LONG_TOOL[:20] in name)

    assert len(vault_chat_names) == 1
    assert len(vault_chat_names[0]) <= 64
    assert bridged.input[1]["output"].endswith(f"callable: {vault_chat_names[0]}")


def test_a_long_namespace_tool_offered_eagerly_and_loaded_again_is_sent_once() -> None:
    bridged: Final = bridge_client_tool_search([_SEARCH_CALL, _search_output(_VAULT)], [_CODEX_TOOL_SEARCH, _VAULT])
    chat_names: Final = _chat_names(bridged.tools)

    assert len(chat_names) == len(set(chat_names)), chat_names


def test_two_loaded_tools_whose_short_names_collide_are_rejected_instead_of_merged() -> None:
    colliding: Final = tuple(
        {
            "type": "namespace",
            "name": f"mcp__codex_apps__vault_connector_with_a_long_name_{suffix}",
            "tools": [{**_SHELL, "name": "read_secret"}],
        }
        for suffix in ("104655", "119871")
    )

    bridged: Final = bridge_client_tool_search([_SEARCH_CALL, _search_output(*colliding)], [_CODEX_TOOL_SEARCH])

    with pytest.raises(ValueError, match="shorten to the same tool name"):
        _chat_names(bridged.tools)
