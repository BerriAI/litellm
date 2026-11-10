import copy
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    ChatAllowedToolsToolChoice,
    hosted_tool_choice_type,
    is_allowed_tools_tool_choice,
    parse_chat_allowed_tools_tool_choice,
    raise_if_hosted_tool_choice_unsupported,
    responses_allowed_tools_tool_choice,
)

_CHAT_LAYOUT: Final[dict[str, object]] = {
    "type": "allowed_tools",
    "allowed_tools": {
        "mode": "auto",
        "tools": [
            {"type": "function", "function": {"name": "get_time"}},
            {"type": "custom", "custom": {"name": "apply_patch", "x-vendor": 1}, "x-trace": "a"},
            {"type": "mcp", "server_label": "docs", "server_url": "https://docs.example"},
            {"type": "function", "name": "already_flat"},
        ],
    },
}


def _parsed(tool_choice: Mapping[str, object]) -> ChatAllowedToolsToolChoice:
    parsed: Final = parse_chat_allowed_tools_tool_choice(tool_choice)
    assert parsed is not None
    return parsed


def test_responses_layout_lifts_mode_and_tools_and_flattens_names_in_order():
    assert responses_allowed_tools_tool_choice(_parsed(_CHAT_LAYOUT)) == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "function", "name": "get_time"},
            {"type": "custom", "name": "apply_patch", "x-vendor": 1, "x-trace": "a"},
            {"type": "mcp", "server_label": "docs", "server_url": "https://docs.example"},
            {"type": "function", "name": "already_flat"},
        ],
    }


def test_responses_layout_leaves_the_caller_dict_untouched():
    original: Final = copy.deepcopy(_CHAT_LAYOUT)
    flat: Final = responses_allowed_tools_tool_choice(_parsed(_CHAT_LAYOUT))
    assert flat is not None
    next(iter(flat["tools"]))["name"] = "mutated"
    assert _CHAT_LAYOUT == original


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function"},
        {"type": "function", "name": ""},
        {"type": "function", "function": {}},
        {"type": "function", "function": "get_time"},
        {"type": "custom", "custom": {"name": ""}},
        {"type": "function", "name": "get_time", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_time", "type": "custom"}},
    ],
)
def test_responses_layout_refuses_to_rewrite_a_malformed_or_conflicting_entry(tool: dict[str, object]):
    tool_choice: Final[dict[str, object]] = {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "required", "tools": [tool]},
    }
    assert responses_allowed_tools_tool_choice(_parsed(tool_choice)) is None


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_time"}]},
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}},
        {"type": "allowed_tools", "allowed_tools": {"mode": "any", "tools": [{"type": "function"}]}},
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "function"}]}, "extra": 1},
        {"type": "function", "function": {"name": "get_time"}},
    ],
)
def test_parse_rejects_anything_but_the_chat_layout(tool_choice: dict[str, object]):
    assert parse_chat_allowed_tools_tool_choice(tool_choice) is None


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (_CHAT_LAYOUT, True),
        ({"type": "allowed_tools"}, True),
        ({"type": "function"}, False),
        ("auto", False),
        (None, False),
    ],
)
def test_is_allowed_tools_tool_choice(tool_choice: Mapping[str, object] | str | None, expected: bool):
    assert is_allowed_tools_tool_choice(tool_choice) is expected


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ({"type": "web_search"}, "web_search"),
        ({"type": "mcp", "server_label": "docs"}, "mcp"),
        ({"type": "function", "function": {"name": "get_time"}}, None),
        ({"type": "allowed_tools"}, None),
        ("required", None),
        (None, None),
    ],
)
def test_hosted_tool_choice_type(tool_choice: Mapping[str, object] | str | None, expected: str | None):
    assert hosted_tool_choice_type(tool_choice) == expected


@pytest.mark.parametrize("provider", ["openai", "azure", "litellm_proxy"])
def test_hosted_tool_choice_passes_where_the_responses_api_runs_the_tool(provider: str):
    assert (
        raise_if_hosted_tool_choice_unsupported({"type": "web_search"}, custom_llm_provider=provider, model="m") is None
    )


@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "gemini", "vertex_ai"])
def test_hosted_tool_choice_is_a_400_where_the_provider_cannot_run_it(provider: str):
    with pytest.raises(BadRequestError, match=r"tool_choice type 'file_search'.*only supported for") as exc:
        raise_if_hosted_tool_choice_unsupported({"type": "file_search"}, custom_llm_provider=provider, model="m")
    assert exc.value.status_code == 400
    assert f"custom_llm_provider={provider}" in str(exc.value)


def test_a_function_tool_choice_is_not_a_hosted_tool_anywhere():
    assert (
        raise_if_hosted_tool_choice_unsupported(
            {"type": "function", "function": {"name": "get_time"}}, custom_llm_provider="anthropic", model="m"
        )
        is None
    )
