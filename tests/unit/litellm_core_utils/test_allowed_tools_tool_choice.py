import copy
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    ChatAllowedToolsToolChoice,
    is_allowed_tools_tool_choice,
    chat_allowed_tools_tool_choice,
    parse_allowed_tools_tool_choice,
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
    parsed: Final = parse_allowed_tools_tool_choice(tool_choice)
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


_RESPONSES_LAYOUT: Final[dict[str, object]] = {
    "type": "allowed_tools",
    "mode": "required",
    "tools": [
        {"type": "function", "name": "get_time"},
        {"type": "custom", "name": "apply_patch", "x-vendor": 1},
        {"type": "mcp", "server_label": "docs"},
        {"type": "function", "function": {"name": "already_nested"}},
    ],
}


def test_responses_layout_is_read_into_the_chat_layout():
    assert chat_allowed_tools_tool_choice(_parsed(_RESPONSES_LAYOUT)) == {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [
                {"type": "function", "function": {"name": "get_time"}},
                {"type": "custom", "x-vendor": 1, "custom": {"name": "apply_patch"}},
                {"type": "mcp", "server_label": "docs"},
                {"type": "function", "function": {"name": "already_nested"}},
            ],
        },
    }


def test_responses_layout_round_trips_back_to_what_the_caller_sent():
    flat_only: Final = {**_RESPONSES_LAYOUT, "tools": list(_RESPONSES_LAYOUT["tools"])[:3]}
    assert responses_allowed_tools_tool_choice(_parsed(flat_only)) == flat_only


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}},
        {"type": "allowed_tools", "allowed_tools": {"mode": "any", "tools": [{"type": "function"}]}},
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "function"}]}, "extra": 1},
        {"type": "allowed_tools", "mode": "auto", "tools": []},
        {"type": "allowed_tools", "mode": "none", "tools": [{"type": "function", "name": "get_time"}]},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function"}]},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "custom", "name": ""}]},
        {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "get_time", "function": {"name": "get_weather"}}],
        },
        {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "get_time"}],
            "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "name": "get_time"}]},
        },
        {"type": "function", "function": {"name": "get_time"}},
    ],
)
def test_parse_rejects_a_malformed_choice_in_either_layout(tool_choice: dict[str, object]):
    assert parse_allowed_tools_tool_choice(tool_choice) is None


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
