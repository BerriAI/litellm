from typing import Final

import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    ALLOWED_TOOL_CHOICE,
    chat_allowed_tool_choice,
    pruned_allowed_tool_choice,
    responses_allowed_tool_choice,
)

CHAT_CHOICE: Final = {
    "type": "allowed_tools",
    "allowed_tools": {
        "mode": "required",
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "strict": True}, "note": "outer"},
            {"type": "custom", "custom": {"name": "apply_patch"}},
            {"type": "mcp", "server_label": "deepwiki"},
        ],
        "future_field": 1,
    },
}

RESPONSES_CHOICE: Final = {
    "type": "allowed_tools",
    "mode": "required",
    "tools": [
        {"type": "function", "name": "get_weather", "strict": True, "note": "outer"},
        {"type": "custom", "name": "apply_patch"},
        {"type": "mcp", "server_label": "deepwiki"},
    ],
}


def test_chat_shape_flattens_for_responses_keeping_every_tool_field():
    assert responses_allowed_tool_choice(ALLOWED_TOOL_CHOICE.validate_python(CHAT_CHOICE)) == RESPONSES_CHOICE


def test_responses_shape_nests_for_chat_keeping_its_tool_names():
    assert chat_allowed_tool_choice(ALLOWED_TOOL_CHOICE.validate_python(RESPONSES_CHOICE)) == {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}, "strict": True, "note": "outer"},
                {"type": "custom", "custom": {"name": "apply_patch"}},
                {"type": "mcp", "server_label": "deepwiki"},
            ],
        },
    }


@pytest.mark.parametrize(
    ("convert", "choice"),
    [(chat_allowed_tool_choice, CHAT_CHOICE), (responses_allowed_tool_choice, RESPONSES_CHOICE)],
    ids=["chat", "responses"],
)
def test_each_shape_is_unchanged_when_already_in_the_target_shape(convert, choice):
    assert convert(ALLOWED_TOOL_CHOICE.validate_python(choice)) == choice


def test_tools_given_as_a_tuple_are_accepted():
    choice: Final = {"type": "allowed_tools", "mode": "auto", "tools": ({"type": "function", "name": "get_weather"},)}
    assert chat_allowed_tool_choice(ALLOWED_TOOL_CHOICE.validate_python(choice))["allowed_tools"] == {
        "mode": "auto",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}},
        {"type": "allowed_tools", "mode": "auto", "tools": []},
        {"type": "allowed_tools", "mode": "sometimes", "tools": [{"type": "function", "name": "a"}]},
        {"type": "function", "function": {"name": "a"}},
    ],
)
def test_a_choice_that_is_not_a_valid_allowed_tools_choice_is_rejected(choice):
    with pytest.raises(ValidationError):
        ALLOWED_TOOL_CHOICE.validate_python(choice)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (
            CHAT_CHOICE,
            {
                **CHAT_CHOICE,
                "allowed_tools": {
                    **CHAT_CHOICE["allowed_tools"],
                    "tools": CHAT_CHOICE["allowed_tools"]["tools"][1:],
                },
            },
        ),
        (RESPONSES_CHOICE, {**RESPONSES_CHOICE, "tools": RESPONSES_CHOICE["tools"][1:]}),
    ],
    ids=["chat", "responses"],
)
def test_removed_functions_are_pruned_from_the_choice_in_its_own_shape(choice, expected):
    assert pruned_allowed_tool_choice({"tool_choice": choice}, {"get_weather"}) == expected


def test_a_choice_left_with_no_tools_becomes_none():
    choice: Final = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "get_weather"}],
    }
    assert pruned_allowed_tool_choice({"tool_choice": choice}, {"get_weather"}) == "none"


@pytest.mark.parametrize(
    ("request_body", "removed"),
    [
        ({"tool_choice": CHAT_CHOICE}, {"get_time"}),
        ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, {"get_weather"}),
        ({"tool_choice": "auto"}, {"get_weather"}),
        ({}, {"get_weather"}),
    ],
    ids=["other_function_removed", "named_choice", "auto", "no_choice"],
)
def test_the_request_keeps_its_choice_when_nothing_it_allows_is_removed(request_body, removed):
    assert pruned_allowed_tool_choice(request_body, removed) is None
