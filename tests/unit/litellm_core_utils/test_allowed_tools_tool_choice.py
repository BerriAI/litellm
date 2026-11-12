import copy
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    ChatAllowedToolsToolChoice,
    GatewayTool,
    GatewayTools,
    allowed_tools_tool_choice_in_auto_mode,
    allowed_tools_tool_choice_in_effect,
    allowed_tools_tool_choice_with_builtin_as_function,
    allowed_tools_tool_choice_with_function,
    allowed_tools_tool_choice_with_gateway_tools_as_functions,
    drops_unsupported_allowed_tools_tool_choice,
    is_allowed_tools_tool_choice,
    chat_allowed_tools_tool_choice,
    parse_allowed_tools_tool_choice,
    refuse_tool_calls_the_choice_excludes,
    tool_calls_the_choice_excludes,
    unenforced_allowed_tools_tool_choice,
    responses_allowed_tools_tool_choice,
)

_CHOICES_WITH_A_NAME_THAT_IS_NOT_A_STRING: Final = (
    {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "name": ["get_time"]}]},
    {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "name": {"first": "get_time"}}]},
    {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "name": 7}]},
    {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "name": ""}]},
    {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "mcp", "name": ["get_time"]}]}},
    {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "mcp", "name": 7}]}},
    {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": ["get_time"]}}]},
    },
    {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "function": {"name": 7}}]},
)

_WHOLE_DOCS_SERVER: Final = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "server_label": "docs"}]}

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


_FLAT_CHOICE: Final = {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_time"}]}


@pytest.mark.parametrize(
    ("tool_choice", "custom_llm_provider"),
    [
        (_FLAT_CHOICE, "anthropic"),
        (_CHAT_LAYOUT, "hosted_vllm"),
        ({"type": "allowed_tools", "mode": "required", "tools": []}, "openai"),
        ({"type": "allowed_tools", "tools": [{"type": "function", "name": "get_time"}]}, "azure"),
    ],
)
def test_a_choice_nothing_would_enforce_counts_as_unenforced(tool_choice: object, custom_llm_provider: str):
    assert unenforced_allowed_tools_tool_choice(tool_choice, custom_llm_provider)


@pytest.mark.parametrize(
    ("tool_choice", "custom_llm_provider"),
    [
        (_FLAT_CHOICE, "openai"),
        (_CHAT_LAYOUT, "azure"),
        (_FLAT_CHOICE, "litellm_proxy"),
        ("required", "anthropic"),
        ({"type": "function", "function": {"name": "get_time"}}, "anthropic"),
        (None, "anthropic"),
    ],
)
def test_a_choice_the_provider_enforces_or_any_other_choice_counts_as_enforced(
    tool_choice: object, custom_llm_provider: str
):
    assert not unenforced_allowed_tools_tool_choice(tool_choice, custom_llm_provider)


@pytest.mark.parametrize(
    ("tool_choice", "tool_names"),
    [(_FLAT_CHOICE, ("get_time",)), (_CHAT_LAYOUT, ("get_time", "apply_patch"))],
)
def test_a_tool_call_the_choice_lists_passes(tool_choice: object, tool_names: tuple[str, ...]):
    assert refuse_tool_calls_the_choice_excludes(tool_choice, tool_names, "the-model") is None


@pytest.mark.parametrize(
    "tool_choice", [_FLAT_CHOICE, _CHAT_LAYOUT, {"type": "allowed_tools", "mode": "required", "tools": []}]
)
def test_a_tool_call_the_choice_does_not_list_is_refused(tool_choice: object):
    with pytest.raises(BadRequestError, match=r"The model called .*get_weather") as exc:
        refuse_tool_calls_the_choice_excludes(tool_choice, ("get_time", "get_weather"), "the-model")

    assert exc.value.status_code == 400


@pytest.mark.parametrize("tool_choice", ["required", None, {"type": "function", "function": {"name": "get_time"}}])
def test_any_other_choice_leaves_the_tool_calls_alone(tool_choice: object):
    assert refuse_tool_calls_the_choice_excludes(tool_choice, ("get_weather",), "the-model") is None


@pytest.mark.parametrize(
    ("api_version", "unenforced"),
    [
        ("2023-07-01-preview", True),
        ("2023-11-30", True),
        ("2023-12-01-preview", False),
        ("2025-04-01-preview", False),
        ("v1", False),
    ],
)
def test_azure_counts_as_enforcing_only_on_a_version_that_takes_tool_choice(api_version: str, unenforced: bool):
    assert unenforced_allowed_tools_tool_choice(_FLAT_CHOICE, "azure", api_version) is unenforced


def test_azure_reads_the_same_api_version_fallback_as_the_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "api_version", "2023-07-01-preview")

    assert unenforced_allowed_tools_tool_choice(_FLAT_CHOICE, "azure")


def test_drop_params_drops_a_choice_the_provider_cannot_take():
    assert drops_unsupported_allowed_tools_tool_choice(_FLAT_CHOICE, "anthropic", "the-model", True)


def test_a_choice_the_provider_cannot_take_answers_400_without_drop_params():
    with pytest.raises(BadRequestError, match="only supported for") as exc:
        drops_unsupported_allowed_tools_tool_choice(_FLAT_CHOICE, "anthropic", "the-model", False)

    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "mode": "required", "tools": []},
        {"type": "allowed_tools", "tools": [{"type": "function", "name": "get_time"}]},
        *_CHOICES_WITH_A_NAME_THAT_IS_NOT_A_STRING,
    ],
)
def test_a_malformed_choice_is_refused_without_drop_params_and_dropped_with_it(tool_choice: object):
    with pytest.raises(BadRequestError, match="Invalid tool choice") as exc:
        allowed_tools_tool_choice_in_effect(tool_choice, "the-model", False)

    assert exc.value.status_code == 400
    assert allowed_tools_tool_choice_in_effect(tool_choice, "the-model", True) is None


@pytest.mark.parametrize("tool_choice", _CHOICES_WITH_A_NAME_THAT_IS_NOT_A_STRING)
def test_a_name_that_is_not_a_string_lists_no_tool(tool_choice: object):
    assert tool_calls_the_choice_excludes(tool_choice, ("get_time",)) == ("get_time",)


@pytest.mark.parametrize("tool_choice", [_FLAT_CHOICE, _CHAT_LAYOUT, "required", None])
@pytest.mark.parametrize("drop_params", [True, False])
def test_a_valid_choice_or_any_other_choice_stays_in_effect(tool_choice: object, drop_params: bool):
    assert allowed_tools_tool_choice_in_effect(tool_choice, "the-model", drop_params) is tool_choice


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "server_label": "docs", "name": None}]},
        {
            "type": "allowed_tools",
            "allowed_tools": {"mode": "auto", "tools": [{"type": "mcp", "server_label": "docs", "name": None}]},
        },
    ],
)
def test_an_mcp_entry_with_a_null_name_is_a_valid_choice(tool_choice: object):
    assert parse_allowed_tools_tool_choice(tool_choice) is not None
    assert allowed_tools_tool_choice_in_effect(tool_choice, "the-model", False) is tool_choice


_DOCS_SEARCH: Final = {
    "type": "allowed_tools",
    "mode": "auto",
    "tools": [{"type": "mcp", "server_label": "docs", "name": "search"}],
}


def _gateway_serving_search(*server_labels: str) -> GatewayTools:
    return GatewayTools(
        server_labels=frozenset({"docs", "other"}),
        tools={
            "docs_server-search": GatewayTool(
                names=frozenset({"docs_server-search", "search"}), server_labels=frozenset(server_labels)
            )
        },
    )


@pytest.mark.parametrize("name", ["search", "docs_server-search"])
def test_a_named_mcp_entry_allows_its_call_on_a_server_its_label_serves(name: str):
    tool_choice: Final = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "mcp", "server_label": "docs", "name": name}],
    }

    assert (
        tool_calls_the_choice_excludes(tool_choice, ("docs_server-search",), _gateway_serving_search("docs", "other"))
        == ()
    )


def test_an_mcp_entry_without_a_name_allows_every_call_its_server_serves():
    assert (
        tool_calls_the_choice_excludes(_WHOLE_DOCS_SERVER, ("docs_server-search",), _gateway_serving_search("docs"))
        == ()
    )


@pytest.mark.parametrize(
    ("tool_choice", "server_labels"),
    [
        (_WHOLE_DOCS_SERVER, frozenset({"other"})),
        (_WHOLE_DOCS_SERVER, frozenset()),
        (
            {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "server_label": "docs", "name": "read"}]},
            frozenset({"docs"}),
        ),
        (_DOCS_SEARCH, frozenset({"other"})),
        (_DOCS_SEARCH, frozenset()),
    ],
)
def test_an_mcp_entry_allows_no_call_outside_its_server_or_its_name(
    tool_choice: object, server_labels: frozenset[str]
):
    assert tool_calls_the_choice_excludes(
        tool_choice, ("docs_server-search",), _gateway_serving_search(*server_labels)
    ) == ("docs_server-search",)


def _is_web_search(tool: Mapping[str, object]) -> bool:
    return tool.get("type") == "web_search"


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search"}, {"type": "function", "name": "get_time"}]},
        {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "web_search"}, {"type": "function", "function": {"name": "get_time"}}],
            },
        },
    ],
)
def test_a_builtin_entry_allows_the_function_that_replaces_it_in_the_same_layout(tool_choice: dict[str, object]):
    rewritten: Final = allowed_tools_tool_choice_with_builtin_as_function(
        tool_choice, _is_web_search, "litellm_web_search"
    )

    assert tool_calls_the_choice_excludes(tool_choice, ("litellm_web_search", "get_time")) == ("litellm_web_search",)
    assert tool_calls_the_choice_excludes(rewritten, ("litellm_web_search", "get_time", "get_weather")) == (
        "get_weather",
    )
    assert isinstance(rewritten, dict) and rewritten.keys() == tool_choice.keys()


def test_a_function_or_mcp_entry_is_never_taken_for_a_builtin():
    tool_choice: Final = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "web_search"}, {"type": "mcp", "server_label": "docs", "name": "web_search"}],
    }

    rewritten: Final = allowed_tools_tool_choice_with_builtin_as_function(
        tool_choice, lambda tool: tool.get("name") == "web_search", "litellm_web_search"
    )

    assert parse_allowed_tools_tool_choice(rewritten) == parse_allowed_tools_tool_choice(tool_choice)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        None,
        {"type": "allowed_tools", "mode": "auto", "tools": []},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search", "name": []}]},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search"}, {"type": "function", "name": 3}]},
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "web_search", "name": {}}]}},
    ],
)
def test_any_other_or_malformed_choice_is_left_as_sent(tool_choice: object):
    assert allowed_tools_tool_choice_with_builtin_as_function(tool_choice, _is_web_search, "litellm_web_search") is tool_choice


_GATEWAY_WITH_TWO_SERVERS: Final = GatewayTools(
    server_labels=frozenset({"docs", "weather"}),
    tools={
        "docs_server-search": GatewayTool(
            names=frozenset({"docs_server-search", "search"}), server_labels=frozenset({"docs"})
        ),
        "docs_server-read": GatewayTool(names=frozenset({"docs_server-read", "read"}), server_labels=frozenset({"docs"})),
        "weather_server-forecast": GatewayTool(
            names=frozenset({"weather_server-forecast", "forecast"}), server_labels=frozenset({"weather"})
        ),
    },
)


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "mcp", "server_label": "docs"}, {"type": "function", "name": "get_time"}],
            },
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "get_time"},
                    {"type": "function", "name": "docs_server-search"},
                    {"type": "function", "name": "docs_server-read"},
                ],
            },
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "mcp", "server_label": "weather", "name": "forecast"}],
                },
            },
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "function", "function": {"name": "weather_server-forecast"}}],
                },
            },
        ),
    ],
)
def test_gateway_mcp_entries_become_the_functions_they_allow_in_the_callers_layout(
    tool_choice: dict[str, object], expected: dict[str, object]
):
    assert allowed_tools_tool_choice_with_gateway_tools_as_functions(tool_choice, _GATEWAY_WITH_TWO_SERVERS) == expected


def test_an_mcp_entry_of_a_label_the_gateway_does_not_serve_is_forwarded_as_sent():
    tool_choice: Final = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "mcp", "server_label": "remote"}, {"type": "mcp", "server_label": "docs", "name": "read"}],
    }

    assert allowed_tools_tool_choice_with_gateway_tools_as_functions(tool_choice, _GATEWAY_WITH_TWO_SERVERS) == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "mcp", "server_label": "remote"}, {"type": "function", "name": "docs_server-read"}],
    }


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_gateway_entries_that_allow_no_converted_tool_send_none_instead_of_an_empty_list(mode: str):
    tool_choice: Final = {
        "type": "allowed_tools",
        "mode": mode,
        "tools": [{"type": "mcp", "server_label": "docs", "name": "delete"}],
    }

    assert allowed_tools_tool_choice_with_gateway_tools_as_functions(tool_choice, _GATEWAY_WITH_TWO_SERVERS) == "none"


@pytest.mark.parametrize("tool_choice", ["auto", None, _FLAT_CHOICE, {"type": "allowed_tools", "mode": "auto", "tools": []}])
def test_a_choice_without_gateway_entries_is_forwarded_as_sent(tool_choice: object):
    assert allowed_tools_tool_choice_with_gateway_tools_as_functions(tool_choice, _GATEWAY_WITH_TWO_SERVERS) is tool_choice


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (
            {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_time"}]},
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_time"}, {"type": "function", "name": "retrieve"}],
            },
        ),
        (
            {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [{"type": "mcp", "server_label": "x"}]}},
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "mcp", "server_label": "x"}, {"type": "function", "function": {"name": "retrieve"}}],
                },
            },
        ),
    ],
)
def test_an_added_function_joins_the_choice_in_the_callers_layout(
    tool_choice: dict[str, object], expected: dict[str, object]
):
    assert allowed_tools_tool_choice_with_function(tool_choice, "retrieve") == expected


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        None,
        {"type": "function", "function": {"name": "get_time"}},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "retrieve"}]},
    ],
)
def test_a_choice_that_is_not_allowed_tools_or_already_lists_the_function_is_left_as_sent(tool_choice: object):
    assert allowed_tools_tool_choice_with_function(tool_choice, "retrieve") is tool_choice


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "required", "tools": [{"type": "function", "function": {"name": "get_time"}}]},
            },
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "get_time"}}]},
            },
        ),
        (
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_time"}, {"type": "mcp", "server_label": "docs"}],
            },
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_time"}, {"type": "mcp", "server_label": "docs"}],
            },
        ),
        (_CHAT_LAYOUT, _CHAT_LAYOUT),
        ("required", "required"),
        (
            {"type": "function", "function": {"name": "get_time"}},
            {"type": "function", "function": {"name": "get_time"}},
        ),
    ],
)
def test_auto_mode_lifts_only_the_requirement_and_keeps_the_callers_layout_and_tools(
    tool_choice: object, expected: object
):
    assert allowed_tools_tool_choice_in_auto_mode(tool_choice) == expected
