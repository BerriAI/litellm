"""
Unit tests for merge_guardrailed_tools, which writes guardrail-returned chat tools back onto the
Responses API request tools they were flattened from
"""

import copy
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Final

import pytest

from litellm.llms.openai.responses.guardrail_translation.tool_merge import merge_guardrailed_tools
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GuardrailToolParam


def _groups(tools):
    return [form.chat_tools for form in LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(tools)]


def _flat(groups):
    return [chat_tool for group in groups for chat_tool in group]


def _function(name, description=""):
    return {"type": "function", "name": name, "description": description, "parameters": {"type": "object"}}


def test_unchanged_tools_come_back_as_the_original_objects():
    original = [
        _function("a"),
        {"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("x"), _function("y")]},
        {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"},
        {"type": "web_search"},
    ]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, _flat(groups))

    assert list(merged) == original
    assert all(merged_tool is original_tool for merged_tool, original_tool in zip(merged, original))


def test_guardrail_reordering_unchanged_tools_keeps_request_order():
    original = [_function("a"), {"type": "namespace", "name": "ns", "tools": [_function("x")]}, {"type": "web_search"}]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, list(reversed(_flat(groups))))

    assert list(merged) == original


def test_duplicate_function_names_are_matched_by_ordinal():
    original = [_function("dup", "first"), _function("dup", "second")]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, _flat(groups)[:1])

    assert list(merged) == [original[0]]


def test_kept_tools_that_differ_in_parameters_match_their_originals_in_any_order():
    original = [
        {**_function("lookup_city"), "parameters": {"type": "object", "properties": {"city": {}}}},
        {**_function("lookup_city"), "parameters": {"type": "object", "properties": {"name": {}, "country": {}}}},
        {**_function("lookup_city"), "parameters": {"type": "object", "properties": {"city": {}, "region": {}}}},
    ]
    groups = _groups(original)
    flat = _flat(groups)

    merged = merge_guardrailed_tools(original, groups, [flat[2], flat[0]])

    assert merged == (original[0], original[2])


@pytest.mark.parametrize(
    "twin,sends_original",
    [({}, True), ({"strict": False}, False)],
    ids=["identical_twins", "twins_differing_only_in_strict"],
)
def test_a_kept_tool_matching_several_twins_keeps_an_original_only_when_the_twins_are_identical(
    twin, sends_original
):
    original = [
        _function("lookup_city", "first"),
        _function("lookup_city", "twin"),
        {**_function("lookup_city", "twin"), **twin},
    ]
    groups = _groups(original)
    flat = _flat(groups)
    guardrails_own_tool = LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(
        [flat[2]]
    )[0]

    merged = merge_guardrailed_tools(original, groups, [flat[2]])

    assert merged is not None
    assert len(merged) == 1
    assert (merged[0] is original[1]) is sends_original
    assert merged[0] == (original[1] if sends_original else guardrails_own_tool)


def test_interleaved_duplicate_names_keep_their_own_ordinals():
    original = [
        _function("dup", "a"),
        _function("other", "x"),
        _function("dup", "b"),
        _function("dup", "c"),
        _function("other", "y"),
    ]
    groups = _groups(original)
    flat = _flat(groups)
    edited = {**flat[3], "function": {**flat[3]["function"], "description": "changed"}}

    merged = merge_guardrailed_tools(original, groups, [*flat[:3], edited, flat[4]])

    assert list(merged) == [*original[:3], {**_function("dup", "changed"), "strict": False}, original[4]]
    assert all(merged[position] is original[position] for position in (0, 1, 2, 4))


_DOC_ID = {"type": "object", "properties": {"doc_id": {"type": "string"}}}


def _doc_getter(name: str, description: str):
    return {**_function(name, description), "parameters": _DOC_ID}


def _top_level_and_member_sharing_a_name():
    return [
        _doc_getter("mcp__docs__get_document", "Top-level document getter."),
        {
            "type": "namespace",
            "name": "mcp__docs",
            "description": "Docs tools",
            "tools": [_doc_getter("get_document", "Namespace document getter.")],
        },
    ]


def test_reordered_tools_sharing_a_name_keep_their_own_descriptions():
    original = _top_level_and_member_sharing_a_name()
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, list(reversed(_flat(groups))))

    assert merged == tuple(original)
    assert all(merged_tool is original_tool for merged_tool, original_tool in zip(merged, original))


def test_a_replacement_for_a_dropped_tool_sharing_a_name_takes_the_dropped_tools_place():
    original = _top_level_and_member_sharing_a_name()
    groups = _groups(original)
    replacement = {
        "type": "function",
        "function": {"name": "mcp__docs__get_document", "description": "Replacement", "parameters": _DOC_ID},
    }

    merged = merge_guardrailed_tools(original, groups, [_flat(groups)[1], replacement])

    assert merged == (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools([replacement])[0],
        original[1],
    )
    assert merged[1] is original[1]


def test_a_member_sharing_a_name_keeps_its_place_when_the_guardrail_edits_its_parameters():
    original = _top_level_and_member_sharing_a_name()
    groups = _groups(original)
    flat = _flat(groups)
    narrowed = {"type": "object", "properties": {"doc_id": {"type": "string", "pattern": "^N-"}}}
    edited = {**flat[1], "function": {**flat[1]["function"], "parameters": narrowed}}

    merged = merge_guardrailed_tools(original, groups, [edited, flat[0]])

    assert merged == (original[0], {**original[1], "tools": [{**original[1]["tools"][0], "parameters": narrowed}]})
    assert merged[0] is original[0]


def test_an_edited_member_is_matched_by_parameters_after_the_unchanged_tools_are_matched():
    member = _doc_getter("get_document", "Namespace document getter.")
    original = [
        {"type": "namespace", "name": "mcp__docs", "description": "Docs tools", "tools": [member]},
        {
            "type": "namespace",
            "name": "mcp",
            "description": "MCP tools",
            "tools": [_doc_getter("docs__get_document", "Second namespace document getter.")],
        },
        {
            **_function("mcp__docs__get_document", "Top-level document getter."),
            "parameters": {"type": "object", "properties": {"name": {}, "country": {}}},
        },
    ]
    groups = _groups(original)
    flat = _flat(groups)
    edited = {**flat[0], "function": {**flat[0]["function"], "description": "edited by the guardrail"}}

    merged = merge_guardrailed_tools(original, groups, [flat[1], edited])

    assert merged == ({**original[0], "tools": [{**member, "description": "edited by the guardrail"}]}, original[1])
    assert merged[1] is original[1]


def test_dropped_and_reordered_same_name_tools_keep_the_surviving_originals():
    original = [_function("dup", "a"), _function("dup", "b"), _function("dup", "c")]
    groups = _groups(original)
    flat = _flat(groups)

    merged = merge_guardrailed_tools(original, groups, [flat[2], flat[0]])

    assert merged is not None
    assert list(merged) == [original[0], original[2]]
    assert merged[0] is original[0] and merged[1] is original[2]


def test_dropping_one_of_two_identical_chat_forms_sends_neither_originals_hidden_fields():
    original = [
        {**_function("dup", "same"), "x_vendor": "first"},
        {**_function("dup", "same"), "x_vendor": "second"},
    ]
    groups = _groups(original)
    flat = _flat(groups)
    assert flat[0] == flat[1]

    merged = merge_guardrailed_tools(original, groups, flat[1:])

    assert merged == tuple(
        LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(flat[1:])
    )
    assert "x_vendor" not in merged[0]
    assert list(merge_guardrailed_tools(original, groups, flat)) == original


def test_merging_many_tools_reads_each_tool_a_bounded_number_of_times():
    reads = []

    class CountingTool(dict):
        def get(self, key, default=None):
            reads.append(key)
            return super().get(key, default)

    tool_count = 200
    original = [_function(f"tool-{index}") for index in range(tool_count)]
    groups = [[CountingTool(tool) for tool in group] for group in _groups(original)]

    merged = merge_guardrailed_tools(original, groups, _flat(_groups(original)))

    assert list(merged) == original
    assert len(reads) <= 8 * tool_count


@pytest.mark.parametrize(
    ("same_named", "unique"),
    [(400, 0), (32, 600)],
    ids=["many_same_named_tools", "same_named_tools_among_many_unique_tools"],
)
def test_dropping_half_of_many_same_named_tools_reads_each_tool_a_bounded_number_of_times(
    same_named: int, unique: int
) -> None:
    reads: Final[Counter[int]] = Counter()

    class CountingTool(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            reads[id(self)] += 1
            return super().get(key, default)

    original: Final = [
        *(_function("search") for _ in range(same_named)),
        *(_function(f"tool-{index}") for index in range(unique)),
    ]
    groups: Final = [[CountingTool(tool) for tool in group] for group in _groups(original)]
    returned: Final = [
        tool for index, tool in enumerate(_flat(_groups(original))) if index >= same_named or index % 2 == 0
    ]

    merged: Final = merge_guardrailed_tools(original, groups, returned)

    assert merged is not None
    assert [tool["name"] for tool in merged] == [
        *(["search"] * ((same_named + 1) // 2)),
        *(f"tool-{index}" for index in range(unique)),
    ]
    assert max(reads.values()) <= 8


def test_a_kept_member_stays_in_its_own_namespace_when_other_same_named_tools_are_too_many_to_match() -> None:
    first_namespace: Final = {
        "type": "namespace",
        "name": "a__b",
        "description": "First namespace",
        "tools": [_doc_getter("c", "First namespace member.")],
    }
    second_namespace: Final = {
        "type": "namespace",
        "name": "a",
        "description": "Second namespace",
        "tools": [_doc_getter("b__c", "Second namespace member.")],
    }
    mcp_tools: Final = [
        {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"} for _ in range(23)
    ]
    original: Final = [first_namespace, second_namespace, *mcp_tools]
    groups: Final = _groups(original)
    flat: Final = _flat(groups)

    merged: Final = merge_guardrailed_tools(original, groups, flat[1:])

    assert merged is not None
    assert merged[0] is second_namespace
    assert list(merged[1:]) == mcp_tools


def test_a_kept_member_stays_in_its_own_namespace_when_its_same_named_group_is_too_large_to_score() -> None:
    first_namespace: Final = {
        "type": "namespace",
        "name": "a__b",
        "description": "First namespace",
        "tools": [_doc_getter("c", "First namespace member.")],
    }
    second_namespace: Final = {
        "type": "namespace",
        "name": "a",
        "description": "Second namespace",
        "tools": [_doc_getter("b__c", "Second namespace member.")],
    }
    top_level: Final = [_doc_getter("a__b__c", f"Top-level getter {index}.") for index in range(22)]
    original: Final = [first_namespace, second_namespace, *top_level]
    groups: Final = _groups(original)
    flat: Final = _flat(groups)

    merged: Final = merge_guardrailed_tools(original, groups, flat[1:])

    assert merged is not None
    assert merged[0] is second_namespace
    assert list(merged[1:]) == top_level


def test_edited_mcp_tool_is_rewritten():
    original = [{"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"}]
    groups = _groups(original)
    edited = [{**groups[0][0], "allowed_tools": ["read_wiki_structure"]}]

    merged = merge_guardrailed_tools(original, groups, edited)

    assert list(merged) == edited


def _always_approved(tool: Mapping[str, object]) -> dict[str, object]:
    return {**tool, "require_approval": "always"}


def _with_memory_limit(tool: Mapping[str, object]) -> dict[str, object]:
    container: Final = tool["container"]
    assert isinstance(container, Mapping)
    return {**tool, "container": {**container, "memory_limit": "4g"}}


@pytest.mark.parametrize("kept", [0, 2], ids=["first_kept", "last_kept"])
@pytest.mark.parametrize(
    ("first", "last", "edit"),
    [
        pytest.param(
            {"type": "mcp", "server_label": "docs", "server_url": "https://a.example/mcp", "require_approval": "never"},
            {"type": "mcp", "server_label": "docs", "server_url": "https://b.example/mcp", "require_approval": "never"},
            _always_approved,
            id="mcp_servers_sharing_a_label",
        ),
        pytest.param(
            {"type": "code_interpreter", "container": {"type": "auto"}},
            {"type": "code_interpreter", "container": {"type": "auto", "file_ids": ["file_1"]}},
            _with_memory_limit,
            id="code_interpreters",
        ),
    ],
)
def test_editing_one_of_two_same_named_tools_and_dropping_the_other_sends_the_edit_in_its_own_slot(
    first: Mapping[str, object],
    last: Mapping[str, object],
    edit: Callable[[Mapping[str, object]], dict[str, object]],
    kept: int,
) -> None:
    between: Final = _function("lookup")
    original: Final = [first, between, last]
    groups: Final = _groups(original)
    flat: Final = _flat(groups)
    edited: Final = edit(flat[kept])

    merged: Final = merge_guardrailed_tools(original, groups, [flat[1], edited])

    assert merged == ((edited, between) if kept == 0 else (between, edited))


@pytest.mark.parametrize("kept", [0, 2], ids=["first_kept", "last_kept"])
def test_editing_one_of_two_same_named_functions_and_dropping_the_other_sends_the_edit_in_its_own_slot(
    kept: int,
) -> None:
    between: Final = _function("lookup")
    original: Final = [_function("search", "Search the web"), between, _function("search", "Search the docs")]
    groups: Final = _groups(original)
    flat: Final = _flat(groups)
    required_query: Final = {"type": "object", "required": ["query"]}
    edited: Final = {**flat[kept], "function": {**flat[kept]["function"], "parameters": required_query}}

    merged: Final = merge_guardrailed_tools(original, groups, [flat[1], edited])

    sent: Final = {**original[kept], "parameters": required_query, "strict": False}
    assert merged == ((sent, between) if kept == 0 else (between, sent))


@pytest.mark.parametrize("returned_reversed", [False, True], ids=["request_order", "reversed"])
def test_dropping_one_of_three_same_named_functions_sends_the_two_edits_in_their_returned_order(
    returned_reversed: bool,
) -> None:
    original: Final = [
        _function("search", "Search the web"),
        _function("search", "Search the docs"),
        _function("search", "Search the news"),
    ]
    groups: Final = _groups(original)
    required_query: Final = {"type": "object", "required": ["query"]}
    edited: Final = [
        {**chat_tool, "function": {**chat_tool["function"], "parameters": required_query}}
        for chat_tool in _flat(groups)[1:]
    ]
    returned: Final = edited[::-1] if returned_reversed else edited

    merged: Final = merge_guardrailed_tools(original, groups, returned)

    assert [tool["description"] for tool in merged] == [tool["function"]["description"] for tool in returned]


@pytest.mark.parametrize("edit_first", [True, False], ids=["returned_before_lookup", "returned_after_lookup"])
def test_an_edit_equally_close_to_two_same_named_tools_keeps_its_returned_place_around_the_other_tools(
    edit_first: bool,
) -> None:
    between: Final = _function("lookup")
    original: Final = [
        {"type": "mcp", "server_label": "docs", "server_url": "https://a.example/mcp"},
        between,
        {"type": "mcp", "server_label": "docs", "server_url": "https://b.example/mcp"},
    ]
    groups: Final = _groups(original)
    lookup: Final = _flat(groups)[1]
    rerouted: Final = {"type": "mcp", "server_label": "docs", "server_url": "https://gateway.example/mcp"}

    merged: Final = merge_guardrailed_tools(original, groups, [rerouted, lookup] if edit_first else [lookup, rerouted])

    assert merged == ((rerouted, between) if edit_first else (between, rerouted))


def test_injected_tool_lands_after_the_request_tools_when_request_had_none():
    injected = {"type": "function", "function": {"name": "b", "description": "d", "parameters": {"type": "object"}}}

    merged = merge_guardrailed_tools([], [], [injected])

    assert list(merged) == [
        {"type": "function", "name": "b", "description": "d", "parameters": {"type": "object"}, "strict": False}
    ]


def test_empty_guardrail_output_keeps_only_tools_never_sent_to_the_guardrail():
    original = [_function("a"), {"type": "web_search"}, {"type": "namespace", "name": "ns", "tools": [_function("x")]}]

    merged = merge_guardrailed_tools(original, _groups(original), [])

    assert list(merged) == [{"type": "web_search"}]


def test_member_edit_strips_only_the_namespace_description_prefix():
    original = [{"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("x", "X doc")]}]
    groups = _groups(original)
    assert groups[0][0]["function"]["description"] == "NS\n\nX doc"
    edited = [{**groups[0][0], "function": {**groups[0][0]["function"], "description": "NS\n\nX doc (guarded)"}}]

    merged = merge_guardrailed_tools(original, groups, edited)

    assert list(merged) == [
        {"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("x", "X doc (guarded)")]}
    ]


def test_namespace_keeps_a_non_function_member_when_a_function_member_is_edited():
    custom_member = {"type": "custom", "name": "grep", "description": "Grep", "format": {"type": "text"}}
    original = [
        {"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("read", "Read"), custom_member]}
    ]
    groups = _groups(original)
    edited = copy.deepcopy(_flat(groups))
    edited[0]["function"]["description"] = "NS\n\nEDITED"

    merged = merge_guardrailed_tools(original, groups, edited)

    assert len(merged) == 1
    assert [member["name"] for member in merged[0]["tools"]] == ["read", "grep"]
    assert merged[0]["tools"][0]["description"] == "EDITED"
    assert merged[0]["tools"][1] == custom_member


def test_namespace_keeps_its_custom_member_when_every_function_member_is_dropped():
    custom_member = {"type": "custom", "name": "grep", "description": "Grep", "format": {"type": "text"}}
    original = [
        {"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("read"), custom_member]},
        _function("a"),
    ]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, [groups[0][1], groups[1][0]])

    assert list(merged) == [{"type": "namespace", "name": "ns", "description": "NS", "tools": [custom_member]}, _function("a")]


def test_namespace_custom_member_is_dropped_when_the_guardrail_drops_its_chat_form():
    custom_member = {"type": "custom", "name": "grep", "description": "Grep", "format": {"type": "text"}}
    original = [
        {"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("read"), custom_member]},
        _function("a"),
    ]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, [groups[1][0]])

    assert list(merged) == [_function("a")]


def test_custom_member_description_edit_lands_without_the_namespace_prefix_or_grammar_block():
    grammar = {"type": "grammar", "syntax": "lark", "definition": "start: X"}
    custom_member = {"type": "custom", "name": "exec", "description": "Run a command", "format": grammar}
    original = [{"type": "namespace", "name": "shell", "description": "Shell", "tools": [custom_member]}]
    groups = _groups(original)
    assert groups[0][0]["function"]["description"] == "Shell\n\nRun a command\n\nFormat:\n```lark\nstart: X\n```"
    edited = copy.deepcopy(_flat(groups))
    edited[0]["function"]["description"] = "Shell\n\nRun a command (guarded)\n\nFormat:\n```lark\nstart: X\n```"

    merged = merge_guardrailed_tools(original, groups, edited)

    guarded_member = {**custom_member, "description": "Run a command (guarded)"}
    assert list(merged) == [{"type": "namespace", "name": "shell", "description": "Shell", "tools": [guarded_member]}]


def test_text_appended_after_the_grammar_block_lands_on_the_member_without_the_block():
    grammar = {"type": "grammar", "syntax": "lark", "definition": "start: X"}
    custom_member = {"type": "custom", "name": "exec", "description": "Run a command", "format": grammar}
    original = [{"type": "namespace", "name": "shell", "description": "Shell", "tools": [custom_member]}]
    groups = _groups(original)
    edited = copy.deepcopy(_flat(groups))
    edited[0]["function"]["description"] = edited[0]["function"]["description"] + " [checked]"

    merged = merge_guardrailed_tools(original, groups, edited)

    assert merged[0]["tools"][0]["description"] == "Run a command [checked]"
    reflattened = _flat(_groups(merged))
    assert reflattened[0]["function"]["description"] == "Shell\n\nRun a command [checked]\n\nFormat:\n```lark\nstart: X\n```"


def test_member_extras_edited_by_the_guardrail_land_on_that_member():
    original = [{"type": "namespace", "name": "ns", "tools": [_function("read")]}]
    groups = _groups(original)
    edited = copy.deepcopy(_flat(groups))
    edited[0]["cache_control"] = {"type": "ephemeral"}

    merged = merge_guardrailed_tools(original, groups, edited)

    assert merged[0]["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert merged[0]["tools"][0]["name"] == "read"


def test_guardrail_output_is_read_once():
    original = [_function("a"), {"type": "namespace", "name": "ns", "tools": [_function("x")]}]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, (chat_tool for chat_tool in _flat(groups)))

    assert list(merged) == original


def test_pydantic_guardrail_tools_round_trip_like_dicts():
    original = [_function("a"), {"type": "namespace", "name": "ns", "tools": [_function("x", "X doc")]}]
    groups = _groups(original)
    models = [GuardrailToolParam.model_validate(chat_tool) for chat_tool in _flat(groups)]

    merged = merge_guardrailed_tools(original, groups, models)

    assert list(merged) == original
    assert all(merged_tool is original_tool for merged_tool, original_tool in zip(merged, original))


def test_pydantic_guardrail_edit_lands_on_the_member():
    original = [{"type": "namespace", "name": "ns", "tools": [_function("x", "X doc")]}]
    groups = _groups(original)
    edited = copy.deepcopy(_flat(groups))
    edited[0]["function"]["description"] = "EDITED"

    merged = merge_guardrailed_tools(original, groups, [GuardrailToolParam.model_validate(edited[0])])

    assert list(merged) == [{"type": "namespace", "name": "ns", "tools": [_function("x", "EDITED")]}]


def test_non_object_guardrail_items_are_dropped():
    original = [_function("a")]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, [*_flat(groups), "junk", None])

    assert list(merged) == original
