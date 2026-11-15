"""
Unit tests for merge_guardrailed_tools, which writes guardrail-returned chat tools back onto the
Responses API request tools they were flattened from
"""

import copy

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


def _doc_getter(name, description):
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


def test_edited_mcp_tool_is_rewritten():
    original = [{"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"}]
    groups = _groups(original)
    edited = [{**groups[0][0], "allowed_tools": ["read_wiki_structure"]}]

    merged = merge_guardrailed_tools(original, groups, edited)

    assert list(merged) == edited


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
