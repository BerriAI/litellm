"""
Unit tests for merge_guardrailed_tools, which writes guardrail-returned chat tools back onto the
Responses API request tools they were flattened from
"""

import copy

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


def test_namespace_keeps_its_non_function_members_when_every_function_member_is_dropped():
    custom_member = {"type": "custom", "name": "grep", "description": "Grep", "format": {"type": "text"}}
    original = [
        {"type": "namespace", "name": "ns", "description": "NS", "tools": [_function("read"), custom_member]},
        _function("a"),
    ]
    groups = _groups(original)

    merged = merge_guardrailed_tools(original, groups, [groups[1][0]])

    assert list(merged) == [{"type": "namespace", "name": "ns", "description": "NS", "tools": [custom_member]}, _function("a")]


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
