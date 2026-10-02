import json

import pytest

from litellm.tracing.ui_format import to_ui_content


def test_message_array_maps_roles_and_keeps_order():
    raw = json.dumps(
        [
            {"role": "system", "content": "be brief"},
            {"role": "human", "content": "hi"},
            {"role": "tool", "name": "lookup", "content": "42"},
            {"role": "narrator", "content": "aside"},
        ]
    )
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": (
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "42", "name": "lookup"},
            {"role": "user", "content": "aside"},
        ),
    }


@pytest.mark.parametrize(
    "call",
    [
        {"name": "get_plan", "args": {"customer_id": "c-1"}},
        {"name": "get_plan", "arguments": '{"customer_id": "c-1"}'},
        {"id": "call_1", "type": "function", "function": {"name": "get_plan", "arguments": '{"customer_id": "c-1"}'}},
    ],
)
def test_single_assistant_message_with_tool_call(call: dict[str, object]):
    content = to_ui_content(json.dumps({"role": "assistant", "content": None, "tool_calls": [call]}))
    assert content["kind"] == "messages"
    (message,) = content["messages"]
    assert message["role"] == "assistant"
    assert message["content"] == ""
    calls = message.get("tool_calls")
    assert calls is not None and len(calls) == 1
    assert calls[0]["name"] == "get_plan"
    assert json.loads(calls[0]["arguments"]) == {"customer_id": "c-1"}


def test_unknown_role_with_tool_calls_is_assistant():
    content = to_ui_content(json.dumps({"role": "model", "content": "", "tool_calls": [{"name": "f", "args": None}]}))
    assert content == {
        "kind": "messages",
        "messages": ({"role": "assistant", "content": "", "tool_calls": ({"name": "f", "arguments": "{}"},)},),
    }


def test_block_list_content_keeps_text_and_drops_reasoning():
    raw = json.dumps(
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "encrypted_content": "opaque"},
                {"type": "thinking", "thinking": "hidden chain"},
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    )
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": ({"role": "assistant", "content": "first\n\nsecond"},),
    }


def test_langchain_kwargs_shape():
    raw = json.dumps(
        [
            {"lc": 1, "type": "constructor", "kwargs": {"type": "human", "content": "question"}},
            {"kwargs": {"type": "ai", "content": "", "tool_calls": [{"name": "search", "args": {"q": "x"}}]}},
        ]
    )
    content = to_ui_content(raw)
    assert content["kind"] == "messages"
    human, ai = content["messages"]
    assert human == {"role": "user", "content": "question"}
    assert ai["role"] == "assistant"
    assert ai.get("tool_calls") == ({"name": "search", "arguments": '{"q": "x"}'},)


def test_plain_object_becomes_fields_in_key_order():
    raw = json.dumps({"zeta": "plain", "alpha": {"nested": [1, 2]}, "count": 3, "missing": None})
    assert to_ui_content(raw) == {
        "kind": "fields",
        "fields": (
            {"key": "zeta", "value": "plain"},
            {"key": "alpha", "value": '{"nested": [1, 2]}'},
            {"key": "count", "value": "3"},
            {"key": "missing", "value": "null"},
        ),
    }


def test_object_with_role_but_no_content_is_fields():
    assert to_ui_content('{"role": "admin", "user_id": "u1"}')["kind"] == "fields"


def test_json_string_becomes_its_text():
    assert to_ui_content(json.dumps('line one\n"quoted"')) == {"kind": "text", "text": 'line one\n"quoted"'}


@pytest.mark.parametrize(
    "raw",
    ['[{"role": "user", "content": "cut of', "plain words", "42", "[1, 2]", "[]"],
)
def test_non_message_non_object_payloads_keep_the_raw_string(raw: str):
    assert to_ui_content(raw) == {"kind": "text", "text": raw}


def test_empty_is_empty_text():
    assert to_ui_content("") == {"kind": "text", "text": ""}
