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


def test_genai_parts_preserve_conversation_text_calls_and_results():
    raw = json.dumps(
        [
            {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]},
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "Checking"},
                    {"type": "tool_call", "name": "weather", "arguments": {"city": "Paris"}},
                    {"type": "tool_call", "name": "time", "arguments": '{"city":"Paris"}'},
                ],
            },
            {"role": "tool", "parts": [{"type": "tool_call_response", "result": {"temperature": 20}}]},
            {"role": "assistant", "parts": [{"type": "text", "content": "It is warm"}]},
        ]
    )
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": (
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "Checking",
                "tool_calls": (
                    {"name": "weather", "arguments": '{"city": "Paris"}'},
                    {"name": "time", "arguments": '{"city":"Paris"}'},
                ),
            },
            {"role": "tool", "content": '{"temperature": 20}'},
            {"role": "assistant", "content": "It is warm"},
        ),
    }


@pytest.mark.parametrize(
    "block",
    [
        {"type": "tool_use", "name": "lookup", "input": {"id": 7}},
        {"type": "function_call", "name": "lookup", "arguments": '{"id": 7}'},
        {"type": "tool_call", "name": "lookup", "args": {"id": 7}},
    ],
)
def test_embedded_tool_calls_remain_visible(block: dict[str, object]):
    raw = json.dumps({"role": "assistant", "content": [{"type": "text", "text": "Looking up"}, block]})
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": (
            {
                "role": "assistant",
                "content": "Looking up",
                "tool_calls": ({"name": "lookup", "arguments": '{"id": 7}'},),
            },
        ),
    }


def test_explicit_tool_calls_do_not_duplicate_content_blocks():
    raw = json.dumps(
        {
            "role": "assistant",
            "content": [{"type": "tool_call", "name": "lookup", "args": {"id": 7}}],
            "tool_calls": [{"name": "lookup", "args": {"id": 7}}],
        }
    )
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": (
            {"role": "assistant", "content": "", "tool_calls": ({"name": "lookup", "arguments": '{"id": 7}'},)},
        ),
    }


@pytest.mark.parametrize("result", ["warm", {"temperature": 20}, [1, 2], False, 0, None])
def test_genai_tool_results_keep_json_values(result: object):
    raw = json.dumps({"role": "tool", "parts": [{"type": "tool_call_response", "result": result}]})
    expected = result if isinstance(result, str) else json.dumps(result)
    assert to_ui_content(raw) == {"kind": "messages", "messages": ({"role": "tool", "content": expected},)}


def test_content_takes_precedence_over_parts():
    raw = json.dumps({"role": "user", "content": "chat content", "parts": [{"type": "text", "content": "parts"}]})
    assert to_ui_content(raw) == {"kind": "messages", "messages": ({"role": "user", "content": "chat content"},)}


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "image", "content": "image-data"}],
        [{"type": "text", "content": "caption"}, {"type": "image", "content": "image-data"}],
        [{"type": "tool_call", "name": 5, "arguments": {"id": 7}}],
        [{"type": "tool_call", "arguments": {"id": 7}}],
        [{"type": "tool_call", "name": "", "arguments": {"id": 7}}],
        [{"type": "tool_call", "function": {"name": "lookup", "arguments": '{"id": 7}'}}],
        [{"type": "tool_call", "name": "lookup"}],
        [{"type": "tool_use", "name": "lookup"}],
        [{"type": "function_call", "name": "lookup"}],
    ],
)
def test_unknown_or_malformed_parts_remain_inspectable(blocks: list[dict[str, object]]):
    raw = json.dumps({"role": "user", "parts": blocks})
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": ({"role": "user", "content": json.dumps(blocks)},),
    }


@pytest.mark.parametrize("arguments", ['{"unfinished":', "", False, 0, None])
def test_embedded_call_arguments_are_displayed_without_repair(arguments: object):
    raw = json.dumps({"role": "model", "parts": [{"type": "tool_call", "name": "lookup", "arguments": arguments}]})
    expected = arguments if isinstance(arguments, str) else "{}" if arguments is None else json.dumps(arguments)
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": ({"role": "assistant", "content": "", "tool_calls": ({"name": "lookup", "arguments": expected},)},),
    }


def test_reasoning_text_does_not_become_visible_message_content():
    raw = json.dumps(
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "text": "private reasoning"},
                {"type": "text", "content": "answer"},
            ],
        }
    )
    assert to_ui_content(raw) == {"kind": "messages", "messages": ({"role": "assistant", "content": "answer"},)}


@pytest.mark.parametrize("content", [None, ""])
def test_explicit_empty_content_is_not_replaced_by_parts(content: object):
    raw = json.dumps({"role": "assistant", "content": content, "parts": [{"type": "text", "content": "fallback"}]})
    assert to_ui_content(raw) == {"kind": "messages", "messages": ({"role": "assistant", "content": ""},)}


def test_named_text_blocks_do_not_create_tool_calls():
    raw = json.dumps({"role": "assistant", "parts": [{"type": "text", "name": "answer", "content": "done"}]})
    assert to_ui_content(raw) == {"kind": "messages", "messages": ({"role": "assistant", "content": "done"},)}


@pytest.mark.parametrize("name", [None, 5, {"source": "exporter"}])
def test_optional_text_name_does_not_hide_neighboring_calls(name: object):
    raw = json.dumps(
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Looking up", "name": name},
                {"type": "tool_call", "name": "lookup", "arguments": {"id": 7}},
            ],
        }
    )
    assert to_ui_content(raw) == {
        "kind": "messages",
        "messages": (
            {
                "role": "assistant",
                "content": "Looking up",
                "tool_calls": ({"name": "lookup", "arguments": '{"id": 7}'},),
            },
        ),
    }


@pytest.mark.parametrize("parts", [[{"type": "text", "text": False}], {"text": "raw"}])
def test_invalid_parts_shapes_stay_available_as_json(parts: object):
    raw = json.dumps({"role": "user", "parts": parts})
    assert to_ui_content(raw) == {"kind": "messages", "messages": ({"role": "user", "content": json.dumps(parts)},)}
