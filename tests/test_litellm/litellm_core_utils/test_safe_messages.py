"""Unit tests for the safe_messages traversal helpers."""

import copy

from litellm.litellm_core_utils.safe_messages import (
    collect_message_text,
    iter_message_texts,
    iter_request_texts,
    set_message_text,
    set_request_text,
)


def test_iter_string_content():
    messages = [{"role": "user", "content": "hello"}]
    assert list(iter_message_texts(messages)) == [(0, None, "hello")]


def test_iter_multimodal_text_part():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert list(iter_message_texts(messages)) == [(0, 0, "hello")]


def test_iter_mixed_parts_skips_non_text():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "https://x"}},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    assert list(iter_message_texts(messages)) == [
        (0, 0, "first"),
        (0, 2, "second"),
    ]


def test_iter_bare_string_in_list():
    messages = [{"role": "user", "content": ["a", "b"]}]
    assert list(iter_message_texts(messages)) == [(0, 0, "a"), (0, 1, "b")]


def test_iter_skips_none_missing_and_garbage():
    messages = [
        {"role": "assistant", "content": None},
        {"role": "user"},
        {"role": "user", "content": 42},
        {"role": "user", "content": [{"type": "image_url"}]},
        {"role": "user", "content": [{"type": "text", "text": None}]},
        "not-a-dict",
    ]
    assert list(iter_message_texts(messages)) == []


def test_set_string_content():
    messages = [{"role": "user", "content": "old"}]
    set_message_text(messages, 0, None, "new")
    assert messages[0]["content"] == "new"


def test_set_text_part_preserves_other_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "old"},
                {"type": "image_url", "image_url": {"url": "https://x"}},
            ],
        }
    ]
    set_message_text(messages, 0, 0, "new")
    assert messages[0]["content"][0] == {"type": "text", "text": "new"}
    assert messages[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://x"},
    }


def test_set_bare_string_in_list():
    messages = [{"role": "user", "content": ["a", "b"]}]
    set_message_text(messages, 0, 1, "B")
    assert messages[0]["content"] == ["a", "B"]


def test_set_out_of_bounds_no_op():
    messages = [{"role": "user", "content": "x"}]
    before = copy.deepcopy(messages)
    set_message_text(messages, 5, None, "y")
    set_message_text(messages, 0, 99, "y")
    assert messages == before


def test_collect_concatenates_in_order():
    messages = [
        {"role": "user", "content": "one "},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "two "},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "three"},
            ],
        },
    ]
    assert collect_message_text(messages) == "one two three"
    assert collect_message_text(messages, separator="|") == "one |two |three"


def test_round_trip_iter_then_set_is_idempotent():
    messages = [
        {"role": "user", "content": "raw"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "mm"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
    ]
    before = copy.deepcopy(messages)
    for msg_idx, part_idx, text in list(iter_message_texts(messages)):
        set_message_text(messages, msg_idx, part_idx, text)
    assert messages == before


def test_iter_request_texts_walks_messages_prompt_and_input():
    data = {
        "messages": [
            {"role": "user", "content": "msg"},
            {"role": "user", "content": [{"type": "text", "text": "mm"}]},
        ],
        "prompt": ["p0", "p1"],
        "input": "in",
    }
    yielded = list(iter_request_texts(data))
    assert yielded == [
        ("messages", (0, None, "msg"), "msg"),
        ("messages", (1, 0, "mm"), "mm"),
        ("prompt", (0, 0, "p0"), "p0"),
        ("prompt", (0, 1, "p1"), "p1"),
        ("input", (0, None, "in"), "in"),
    ]


def test_set_request_text_writes_back_for_each_source():
    data = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "m"}]}],
        "prompt": ["p"],
        "input": "i",
    }
    set_request_text(data, "messages", (0, 0, "m"), "M")
    set_request_text(data, "prompt", (0, 0, "p"), "P")
    set_request_text(data, "input", (0, None, "i"), "I")
    assert data["messages"][0]["content"][0]["text"] == "M"
    assert data["prompt"] == ["P"]
    assert data["input"] == "I"


def test_set_request_text_unknown_source_is_no_op():
    data = {"messages": [{"role": "user", "content": "x"}]}
    before = copy.deepcopy(data)
    set_request_text(data, "bogus", (0, None, "x"), "y")
    assert data == before
