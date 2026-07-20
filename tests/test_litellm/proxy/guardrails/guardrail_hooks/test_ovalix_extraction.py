import base64

from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix_extraction import (
    extract_file_parts_from_images,
    extract_file_parts_from_messages,
    extract_tool_results,
    make_tool_data,
    tool_call_to_tool_data,
)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def test_file_block_data_url_decoded():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"filename": "a.txt", "file_data": f"data:text/plain;base64,{_b64(b'hi')}"}}
            ],
        }
    ]
    parts = extract_file_parts_from_messages(msgs, size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"hi" and parts[0].name == "a.txt" and parts[0].inline


def test_image_url_reference_tracked_by_name():
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x.test/pic.png"}}]}]
    parts = extract_file_parts_from_messages(msgs, size_limit=1000)
    assert len(parts) == 1 and parts[0].data is None and parts[0].inline is False and parts[0].name == "pic.png"


def test_oversize_file_flagged_no_data():
    big = _b64(b"x" * 100)
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {"filename": "big.bin", "file_data": f"data:application/octet-stream;base64,{big}"},
                }
            ],
        }
    ]
    parts = extract_file_parts_from_messages(msgs, size_limit=10)
    assert len(parts) == 1 and parts[0].oversize is True and parts[0].data is None


def test_images_field_data_url():
    parts = extract_file_parts_from_images([f"data:image/png;base64,{_b64(b'png')}"], size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"png" and parts[0].inline


def test_tool_call_to_tool_data_parses_arguments():
    td = tool_call_to_tool_data(
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "TLV"}'}}
    )
    assert (
        td["content"] == '{"city": "TLV"}'
        and td["tool_name"] == "get_weather"
        and td["action_name"] == "get_weather"
        and td["tool_input"] == {"city": "TLV"}
    )


def test_tool_call_malformed_dropped():
    assert tool_call_to_tool_data({"id": "c1", "type": "function", "function": {"name": ""}}) is None
    assert tool_call_to_tool_data({"id": "c1"}) is None


def test_extract_tool_results_correlates_name():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "unknown", "content": "orphan"},
    ]
    results = extract_tool_results(msgs)
    assert ("get_weather", "sunny", "c1") in results
    assert ("tool_result", "orphan", "unknown") in results


def test_make_tool_data_truncates_and_defaults_name():
    td = make_tool_data("   ", "content")
    assert td["tool_name"] == "tool_result" and td["action_name"] == "tool_result"
    long = "x" * 200
    td2 = make_tool_data(long, "c")
    assert len(td2["tool_name"]) == 100 and td2["action_name"] == long


def test_unhashable_block_type_skipped_without_raising():
    msgs = [{"role": "user", "content": [{"type": ["file"], "file": {"filename": "a.txt"}}]}]
    parts = extract_file_parts_from_messages(msgs, size_limit=1000)
    assert parts == []


def test_unhashable_tool_call_id_skipped_without_raising():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": ["c1"], "function": {"name": "get_weather"}}]},
        {"role": "tool", "tool_call_id": ["c1"], "content": "sunny"},
    ]
    results = extract_tool_results(msgs)
    assert results == [("tool_result", "sunny", ["c1"])]


def test_extract_tool_results_list_form_content():
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "part1"}, "part2"]}]
    results = extract_tool_results(msgs)
    assert ("tool_result", "part1\npart2", "c1") in results


def test_extract_tool_results_dict_form_content():
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": {"city": "TLV"}}]
    results = extract_tool_results(msgs)
    assert ("tool_result", '{"city": "TLV"}', "c1") in results


def test_images_field_oversize_flagged_no_data():
    big = _b64(b"x" * 100)
    parts = extract_file_parts_from_images([f"data:image/png;base64,{big}"], size_limit=10)
    assert len(parts) == 1 and parts[0].oversize is True and parts[0].data is None


class _StubFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, function):
        self.function = function


def test_tool_call_to_tool_data_accepts_object_style_tool_call():
    tool_call = _StubToolCall(_StubFunction("get_weather", '{"city": "TLV"}'))
    td = tool_call_to_tool_data(tool_call)
    assert td["tool_name"] == "get_weather" and td["tool_input"] == {"city": "TLV"}
