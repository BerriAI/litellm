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


def _msgs(block):
    return [{"role": "user", "content": [block]}]


def test_image_url_block_data_url_decoded_from_messages():
    block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(b'png')}"}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"png" and parts[0].inline and parts[0].name is None


def test_image_url_block_non_string_url_skipped():
    block = {"type": "image_url", "image_url": {"url": 123}}
    assert extract_file_parts_from_messages(_msgs(block), size_limit=1000) == []


def test_input_image_block_data_url_decoded():
    block = {"type": "input_image", "image_url": f"data:image/png;base64,{_b64(b'img')}"}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"img" and parts[0].inline


def test_file_block_non_dict_file_skipped():
    block = {"type": "file", "file": "not-a-dict"}
    assert extract_file_parts_from_messages(_msgs(block), size_limit=1000) == []


def test_file_block_reference_without_bytes_is_name_only():
    block = {"type": "file", "file": {"file_id": "file-abc"}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].name == "file-abc" and parts[0].data is None and parts[0].inline is False


def test_file_block_urlsafe_base64_decoded_via_fallback():
    raw = b"\xff\xff\xfe"  # encodes with url-unsafe chars '+'/'/' in standard b64
    urlsafe = base64.urlsafe_b64encode(raw).decode()
    assert "-" in urlsafe or "_" in urlsafe
    block = {
        "type": "file",
        "file": {"filename": "b.bin", "file_data": f"data:application/octet-stream;base64,{urlsafe}"},
    }
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data == raw


def test_file_block_data_url_without_base64_marker_has_no_bytes():
    block = {"type": "file", "file": {"filename": "n.txt", "file_data": "data:text/plain,hello"}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data is None and parts[0].inline is False and parts[0].name == "n.txt"


def test_input_file_block_data_url_decoded():
    block = {"type": "input_file", "filename": "doc.pdf", "file_data": f"data:application/pdf;base64,{_b64(b'pdf')}"}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"pdf" and parts[0].name == "doc.pdf" and parts[0].inline


def test_input_file_block_file_url_reference_name_only():
    block = {"type": "input_file", "file_url": "https://x.test/report.csv"}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].name == "report.csv" and parts[0].data is None and parts[0].inline is False


def test_input_audio_block_decoded():
    block = {"type": "input_audio", "input_audio": {"data": _b64(b"wav"), "format": "wav"}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"wav" and parts[0].name == "audio.wav" and parts[0].inline


def test_input_audio_block_undecodable_is_name_only():
    block = {"type": "input_audio", "input_audio": {"data": "!!!not-base64!!!"}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].name == "audio.bin" and parts[0].data is None and parts[0].inline is False


def test_input_audio_block_non_dict_skipped():
    block = {"type": "input_audio", "input_audio": "nope"}
    assert extract_file_parts_from_messages(_msgs(block), size_limit=1000) == []


def test_tool_call_dict_arguments_serialized_and_parsed():
    td = tool_call_to_tool_data({"function": {"name": "f", "arguments": {"a": 1}}})
    assert td["content"] == '{"a": 1}' and td["tool_input"] == {"a": 1}


def test_tool_call_none_arguments_yields_empty_content():
    td = tool_call_to_tool_data({"function": {"name": "f", "arguments": None}})
    assert td["content"] == "" and td["tool_input"] == {}


def test_tool_call_non_string_non_dict_arguments_serialized():
    td = tool_call_to_tool_data({"function": {"name": "f", "arguments": [1, 2]}})
    assert td["content"] == "[1, 2]" and td["tool_input"] == {}


def test_tool_call_invalid_json_string_arguments_kept_as_content():
    td = tool_call_to_tool_data({"function": {"name": "f", "arguments": "{not json"}})
    assert td["content"] == "{not json" and td["tool_input"] == {}


def test_tool_result_with_non_string_tool_call_id_uses_default_name():
    msgs = [{"role": "tool", "tool_call_id": ["c1"], "content": "orphan"}]
    results = extract_tool_results(msgs)
    assert results == [("tool_result", "orphan", ["c1"])]


def test_images_field_http_url_is_name_only_reference():
    parts = extract_file_parts_from_images(["https://x.test/pic.png"], size_limit=1000)
    assert len(parts) == 1 and parts[0].name == "pic.png" and parts[0].data is None and parts[0].inline is False


def test_messages_skip_non_dict_and_unknown_blocks():
    msgs = [
        "not-a-message",
        {
            "role": "user",
            "content": [
                "bare-string-block",
                {"type": "text", "text": "hi"},
                {"type": "file", "file": {"filename": "a.txt", "file_data": f"data:text/plain;base64,{_b64(b'x')}"}},
            ],
        },
    ]
    parts = extract_file_parts_from_messages(msgs, size_limit=1000)
    assert len(parts) == 1 and parts[0].name == "a.txt" and parts[0].data == b"x"


def test_image_url_block_invalid_data_url_returns_no_part():
    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%invalid%%%"}}
    assert extract_file_parts_from_messages(_msgs(block), size_limit=1000) == []


def test_tool_message_with_empty_content_is_skipped():
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": "   "}]
    assert extract_tool_results(msgs) == []


def test_extract_tool_results_skips_non_dict_messages_and_tool_calls():
    msgs = [
        "junk",
        {"role": "assistant", "tool_calls": ["not-a-dict", {"id": "c1", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert extract_tool_results(msgs) == [("f", "ok", "c1")]


def test_malformed_data_url_yields_no_bytes():
    block = {"type": "file", "file": {"filename": "x.bin", "file_data": "data:garbage-no-comma"}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data is None and parts[0].inline is False and parts[0].name == "x.bin"


def test_input_audio_block_without_data_skipped():
    block = {"type": "input_audio", "input_audio": {"format": "wav"}}
    assert extract_file_parts_from_messages(_msgs(block), size_limit=1000) == []


def test_images_field_non_string_entries_skipped():
    assert extract_file_parts_from_images([123, None, ""], size_limit=1000) == []


def test_make_tool_data_whitespace_after_truncation_defaults_name():
    name = " " * 100 + "x"
    td = make_tool_data(name, "c")
    assert td["tool_name"] == "tool_result" and td["action_name"] == name


def test_raw_base64_file_data_without_data_url_prefix_decoded():
    block = {"type": "file", "file": {"filename": "a.bin", "file_data": _b64(b"rawbytes")}}
    parts = extract_file_parts_from_messages(_msgs(block), size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"rawbytes" and parts[0].mime_hint is None


def test_images_field_raw_base64_without_data_url_prefix_decoded():
    parts = extract_file_parts_from_images([_b64(b"rawimg")], size_limit=1000)
    assert len(parts) == 1 and parts[0].data == b"rawimg"
