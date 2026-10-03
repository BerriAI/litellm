import json

import pytest

from litellm.llms.base_llm.harness.utils import decode_json_line, structured_output_instruction


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (b'{"type": "text", "part": {"text": "hi"}}', {"type": "text", "part": {"text": "hi"}}),
        ('  {"b": 1, "a": [null, 2.5, true], "c": {"d": {}}}\n', {"b": 1, "a": [None, 2.5, True], "c": {"d": {}}}),
        (b"{}", {}),
        ('{"dup": 1, "dup": 2}', {"dup": 2}),
    ],
)
def test_decode_json_line_returns_the_json_object(line: bytes | str, expected: dict[str, object]):
    decoded = decode_json_line(line)

    assert decoded == expected
    assert json.dumps(decoded) == json.dumps(expected)


@pytest.mark.parametrize(
    "line",
    [
        b"",
        "   ",
        b"\n",
        b"not json",
        b"{",
        b'{"a": 1} trailing',
        b"[1, 2]",
        b'[{"a": 1}]',
        b'"text"',
        b"7",
        b"1.5",
        b"true",
        b"null",
    ],
)
def test_decode_json_line_is_none_for_blank_non_json_and_non_object_lines(line: bytes | str):
    assert decode_json_line(line) is None


def test_decode_json_line_lets_undecodable_bytes_raise():
    with pytest.raises(UnicodeDecodeError):
        decode_json_line(b'{"a": "\xff"}')


def test_structured_output_instruction_embeds_the_schema_as_json():
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}

    assert structured_output_instruction(schema).endswith(f"\n{json.dumps(schema)}")
