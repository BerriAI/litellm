import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import json
from io import BytesIO
from typing import Dict, List
from litellm.router_utils.batch_utils import (
    replace_model_in_jsonl,
    _get_router_metadata_variable_name,
    InMemoryFile,
    parse_jsonl_with_embedded_newlines,
)


# Fixtures
@pytest.fixture
def sample_jsonl_data() -> List[Dict]:
    """Fixture providing sample JSONL data"""
    return [
        {
            "body": {
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        },
        {"body": {"model": "gpt-5.5", "messages": [{"role": "user", "content": "Hi"}]}},
    ]


@pytest.fixture
def sample_jsonl_bytes(sample_jsonl_data) -> bytes:
    """Fixture providing sample JSONL as bytes"""
    jsonl_str = "\n".join(json.dumps(line) for line in sample_jsonl_data)
    return jsonl_str.encode("utf-8")


@pytest.fixture
def sample_file_like(sample_jsonl_bytes):
    """Fixture providing a file-like object"""
    return BytesIO(sample_jsonl_bytes)


# Test cases
def test_bytes_input(sample_jsonl_bytes):
    """Test with bytes input"""
    new_model = "claude-3"
    result = replace_model_in_jsonl(sample_jsonl_bytes, new_model)

    assert result is not None
    assert isinstance(result, InMemoryFile)
    assert result.name == "modified_file.jsonl"
    assert result.content_type == "application/jsonl"


def test_tuple_input(sample_jsonl_bytes):
    """Test with tuple input"""
    new_model = "claude-3"
    test_tuple = ("test.jsonl", sample_jsonl_bytes, "application/json")
    result = replace_model_in_jsonl(test_tuple, new_model)

    assert result is not None
    assert isinstance(result, InMemoryFile)
    assert result.name == "modified_file.jsonl"
    assert result.content_type == "application/jsonl"


def test_tuple_with_file_handle_rewrites_model(sample_jsonl_bytes):
    """Security regression: when the tuple's content element is a file handle
    (batch uploads stream from the spooled upload handle), the model must still
    be rewritten. Otherwise a restricted body.model survives unmodified and
    bypasses the batch model allowlist, which only checks the upload target."""
    new_model = "approved-target-model"
    handle = BytesIO(sample_jsonl_bytes)
    test_tuple = ("test.jsonl", handle, "application/json")

    result = replace_model_in_jsonl(test_tuple, new_model)

    assert isinstance(result, InMemoryFile)
    rows = [
        json.loads(line)
        for line in result.getvalue().decode("utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "rewrite must produce rows"
    # every row now carries the rewritten target, not the original (restricted) model
    assert all(row["body"]["model"] == new_model for row in rows)
    assert all(row["body"]["model"] != "gpt-5.5" for row in rows)


def test_file_like_object(sample_file_like):
    """Test with file-like object input"""
    new_model = "claude-3"
    result = replace_model_in_jsonl(sample_file_like, new_model)

    assert result is not None
    assert isinstance(result, InMemoryFile)
    assert result.name == "modified_file.jsonl"
    assert result.content_type == "application/jsonl"


def test_router_metadata_variable_name():
    """Test that the variable name is correct"""
    assert _get_router_metadata_variable_name(function_name="completion") == "metadata"
    assert (
        _get_router_metadata_variable_name(function_name="batch") == "litellm_metadata"
    )
    assert (
        _get_router_metadata_variable_name(function_name="acreate_file")
        == "litellm_metadata"
    )
    assert (
        _get_router_metadata_variable_name(function_name="aget_file")
        == "litellm_metadata"
    )


def test_non_json_input():
    """Test that replace_model_in_jsonl returns original content for non-JSON input"""
    from litellm.router_utils.batch_utils import replace_model_in_jsonl

    # Test with non-JSON string
    non_json_str = "This is not a JSON string"
    result = replace_model_in_jsonl(non_json_str, "gpt-4")
    assert result == non_json_str

    # Test with non-JSON bytes
    non_json_bytes = b"This is not JSON bytes"
    result = replace_model_in_jsonl(non_json_bytes, "gpt-4")
    assert result == non_json_bytes

    # Test with non-JSON file-like object
    from io import BytesIO

    non_json_file = BytesIO(b"This is not JSON in a file")
    result = replace_model_in_jsonl(non_json_file, "gpt-4")
    assert result == non_json_file


def test_should_replace_model_in_jsonl():
    """Test that should_replace_model_in_jsonl returns the correct value"""
    from litellm.router_utils.batch_utils import should_replace_model_in_jsonl

    assert should_replace_model_in_jsonl(purpose="batch") is True
    assert should_replace_model_in_jsonl(purpose="test") is False
    assert should_replace_model_in_jsonl(purpose="user_data") is False


def test_parse_jsonl_with_embedded_newlines_simple():
    """Test parsing simple JSONL without embedded newlines"""
    content = '{"id": 1, "name": "test"}\n{"id": 2, "name": "test2"}'
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 2
    assert result[0] == {"id": 1, "name": "test"}
    assert result[1] == {"id": 2, "name": "test2"}


def test_parse_jsonl_with_embedded_newlines_in_strings():
    """Test parsing JSONL with newlines embedded in string values"""
    content = (
        '{"id": 1, "message": "Line 1\\nLine 2\\nLine 3"}\n{"id": 2, "message": "test"}'
    )
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 2
    assert result[0] == {"id": 1, "message": "Line 1\nLine 2\nLine 3"}
    assert result[1] == {"id": 2, "message": "test"}


def test_parse_jsonl_with_embedded_newlines_real_world_example():
    """Test with the real-world example from the Cooler Master Shark X case"""
    # This simulates the actual problem case from the user's log
    content = """{"custom_id":"16546277850245725","method":"POST","url":"/v1/chat/completions","body":{"model":"openai-gpt-4o-mini-dp-items-translation-dag","messages":[{"role":"system","content":"Translate the product title and description for an e-commerce marketplace in Saudi Arabia and the UAE. Text may be in English or Arabic.\\n"},{"role":"user","content":"\\nOriginal Title: ```Cooler Master Shark X PC Case```\\nOriginal Description: ```UNIQUE MASTERPIECEShark X is a system that provides an impressive  unique alternative to traditional PC systems.  Shark X will stand out and can be the ultimate  trophy or conversation piece for people looking  for a unique setup that stands head and fins  above the res.```\\nStore Name: ```geekay```\\n"}]}}"""

    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 1
    assert result[0]["custom_id"] == "16546277850245725"
    assert result[0]["method"] == "POST"
    assert result[0]["body"]["model"] == "openai-gpt-4o-mini-dp-items-translation-dag"
    assert len(result[0]["body"]["messages"]) == 2
    assert "Translate the product title" in result[0]["body"]["messages"][0]["content"]
    assert (
        "Cooler Master Shark X PC Case" in result[0]["body"]["messages"][1]["content"]
    )
    assert "UNIQUE MASTERPIECEShark X" in result[0]["body"]["messages"][1]["content"]


def test_parse_jsonl_with_embedded_newlines_multiple_complex_objects():
    """Test parsing multiple complex JSON objects with embedded newlines"""
    content = """{"id":1,"text":"Line 1\\nLine 2"}
{"id":2,"nested":{"field":"Value\\nWith\\nNewlines"}}
{"id":3,"simple":"test"}"""

    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 3
    assert result[0]["id"] == 1
    assert result[0]["text"] == "Line 1\nLine 2"
    assert result[1]["id"] == 2
    assert result[1]["nested"]["field"] == "Value\nWith\nNewlines"
    assert result[2]["id"] == 3
    assert result[2]["simple"] == "test"


def test_parse_jsonl_with_embedded_newlines_no_trailing_newline():
    """Test parsing JSONL without trailing newline"""
    content = '{"id": 1, "name": "test"}'
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 1
    assert result[0] == {"id": 1, "name": "test"}


def test_parse_jsonl_with_embedded_newlines_empty_string():
    """Test parsing empty string"""
    content = ""
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 0


def test_parse_jsonl_with_embedded_newlines_whitespace_only():
    """Test parsing whitespace-only content"""
    content = "   \n  \n  "
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 0


def test_replace_model_in_jsonl_malformed_middle_row_returns_original():
    """Regression: a malformed/truncated middle row must not silently drop the
    rows that follow it. The streaming rewrite accumulates physical lines into a
    buffer; a row that never parses poisons the buffer so every later valid row
    is concatenated into it and dropped. Returning that partial rewrite would
    ship a truncated batch with no error to the caller. Instead the original
    content is returned unchanged so the provider rejects the bad batch loudly."""
    content = (
        b'{"custom_id":"a","body":{"model":"x"}}\n'
        b'{"custom_id":"b","body":{"model":\n'  # truncated, never completes
        b'{"custom_id":"c","body":{"model":"x"}}\n'
    )

    result = replace_model_in_jsonl(content, "new-model")

    assert (
        result == content
    ), "must return the original unchanged, not a partial rewrite"


def test_replace_model_in_jsonl_malformed_row_seekable_handle_rewound():
    """When the source is a seekable handle that gets consumed during the failed
    rewrite, it must be rewound to 0 so the caller can re-read the full original."""
    content = (
        b'{"custom_id":"a","body":{"model":"x"}}\n'
        b'{"custom_id":"b","body":{"model":\n'
        b'{"custom_id":"c","body":{"model":"x"}}\n'
    )
    handle = BytesIO(content)

    result = replace_model_in_jsonl(handle, "new-model")

    assert result is handle
    assert handle.read() == content, "handle must be rewound for the caller to re-read"


def test_replace_model_in_jsonl_multi_row_rewrites_every_model():
    """Happy path: a well-formed multi-row file gets every row's model rewritten
    and no row is dropped."""
    content = (
        b'{"custom_id":"a","body":{"model":"old1"}}\n'
        b'{"custom_id":"b","body":{"model":"old2"}}\n'
        b'{"custom_id":"c","body":{"model":"old3"}}\n'
    )

    result = replace_model_in_jsonl(content, "new-model")

    assert isinstance(result, InMemoryFile)
    rows = [
        json.loads(line)
        for line in result.getvalue().decode("utf-8").splitlines()
        if line.strip()
    ]
    assert [row["custom_id"] for row in rows] == ["a", "b", "c"]
    assert all(row["body"]["model"] == "new-model" for row in rows)


def test_replace_model_in_jsonl_with_embedded_newlines():
    """Test that replace_model_in_jsonl works correctly with embedded newlines in content"""
    # Create a JSONL with embedded newlines in the message content
    jsonl_data = {
        "custom_id": "test123",
        "body": {
            "model": "old-model",
            "messages": [
                {"role": "user", "content": "This is a message\nwith multiple\nlines"}
            ],
        },
    }

    jsonl_bytes = json.dumps(jsonl_data).encode("utf-8")
    new_model = "new-model"

    result = replace_model_in_jsonl(jsonl_bytes, new_model)

    assert isinstance(result, InMemoryFile)

    # Read and parse the result
    result_content = result.read().decode("utf-8")
    result_json = json.loads(result_content)

    # Verify the model was replaced
    assert result_json["body"]["model"] == "new-model"
    # Verify the content with newlines is preserved
    assert (
        result_json["body"]["messages"][0]["content"]
        == "This is a message\nwith multiple\nlines"
    )
    assert result_json["custom_id"] == "test123"
