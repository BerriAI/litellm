import pytest
from litellm.experimental_mcp_client.tolerant_result import (
    TolerantCallToolResult,
    _as_text_block,
    _resource_text,
    _ResourcePayload,
    _validate_resource,
)
from pydantic import ValidationError


def test_resource_text_returns_text_field_directly():
    resource = _ResourcePayload(text="already text")
    assert _resource_text(resource) == "already text"


def test_resource_text_returns_none_when_nothing_decodable():
    resource = _ResourcePayload(text=None, blob=None, mimeType=None)
    assert _resource_text(resource) is None


def test_resource_text_returns_none_on_undecodable_base64():
    resource = _ResourcePayload(blob="not valid base64!!!", mimeType="text/plain")
    assert _resource_text(resource) is None


def test_resource_text_returns_none_on_non_utf8_bytes():
    non_utf8_blob = b"\xff\xfe".hex()
    import base64

    resource = _ResourcePayload(blob=base64.b64encode(bytes.fromhex(non_utf8_blob)).decode(), mimeType="text/plain")
    assert _resource_text(resource) is None


def test_validate_resource_returns_none_for_wrong_typed_field():
    assert _validate_resource({"mimeType": 12345}) is None


def test_as_text_block_falls_back_to_json_dump_for_unrecognized_block():
    block = _as_text_block({"type": "nonsense", "foo": "bar"})
    assert block["type"] == "text"
    assert block["text"] == '{"type": "nonsense", "foo": "bar"}'


def test_degrade_invalid_content_blocks_passthrough_for_non_mapping_input():
    with pytest.raises(ValidationError):
        TolerantCallToolResult.model_validate("not a mapping at all")


def test_degrade_invalid_content_blocks_passthrough_for_non_sequence_content():
    with pytest.raises(ValidationError):
        TolerantCallToolResult.model_validate({"content": 42, "isError": False})


def test_degrade_invalid_content_blocks_passthrough_for_string_content():
    with pytest.raises(ValidationError):
        TolerantCallToolResult.model_validate({"content": "not-a-list-of-blocks", "isError": False})
