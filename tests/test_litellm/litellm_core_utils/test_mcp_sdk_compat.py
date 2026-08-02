import pytest
from pydantic import BaseModel

from litellm.litellm_core_utils.mcp_sdk_compat import (
    mcp_content_mime_type,
    mcp_tool_input_schema,
)


class ToolV1(BaseModel):
    """Shape of ``mcp.types.Tool`` on the SDK 1.x line."""

    name: str
    inputSchema: dict


class ToolV2(BaseModel):
    """Shape of ``mcp.types.Tool`` on the SDK 2.x line."""

    name: str
    input_schema: dict


class ContentsV1(BaseModel):
    mimeType: str | None = None


class ContentsV2(BaseModel):
    mime_type: str | None = None


def test_input_schema_read_from_v1_camel_case():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert mcp_tool_input_schema(ToolV1(name="t", inputSchema=schema)) == schema


def test_input_schema_read_from_v2_snake_case():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert mcp_tool_input_schema(ToolV2(name="t", input_schema=schema)) == schema


def test_empty_input_schema_is_returned_not_treated_as_absent():
    """An empty schema is falsy; a truthiness-based fallback would wrongly skip it."""
    assert mcp_tool_input_schema(ToolV2(name="t", input_schema={})) == {}
    assert mcp_tool_input_schema(ToolV1(name="t", inputSchema={})) == {}


def test_input_schema_on_unsupported_sdk_shape_raises():
    class ToolUnknown(BaseModel):
        name: str

    with pytest.raises(AttributeError, match="unsupported MCP SDK version"):
        mcp_tool_input_schema(ToolUnknown(name="t"))


def test_input_schema_rejects_non_dict():
    class ToolBad(BaseModel):
        input_schema: str

    with pytest.raises(TypeError, match="must be a dict"):
        mcp_tool_input_schema(ToolBad(input_schema="nope"))


def test_mime_type_read_from_both_sdk_lines():
    assert mcp_content_mime_type(ContentsV1(mimeType="text/plain")) == "text/plain"
    assert mcp_content_mime_type(ContentsV2(mime_type="text/plain")) == "text/plain"


def test_absent_mime_type_is_none_not_an_error():
    """The field exists but is unset on both lines, which is distinct from an unknown SDK."""
    assert mcp_content_mime_type(ContentsV1()) is None
    assert mcp_content_mime_type(ContentsV2()) is None


def test_mime_type_on_unsupported_sdk_shape_raises():
    class ContentsUnknown(BaseModel):
        blob: str

    with pytest.raises(AttributeError, match="unsupported MCP SDK version"):
        mcp_content_mime_type(ContentsUnknown(blob="x"))


def test_accessors_work_against_the_installed_mcp_sdk():
    """Guards the rename against whichever SDK line is actually resolved."""
    from mcp.types import TextResourceContents, Tool

    schema = {"type": "object", "properties": {}}
    tool = Tool(name="t", inputSchema=schema)
    assert mcp_tool_input_schema(tool) == schema

    contents = TextResourceContents(uri="https://example.com/x", text="hi", mimeType="text/plain")
    assert mcp_content_mime_type(contents) == "text/plain"
