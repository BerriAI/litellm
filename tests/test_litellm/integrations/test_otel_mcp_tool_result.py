"""Regression tests for the OTEL callback crashing on MCP tool-call results.

When an MCP tool call is traced (call_type ``call_mcp_tool``), ``response_obj``
is a Pydantic ``mcp.types.CallToolResult``, which has no ``.get()``.
``set_attributes`` assumed a dict and raised
``AttributeError: 'CallToolResult' object has no attribute 'get'``, which the
outer try/except swallowed, so the span silently lost every response attribute
and the tool output. See https://github.com/BerriAI/litellm/issues/30651
"""

from unittest.mock import MagicMock

from mcp.types import CallToolResult, TextContent

from litellm.integrations.opentelemetry import OpenTelemetry


def _span_attrs(mock_span):
    return {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}


def _kwargs():
    return {
        "optional_params": {},
        "litellm_params": {},
        "standard_logging_object": {
            "metadata": {},
            "call_type": "call_mcp_tool",
            "id": "resp-1",
            "litellm_call_id": "call-1",
        },
    }


def test_set_attributes_handles_mcp_call_tool_result():
    """A CallToolResult response must not crash set_attributes, and its output
    must be captured rather than silently dropped."""
    otel = OpenTelemetry()
    otel._capture_in_span = lambda: True  # enable content capture for tool output
    span = MagicMock()
    response_obj = CallToolResult(content=[TextContent(type="text", text="20")], isError=False)

    otel.set_attributes(span, _kwargs(), response_obj)

    attrs = _span_attrs(span)
    # Crash fix: execution got past the `.get("id")` line, so the response id
    # (falling back to the standard-logging-object id) is set on the span.
    # Before the fix this attribute is never set because the AttributeError is
    # raised first and swallowed.
    assert attrs.get("gen_ai.response.id") == "resp-1"
    # Tool output content is captured instead of being dropped.
    assert attrs.get("gen_ai.tool.output") == "20"


def test_set_attributes_non_dict_without_model_dump_does_not_crash():
    """A non-dict response object lacking both .get and .model_dump must
    coalesce to {} rather than crash; response id still falls back to the
    standard-logging-object id."""
    otel = OpenTelemetry()
    otel._capture_in_span = lambda: True
    span = MagicMock()

    otel.set_attributes(span, _kwargs(), object())

    attrs = _span_attrs(span)
    assert attrs.get("gen_ai.response.id") == "resp-1"
    # No content to record, so the tool-output attribute is not set.
    assert "gen_ai.tool.output" not in attrs


def test_extract_tool_result_output():
    result = CallToolResult(content=[TextContent(type="text", text="hello")], isError=False)
    assert OpenTelemetry._extract_tool_result_output(result) == "hello"

    # Multiple text blocks are joined with newlines.
    multi = CallToolResult(
        content=[
            TextContent(type="text", text="line1"),
            TextContent(type="text", text="line2"),
        ],
        isError=False,
    )
    assert OpenTelemetry._extract_tool_result_output(multi) == "line1\nline2"

    # No text content -> None (nothing to record).
    assert OpenTelemetry._extract_tool_result_output(object()) is None
