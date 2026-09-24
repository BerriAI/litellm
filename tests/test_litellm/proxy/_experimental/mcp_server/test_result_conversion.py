import json
from typing import Final

import pytest
from mcp.types import CallToolResult, InputRequiredResult, TextContent, Tool
from mcp_types.methods import serialize_server_result
from mcp_types.version import KNOWN_PROTOCOL_VERSIONS, MODERN_PROTOCOL_VERSIONS
from pydantic import ValidationError

from litellm.proxy._experimental.mcp_server.result_conversion import (
    INPUT_REQUIRED_UNSUPPORTED_MESSAGE,
    JsonResult,
    TextResult,
    WireCompat,
    complete_call_tool_result,
    error_text_result,
    handler_outcome,
    parse_http_body,
    to_call_tool_result,
    to_gateway_tool,
    wire_compat_for,
)

BOTH: Final = (WireCompat.LEGACY, WireCompat.MODERN)


def _interim() -> InputRequiredResult:
    return InputRequiredResult.model_validate(
        {
            "resultType": "input_required",
            "inputRequests": {
                "req-1": {
                    "method": "elicitation/create",
                    "params": {"message": "Pick one", "requestedSchema": {"type": "object", "properties": {}}},
                }
            },
            "requestState": "abc",
        }
    )


def _wire(result: CallToolResult | InputRequiredResult, version: str) -> dict[str, object]:
    return serialize_server_result(
        "tools/call", version, result.model_dump(by_alias=True, mode="json", exclude_none=True)
    )


class TestWireCompatFor:
    def test_only_modern_revisions_map_to_modern(self):
        for version in KNOWN_PROTOCOL_VERSIONS:
            expected: Final = WireCompat.MODERN if version in MODERN_PROTOCOL_VERSIONS else WireCompat.LEGACY
            assert wire_compat_for(version) is expected, version
        assert wire_compat_for("1999-01-01") is WireCompat.LEGACY


class TestParseHttpBody:
    @pytest.mark.parametrize("body", ["", "   ", "{not json", "null"])
    def test_non_structured_bodies_stay_text(self, body: str):
        assert parse_http_body(body) == TextResult(body)

    @pytest.mark.parametrize(
        "body, value",
        [
            ('{"a": 1}', {"a": 1}),
            ("[1, 2]", [1, 2]),
            ("1.10", 1.1),
            ("true", True),
            ('"hi"', "hi"),
        ],
    )
    def test_json_bodies_keep_original_text(self, body: str, value: object):
        assert parse_http_body(body) == JsonResult(value=value, original_text=body)

    def test_handler_outcome_stringifies_unknown_values(self):
        assert handler_outcome(42) == TextResult("42")
        assert handler_outcome(TextResult("x")) == TextResult("x")


class TestTextAndJsonArms:
    @pytest.mark.parametrize("compat", BOTH)
    def test_text_result(self, compat: WireCompat):
        result = to_call_tool_result(TextResult("plain"), compat)
        assert isinstance(result, CallToolResult)
        assert result.is_error is False
        assert [c.text for c in result.content if isinstance(c, TextContent)] == ["plain"]
        assert result.structured_content is None

    @pytest.mark.parametrize("compat", BOTH)
    def test_json_object_is_structured_everywhere_and_text_is_verbatim(self, compat: WireCompat):
        body: Final = '{"n":  1.10,\n"k": "v"}'
        result = to_call_tool_result(parse_http_body(body), compat)
        assert isinstance(result, CallToolResult)
        assert result.structured_content == {"n": 1.1, "k": "v"}
        assert [c.text for c in result.content if isinstance(c, TextContent)] == [body]

    @pytest.mark.parametrize("body", ["[1, 2]", "3", "true", '"s"'])
    def test_non_object_json_is_structured_only_on_modern(self, body: str):
        legacy = to_call_tool_result(parse_http_body(body), WireCompat.LEGACY)
        modern = to_call_tool_result(parse_http_body(body), WireCompat.MODERN)
        assert isinstance(legacy, CallToolResult) and isinstance(modern, CallToolResult)
        assert legacy.structured_content is None
        assert modern.structured_content == json.loads(body)
        for result in (legacy, modern):
            assert [c.text for c in result.content if isinstance(c, TextContent)] == [body]

    @pytest.mark.parametrize("compat", BOTH)
    def test_json_null_keeps_text_and_claims_no_structured_field(self, compat: WireCompat):
        result = to_call_tool_result(parse_http_body("null"), compat)
        assert isinstance(result, CallToolResult)
        assert [c.text for c in result.content if isinstance(c, TextContent)] == ["null"]
        assert "structuredContent" not in _wire(result, "2026-07-28")


class TestSdkResultArm:
    def _incoming(self, content: list[TextContent]) -> CallToolResult:
        return CallToolResult(content=content, structured_content=[1, 2], meta={"trace": "t1"}, is_error=False)

    def test_modern_passes_through_the_same_object(self):
        incoming = self._incoming([])
        assert to_call_tool_result(incoming, WireCompat.MODERN) is incoming

    def test_legacy_downgrade_with_empty_content_appends_json_text(self):
        incoming = self._incoming([])
        result = to_call_tool_result(incoming, WireCompat.LEGACY)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is None
        assert [c.text for c in result.content if isinstance(c, TextContent)] == ["[1, 2]"]
        assert result.meta == {"trace": "t1"}
        assert incoming.structured_content == [1, 2] and incoming.content == []

    def test_legacy_downgrade_keeps_unrelated_content_and_appends_json_text(self):
        incoming = self._incoming([TextContent(type="text", text="Done")])
        result = to_call_tool_result(incoming, WireCompat.LEGACY)
        assert isinstance(result, CallToolResult)
        assert [c.text for c in result.content if isinstance(c, TextContent)] == ["Done", "[1, 2]"]
        assert incoming.content == [TextContent(type="text", text="Done")]
        assert incoming.structured_content == [1, 2]

    def test_legacy_keeps_object_structured_content(self):
        incoming = CallToolResult(content=[], structured_content={"a": 1}, is_error=False)
        assert to_call_tool_result(incoming, WireCompat.LEGACY) is incoming

    def test_is_error_survives_downgrade(self):
        incoming = CallToolResult(content=[], structured_content=7, is_error=True)
        result = to_call_tool_result(incoming, WireCompat.LEGACY)
        assert isinstance(result, CallToolResult) and result.is_error is True


class TestInterimAndExceptionArms:
    def test_modern_interim_passes_through(self):
        interim = _interim()
        assert to_call_tool_result(interim, WireCompat.MODERN) is interim

    def test_legacy_interim_becomes_error_result(self):
        result = to_call_tool_result(_interim(), WireCompat.LEGACY)
        assert isinstance(result, CallToolResult)
        assert result.is_error is True
        assert [c.text for c in result.content if isinstance(c, TextContent)] == [INPUT_REQUIRED_UNSUPPORTED_MESSAGE]

    def test_complete_call_tool_result_never_returns_interim(self):
        result = complete_call_tool_result(_interim(), WireCompat.MODERN)
        assert isinstance(result, CallToolResult) and result.is_error is True

    @pytest.mark.parametrize("compat", BOTH)
    def test_exception_arm_matches_error_text_result(self, compat: WireCompat):
        exc = ValueError("boom")
        result = to_call_tool_result(exc, compat)
        assert result == error_text_result(exc)
        assert isinstance(result, CallToolResult) and result.is_error is True
        assert [c.text for c in result.content if isinstance(c, TextContent)] == ["ValueError: boom"]


class TestSdkWireSerialization:
    @pytest.mark.parametrize("version", KNOWN_PROTOCOL_VERSIONS)
    def test_converted_results_serialize_on_their_negotiated_revision(self, version: str):
        compat = wire_compat_for(version)
        for body in ('{"a": 1}', "[1, 2]", "3", "null", "text"):
            result = to_call_tool_result(parse_http_body(body), compat)
            frame = _wire(result, version)
            assert frame["content"] == [{"type": "text", "text": body}]
            structured = json.loads(body) if body != "text" else None
            expects_structured = structured is not None and (
                compat is WireCompat.MODERN or isinstance(structured, dict)
            )
            assert ("structuredContent" in frame) is expects_structured, (version, body)
            if expects_structured:
                assert frame["structuredContent"] == structured
            assert ("resultType" in frame) is (compat is WireCompat.MODERN), (version, body)

    @pytest.mark.parametrize("version", KNOWN_PROTOCOL_VERSIONS)
    def test_downgraded_sdk_result_serializes_where_the_raw_one_would_not(self, version: str):
        incoming = CallToolResult(content=[TextContent(type="text", text="Done")], structured_content=[1, 2])
        converted = to_call_tool_result(incoming, wire_compat_for(version))
        frame = _wire(converted, version)
        if version in MODERN_PROTOCOL_VERSIONS:
            assert frame["structuredContent"] == [1, 2]
            return
        with pytest.raises(ValidationError):
            _wire(incoming, version)
        assert "structuredContent" not in frame
        assert frame["content"] == [{"type": "text", "text": "Done"}, {"type": "text", "text": "[1, 2]"}]

    def test_modern_interim_serializes_with_its_fields_intact(self):
        frame = _wire(_interim(), "2026-07-28")
        assert frame["resultType"] == "input_required"
        assert frame["requestState"] == "abc"
        assert frame["inputRequests"]["req-1"]["params"]["message"] == "Pick one"


class TestToGatewayTool:
    def test_rename_is_a_deep_copy_that_keeps_every_other_field(self):
        tool = Tool(
            name="orig",
            description="d",
            inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
            _meta={"owner": "x"},
        )
        renamed = to_gateway_tool(tool, "srv-orig")
        assert renamed.name == "srv-orig"
        assert tool.name == "orig"
        assert renamed.input_schema == tool.input_schema and renamed.input_schema is not tool.input_schema
        assert renamed.meta == {"owner": "x"}
        assert renamed.description == "d"
