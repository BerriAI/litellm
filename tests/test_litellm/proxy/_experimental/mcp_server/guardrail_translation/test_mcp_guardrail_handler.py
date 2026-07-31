"""Tests for the MCP guardrail translation handler."""

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from litellm.exceptions import BlockedPiiEntityError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs


class MockGuardrail(CustomGuardrail):
    """Simple guardrail mock that records invocations."""

    def __init__(self):
        super().__init__(guardrail_name="mock-mcp-guardrail")
        self.call_count = 0
        self.last_inputs = None
        self.last_request_data = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_inputs = inputs
        self.last_request_data = request_data
        return None  # Guardrail doesn't modify for MCP tools


@pytest.mark.asyncio
async def test_process_input_messages_updates_content():
    """Handler should pass tool definition to guardrail when mcp_tool_name is present."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "weather",
        "mcp_arguments": {"city": "tokyo"},
        "mcp_tool_description": "Get weather for a city",
    }

    result = await handler.process_input_messages(data, guardrail)

    # Handler passes data through unchanged
    assert result == data
    # Guardrail was called
    assert guardrail.call_count == 1
    # Guardrail received tools (not texts) with tool definition
    assert guardrail.last_inputs is not None
    tools = guardrail.last_inputs.get("tools", [])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "weather"
    # Request data was passed to guardrail
    assert guardrail.last_request_data == data


@pytest.mark.asyncio
async def test_process_input_messages_skips_when_no_tool_name():
    """Handler should skip guardrail invocation if mcp_tool_name is missing."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    # No mcp_tool_name means nothing to process
    data = {"some_other_field": "value"}
    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 0


@pytest.mark.asyncio
async def test_process_input_messages_handles_minimal_data():
    """Handler should work with just mcp_tool_name (minimal required field)."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {"mcp_tool_name": "simple_tool"}

    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 1
    tools = guardrail.last_inputs.get("tools", [])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "simple_tool"


class MaskingGuardrail(CustomGuardrail):
    """Guardrail that rewrites every scanned text, recording what it saw."""

    def __init__(self, masked_texts=None, raises=None):
        super().__init__(guardrail_name="masking-mcp-guardrail")
        self.masked_texts = masked_texts
        self.raises = raises
        self.call_count = 0
        self.last_inputs = None
        self.last_input_type = None
        self.last_request_data = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_inputs = inputs
        self.last_input_type = input_type
        self.last_request_data = request_data
        if self.raises is not None:
            raise self.raises
        if self.masked_texts is None:
            return inputs
        return GenericGuardrailAPIInputs(texts=list(self.masked_texts))


@pytest.mark.asyncio
async def test_process_output_response_masks_text_content():
    """Masked text returned by the guardrail must land in the tool result."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(masked_texts=["email <EMAIL_ADDRESS>", "call <PHONE_NUMBER>"])
    result = CallToolResult(
        content=[
            TextContent(type="text", text="email jane@example.com"),
            TextContent(type="text", text="call 415-555-0132"),
        ],
        isError=False,
    )

    returned = await handler.process_output_response(
        response=result,
        guardrail_to_apply=guardrail,
        request_data={"mcp_tool_name": "echo"},
    )

    assert guardrail.call_count == 1
    assert guardrail.last_input_type == "response"
    assert guardrail.last_inputs["texts"] == ["email jane@example.com", "call 415-555-0132"]
    assert [item.text for item in returned.content] == ["email <EMAIL_ADDRESS>", "call <PHONE_NUMBER>"]
    assert [item.text for item in result.content] == ["email <EMAIL_ADDRESS>", "call <PHONE_NUMBER>"]


@pytest.mark.asyncio
async def test_process_output_response_masks_dict_shaped_result():
    """A dict-shaped tool result (REST/JSON-RPC payload) must be masked too."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(masked_texts=["<EMAIL_ADDRESS>"])
    result = {"content": [{"type": "text", "text": "jane@example.com"}], "isError": False}

    returned = await handler.process_output_response(response=result, guardrail_to_apply=guardrail)

    assert returned["content"][0]["text"] == "<EMAIL_ADDRESS>"
    assert returned["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_process_output_response_propagates_block():
    """A guardrail rejecting the tool result must not be swallowed."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(
        raises=BlockedPiiEntityError(entity_type="EMAIL_ADDRESS", guardrail_name="masking-mcp-guardrail")
    )
    result = CallToolResult(content=[TextContent(type="text", text="jane@example.com")], isError=False)

    with pytest.raises(BlockedPiiEntityError):
        await handler.process_output_response(response=result, guardrail_to_apply=guardrail)


@pytest.mark.asyncio
async def test_process_output_response_skips_non_text_content():
    """A result carrying no text content must not be sent to the guardrail."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(masked_texts=["should not be used"])
    result = CallToolResult(
        content=[ImageContent(type="image", data="aGk=", mimeType="image/png")],
        isError=False,
    )

    returned = await handler.process_output_response(response=result, guardrail_to_apply=guardrail)

    assert guardrail.call_count == 0
    assert returned is result


@pytest.mark.asyncio
async def test_process_output_response_handles_result_without_content():
    """An unexpected result shape must be passed through, not crash the tool call."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(masked_texts=["should not be used"])

    returned = await handler.process_output_response(response={"error": "boom"}, guardrail_to_apply=guardrail)

    assert guardrail.call_count == 0
    assert returned == {"error": "boom"}


@pytest.mark.asyncio
async def test_process_output_response_leaves_result_unmasked_on_text_count_mismatch():
    """A guardrail returning the wrong number of texts must not shuffle content."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(masked_texts=["<EMAIL_ADDRESS>"])
    result = CallToolResult(
        content=[
            TextContent(type="text", text="jane@example.com"),
            TextContent(type="text", text="415-555-0132"),
        ],
        isError=False,
    )

    returned = await handler.process_output_response(response=result, guardrail_to_apply=guardrail)

    assert [item.text for item in returned.content] == ["jane@example.com", "415-555-0132"]


class SubstitutingGuardrail(CustomGuardrail):
    """Masks one substring wherever it appears, across however many texts it is given."""

    def __init__(self, needle: str, replacement: str):
        super().__init__(guardrail_name="substituting-mcp-guardrail")
        self.needle = needle
        self.replacement = replacement
        self.seen_texts: list = []

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.seen_texts = list(inputs.get("texts") or [])
        return GenericGuardrailAPIInputs(
            texts=[text.replace(self.needle, self.replacement) for text in self.seen_texts]
        )


@pytest.mark.asyncio
async def test_structured_content_is_masked_alongside_content():
    """structuredContent goes to the client too, so it must be masked, not just content."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("jane@example.com", "<EMAIL_ADDRESS>")
    response = CallToolResult(
        content=[TextContent(type="text", text="email jane@example.com")],
        structuredContent={"contact": {"email": "jane@example.com"}, "balance": 42.0},
        isError=False,
    )

    returned = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert returned.content[0].text == "email <EMAIL_ADDRESS>"
    assert returned.structuredContent == {"contact": {"email": "<EMAIL_ADDRESS>"}, "balance": 42.0}


@pytest.mark.asyncio
async def test_value_present_only_in_structured_content_is_masked():
    """The gap this closes: a sensitive value that never appears in the text content.

    Scanning only content would hand it to the guardrail never, so it would reach
    the client unscanned behind a result that looks inspected.
    """
    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("jane@example.com", "<EMAIL_ADDRESS>")
    response = CallToolResult(
        content=[TextContent(type="text", text="lookup complete")],
        structuredContent={"records": [{"email": "jane@example.com"}]},
        isError=False,
    )

    returned = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert "jane@example.com" in guardrail.seen_texts
    assert returned.structuredContent == {"records": [{"email": "<EMAIL_ADDRESS>"}]}
    assert returned.content[0].text == "lookup complete"


@pytest.mark.asyncio
async def test_structured_content_without_a_match_is_untouched():
    """Unrelated structured data keeps its values and its types."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("jane@example.com", "<EMAIL_ADDRESS>")
    response = CallToolResult(
        content=[TextContent(type="text", text="lookup complete")],
        structuredContent={"record_id": "C-1001", "balance": 42.0, "active": True, "note": None},
        isError=False,
    )

    returned = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert returned.structuredContent == {"record_id": "C-1001", "balance": 42.0, "active": True, "note": None}


@pytest.mark.asyncio
async def test_structured_content_nested_too_deeply_is_blocked():
    """Too deep to walk must block rather than pass the deeper values unscanned."""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.utils import MAX_STRUCTURED_CONTENT_SCAN_DEPTH

    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("jane@example.com", "<EMAIL_ADDRESS>")
    nested: dict = {"leaf": "jane@example.com"}
    for _ in range(MAX_STRUCTURED_CONTENT_SCAN_DEPTH + 1):
        nested = {"next": nested}
    response = CallToolResult(
        content=[TextContent(type="text", text="lookup complete")],
        structuredContent=nested,
        isError=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert exc_info.value.status_code == 400


def test_too_deep_json_returns_a_sentinel_rather_than_raising():
    """The too-deep signal must be a return value, not a custom exception.

    mcp_server/utils.py is reloaded by tests that override its environment-backed
    constants, which gives any exception class defined there a fresh identity and
    lets it escape a caller's except clause; under xdist that surfaced as a failure
    in an unrelated shard. A sentinel has no identity to lose. Asserted directly on
    the helper so this pins the contract without reloading the module and leaking
    that reload into other tests.
    """
    from litellm.proxy._experimental.mcp_server.utils import (
        MAX_STRUCTURED_CONTENT_SCAN_DEPTH,
        json_string_leaves,
    )

    nested: dict = {"leaf": "jane@example.com"}
    for _ in range(MAX_STRUCTURED_CONTENT_SCAN_DEPTH + 1):
        nested = {"next": nested}

    assert json_string_leaves(nested) is None
    assert json_string_leaves({"a": "b"}) == ((("a",), "b"),)


@pytest.mark.asyncio
async def test_sensitive_structured_content_key_is_blocked():
    """A dict key is client-visible but not rewritable, so a match must block.

    Maps keyed by an identifier are a common API shape, and renaming the key would
    change the payload contract rather than redact a value; the content filter takes
    the same position on MCP tool call arguments.
    """
    from fastapi import HTTPException

    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("jane@example.com", "<EMAIL_ADDRESS>")
    response = CallToolResult(
        content=[TextContent(type="text", text="lookup complete")],
        structuredContent={"jane@example.com": {"balance": 42.0}},
        isError=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert exc_info.value.status_code == 400
    assert "non-rewritable" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_sensitive_structured_content_numeric_value_is_blocked():
    """A numeric value cannot be masked in place either, so a match must block."""
    from fastapi import HTTPException

    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("4155550199", "<PHONE_NUMBER>")
    response = CallToolResult(
        content=[TextContent(type="text", text="lookup complete")],
        structuredContent={"phone": 4155550199},
        isError=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_clean_structured_content_keys_do_not_block():
    """Ordinary keys and numbers must pass through untouched."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = SubstitutingGuardrail("jane@example.com", "<EMAIL_ADDRESS>")
    response = CallToolResult(
        content=[TextContent(type="text", text="email jane@example.com")],
        structuredContent={"record_id": "C-1001", "balance": 42.0, "count": 3},
        isError=False,
    )

    returned = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert returned.content[0].text == "email <EMAIL_ADDRESS>"
    assert returned.structuredContent == {"record_id": "C-1001", "balance": 42.0, "count": 3}
