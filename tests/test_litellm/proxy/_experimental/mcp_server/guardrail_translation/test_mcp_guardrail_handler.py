"""Tests for the MCP guardrail translation handler."""

import asyncio

import pytest
from fastapi import HTTPException
from mcp.types import CallToolResult, ImageContent, TextContent

import litellm
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.proxy._experimental.mcp_server.utils import MAX_STRUCTURED_CONTENT_SCAN_DEPTH
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import unified_guardrail
from litellm.proxy.utils import ProxyLogging
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


@pytest.mark.asyncio
async def test_process_input_messages_updates_content():
    """Handler should pass the tool definition and the argument strings to the guardrail."""
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
    # Guardrail received tools with the tool definition
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


class ArgumentMaskingGuardrail(CustomGuardrail):
    """Unified guardrail that rewrites every text it is handed, like presidio does."""

    def __init__(
        self,
        secret: str = "jane.doe@example.com",
        replacement: str = "<EMAIL_ADDRESS>",
        texts_override: list[str] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("guardrail_name", "argument-masking-mcp-guardrail")
        super().__init__(**kwargs)
        self.secret = secret
        self.replacement = replacement
        self.texts_override = texts_override
        self.seen_texts: list[str] | None = None

    def _mask(self, text: str) -> str:
        return text.replace(self.secret, self.replacement)

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.seen_texts = list(inputs.get("texts") or [])
        if self.texts_override is not None:
            inputs["texts"] = self.texts_override
        else:
            inputs["texts"] = [self._mask(text) for text in self.seen_texts]
        return inputs


@pytest.fixture
def restore_callbacks(monkeypatch):
    """Restore the process-wide state driving pre_call_hook through unified_guardrail.

    unified_guardrail memoizes its translation mappings in a module global, and
    ProxyLogging caches callback capabilities keyed on id()s of litellm.callbacks,
    so leaving either populated leaks into unrelated tests in the same worker.
    """
    monkeypatch.setattr(litellm, "callbacks", litellm.callbacks)
    monkeypatch.setattr(
        unified_guardrail,
        "endpoint_guardrail_translation_mappings",
        unified_guardrail.endpoint_guardrail_translation_mappings,
    )
    yield
    ProxyLogging._callback_capabilities_cache.clear()


@pytest.mark.asyncio
async def test_argument_strings_are_handed_to_the_guardrail():
    """A guardrail must see the argument values, not just the tool definition.

    Without this the guardrail is handed a name and an empty schema, so no
    sensitive-data detection can ever fire on an MCP tool call.
    """
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "contact jane.doe@example.com about the invoice"},
    }

    await handler.process_input_messages(data, guardrail)

    assert guardrail.last_inputs is not None
    assert guardrail.last_inputs.get("texts") == ["contact jane.doe@example.com about the invoice"]


@pytest.mark.asyncio
async def test_masked_arguments_are_written_back_for_the_call_path():
    """A mask only takes effect once it lands in modified_arguments."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = ArgumentMaskingGuardrail()

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "contact jane.doe@example.com about the invoice"},
    }

    result = await handler.process_input_messages(data, guardrail)

    masked = {"query": "contact <EMAIL_ADDRESS> about the invoice"}
    assert result["modified_arguments"] == masked
    assert result["mcp_arguments"] == masked


@pytest.mark.asyncio
async def test_nested_arguments_keep_their_shape_when_masked():
    """Masking rewrites string leaves in place and preserves non-string values."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = ArgumentMaskingGuardrail()

    arguments = {
        "recipients": ["jane.doe@example.com", "ops@example.net"],
        "envelope": {"reply_to": "jane.doe@example.com", "retries": 3, "urgent": True, "cc": None},
        "count": 2,
    }
    data = {"mcp_tool_name": "send_email", "mcp_arguments": arguments}

    result = await handler.process_input_messages(data, guardrail)

    assert guardrail.seen_texts == [
        "jane.doe@example.com",
        "ops@example.net",
        "jane.doe@example.com",
    ]
    assert result["modified_arguments"] == {
        "recipients": ["<EMAIL_ADDRESS>", "ops@example.net"],
        "envelope": {"reply_to": "<EMAIL_ADDRESS>", "retries": 3, "urgent": True, "cc": None},
        "count": 2,
    }


@pytest.mark.asyncio
async def test_clean_arguments_are_not_overridden():
    """A guardrail that changes nothing must not set modified_arguments."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = ArgumentMaskingGuardrail()

    data = {"mcp_tool_name": "search", "mcp_arguments": {"query": "quarterly revenue"}}

    result = await handler.process_input_messages(data, guardrail)

    assert "modified_arguments" not in result
    assert result["mcp_arguments"] == {"query": "quarterly revenue"}


@pytest.mark.asyncio
async def test_guardrail_returning_wrong_text_count_leaves_arguments_alone():
    """Write-back is positional, so a length mismatch must not scramble arguments."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = ArgumentMaskingGuardrail(texts_override=["only", "two", "texts"])

    arguments = {"query": "contact jane.doe@example.com about the invoice"}
    data = {"mcp_tool_name": "search", "mcp_arguments": arguments}

    result = await handler.process_input_messages(data, guardrail)

    assert "modified_arguments" not in result
    assert result["mcp_arguments"] == arguments


@pytest.mark.asyncio
async def test_deeply_nested_arguments_are_blocked_rather_than_skipped():
    """Arguments too deep to walk must block instead of passing unscanned."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = ArgumentMaskingGuardrail()

    nested: dict = {"leaf": "jane.doe@example.com"}
    for _ in range(MAX_STRUCTURED_CONTENT_SCAN_DEPTH + 1):
        nested = {"next": nested}

    data = {"mcp_tool_name": "search", "mcp_arguments": nested}

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_input_messages(data, guardrail)

    assert exc_info.value.status_code == 400


class SelfWritingMaskingGuardrail(ArgumentMaskingGuardrail):
    """Masks through ``texts`` and writes the masked arguments itself.

    The shape the bundled content filter guardrail already has: it rewrites
    ``request_data["mcp_arguments"]`` from inside ``apply_guardrail`` as well as
    returning masked texts.
    """

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        returned = await super().apply_guardrail(inputs, request_data, input_type, **kwargs)
        arguments = request_data.get("mcp_arguments") or {}
        masked = {key: self._mask(value) if isinstance(value, str) else value for key, value in arguments.items()}
        request_data["mcp_arguments"] = masked
        request_data["modified_arguments"] = masked
        return returned


@pytest.mark.asyncio
async def test_guardrail_that_masks_the_arguments_itself_is_not_treated_as_a_conflict():
    """Converging on the same replacement is not an unmergeable rewrite.

    A guardrail that both returns masked texts and rewrites the arguments in
    request_data must still mask, not be rejected as if a second guardrail had
    clobbered the leaf.
    """
    handler = MCPGuardrailTranslationHandler()
    guardrail = SelfWritingMaskingGuardrail()

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "contact jane.doe@example.com about the invoice"},
    }

    result = await handler.process_input_messages(data, guardrail)

    assert result["modified_arguments"] == {"query": "contact <EMAIL_ADDRESS> about the invoice"}


class ReshapingGuardrail(ArgumentMaskingGuardrail):
    """Masks through ``texts`` while moving the secret to a different path."""

    def __init__(self, reshaped: dict, **kwargs):
        super().__init__(**kwargs)
        self.reshaped = reshaped

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        returned = await super().apply_guardrail(inputs, request_data, input_type, **kwargs)
        request_data["mcp_arguments"] = self.reshaped
        return returned


@pytest.mark.asyncio
async def test_arguments_reshaped_under_the_guardrail_fail_closed():
    """A payload that no longer lines up leaf for leaf must block, not be written blind.

    Write-back pairs masked texts to leaves positionally, so a tree another guardrail
    reshaped would take the redaction on the wrong value.
    """
    handler = MCPGuardrailTranslationHandler()
    guardrail = ReshapingGuardrail({"query": "contact jane.doe@example.com", "note": "added"})

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "contact jane.doe@example.com about the invoice"},
    }

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_input_messages(data, guardrail)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_a_renamed_argument_key_blocks_rather_than_dropping_the_mask():
    """The leak this closes: same text, new path, so the write-back would find nothing.

    Matching purely on position would see an unchanged value and write the mask to a
    path that no longer exists, shipping the secret while reporting a clean scan.
    """
    handler = MCPGuardrailTranslationHandler()
    guardrail = ReshapingGuardrail({"renamed": "jane.doe@example.com", "other": "kept"})

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "jane.doe@example.com", "other": "kept"},
    }

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_input_messages(data, guardrail)

    assert exc_info.value.status_code == 400
    assert "jane.doe@example.com" not in str(data.get("modified_arguments"))


@pytest.mark.parametrize("run_in_parallel", [False, True])
@pytest.mark.asyncio
async def test_masked_arguments_reach_the_outbound_mcp_call(restore_callbacks, monkeypatch, run_in_parallel):
    """End to end over the real MCP pre-call path, not just the handler.

    Drives the same sequence mcp_server_manager.call_tool uses:
    synthetic payload -> pre_call_hook -> arguments sent upstream.

    Covers run_in_parallel both ways: that path shares one payload snapshot and
    discards whatever a guardrail returns, so the mask has to land on the caller's
    dict rather than on a copy of it.
    """
    guardrail = ArgumentMaskingGuardrail(
        event_hook="pre_mcp_call",
        default_on=True,
        run_in_parallel=run_in_parallel,
    )
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    arguments = {"query": "contact jane.doe@example.com about the invoice"}
    pre_hook_kwargs = {
        "name": "search",
        "arguments": arguments,
        "server_name": "test-server",
        "user_api_key_auth": UserAPIKeyAuth(api_key="sk-test", user_id="test-user"),
    }

    request_obj = proxy_logging_obj._create_mcp_request_object_from_kwargs(pre_hook_kwargs)
    synthetic_data = proxy_logging_obj._convert_mcp_to_llm_format(request_obj, pre_hook_kwargs)

    modified_data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=pre_hook_kwargs["user_api_key_auth"],
        data=synthetic_data,
        call_type="call_mcp_tool",
    )
    modified_kwargs = proxy_logging_obj._convert_mcp_hook_response_to_kwargs(modified_data, pre_hook_kwargs)

    assert modified_kwargs["arguments"] == {"query": "contact <EMAIL_ADDRESS> about the invoice"}


class SlowSubstitutionGuardrail(CustomGuardrail):
    """Rewrites one substring, after a delay, so two instances genuinely interleave."""

    def __init__(self, needle: str, replacement: str, delay: float, **kwargs):
        super().__init__(**kwargs)
        self.needle = needle
        self.replacement = replacement
        self.delay = delay

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        await asyncio.sleep(self.delay)
        inputs["texts"] = [text.replace(self.needle, self.replacement) for text in (inputs.get("texts") or [])]
        return inputs


def _two_interleaving_maskers(run_in_parallel: bool):
    return [
        SlowSubstitutionGuardrail(
            "jane.doe@example.com",
            "<EMAIL_ADDRESS>",
            0.02,
            guardrail_name="mask-email",
            event_hook="pre_mcp_call",
            default_on=True,
            run_in_parallel=run_in_parallel,
        ),
        SlowSubstitutionGuardrail(
            "415-555-0132",
            "<PHONE_NUMBER>",
            0.04,
            guardrail_name="mask-phone",
            event_hook="pre_mcp_call",
            default_on=True,
            run_in_parallel=run_in_parallel,
        ),
    ]


async def _arguments_sent_upstream(arguments: dict):
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    pre_hook_kwargs = {
        "name": "search",
        "arguments": arguments,
        "server_name": "test-server",
        "user_api_key_auth": UserAPIKeyAuth(api_key="sk-test", user_id="test-user"),
    }
    request_obj = proxy_logging_obj._create_mcp_request_object_from_kwargs(pre_hook_kwargs)
    modified_data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=pre_hook_kwargs["user_api_key_auth"],
        data=proxy_logging_obj._convert_mcp_to_llm_format(request_obj, pre_hook_kwargs),
        call_type="call_mcp_tool",
    )
    return proxy_logging_obj._convert_mcp_hook_response_to_kwargs(modified_data, pre_hook_kwargs)["arguments"]


@pytest.mark.asyncio
async def test_two_sequential_guardrails_both_masks_survive(restore_callbacks, monkeypatch):
    """The recommended config: each guardrail sees the previous one's output."""
    monkeypatch.setattr(litellm, "callbacks", _two_interleaving_maskers(run_in_parallel=False))

    sent = await _arguments_sent_upstream({"note": "mail jane.doe@example.com or call 415-555-0132"})

    assert sent == {"note": "mail <EMAIL_ADDRESS> or call <PHONE_NUMBER>"}


@pytest.mark.asyncio
async def test_two_parallel_guardrails_on_separate_arguments_both_masks_survive(restore_callbacks, monkeypatch):
    """Concurrent rewrites of different leaves compose; neither is lost."""
    monkeypatch.setattr(litellm, "callbacks", _two_interleaving_maskers(run_in_parallel=True))

    sent = await _arguments_sent_upstream({"email": "jane.doe@example.com", "phone": "415-555-0132"})

    assert sent == {"email": "<EMAIL_ADDRESS>", "phone": "<PHONE_NUMBER>"}


@pytest.mark.asyncio
async def test_two_parallel_guardrails_on_one_argument_block_instead_of_losing_a_mask(restore_callbacks, monkeypatch):
    """Unmergeable concurrent rewrites must fail closed, not ship one redaction.

    Both guardrails derive a full replacement string from the same snapshot, so
    writing either result would silently discard the other's redaction and leak
    the value it was configured to mask.
    """
    monkeypatch.setattr(litellm, "callbacks", _two_interleaving_maskers(run_in_parallel=True))
    original = "mail jane.doe@example.com or call 415-555-0132"

    with pytest.raises(HTTPException) as exc_info:
        await _arguments_sent_upstream({"note": original})

    assert exc_info.value.status_code == 400
    assert "note" in str(exc_info.value.detail)


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
