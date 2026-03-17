"""Tests for unified guardrail."""

import pytest

from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse
from litellm.llms.mistral.ocr.guardrail_translation.handler import OCRHandler
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
    unified_guardrail as unified_module,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, Delta, ModelResponseStream, StreamingChoices


class RecordingGuardrail(CustomGuardrail):
    """Records the event types it is asked to run for."""

    def __init__(self):
        super().__init__(guardrail_name="recording-guardrail")
        self.event_history = []
        self.apply_calls = []

    def should_run_guardrail(self, data, event_type):  # type: ignore[override]
        self.event_history.append(event_type)
        return True

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.apply_calls.append({"inputs": inputs, "input_type": input_type})
        return {"texts": inputs.get("texts", [])}


class _NoopTranslation(BaseTranslation):
    """Test translation handler that simply echoes input/output."""

    async def process_input_messages(self, data, guardrail_to_apply, litellm_logging_obj=None):  # type: ignore[override]
        return data

    async def process_output_response(  # type: ignore[override]
        self,
        response,
        guardrail_to_apply,
        litellm_logging_obj=None,
        user_api_key_dict=None,
    ):
        return response


@pytest.fixture(autouse=True)
def _inject_mcp_handler_mapping():
    """Inject MCP handler mapping so the unified guardrail can run inside tests."""
    unified_module.endpoint_guardrail_translation_mappings = {
        CallTypes.call_mcp_tool: MCPGuardrailTranslationHandler,
        CallTypes.anthropic_messages: _NoopTranslation,
        CallTypes.ocr: OCRHandler,
        CallTypes.aocr: OCRHandler,
    }
    yield
    unified_module.endpoint_guardrail_translation_mappings = None


class TestUnifiedLLMGuardrails:
    class TestAsyncPreCallHook:
        @pytest.mark.asyncio
        async def test_uses_mcp_event_type(self):
            """pre_call hook should swap to GuardrailEventHooks.pre_mcp_call for MCP calls."""
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()
            cache = DualCache()

            data = {
                "guardrail_to_apply": guardrail,
                "messages": [
                    {"role": "user", "content": "Tool: test\nArguments: {}"}
                ],
                "model": "mcp-tool-call",
            }

            await handler.async_pre_call_hook(
                user_api_key_dict=None,
                cache=cache,
                data=data,
                call_type=CallTypes.call_mcp_tool.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.pre_mcp_call]

    class TestAsyncModerationHook:
        @pytest.mark.asyncio
        async def test_uses_mcp_event_type(self):
            """moderation hook should request GuardrailEventHooks.during_mcp_call for MCP calls."""
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            data = {
                "guardrail_to_apply": guardrail,
                "messages": [
                    {"role": "user", "content": "Tool: test\nArguments: {}"}
                ],
                "model": "mcp-tool-call",
            }

            await handler.async_moderation_hook(
                data=data,
                user_api_key_dict=None,
                call_type=CallTypes.call_mcp_tool.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.during_mcp_call]

        @pytest.mark.asyncio
        async def test_runs_for_anthropic_messages(self):
            """Ensure anthropic_messages requests still trigger guardrail moderation."""
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            data = {
                "guardrail_to_apply": guardrail,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello Anthropics",
                    }
                ],
                "model": "anthropic.claude-3",
            }

            await handler.async_moderation_hook(
                data=data,
                user_api_key_dict=None,
                call_type=CallTypes.anthropic_messages.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.during_call]

    class TestAsyncPostCallStreamingIteratorHook:
        @pytest.mark.asyncio
        async def test_streaming_content_not_lost_on_sampled_chunks(self):
            """
            Verify that every chunk's content is preserved in the output stream.

            The bug: process_output_streaming_response puts the combined
            guardrailed text in the first chunk and clears all subsequent
            chunks to "". The hook then yielded processed_items[-1] (the
            cleared last item), permanently losing every Nth chunk's content.
            """

            class _ContentClearingTranslation(BaseTranslation):
                """Simulates the real OpenAI handler behavior that triggers the bug."""

                async def process_input_messages(self, data, guardrail_to_apply, litellm_logging_obj=None):  # type: ignore[override]
                    return data

                async def process_output_response(self, response, guardrail_to_apply, litellm_logging_obj=None, user_api_key_dict=None):  # type: ignore[override]
                    return response

                async def process_output_streaming_response(
                    self,
                    responses_so_far,
                    guardrail_to_apply,
                    litellm_logging_obj=None,
                    user_api_key_dict=None,
                ):
                    # Simulate what the real handler does:
                    # put combined text in first chunk, clear the rest
                    combined = ""
                    for resp in responses_so_far:
                        for choice in resp.choices:
                            if choice.delta and choice.delta.content:
                                combined += choice.delta.content

                    first_set = False
                    for resp in responses_so_far:
                        for choice in resp.choices:
                            if not first_set:
                                choice.delta.content = combined
                                first_set = True
                            else:
                                choice.delta.content = ""

                    return responses_so_far

            # Override the mapping to use our content-clearing translation
            unified_module.endpoint_guardrail_translation_mappings = {
                CallTypes.acompletion: _ContentClearingTranslation,
            }

            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            # Create 10 streaming chunks with distinct content
            chunks = []
            for i in range(10):
                chunk = ModelResponseStream(
                    choices=[StreamingChoices(
                        delta=Delta(content=f"word{i} ", role="assistant"),
                        finish_reason=None,
                    )],
                )
                chunks.append(chunk)

            async def mock_stream():
                for chunk in chunks:
                    yield chunk

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                request_route="/v1/chat/completions",
            )

            request_data = {
                "guardrail_to_apply": guardrail,
                "model": "gpt-4",
            }

            # Collect all yielded chunks
            yielded_contents = []
            async for item in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data,
            ):
                content = item.choices[0].delta.content if item.choices[0].delta else None
                yielded_contents.append(content)

            # Every chunk should have non-empty content
            for i, content in enumerate(yielded_contents):
                assert content is not None and content != "", (
                    f"Chunk {i} lost its content (got {content!r}). "
                    f"Expected non-empty content for every streamed chunk."
                )

    class TestOCRGuardrailE2E:
        """End-to-end tests: UnifiedLLMGuardrails -> OCRHandler."""

        @pytest.mark.asyncio
        async def test_pre_call_hook_invokes_ocr_handler_for_input(self):
            """
            Verify that async_pre_call_hook with call_type=aocr routes through
            the OCR handler and calls apply_guardrail with the document URL.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()
            cache = DualCache()

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": "https://arxiv.org/pdf/2201.04234",
                },
            }

            result = await handler.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
                cache=cache,
                data=data,
                call_type=CallTypes.aocr.value,
            )

            # Guardrail should have been checked and invoked
            assert guardrail.event_history == [GuardrailEventHooks.pre_call]
            assert len(guardrail.apply_calls) == 1
            assert guardrail.apply_calls[0]["input_type"] == "request"
            assert "https://arxiv.org/pdf/2201.04234" in guardrail.apply_calls[0]["inputs"]["texts"]

            # Data should be returned with document intact
            assert result["document"]["document_url"] == "https://arxiv.org/pdf/2201.04234"

        @pytest.mark.asyncio
        async def test_moderation_hook_invokes_ocr_handler(self):
            """
            Verify that async_moderation_hook with call_type=aocr routes through
            the OCR handler correctly.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
                "document": {
                    "type": "image_url",
                    "image_url": "https://example.com/scan.png",
                },
            }

            await handler.async_moderation_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
                call_type=CallTypes.aocr.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.during_call]
            assert len(guardrail.apply_calls) == 1
            assert "https://example.com/scan.png" in guardrail.apply_calls[0]["inputs"]["texts"]

        @pytest.mark.asyncio
        async def test_post_call_success_hook_guardrails_ocr_output(self):
            """
            Verify that async_post_call_success_hook resolves the OCR route
            to the OCR handler and applies guardrails to page markdown.
            """

            class TextModifyingGuardrail(CustomGuardrail):
                def __init__(self):
                    super().__init__(guardrail_name="text-modifier")

                def should_run_guardrail(self, data, event_type):  # type: ignore[override]
                    return True

                async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
                    texts = inputs.get("texts", [])
                    return {"texts": [t.replace("SECRET", "[REDACTED]") for t in texts]}

            handler = UnifiedLLMGuardrails()
            guardrail = TextModifyingGuardrail()

            ocr_response = OCRResponse(
                pages=[
                    OCRPage(index=0, markdown="Page 1 has a SECRET value"),
                    OCRPage(index=1, markdown="Page 2 is clean"),
                    OCRPage(index=2, markdown="Page 3 also has SECRET data"),
                ],
                model="mistral/mistral-ocr-latest",
            )

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                request_route="/v1/ocr",
            )

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
            }

            result = await handler.async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=ocr_response,
            )

            # Verify the SECRET text was redacted across pages
            assert result.pages[0].markdown == "Page 1 has a [REDACTED] value"
            assert result.pages[1].markdown == "Page 2 is clean"
            assert result.pages[2].markdown == "Page 3 also has [REDACTED] data"

        @pytest.mark.asyncio
        async def test_post_call_success_hook_ocr_route_resolves_call_type(self):
            """
            Verify that request_route=/v1/ocr correctly resolves to the OCR
            call type and the handler is invoked (not skipped).
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            ocr_response = OCRResponse(
                pages=[OCRPage(index=0, markdown="Some text")],
                model="mistral/mistral-ocr-latest",
            )

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                request_route="/v1/ocr",
            )

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
            }

            result = await handler.async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=ocr_response,
            )

            # Guardrail was invoked
            assert guardrail.event_history == [GuardrailEventHooks.post_call]
            assert len(guardrail.apply_calls) == 1
            assert guardrail.apply_calls[0]["input_type"] == "response"
            assert guardrail.apply_calls[0]["inputs"]["texts"] == ["Some text"]

            # Response returned with pages intact
            assert result.pages[0].markdown == "Some text"
