"""Tests for unified guardrail."""

import pytest

from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import unified_guardrail as unified_module
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

    def should_run_guardrail(self, data, event_type):  # type: ignore[override]
        self.event_history.append(event_type)
        return True

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
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
