"""Tests for unified guardrail."""

import pytest

from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import unified_guardrail as unified_module
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes


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
