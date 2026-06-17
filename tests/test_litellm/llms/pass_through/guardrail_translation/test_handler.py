"""
Tests for LlmPassthroughRouteHandler and the guardrail_translation_mappings registry.

Validates:
- allm_passthrough_route is registered in the mappings (regression: this was the bug)
- Bedrock provider is dispatched to BedrockPassthroughGuardrailHandler
- Unknown provider skips apply_guardrail
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.llms.pass_through.guardrail_translation import (
    guardrail_translation_mappings,
)
from litellm.llms.pass_through.guardrail_translation.handler import (
    LlmPassthroughRouteHandler,
)
from litellm.types.utils import CallTypes


class TestRegistry:
    def test_allm_passthrough_route_registered(self):
        """Regression: missing this mapping was the root cause of the bug."""
        assert CallTypes.allm_passthrough_route in guardrail_translation_mappings

    def test_allm_passthrough_route_maps_to_llm_passthrough_route_handler(self):
        assert (
            guardrail_translation_mappings[CallTypes.allm_passthrough_route]
            is LlmPassthroughRouteHandler
        )

    def test_pass_through_still_registered(self):
        from litellm.llms.pass_through.guardrail_translation.handler import (
            PassThroughEndpointHandler,
        )

        assert (
            guardrail_translation_mappings[CallTypes.pass_through]
            is PassThroughEndpointHandler
        )


def _make_guardrail() -> MagicMock:
    g = MagicMock()
    g.guardrail_name = "test-guard"
    g.apply_guardrail = AsyncMock(return_value={"texts": []})
    g.skip_system_message_in_guardrail = False
    g.skip_tool_message_in_guardrail = False
    return g


class TestLlmPassthroughRouteHandlerInput:
    @pytest.mark.asyncio
    async def test_bedrock_provider_delegates_to_bedrock_handler(self):
        handler = LlmPassthroughRouteHandler()
        data = {
            "custom_llm_provider": "bedrock",
            "endpoint": "model/anthropic.claude-3-sonnet/converse",
            "data": {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        }
        guardrail = _make_guardrail()

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        guardrail.apply_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_provider_skips_apply_guardrail(self):
        handler = LlmPassthroughRouteHandler()
        data = {
            "custom_llm_provider": "some_unknown_provider",
            "endpoint": "v1/chat/completions",
            "data": {"messages": [{"role": "user", "content": "hi"}]},
        }
        guardrail = _make_guardrail()

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail
        )

        guardrail.apply_guardrail.assert_not_called()
        assert result is data

    @pytest.mark.asyncio
    async def test_missing_provider_skips(self):
        handler = LlmPassthroughRouteHandler()
        data = {"endpoint": "foo/bar", "data": {}}
        guardrail = _make_guardrail()

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail
        )

        guardrail.apply_guardrail.assert_not_called()
        assert result is data


class TestLlmPassthroughRouteHandlerOutput:
    @pytest.mark.asyncio
    async def test_bedrock_provider_delegates_output_to_bedrock_handler(self):
        handler = LlmPassthroughRouteHandler()
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello"}],
                }
            }
        }
        request_data = {
            "custom_llm_provider": "bedrock",
            "endpoint": "model/anthropic.claude-3-sonnet/converse",
        }
        guardrail = _make_guardrail()

        await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
            request_data=request_data,
        )

        guardrail.apply_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_provider_skips_output(self):
        handler = LlmPassthroughRouteHandler()
        response = {"some": "response"}
        request_data = {"custom_llm_provider": "unknown"}
        guardrail = _make_guardrail()

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
            request_data=request_data,
        )

        guardrail.apply_guardrail.assert_not_called()
        assert result is response


class TestDeAnonymizeEventStream:
    @pytest.mark.asyncio
    async def test_bedrock_provider_dispatches_to_handler(self):
        body = b"original-stream-bytes"
        expected = b"de-anonymized-bytes"
        proxy_logging_obj = MagicMock()
        user_api_key_dict = MagicMock()

        with patch(
            "litellm.llms.bedrock.passthrough.guardrail_translation.handler."
            "BedrockPassthroughGuardrailHandler.de_anonymize_event_stream",
            new=AsyncMock(return_value=expected),
        ) as mock_handler:
            result = await LlmPassthroughRouteHandler.de_anonymize_event_stream(
                body_bytes=body,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                data={"custom_llm_provider": "bedrock"},
            )

        mock_handler.assert_awaited_once()
        assert result == expected

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_original_bytes(self):
        body = b"original-stream-bytes"

        result = await LlmPassthroughRouteHandler.de_anonymize_event_stream(
            body_bytes=body,
            proxy_logging_obj=MagicMock(),
            user_api_key_dict=MagicMock(),
            data={"custom_llm_provider": "anthropic"},
        )

        assert result is body

    @pytest.mark.asyncio
    async def test_missing_provider_returns_original_bytes(self):
        body = b"original-stream-bytes"

        result = await LlmPassthroughRouteHandler.de_anonymize_event_stream(
            body_bytes=body,
            proxy_logging_obj=MagicMock(),
            user_api_key_dict=MagicMock(),
            data={},
        )

        assert result is body


class TestSupportsEventStreamDeAnonymization:
    def test_bedrock_converse_stream_is_supported(self):
        assert (
            LlmPassthroughRouteHandler.supports_event_stream_de_anonymization(
                "bedrock", "model/us.amazon.nova-lite-v1:0/converse-stream"
            )
            is True
        )

    def test_bedrock_invoke_stream_is_not_supported(self):
        assert (
            LlmPassthroughRouteHandler.supports_event_stream_de_anonymization(
                "bedrock",
                "model/us.amazon.nova-lite-v1:0/invoke-with-response-stream",
            )
            is False
        )

    def test_unknown_provider_is_not_supported(self):
        assert (
            LlmPassthroughRouteHandler.supports_event_stream_de_anonymization(
                "anthropic", "model/foo/converse-stream"
            )
            is False
        )

    def test_missing_provider_is_not_supported(self):
        assert (
            LlmPassthroughRouteHandler.supports_event_stream_de_anonymization(
                None, "model/foo/converse-stream"
            )
            is False
        )
