"""
Tests for LlmPassthroughRouteHandler.

Validates fix for issue #37638:
  Guardrail mode=block silently behaves as redact on provider-native
  passthrough routes (e.g. Gemini generateContent) — only bedrock was wired up.

Before the fix, process_input_messages/process_output_response for any
provider other than "bedrock" would return early (skipping the guardrail).
After the fix they delegate to PassThroughEndpointHandler (the same generic
fallback that Bedrock itself uses for non-Converse routes), so blocking
guardrails are honoured on ALL passthrough providers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.llms.pass_through.guardrail_translation.handler import (
    LlmPassthroughRouteHandler,
)


class GuardrailBlocked(Exception):
    """Stand-in for a guardrail blocking a request; handler must let it propagate."""


def _make_guardrail(apply_result: dict) -> MagicMock:
    """Helper: build a minimal CustomGuardrail mock."""
    g = MagicMock()
    g.guardrail_name = "test-guard"
    g.apply_guardrail = AsyncMock(return_value=apply_result)
    return g


def _make_blocking_guardrail() -> MagicMock:
    """Helper: build a guardrail whose apply_guardrail raises GuardrailBlocked."""
    g = MagicMock()
    g.guardrail_name = "blocking-guard"
    g.apply_guardrail = AsyncMock(side_effect=GuardrailBlocked("blocked by guardrail"))
    return g


def _gemini_data(text: str = "Hello, my key is sk-ant-api03-secret") -> dict:
    """Minimal Gemini generateContent passthrough payload."""
    return {
        "custom_llm_provider": "gemini",
        "model": "gemini-2.5-flash",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": text}],
            }
        ],
    }


def _vertex_data(text: str = "Hello world") -> dict:
    """Minimal Vertex AI passthrough payload."""
    return {
        "custom_llm_provider": "vertex_ai",
        "model": "gemini-1.5-pro",
        "contents": [{"role": "user", "parts": [{"text": text}]}],
    }


# ---------------------------------------------------------------------------
# process_input_messages
# ---------------------------------------------------------------------------


class TestLlmPassthroughRouteHandlerInput:
    """
    Verify that process_input_messages applies the guardrail for every
    non-bedrock passthrough provider (fix for #37638).
    """

    @pytest.mark.asyncio
    async def test_gemini_provider_calls_guardrail(self):
        """
        Regression test for #37638:
        Before the fix, the guardrail was silently skipped for Gemini routes.
        After the fix, apply_guardrail must be called at least once.
        """
        data = _gemini_data()
        guardrail = _make_guardrail({"texts": ["Hello, my key is [REDACTED]"]})

        handler = LlmPassthroughRouteHandler()
        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
        )

        # The guardrail should have been invoked — not silently skipped.
        guardrail.apply_guardrail.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_vertex_ai_provider_calls_guardrail(self):
        """Same regression check for vertex_ai provider."""
        data = _vertex_data()
        guardrail = _make_guardrail({"texts": ["Hello world"]})

        handler = LlmPassthroughRouteHandler()
        await handler.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
        )

        guardrail.apply_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_gemini_blocking_guardrail_raises(self):
        """
        A guardrail configured with mode=block must raise (not silently pass)
        when sensitive content is detected on a Gemini passthrough route.

        This is the core failure described in #37638: before the fix the request
        succeeded with HTTP 200 instead of raising an error.
        """
        data = _gemini_data(text="My API key is sk-ant-api03-abcdefghijklmnop")
        guardrail = _make_blocking_guardrail()

        handler = LlmPassthroughRouteHandler()
        with pytest.raises(GuardrailBlocked):
            await handler.process_input_messages(
                data=data,
                guardrail_to_apply=guardrail,
            )

    @pytest.mark.asyncio
    async def test_vertex_blocking_guardrail_raises(self):
        """Same block-mode check for vertex_ai provider."""
        data = _vertex_data(text="secret token: sk-ant-api03-abcdefghijklmnop")
        guardrail = _make_blocking_guardrail()

        handler = LlmPassthroughRouteHandler()
        with pytest.raises(GuardrailBlocked):
            await handler.process_input_messages(
                data=data,
                guardrail_to_apply=guardrail,
            )

    @pytest.mark.asyncio
    async def test_unknown_provider_calls_guardrail(self):
        """
        An unrecognised provider should also fall through to the generic handler
        rather than silently skip the guardrail.
        """
        data = {
            "custom_llm_provider": "some_new_provider",
            "model": "my-model",
            "messages": [{"role": "user", "content": "check this"}],
        }
        guardrail = _make_guardrail({"texts": ["check this"]})

        handler = LlmPassthroughRouteHandler()
        await handler.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
        )

        guardrail.apply_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_bedrock_provider_still_works(self):
        """
        Bedrock must continue to use its dedicated handler; this test confirms
        the generic fallback doesn't interfere with bedrock dispatch.
        """
        data = {
            "custom_llm_provider": "bedrock",
            "model": "anthropic.claude-3-sonnet",
            "endpoint": "model/anthropic.claude-3-sonnet/invoke",
            "data": {"prompt": "hello"},
        }
        guardrail = _make_guardrail({"texts": ["hello"]})

        handler = LlmPassthroughRouteHandler()
        # Should not raise; bedrock invoke routes use the generic fallback
        # inside BedrockPassthroughGuardrailHandler._generic_passthrough_handler.
        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# process_output_response
# ---------------------------------------------------------------------------


class TestLlmPassthroughRouteHandlerOutput:
    """
    Verify that process_output_response applies the guardrail for every
    non-bedrock passthrough provider (fix for #37638).
    """

    @pytest.mark.asyncio
    async def test_gemini_provider_output_calls_guardrail(self):
        """Guardrail must be called on Gemini response payloads."""
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "safe reply"}], "role": "model"}}
            ]
        }
        request_data = _gemini_data()
        guardrail = _make_guardrail({"texts": ["safe reply"]})

        handler = LlmPassthroughRouteHandler()
        await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
            request_data=request_data,
        )

        guardrail.apply_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_gemini_blocking_output_guardrail_raises(self):
        """A blocking output guardrail must raise for Gemini responses."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "secret: sk-ant-api03-abc"}],
                        "role": "model",
                    }
                }
            ]
        }
        request_data = _gemini_data()
        guardrail = _make_blocking_guardrail()

        handler = LlmPassthroughRouteHandler()
        with pytest.raises(GuardrailBlocked):
            await handler.process_output_response(
                response=response,
                guardrail_to_apply=guardrail,
                request_data=request_data,
            )

    @pytest.mark.asyncio
    async def test_non_dict_response_is_returned_unchanged(self):
        """
        Non-dict responses (e.g. raw bytes or strings) should pass through
        without error, matching the PassThroughEndpointHandler behaviour.
        """
        response = b"raw bytes response"
        request_data = _gemini_data()
        guardrail = _make_guardrail({})

        handler = LlmPassthroughRouteHandler()
        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
            request_data=request_data,
        )

        assert result == response
        guardrail.apply_guardrail.assert_not_called()
