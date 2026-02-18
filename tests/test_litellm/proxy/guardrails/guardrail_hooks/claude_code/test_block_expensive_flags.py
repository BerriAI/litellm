"""
Tests for the Claude Code Block Expensive Flags Guardrail.
"""

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_hooks.claude_code.block_expensive_flags import (
    ClaudeCodeBlockExpensiveFlagsGuardrail,
)


class TestClaudeCodeBlockExpensiveFlagsGuardrail:
    """Test ClaudeCodeBlockExpensiveFlagsGuardrail."""

    def test_initialization(self):
        """Test that guardrail initializes with pre_call hook."""
        guardrail = ClaudeCodeBlockExpensiveFlagsGuardrail(
            guardrail_name="test-block-expensive"
        )
        assert guardrail.guardrail_name == "test-block-expensive"
        assert "pre_call" in str(guardrail.supported_event_hooks)

    @pytest.mark.asyncio
    async def test_blocks_speed_fast(self):
        """Test that speed=fast is blocked."""
        guardrail = ClaudeCodeBlockExpensiveFlagsGuardrail(
            guardrail_name="test-block-expensive"
        )
        request_data = {"speed": "fast"}
        inputs = {"tools": None}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

        assert exc_info.value.status_code == 403
        assert "fast" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_blocks_inference_geo(self):
        """Test that inference_geo is blocked."""
        guardrail = ClaudeCodeBlockExpensiveFlagsGuardrail(
            guardrail_name="test-block-expensive"
        )
        request_data = {"inference_geo": "us"}
        inputs = {"tools": None}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

        assert exc_info.value.status_code == 403
        assert "inference" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_blocks_thinking_type_enabled(self):
        """Test that thinking.type=enabled is blocked."""
        guardrail = ClaudeCodeBlockExpensiveFlagsGuardrail(
            guardrail_name="test-block-expensive"
        )
        request_data = {"thinking": {"type": "enabled"}}
        inputs = {"tools": None}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

        assert exc_info.value.status_code == 403
        assert "thinking" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_allows_clean_request(self):
        """Test that requests without expensive flags pass through."""
        guardrail = ClaudeCodeBlockExpensiveFlagsGuardrail(
            guardrail_name="test-block-expensive"
        )
        request_data = {"model": "claude-3-5-sonnet"}
        inputs = {"tools": None}

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )
        assert result == inputs

    @pytest.mark.asyncio
    async def test_skips_response_input_type(self):
        """Test that response input_type is skipped."""
        guardrail = ClaudeCodeBlockExpensiveFlagsGuardrail(
            guardrail_name="test-block-expensive"
        )
        request_data = {"speed": "fast"}
        inputs = {"tools": None}

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
        assert result == inputs

    @pytest.mark.asyncio
    async def test_http_403_classified_as_guardrail_intervention(self):
        """Test that HTTP 403 from guardrail is classified as guardrail_intervened."""
        assert CustomGuardrail._is_guardrail_intervention(
            HTTPException(status_code=403, detail="blocked")
        ) is True
