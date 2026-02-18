"""
Tests for the Block Hosted Tools Guardrail.
"""

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_hooks.block_hosted_tools.guardrail import \
    BlockHostedToolsGuardrail


class TestBlockHostedToolsGuardrail:
    """Test BlockHostedToolsGuardrail."""

    def test_initialization(self):
        """Test that guardrail initializes with pre_call hook."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        assert guardrail.guardrail_name == "test-block-hosted-tools"
        assert "pre_call" in str(guardrail.supported_event_hooks)

    @pytest.mark.asyncio
    async def test_blocks_anthropic_bash_tool(self):
        """Test that Anthropic bash_* hosted tool is blocked."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        tools = [{"type": "bash_20250124", "name": "run_bash"}]
        inputs = {"tools": tools}
        request_data = {}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

        assert exc_info.value.status_code == 403
        assert "bash" in str(exc_info.value.detail).lower()
        assert "disabled" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_blocks_openai_code_interpreter(self):
        """Test that OpenAI code_interpreter hosted tool is blocked."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        tools = [{"type": "code_interpreter"}]
        inputs = {"tools": tools}
        request_data = {}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

        assert exc_info.value.status_code == 403
        assert "disabled" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_blocks_gemini_google_search_top_level_key(self):
        """Test that Gemini native googleSearch top-level key is blocked."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        tools = [{"googleSearch": {}}]
        inputs = {"tools": tools}
        request_data = {}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

        assert exc_info.value.status_code == 403
        assert "disabled" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_allows_function_tools(self):
        """Test that user-defined function tools pass through."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                },
            }
        ]
        inputs = {"tools": tools}
        request_data = {}

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )
        assert result == inputs

    @pytest.mark.asyncio
    async def test_allows_empty_tools(self):
        """Test that empty tools list passes through."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        inputs = {"tools": []}
        request_data = {}

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )
        assert result == inputs

    @pytest.mark.asyncio
    async def test_skips_response_input_type(self):
        """Test that response input_type is skipped."""
        guardrail = BlockHostedToolsGuardrail(
            guardrail_name="test-block-hosted-tools"
        )
        tools = [{"type": "bash_20250124"}]
        inputs = {"tools": tools}
        request_data = {}

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
